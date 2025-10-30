import os, json
import numpy as np
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import math, numpy as np, torch
from collections import defaultdict

# ======================= Config =======================
SEED = 12345
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS   = 120
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.0
WEIGHT_DECAY = 0.0

K_WINDOW     = 22          # sliding window for FIE/GIE
FLIP_SEED    = 777         # which cats get flipped to dog
N_CATS_FLIP  = 0         # exactly 100 cats → dog

EPS = 1e-12

USE_CKL = True   # <— set to True to enable CKL/Bayes-GIE; False = plain CE baseline

# =================== Determinism ======================
def set_all_seeds(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    import random as pyrand
    pyrand.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# ---------- CKL (finite-boundary, ID form) and CKL→alpha ----------

def _ckl_equal_W(W, d1, d2):
    return W * ((d2 - d1) ** 2) / (((d1 + 1.0) ** 2) * (d2 + 1.0) + EPS)

def _ckl_case_A(W1, d1, W2, d2):
    term = ( W1 * ((d2/(d1+1.0)) * math.log(max(W2/W1, EPS)) - (d1 - d2)/((d1+1.0)**2))
           + d2 * ((W2 - W1) + W1 * math.log(max(W1/W2, EPS))) )
    return term + (d1/(d1+1.0))*W1 - (d2/(d2+1.0))*W2

def _ckl_case_B(W1, d1, W2, d2):
    ratio = max(W2/W1, EPS)
    term = (W1 / ((d1 + 1.0)**2)) * ( d2 * (ratio**(d1 + 1.0)) - d1 )
    return term + (d1/(d1+1.0))*W1 - (d2/(d2+1.0))*W2

def ckl_finite(W1, d1, W2, d2):
    if not all(np.isfinite([W1,W2,d1,d2])) or min(W1,W2,d1,d2) <= 0:
        return np.nan
    if abs(W1 - W2) < 1e-12:
        return _ckl_equal_W(W1, d1, d2)
    return _ckl_case_A(W1, d1, W2, d2) if W1 < W2 else _ckl_case_B(W1, d1, W2, d2)

def ckl_to_alpha(scores_np, thr, kappa=2.0, alpha_floor=0.05):
    scores = np.asarray(scores_np, dtype=np.float64)
    rel = np.maximum((scores - thr) / (thr + EPS), 0.0)
    a = np.exp(-kappa * rel)
    return np.clip(a, alpha_floor, 1.0)

class EpochCklTracker:
    """Online union-reference + streaming threshold + run-gate, reset each epoch."""
    def __init__(self, thr_mode="mean", min_run=5):
        assert thr_mode in ("mean","median")
        self.thr_mode = thr_mode; self.min_run = min_run
        self.reset()
    def reset(self):
        self.logW_sum = 0.0; self.logd_sum = 0.0; self.ref_cnt = 0
        self._thr_n = 0; self._thr_s = 0.0; self.ckl_buf = []
        self.curr_thr = 0.0; self.runlen = defaultdict(int)
    def current_ref(self):
        if self.ref_cnt == 0: return None
        return math.exp(self.logW_sum/self.ref_cnt), math.exp(self.logd_sum/self.ref_cnt)
    def current_thr(self):
        if self.thr_mode == "mean": return float(self.curr_thr)
        return float(np.nanmedian(np.asarray(self.ckl_buf))) if self.ckl_buf else float(self.curr_thr)
    def update_ref(self, W_batch, d_batch):
        self.logW_sum += float(np.log(np.maximum(W_batch, EPS)).sum())
        self.logd_sum += float(np.log(np.maximum(d_batch, EPS)).sum())
        self.ref_cnt  += int(len(W_batch))
    def update_thr(self, ckl_vals):
        if self.thr_mode == "mean":
            self._thr_n += len(ckl_vals); self._thr_s += float(np.nansum(ckl_vals))
            self.curr_thr = self._thr_s / max(self._thr_n, 1)
        else:
            self.ckl_buf.extend(list(ckl_vals))
    def update_gates(self, sample_ids, raw_flags):
        gated = []
        for sid, raw in zip(sample_ids, raw_flags):
            self.runlen[sid] = (self.runlen[sid] + 1) if raw else 0
            gated.append(self.runlen[sid] >= self.min_run)
        return np.array(gated, dtype=bool)

# =================== Model (ResNet32) =================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.penultimate = None
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        self.penultimate = out
        return self.linear(out)

def ResNet32(): return ResNet(BasicBlock, [5,5,5])

# =================== Controlled noise =================
class ControlledCatDogNoise(Dataset):
    """
    Wrap CIFAR10 train set.
    - Flip exactly N_CATS_FLIP items with true class 'cat' → label 'dog'.
    - Keep all others unchanged.
    - Exposes:
        .noisy_mask[idx]   -> True if this idx is flipped cat→dog
        .group_map[idx]    -> 'cat' (clean cat), 'dog' (clean dog), 'noisy' (flipped cat), or 'other'
    """
    def __init__(self, base_dataset, n_flip=100, seed=777):
        self.base = base_dataset
        self.n_flip = int(n_flip)

        # resolve class indices robustly
        cls2idx = self.base.class_to_idx
        self.cat_idx = int(cls2idx['cat'])
        self.dog_idx = int(cls2idx['dog'])

        # original labels
        self.labels = [int(lbl) for _, lbl in self.base]

        # all cat/dog indices
        self.cat_ids = [i for i,l in enumerate(self.labels) if l == self.cat_idx]
        self.dog_ids = [i for i,l in enumerate(self.labels) if l == self.dog_idx]

        rng = np.random.default_rng(seed)
        if self.n_flip > len(self.cat_ids):
            raise ValueError(f"Requested {self.n_flip} flips but only {len(self.cat_ids)} cats exist.")
        self.flipped_ids = sorted(rng.choice(self.cat_ids, size=self.n_flip, replace=False).tolist())

        # build noisy labels
        self.noisy_labels = list(self.labels)
        for i in self.flipped_ids:
            self.noisy_labels[i] = self.dog_idx

        # masks & groups
        n = len(self.labels)
        self.noisy_mask = np.zeros(n, dtype=bool)
        self.noisy_mask[self.flipped_ids] = True

        self.group_map = {}
        flipped_set = set(self.flipped_ids)
        for i in range(n):
            if i in flipped_set:
                self.group_map[i] = 'noisy'       # cat flipped to dog
            elif self.labels[i] == self.cat_idx:
                self.group_map[i] = 'cat'         # clean cat
            elif self.labels[i] == self.dog_idx:
                self.group_map[i] = 'dog'         # clean dog
            else:
                self.group_map[i] = 'other'       # other classes

        print(f"[Noise] Flipped exactly {self.n_flip} cats → dog.")

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, _true = self.base[idx]
        label = int(self.noisy_labels[idx])
        is_noisy = bool(self.noisy_mask[idx])
        return x, label, int(idx), is_noisy

# =================== LID Estimators ===================
    
class LIDEstimators:
    def __init__(self, device='cpu'):
        self.device = device

    @staticmethod
    def _hill(V, w):
        V = np.asarray(V, float)
        V = V[np.isfinite(V)]
        if V.size == 0: return 0.0
        w = max(float(abs(w)), EPS)
        m = np.abs(V) > EPS
        Vn = V[m]
        if Vn.size < 2: return 0.0
        k = Vn.size - 1
        denom = np.sum(np.log(np.abs(Vn / w)))
        if not np.isfinite(denom) or abs(denom) < 1e-30: return 0.0
        return float(-k / denom)

    def compute_GIE_LID(self, phi, G):
        epsilon = 1e-7
    
        # --- Compute deviations ---
        limit0 = np.mean(phi[-3:])
        R = np.abs(phi - limit0)
        w0 = np.max(R)
    
        limit1 = np.mean(G[-3:])
        FR = np.abs(G - limit1)
        w1 = np.max(FR)
    
        # --- Paired filtering: keep only indices where both are non-zero ---
        mask = (R != 0) & (FR != 0)
        R_non_zero = R[mask]
        FR_non_zero = FR[mask]
        
        Wmax = w0#max(w0, w1)
        # --- Number of samples for Hill ---
        k = R_non_zero.shape[0] - 1
        if k <= 4:
            return EPS, float(Wmax)
    
        # --- Hill estimates ---
        hill_num = - (k / np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon)))))
        hill_den = - (k / np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon)))))
    
        gie = hill_num / hill_den if hill_den != 0 else np.nan
        return float(gie), float(Wmax)
    
    def compute_Bayes_GIE(self, phi, G, Num0, Den0):
        epsilon = 1e-7
    
        # --- Compute deviations ---
        limit0 = np.mean(phi[-3:])
        R = np.abs(phi - limit0)
        w0 = np.max(R)
    
        limit1 = np.mean(G[-3:])
        FR = np.abs(G - limit1)
        w1 = np.max(FR)
    
        # --- Paired filtering: remove any index where either deviation is zero ---
        mask = (R > EPS) & (FR > EPS)
        R_non_zero = R[mask]
        FR_non_zero = FR[mask]
    
        # --- k value ---
        k = R_non_zero.shape[0] - 1
        if k <= 4:
            return EPS, float(w0), 0.0, 0.0  # No valid samples → return NaNs and zero increments
    
        # --- Hill estimates ---
        hill_num = - (k / np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon)))))
        hill_den = - (k / np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon)))))
    
        # --- Check validity ---
        if hill_num == 0 or hill_den == 0 or np.isnan(hill_num) or np.isnan(hill_den):
            Num1, Den1 = 0.0, 0.0
        else:
            Num1 = 1.0 / hill_den   # Denominator's Hill goes in numerator's sum
            Den1 = 1.0 / hill_num   # Numerator's Hill goes in denominator's sum
    
        # --- Update cumulative sums ---
        Num_cumulative = Num0 + Num1
        Den_cumulative = Den0 + Den1
    
        # --- Compute Bayesian GIE ---
        LID_Bayes = Num_cumulative / Den_cumulative if Den_cumulative != 0 else EPS
    
        return LID_Bayes, float(w0), Num1, Den1

    def compute_FIE_LID(self, phi_prev, phi_next):
        phi_prev = np.asarray(phi_prev, float)
        phi_next = np.asarray(phi_next, float)
        lim = np.mean(phi_prev[-3:]) if phi_prev.size >= 3 else np.mean(phi_prev)
        R  = np.abs(phi_prev - lim); w0 = max(float(np.max(R)), EPS)
        FR = np.abs(phi_next - lim); w1 = max(float(np.max(FR)), EPS)
        hn = self._hill(R, w0); hd = self._hill(FR, w1)
        fie = 0.0 if hd == 0.0 else float(hn/hd)
        return float(fie), float(w0)

# =================== Utils ============================
def save_pairs_to_file(pairs_dict, filename):
    serializable = {str(idx): [[float(w), float(val)] for (w, val) in pairs]
                    for idx, pairs in pairs_dict.items()}
    with open(filename, "w") as fp:
        json.dump(serializable, fp, indent=2)

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    model.train()
    return 100 * correct / max(total, 1)

def get_phi_for_indices(distance_queues, indices, k_required):
    """
    Collect per-sample phi (recent losses) for each index.
    Returns: list[ndarray or None]
    """
    phis = []
    for idx in indices:
        dq = distance_queues.get(int(idx), None)
        if dq is None or len(dq) < k_required:
            phis.append(None)
        else:
            vals = [v for (v) in dq]  # dq stores raw losses (floats)
            phis.append(np.asarray(vals, dtype=float))
    return phis

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, num_epochs, k, device):
    """
    If USE_CKL is False: trains with plain CE (baseline).
    If USE_CKL is True : trains with online Bayes-GIE -> CKL -> alpha_i -> D2L soft targets.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    per_sample_ce = nn.CrossEntropyLoss(reduction='none')
    global_ce     = nn.CrossEntropyLoss()

    loss_history, test_loss_history = [], []

    # ---------- CKL/Bayes-GIE state (only if enabled) ----------
    if USE_CKL:
        lid = LIDEstimators(device=device)
        # per-sample sliding windows of recent CE (phi)
        distance_queues: dict[int, deque] = {}  # idx -> deque[float] of length k
        # (optional bookkeeping; latest increments only in overwrite mode)
        cum_num_bgie: dict[int, float] = defaultdict(float)
        cum_den_bgie: dict[int, float] = defaultdict(float)
        # CKL gate run-length persists across epochs
        runlen_global: dict[int, int] = defaultdict(int)
        # Global training-loss reference G (length k)
        G_global = deque(maxlen=k)

    for epoch in range(num_epochs):
        model.train()
        running_loss_sum, running_count = 0.0, 0

        # CKL tracker: reset reference/threshold per epoch; persist run-lengths
        if USE_CKL:
            tracker = EpochCklTracker(thr_mode="mean", min_run=5)
            tracker.runlen = runlen_global

        for batch_idx, (inputs, labels, indices, is_noisy) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            indices_np = indices.cpu().numpy()

            # forward
            logits = model(inputs)

            # ===== Baseline: no CKL/Bayes-GIE =====
            if not USE_CKL:
                loss = global_ce(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss_sum += float(loss.detach().cpu()) * labels.size(0)
                running_count    += int(labels.numel())
                continue
            # ======================================

            # per-sample CE (used for phi and warmup)
            ce_vec = per_sample_ce(logits, labels).detach().cpu().numpy()
            batch_mean_ce = float(np.mean(ce_vec))
            G_global.append(batch_mean_ce)

            # update phi windows BEFORE Bayes-GIE
            for i, idx in enumerate(indices_np):
                dq = distance_queues.get(int(idx))
                if dq is None:
                    dq = deque(maxlen=k)
                    distance_queues[int(idx)] = dq
                dq.append(float(ce_vec[i]))

            # collect phi (None if not enough history)
            phi_list = get_phi_for_indices(distance_queues, indices_np, k_required=k)

            # Bayes-GIE (W,d) for this batch (overwrite/window-only mode)
            W_list, d_list = [], []
            have_bg = []
            for i, idx in enumerate(indices_np):
                if phi_list[i] is None or len(G_global) < k:
                    W_list.append(np.nan); d_list.append(np.nan); have_bg.append(False)
                    continue

                phi = phi_list[i]                                 # length k
                G_tr = np.asarray(list(G_global), dtype=float)    # length k

                # OVERWRITE mode: do not carry cumulants
                NG0, DG0 = 0.0, 0.0
                bayes_val, W_i, Num_inc, Den_inc = lid.compute_Bayes_GIE(phi, G_tr, NG0, DG0)

                # (optional bookkeeping of latest increments)
                cum_num_bgie[int(idx)] = Num_inc
                cum_den_bgie[int(idx)] = Den_inc

                W_list.append(W_i if np.isfinite(W_i) and W_i > 0 else np.nan)
                d_list.append(bayes_val if np.isfinite(bayes_val) and bayes_val > 0 else np.nan)
                have_bg.append(True)

            W_b = np.asarray(W_list, dtype=np.float64)
            d_b = np.asarray(d_list, dtype=np.float64)

            # yhat, one-hot
            with torch.no_grad():
                yhat = torch.softmax(logits, dim=1).detach()
                yone = F.one_hot(labels, num_classes=logits.size(1)).to(logits.dtype)

            # CKL union reference (bootstrap from current batch if needed)
            ref = tracker.current_ref()
            if ref is None:
                valid = np.isfinite(W_b) & np.isfinite(d_b) & (W_b > 0) & (d_b > 0)
                if not np.any(valid):
                    # still warming up: plain CE for this batch
                    loss = per_sample_ce(logits, labels).mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    running_loss_sum += float(loss.detach().cpu()) * labels.size(0)
                    running_count    += int(labels.numel())
                    continue
                W_ref = float(np.exp(np.log(W_b[valid]).mean()))
                d_ref = float(np.exp(np.log(d_b[valid]).mean()))
            else:
                W_ref, d_ref = ref

            # CKL for samples with BGIE
            ckl_vals = np.full(len(indices_np), np.nan, dtype=np.float64)
            for i in range(len(indices_np)):
                if have_bg[i] and np.isfinite(W_b[i]) and np.isfinite(d_b[i]) and W_b[i] > 0 and d_b[i] > 0:
                    ckl_vals[i] = ckl_finite(W_b[i], d_b[i], W_ref, d_ref)

            # threshold BEFORE updating with this batch
            thr_now = tracker.current_thr()

            # α from CKL where available; else 1.0. Apply gate (persistent runlen).
            raw_flags = np.isfinite(ckl_vals) & (ckl_vals > thr_now)
            gated     = tracker.update_gates(indices_np.tolist(), raw_flags)

            alphas_np = np.ones(len(indices_np), dtype=np.float64)
            if np.any(np.isfinite(ckl_vals)):
                a_calc = ckl_to_alpha(ckl_vals[np.isfinite(ckl_vals)], thr_now, kappa=3.0, alpha_floor=0.05)
                alphas_np[np.isfinite(ckl_vals)] = a_calc
            alphas_np = np.where(gated, 0.05, alphas_np)  # hard clamp on gate
            alphas = torch.tensor(alphas_np, device=device, dtype=logits.dtype)

            # D2L loss with y* = α y + (1-α) ŷ
            y_star = alphas.unsqueeze(1) * yone + (1.0 - alphas).unsqueeze(1) * yhat
            loss_vec = -(y_star * F.log_softmax(logits, dim=1)).sum(dim=1)
            loss = loss_vec.mean()

            # step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update CKL reference & threshold with current BGIE
            valid = np.isfinite(W_b) & np.isfinite(d_b) & (W_b > 0) & (d_b > 0)
            if np.any(valid):
                tracker.update_ref(W_b[valid], d_b[valid])
                tracker.update_thr(ckl_vals[valid])

            running_loss_sum += float(loss.detach().cpu()) * labels.size(0)
            running_count    += int(labels.numel())

        # epoch metrics
        loss_history.append(running_loss_sum / max(running_count, 1))

        model.eval()
        t_sum, t_cnt = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                t_sum += float(global_ce(logits, y).detach().cpu()) * y.size(0)
                t_cnt += int(y.numel())
        test_loss_history.append(t_sum / max(t_cnt, 1))

        train_acc = compute_accuracy(model, train_loader, device)
        test_acc  = compute_accuracy(model, test_loader,  device)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"TrainLoss={loss_history[-1]:.4f} TestLoss={test_loss_history[-1]:.4f} "
              f"TrainAcc={train_acc:.2f}% TestAcc={test_acc:.2f}%")

# =================== Main =============================
def main():
    device = DEVICE
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])

    train_set_raw = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_set      = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # single dataset with exactly 100 cat→dog flips
    train_set = ControlledCatDogNoise(train_set_raw, n_flip=N_CATS_FLIP, seed=FLIP_SEED)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet32()
    train_model(model, train_loader, test_loader, NUM_EPOCHS, K_WINDOW, device)

if __name__ == '__main__':
    main()

