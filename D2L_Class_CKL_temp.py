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
from torch.optim.lr_scheduler import CosineAnnealingLR

# ======================= Config =======================
SEED = 12345
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS   = 120
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

K_WINDOW     = 22          # sliding window for FIE/GIE
FLIP_SEED    = 777         # which cats get flipped to dog

EPS = 1e-12

USE_CKL = False   # <— set to True to enable CKL/Bayes-GIE; False = plain CE baseline

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

def ckl_to_alpha(scores_np, thr, kappa=1.0, alpha_floor=0.05):
    scores = np.asarray(scores_np, dtype=np.float64)
    rel = np.maximum((scores - thr) / (thr + EPS), 0.0)
    a = np.exp(-kappa * rel)
    return np.clip(a, alpha_floor, 1.0)

# =================== Class-wise CKL tracker =======================
class ClassCklTracker:
    """
    Per-epoch, per-class CKL reference (W_ref, d_ref) and threshold.
    Uses observed labels for grouping. Run-length gate persists via .runlen.
    """
    def __init__(self, num_classes=10, thr_mode="mean", min_run=5):
        assert thr_mode in ("mean", "median")
        self.num_classes = int(num_classes)
        self.thr_mode = thr_mode
        self.min_run = int(min_run)
        self.reset()
        self.runlen = defaultdict(int)  # may be overwritten from outside to persist across epochs

    def reset(self):
        self.logW_sum = [0.0]*self.num_classes
        self.logd_sum = [0.0]*self.num_classes
        self.ref_cnt  = [0]*self.num_classes

        self._thr_n   = [0]*self.num_classes
        self._thr_s   = [0.0]*self.num_classes
        self.ckl_buf  = [[] for _ in range(self.num_classes)]
        self.curr_thr = [0.0]*self.num_classes

    # references
    def has_ref(self, c:int) -> bool:
        return self.ref_cnt[c] > 0

    def current_ref(self, c:int):
        if self.ref_cnt[c] == 0: return None
        W_ref = math.exp(self.logW_sum[c] / self.ref_cnt[c])
        d_ref = math.exp(self.logd_sum[c] / self.ref_cnt[c])
        return float(W_ref), float(d_ref)

    def update_ref_class(self, c:int, W_batch_c, d_batch_c):
        if len(W_batch_c) == 0: return
        Wb = np.maximum(W_batch_c, EPS)
        db = np.maximum(d_batch_c, EPS)
        self.logW_sum[c] += float(np.log(Wb).sum())
        self.logd_sum[c] += float(np.log(db).sum())
        self.ref_cnt[c]  += int(len(W_batch_c))

    # thresholds
    def current_thr_val(self, c:int) -> float:
        if self.thr_mode == "mean":
            return float(self.curr_thr[c])
        return float(np.nanmedian(np.asarray(self.ckl_buf[c]))) if self.ckl_buf[c] else float(self.curr_thr[c])

    def update_thr_class(self, c:int, ckl_vals_c):
        if len(ckl_vals_c) == 0: return
        if self.thr_mode == "mean":
            self._thr_n[c] += len(ckl_vals_c)
            self._thr_s[c] += float(np.nansum(ckl_vals_c))
            self.curr_thr[c] = self._thr_s[c] / max(self._thr_n[c], 1)
        else:
            self.ckl_buf[c].extend(list(ckl_vals_c))

    # persistent run-length gate
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

class SymmetricNoisyCIFAR10(Dataset):
    """
    Symmetric noise: each true label is flipped to any *other* class with probability `noise_ratio`.
    """
    def __init__(self, dataset, noise_ratio=0.4, seed=777):
        self.dataset = dataset
        self.noise_ratio = float(noise_ratio)
        self.seed = seed

        # ---- extract clean labels ------------------------------------------------
        if hasattr(dataset, "targets"):
            self.clean_labels = list(map(int, dataset.targets))
        else:
            self.clean_labels = [int(lbl) for _, lbl in dataset]

        self.noisy_labels = self.clean_labels.copy()
        self.noisy_mask   = [False] * len(self.clean_labels)

        rng = np.random.default_rng(seed)
        n = len(self.clean_labels)
        n_flip = int(round(self.noise_ratio * n))
        flip_idx = rng.choice(n, size=n_flip, replace=False)

        for idx in flip_idx:
            true = self.clean_labels[idx]
            others = [c for c in range(10) if c != true]
            self.noisy_labels[idx] = int(rng.choice(others))
            self.noisy_mask[idx]   = True

        # report real flip rate
        real_rate = sum(self.noisy_mask) / n
        print(f"Symmetric noise: {sum(self.noisy_mask)} samples flipped "
              f"({real_rate:.2%} of total, target {self.noise_ratio:.2%})")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]                 # ignore original label
        return img, int(self.noisy_labels[idx]), int(idx), bool(self.noisy_mask[idx])

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
        limit0 = np.mean(phi[-3:])
        R = np.abs(phi - limit0)
        w0 = np.max(R)
        limit1 = np.mean(G[-3:])
        FR = np.abs(G - limit1)
        w1 = np.max(FR)
        mask = (R != 0) & (FR != 0)
        R_non_zero = R[mask]
        FR_non_zero = FR[mask]
        Wmax = w0
        k = R_non_zero.shape[0] - 1
        if k <= 4:
            return EPS, float(Wmax)
        hill_num = - (k / np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon)))))
        hill_den = - (k / np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon)))))
        gie = hill_num / hill_den if hill_den != 0 else np.nan
        return float(gie), float(Wmax)

    def compute_Bayes_GIE(self, phi, G, Num0, Den0):
        epsilon = 1e-7
        limit0 = np.mean(phi[-3:])
        R = np.abs(phi - limit0)
        w0 = np.max(R)
        limit1 = np.mean(G[-3:])
        FR = np.abs(G - limit1)
        w1 = np.max(FR)
        mask = (R > EPS) & (FR > EPS)
        R_non_zero = R[mask]
        FR_non_zero = FR[mask]
        k = R_non_zero.shape[0] - 1
        if k <= 4:
            return EPS, float(w0), 0.0, 0.0
        hill_num = - (k / np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon)))))
        hill_den = - (k / np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon)))))
        if hill_num == 0 or hill_den == 0 or np.isnan(hill_num) or np.isnan(hill_den):
            Num1, Den1 = 0.0, 0.0
        else:
            Num1 = 1.0 / hill_den
            Den1 = 1.0 / hill_num
        Num_cumulative = Num0 + Num1
        Den_cumulative = Den0 + Den1
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
    phis = []
    for idx in indices:
        dq = distance_queues.get(int(idx), None)
        if dq is None or len(dq) < k_required:
            phis.append(None)
        else:
            vals = [v for (v) in dq]
            phis.append(np.asarray(vals, dtype=float))
    return phis

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, num_epochs, k, device):
    """
    Step 1 applied:
      - After warm-up, build/update CKL references per *predicted* class
        using only high-confidence samples (CONF_MIN).
      - When computing a sample's CKL and threshold, compare to the
        predicted-class reference when available/credible; otherwise fall back
        to the observed label class.

    Other behavior (queues, Bayes-GIE, α mapping, loss, logging) unchanged.
    """
    WARMUP_EPOCHS = k                   # wait until ~k CE points collected
    CONF_MIN      = 0.60                # confidence gate for predicted-class refs

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    per_sample_ce = nn.CrossEntropyLoss(reduction='none')
    global_ce     = nn.CrossEntropyLoss()

    loss_history, test_loss_history = [], []

    # ---------- CKL/Bayes-GIE state ----------
    if USE_CKL:
        lid = LIDEstimators(device=device)
        distance_queues: dict[int, deque] = {}    # idx -> deque[float] of length k
        cum_num_bgie: dict[int, float] = defaultdict(float)
        cum_den_bgie: dict[int, float] = defaultdict(float)
        runlen_global: dict[int, int] = defaultdict(int)  # persists across epochs
        G_global = deque(maxlen=k)                        # global train-loss reference (length k)
    else:
        lid = None
        distance_queues = {}
        cum_num_bgie = {}
        cum_den_bgie = {}
        runlen_global = {}
        G_global = deque(maxlen=k)

    for epoch in range(num_epochs):
        model.train()
        running_loss_sum, running_count = 0.0, 0

        # Per-epoch class-wise tracker (refs + thresholds); gate persists
        if USE_CKL:
            tracker = ClassCklTracker(num_classes=10, thr_mode="mean", min_run=5)
            tracker.runlen = runlen_global
        else:
            tracker = None

        for batch_idx, (inputs, labels, indices, is_noisy) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            indices_np = indices.cpu().numpy()

            # forward
            logits = model(inputs)

            # ---- predictions + confidence (used after warm-up) ----
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)           # predicted class & its conf
            pred_np  = pred.cpu().numpy()
            conf_np  = conf.cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # --- ALWAYS record per-sample CE history (even during warm-up) ---
            if USE_CKL:
                ce_vec = per_sample_ce(logits, labels).detach().cpu().numpy()
                batch_mean_ce = float(np.mean(ce_vec))
                G_global.append(batch_mean_ce)

                for i, idx in enumerate(indices_np):
                    dq = distance_queues.get(int(idx))
                    if dq is None:
                        dq = deque(maxlen=k)
                        distance_queues[int(idx)] = dq
                    dq.append(float(ce_vec[i]))
            # ----------------------------------------------------------------

            # ===== Warm-up: optimize CE only (no CKL/D2L used yet) =====
            if (not USE_CKL) or (epoch < WARMUP_EPOCHS):
                loss = global_ce(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss_sum += float(loss.detach().cpu()) * labels.size(0)
                running_count    += int(labels.numel())
                continue
            # ============================================================

            # ----------------- CKL/D2L branch -----------------
            # collect phi (None if not enough history)
            phi_list = get_phi_for_indices(distance_queues, indices_np, k_required=k)

            # Bayes-GIE (W,d) for this batch (window-only as in your code)
            W_list, d_list, have_bg = [], [], []
            for i, idx in enumerate(indices_np):
                if phi_list[i] is None or len(G_global) < k:
                    W_list.append(np.nan); d_list.append(np.nan); have_bg.append(False)
                    continue

                phi = phi_list[i]                                 # length k
                G_tr = np.asarray(list(G_global), dtype=float)    # length k

                NG0, DG0 = cum_num_bgie[int(idx)], cum_den_bgie[int(idx)]
                bayes_val, W_i, Num_inc, Den_inc = lid.compute_Bayes_GIE(phi, G_tr, NG0, DG0)

                # overwrite (window-only) like your original
                cum_num_bgie[int(idx)] = Num_inc
                cum_den_bgie[int(idx)] = Den_inc

                W_list.append(W_i if np.isfinite(W_i) and W_i > 0 else np.nan)
                d_list.append(bayes_val if np.isfinite(bayes_val) and bayes_val > 0 else np.nan)
                have_bg.append(True)

            W_b = np.asarray(W_list, dtype=np.float64)
            d_b = np.asarray(d_list, dtype=np.float64)
            valid_all = np.isfinite(W_b) & np.isfinite(d_b) & (W_b > 0) & (d_b > 0)

            # --------- Bootstrapping/Updating references & thresholds ----------
            # AFTER warm-up: use predicted class with confidence
            for c in range(10):
                mask_c_valid = (pred_np == c) & (conf_np >= CONF_MIN) & valid_all
                if np.any(mask_c_valid):
                    tracker.update_ref_class(c, W_b[mask_c_valid], d_b[mask_c_valid])
                    # threshold update will use current batch CKL below (requires ckl_vals)
            # -------------------------------------------------------------------

            # Compute CKL per sample against chosen class reference
            ckl_vals = np.full(len(indices_np), np.nan, dtype=np.float64)
            chosen_class = np.zeros(len(indices_np), dtype=np.int32)
            for i in range(len(indices_np)):
                if not have_bg[i] or not valid_all[i]:
                    continue
                c_pred = int(pred_np[i]); c_obs = int(labels_np[i])
                use_pred = tracker.has_ref(c_pred) or (conf_np[i] >= CONF_MIN)
                c_i = c_pred if use_pred else c_obs
                ref_i = tracker.current_ref(c_i)
                if ref_i is None:
                    continue
                W_ref_i, d_ref_i = ref_i
                ckl_vals[i] = ckl_finite(W_b[i], d_b[i], W_ref_i, d_ref_i)
                chosen_class[i] = c_i

            # Per-sample thresholds for the same chosen class
            thr_now_per_i = np.array([tracker.current_thr_val(int(chosen_class[i]))
                                      for i in range(len(indices_np))], dtype=np.float64)

            # Gate using class-wise thresholds
            raw_flags = np.isfinite(ckl_vals) & (ckl_vals > thr_now_per_i)
            gated     = tracker.update_gates(indices_np.tolist(), raw_flags)

            # α from CKL (class-wise threshold already used above)
            alphas_np = np.ones(len(indices_np), dtype=np.float64)
            finite_mask = np.isfinite(ckl_vals)
            if np.any(finite_mask):
                for c in range(10):
                    sel = finite_mask & (chosen_class == c)
                    if np.any(sel):
                        a_c = ckl_to_alpha(ckl_vals[sel], tracker.current_thr_val(c),
                                           kappa=3.0, alpha_floor=0.05)
                        alphas_np[sel] = a_c
            alphas = torch.tensor(alphas_np, device=device, dtype=logits.dtype)

            # D2L loss with y* = α y + (1-α) ŷ
            yone = F.one_hot(labels, num_classes=logits.size(1)).to(logits.dtype)
            yhat = probs.detach()  # from above
            y_star = alphas.unsqueeze(1) * yone + (1.0 - alphas).unsqueeze(1) * yhat
            loss_vec = -(y_star * F.log_softmax(logits, dim=1)).sum(dim=1)
            loss = loss_vec.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Now that we have CKL values, update thresholds per chosen class
            for c in range(10):
                mask_c_valid = (chosen_class == c) & np.isfinite(ckl_vals)
                if np.any(mask_c_valid):
                    tracker.update_thr_class(c, ckl_vals[mask_c_valid])

            running_loss_sum += float(loss.detach().cpu()) * labels.size(0)
            running_count    += int(labels.numel())
            # ----------------- end CKL/D2L branch -----------------

        # === Periodic detection stats (same logic for class choice) ===
        if USE_CKL and (epoch + 1) % 10 == 0:
            model.eval()
            all_true_noisy = []
            all_detected = []
            with torch.no_grad():
                for inputs, labels, indices, is_noisy in train_loader:
                    inputs = inputs.to(device)
                    logits = model(inputs)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    pred_np = pred.cpu().numpy()
                    conf_np = conf.cpu().numpy()
                    labels_np = labels.cpu().numpy()

                    indices_np = indices.cpu().numpy()
                    is_noisy_np = np.array(is_noisy.cpu(), dtype=bool)

                    ce_vec = per_sample_ce(logits, labels.to(device)).detach().cpu().numpy()
                    G_global.append(float(np.mean(ce_vec)))

                    for i, idx in enumerate(indices_np):
                        dq = distance_queues.get(int(idx))
                        if dq is None:
                            dq = deque(maxlen=k)
                            distance_queues[int(idx)] = dq
                        dq.append(float(ce_vec[i]))

                    phi_list = get_phi_for_indices(distance_queues, indices_np, k_required=k)
                    W_list, d_list, have_bg = [], [], []
                    for i, idx in enumerate(indices_np):
                        if phi_list[i] is None or len(G_global) < k:
                            W_list.append(np.nan); d_list.append(np.nan); have_bg.append(False)
                            continue
                        phi = phi_list[i]
                        G_tr = np.asarray(list(G_global), dtype=float)
                        NG0, DG0 = cum_num_bgie[int(idx)], cum_den_bgie[int(idx)]
                        bayes_val, W_i, Num_inc, Den_inc = lid.compute_Bayes_GIE(phi, G_tr, NG0, DG0)
                        cum_num_bgie[int(idx)] = Num_inc
                        cum_den_bgie[int(idx)] = Den_inc
                        W_list.append(W_i if np.isfinite(W_i) and W_i > 0 else np.nan)
                        d_list.append(bayes_val if np.isfinite(bayes_val) and bayes_val > 0 else np.nan)
                        have_bg.append(True)

                    W_b = np.asarray(W_list, dtype=np.float64)
                    d_b = np.asarray(d_list, dtype=np.float64)
                    valid_all = np.isfinite(W_b) & np.isfinite(d_b) & (W_b > 0) & (d_b > 0)

                    ckl_vals = np.full(len(indices_np), np.nan, dtype=np.float64)
                    thr_now_per_i = np.zeros(len(indices_np), dtype=np.float64)
                    for i in range(len(indices_np)):
                        if not have_bg[i] or not valid_all[i]: 
                            continue
                        c_pred = int(pred_np[i]); c_obs = int(labels_np[i])
                        use_pred = tracker.has_ref(c_pred) or (conf_np[i] >= CONF_MIN)
                        c_i = c_pred if use_pred else c_obs
                        ref_i = tracker.current_ref(c_i)
                        if ref_i is None: 
                            continue
                        W_ref_i, d_ref_i = ref_i
                        ckl_vals[i] = ckl_finite(W_b[i], d_b[i], W_ref_i, d_ref_i)
                        thr_now_per_i[i] = tracker.current_thr_val(c_i)

                    raw_flags = np.isfinite(ckl_vals) & (ckl_vals > thr_now_per_i)
                    gated = tracker.update_gates(indices_np.tolist(), raw_flags)

                    all_true_noisy.extend(is_noisy_np.tolist())
                    all_detected.extend(gated.tolist())

            all_true_noisy = np.array(all_true_noisy, dtype=bool)
            all_detected   = np.array(all_detected, dtype=bool)
            tp = np.sum(all_true_noisy & all_detected)
            fp = np.sum(~all_true_noisy & all_detected)
            fn = np.sum(all_true_noisy & ~all_detected)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            print(f"  [GLOBAL Detection @ epoch {epoch+1}] "
                  f"Prec:{prec:.3f} Rec:{rec:.3f} TP/FP/FN:{tp}/{fp}/{fn} "
                  f"(out of {np.sum(all_true_noisy)} noisy samples)")
            model.train()

        # === Epoch metrics ===
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

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])

    # ---- data -------------------------------------------------
    train_set_raw = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
    test_set      = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    train_set = SymmetricNoisyCIFAR10(train_set_raw, noise_ratio=0.04, seed=FLIP_SEED)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ---- model ------------------------------------------------
    model = ResNet32().to(device)

    # ---- train ------------------------------------------------
    train_model(model, train_loader, test_loader,
                num_epochs=NUM_EPOCHS,
                k=K_WINDOW,               # used for queues
                device=device)

if __name__ == '__main__':
    main()
