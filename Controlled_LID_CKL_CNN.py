import os, json
import numpy as np
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import math

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
N_CATS_FLIP  = 100         # exactly 100 cats → dog
MIN_RUNS = 5 
STICKY_FLAG = True         # True = once flagged, stays True; False = can turn back to False
KAPPA = 2.0
ALPHA_FLOOR = 0.05
USE_CKL      = True        # True = CKL+α (NSES), False = plain CE baseline

EPS = 1e-12

# =================== Determinism ======================
def set_all_seeds(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    import random as pyrand
    pyrand.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# =================== Model (12-layer CNN) =================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x

class CNN12(nn.Module):
    """
    12 conv layers total: 3 stages × 4 conv each
    Stage1: 64 channels (x4 conv)  -> MaxPool
    Stage2: 128 channels (x4 conv) -> MaxPool
    Stage3: 256 channels (x4 conv) -> MaxPool
    Then GAP + linear classifier.
    """
    def __init__(self, num_classes=10, p_drop=0.1):
        super().__init__()
        c1, c2, c3 = 64, 128, 256

        # Stage 1 (4 conv layers)
        self.s1 = nn.Sequential(
            ConvBlock(3,  c1, p_drop),
            ConvBlock(c1, c1, p_drop),
            ConvBlock(c1, c1, p_drop),
            ConvBlock(c1, c1, p_drop),
        )
        # Stage 2 (4 conv layers)
        self.s2 = nn.Sequential(
            ConvBlock(c1, c2, p_drop),
            ConvBlock(c2, c2, p_drop),
            ConvBlock(c2, c2, p_drop),
            ConvBlock(c2, c2, p_drop),
        )
        # Stage 3 (4 conv layers)
        self.s3 = nn.Sequential(
            ConvBlock(c2, c3, p_drop),
            ConvBlock(c3, c3, p_drop),
            ConvBlock(c3, c3, p_drop),
            ConvBlock(c3, c3, p_drop),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(c3, num_classes)
        self.penultimate = None

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        # Input: 3×32×32 (CIFAR-10)
        x = self.s1(x)          # -> 64×32×32
        x = self.pool(x)        # -> 64×16×16
        x = self.s2(x)          # -> 128×16×16
        x = self.pool(x)        # -> 128×8×8
        x = self.s3(x)          # -> 256×8×8
        x = self.pool(x)        # -> 256×4×4

        # Global Average Pool to 1×1
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # -> (B, 256)
        self.penultimate = x
        return self.classifier(x)

def CNN12_Model():
    return CNN12(num_classes=10)

# =================== Controlled noise =================
class ControlledCatDogNoise(Dataset):
    """
    Wrap CIFAR10 train set.
    - Flip exactly N_CATS_FLIP items with true class 'cat' → label 'dog'.
    - Exposes:
        .noisy_mask[idx] -> True if this idx is flipped cat→dog
        .group_map[idx]  -> 'cat' (clean cat), 'dog' (clean dog), 'noisy' (flipped cat), or 'other'
    """
    def __init__(self, base_dataset, n_flip=100, seed=777):
        self.base = base_dataset
        self.n_flip = int(n_flip)

        # CIFAR-10 has .classes (list of names) and .targets (list of ints)
        classes = list(self.base.classes)          # e.g., [..., 'cat', ..., 'dog', ...]
        self.cat_idx = classes.index('cat')        # 3
        self.dog_idx = classes.index('dog')        # 5

        self.labels = list(self.base.targets)      # len=50000

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

        flipped_set = set(self.flipped_ids)
        self.group_map = {}
        for i in range(n):
            if i in flipped_set:
                self.group_map[i] = 'noisy'
            elif self.labels[i] == self.cat_idx:
                self.group_map[i] = 'cat'
            elif self.labels[i] == self.dog_idx:
                self.group_map[i] = 'dog'
            else:
                self.group_map[i] = 'other'

        print(f"[Noise] Flipped exactly {self.n_flip} cats → dog.")

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, int(self.noisy_labels[idx]), int(idx), bool(self.noisy_mask[idx])

# =================== LID Estimators ===================
   
class LIDEstimators:
    def __init__(self, device='cpu'):
        self.device = device

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
        mask = (R > EPS) & (FR > EPS)
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

def huber_mean(x, c=1.345, iters=15):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m)) + EPS
    for _ in range(iters):
        r = (x - m) / s
        w = np.where(np.abs(r) <= c, 1.0, c / (np.abs(r) + EPS))
        m_new = float(np.sum(w * x) / (np.sum(w) + EPS))
        if abs(m_new - m) < 1e-12:
            break
        m = m_new
    return m

def one_hot(labels: torch.Tensor, num_classes: int, device):
    y = torch.zeros((labels.size(0), num_classes), device=device, dtype=torch.float32)
    y.scatter_(1, labels.view(-1,1), 1.0)
    return y

@torch.no_grad()
def per_class_accuracy(model, data_loader, device, class_ids, true_labels_array=None):
    """
    Returns dict {cls_id: (correct, total)} using:
      - true_labels_array (np.array of ints, length=len(dataset)) if provided, else
      - the labels coming from the loader batches (observed/noisy labels).
    """
    model.eval()
    counts = {c: [0, 0] for c in class_ids}  # [correct, total]
    for batch in data_loader:
        inputs = batch[0].to(device)
        if true_labels_array is None:
            # use observed labels from the loader
            labels = batch[1].to(device)
            idx_for_true = None
        else:
            # use ground-truth labels from array indexed by sample indices
            indices = batch[2].cpu().numpy()
            true_labels = torch.tensor(true_labels_array[indices], device=device, dtype=torch.long)
            labels = true_labels
            idx_for_true = indices  # not used further; just clarifies logic

        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)

        for c in class_ids:
            mask = (labels == c)
            if mask.any():
                correct = (preds[mask] == labels[mask]).sum().item()
                total   = int(mask.sum().item())
                counts[c][0] += correct
                counts[c][1] += total
    # convert to percentages (safe divide)
    acc = {c: (100.0 * counts[c][0] / max(counts[c][1], 1)) for c in class_ids}
    return acc

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, probe_loader, num_epochs, k, device):
    print("=== Mode:", "CKL+α(NSES)" if USE_CKL else "Baseline CE", "===")

    model = model.to(device)

    # ---- Optimizer & losses ----
    optimizer      = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    ce_hard_vec    = nn.CrossEntropyLoss(reduction='none')  # per-sample (probe or baseline train)
    ce_hard_mean   = nn.CrossEntropyLoss()                  # scalar (baseline train or test)
    ce_test_mean   = nn.CrossEntropyLoss()                  # scalar (test)

    # ---- Buffers & state ----
    distance_queues = {}  # idx -> deque of (hard-CE loss, epoch)  (only used when USE_CKL)
    probe_loss_history, train_loss_history, test_loss_history = [], [], []
    lid = LIDEstimators(device=device)

    n_samples   = len(train_loader.dataset)
    num_classes = len(train_loader.dataset.base.classes)

    # For CKL path only
    above_streak  = defaultdict(int)                # idx -> consecutive "above mean" count
    flagged_noisy = np.zeros(n_samples, dtype=bool) # sticky or non-sticky depending on STICKY_FLAG
    alpha_buffer  = np.full(n_samples, np.nan, dtype=np.float32)  # α used to build y*

    if USE_CKL:
        assert MIN_RUNS <= k, "MIN_RUNS should be <= K_WINDOW"
        assert num_epochs >= k, "Need at least K_WINDOW epochs to start detection"
        print("===== CKL Mode:", "Sticky =====" if STICKY_FLAG else "No-Sticky =====")

    for epoch in range(num_epochs):

        # =========================================================
        # 1) (Optional) PROBE PASS for CKL: collect hard-label CE
        # =========================================================
        alpha_this_epoch = {}  # idx -> α (only for samples seen/flagged this epoch)
        if USE_CKL:
            model.eval()
            probe_sum, probe_cnt = 0.0, 0
            with torch.no_grad():
                for inputs, labels, indices, _ in probe_loader: 
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    losses_vec = ce_hard_vec(logits, labels)  # hard CE, NOT α-mixed

                    probe_sum += float(losses_vec.sum().cpu())
                    probe_cnt += int(labels.numel())

                    # maintain k-window of probe losses per sample
                    for i in range(len(indices)):
                        idx = int(indices[i])
                        if idx not in distance_queues:
                            distance_queues[idx] = deque(maxlen=k)
                        distance_queues[idx].append((float(losses_vec[i].cpu()), epoch))

            probe_epoch_loss = probe_sum / max(probe_cnt, 1)
            probe_loss_history.append(probe_epoch_loss)

            # =========================================================
            # 2) DETECTION: compute GIE/CKL; update flags and α for THIS epoch
            # =========================================================
            if epoch >= k - 1:
                # Build class-wise refs for d and W using log-Huber on GIE/W (robust center),
                # but compare CKL to class arithmetic mean.
                per_cls_log_gie = {c: [] for c in range(num_classes)}
                per_cls_log_w   = {c: [] for c in range(num_classes)}
                sample_stats    = {}  # idx -> (gie_tr, w_tr, cls)

                # First pass: per-sample GIE/W using probe history and probe epoch means
                for idx, dq in distance_queues.items():
                    if len(dq) < k:
                        continue
                    dists, epochs_ = zip(*dq)
                    phi  = np.asarray(dists, dtype=float)
                    G_tr = np.asarray([probe_loss_history[e] for e in epochs_], dtype=float)

                    gie_tr, w_tr = lid.compute_GIE_LID(phi, G_tr)
                    cls = int(train_loader.dataset.noisy_labels[idx])

                    if np.isfinite(gie_tr) and gie_tr > 0.0:
                        per_cls_log_gie[cls].append(np.log(max(gie_tr, EPS)))
                    if np.isfinite(w_tr) and w_tr > 0.0:
                        per_cls_log_w[cls].append(np.log(max(w_tr, EPS)))

                    sample_stats[idx] = (gie_tr, w_tr, cls)

                # Class refs
                cls_log_gie_huber = [huber_mean(per_cls_log_gie[c]) if per_cls_log_gie[c] else float('nan')
                                     for c in range(num_classes)]
                cls_log_w_huber   = [huber_mean(per_cls_log_w[c])   if per_cls_log_w[c]   else float('nan')
                                     for c in range(num_classes)]
                ref_d = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_gie_huber]
                ref_w = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_w_huber]

                # CKL per sample (vs class refs)
                per_cls_ckl = {c: [] for c in range(num_classes)}
                sample_ckl  = {}  # idx -> (ckl_val, cls)
                for idx, (gie_tr, w_tr, cls) in sample_stats.items():
                    if not (np.isfinite(gie_tr) and gie_tr > 0.0 and np.isfinite(w_tr) and w_tr > 0.0):
                        continue
                    d2, w2 = ref_d[cls], ref_w[cls]
                    if not (np.isfinite(d2) and d2 > 0.0 and np.isfinite(w2) and w2 > 0.0):
                        continue
                    ckl_val = ckl_finite(w_tr, gie_tr, w2, d2)
                    if np.isfinite(ckl_val):
                        per_cls_ckl[cls].append(float(ckl_val))
                        sample_ckl[idx] = (float(ckl_val), cls)

                # Arithmetic mean CKL per class
                cls_ckl_mean = [(float(np.mean(per_cls_ckl[c])) if per_cls_ckl[c] else float('nan'))
                                for c in range(num_classes)]

                # Update streaks/flags; compute α for flagged
                for idx, (val, cls) in sample_ckl.items():
                    mean_c = cls_ckl_mean[cls]
                    if not np.isfinite(mean_c):
                        continue

                    # streaks
                    if val > mean_c:
                        above_streak[idx] += 1
                    else:
                        above_streak[idx] = 0

                    # sticky vs non-sticky flag
                    if STICKY_FLAG:
                        if above_streak[idx] >= MIN_RUNS:
                            flagged_noisy[idx] = True
                    else:
                        flagged_noisy[idx] = (above_streak[idx] >= MIN_RUNS)

                    # α for this epoch (only if flagged right now)
                    if flagged_noisy[idx]:
                        alpha_val = float(ckl_to_alpha(val, mean_c, kappa=KAPPA, alpha_floor=ALPHA_FLOOR))
                        alpha_this_epoch[idx] = alpha_val

                # Update α buffer for TRAIN pass this SAME epoch
                for idx, a in alpha_this_epoch.items():
                    alpha_buffer[idx] = a

                if not STICKY_FLAG:
                    seen = set(sample_ckl.keys())
                    for idx in seen:
                        if not flagged_noisy[idx]:
                            alpha_buffer[idx] = np.nan
                    for idx in np.where(~flagged_noisy)[0]:
                        if idx not in seen:
                            alpha_buffer[idx] = np.nan

                # Diagnostic: dog flagged correctness
                dog_idx = train_loader.dataset.dog_idx
                flagged_idxs = np.where(flagged_noisy)[0]
                dog_flagged = [i for i in flagged_idxs if int(train_loader.dataset.noisy_labels[i]) == dog_idx]
                correct = sum(1 for i in dog_flagged if train_loader.dataset.group_map.get(i) == 'noisy')
                incorrect = sum(1 for i in dog_flagged if train_loader.dataset.group_map.get(i) == 'dog')
                print(f"[Epoch {epoch+1}] Dog flagged: total={len(dog_flagged)}, correct={correct}, incorrect={incorrect}")

        # =========================================================
        # 3) TRAIN PASS
        #   - Always start from plain CE per-sample
        #   - If USE_CKL: only replace losses for flagged samples with α<1
        # =========================================================
        model.train()
        train_sum, train_cnt = 0.0, 0

        for inputs, labels, indices, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

            # 3.a Plain CE per-sample (baseline for all)
            ce_vec = ce_hard_vec(logits, labels)  # shape [B]

            if USE_CKL:
                # Build mask of positions that have an α strictly less than 1.0 (i.e., flagged & softened)
                a_list = []
                flagged_mask_list = []
                for i in indices:
                    a = alpha_buffer[int(i)]
                    if np.isnan(a):  # treat NaN as 1.0 (no change)
                        a = 1.0
                    a_list.append(a)
                    flagged_mask_list.append(a < 1.0)

                alphas = torch.tensor(a_list, device=device, dtype=torch.float32).unsqueeze(1)  # [B,1]
                flagged_mask = torch.tensor(flagged_mask_list, device=device, dtype=torch.bool) # [B]

                if flagged_mask.any():
                    # Compute NSES only for the flagged subset
                    probs = F.softmax(logits.detach(), dim=1)        # ŷ (stop-grad in target path)
                    yhard = one_hot(labels, num_classes=num_classes, device=device)

                    # y* for the whole batch (cheap), but we will only use it on flagged indices
                    ystar = alphas * yhard + (1.0 - alphas) * probs  # [B,C]
                    logp  = F.log_softmax(logits, dim=1)
                    nses_vec = -(ystar * logp).sum(dim=1)            # [B]

                    # Start from CE and override ONLY flagged positions
                    loss_vec = ce_vec.clone()
                    loss_vec[flagged_mask] = nses_vec[flagged_mask]
                else:
                    # Nobody flagged this batch → pure CE
                    loss_vec = ce_vec
            else:
                # Baseline run → pure CE
                loss_vec = ce_vec

            loss = loss_vec.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_sum += float(loss.detach().cpu()) * labels.size(0)
            train_cnt += int(labels.numel())

        train_epoch_loss = train_sum / max(train_cnt, 1)
        train_loss_history.append(train_epoch_loss)


        # =========================================================
        # 4) TEST (hard-label CE)
        # =========================================================
        model.eval()
        t_sum, t_cnt = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                t_sum += float(ce_test_mean(logits, y).cpu()) * y.size(0)
                t_cnt += int(y.numel())
        test_loss_history.append(t_sum / max(t_cnt, 1))

        # ---------------- Metrics ----------------
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc  = compute_accuracy(model, test_loader,  device)

        if USE_CKL:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"ProbeCE={probe_loss_history[-1]:.4f}  "
                  f"TrainLoss(y*)={train_loss_history[-1]:.4f}  "
                  f"TestCE={test_loss_history[-1]:.4f}  "
                  f"TrainAcc={train_acc:.2f}%  TestAcc={test_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"TrainCE={train_loss_history[-1]:.4f}  "
                  f"TestCE={test_loss_history[-1]:.4f}  "
                  f"TrainAcc={train_acc:.2f}%  TestAcc={test_acc:.2f}%")

        # ---- Per-class accuracies (all 10) ----
        class_ids   = list(range(num_classes))
        class_names = train_loader.dataset.base.classes

        # Train per-class using observed (noisy) labels
        train_obs_acc = per_class_accuracy(model, train_loader, device, class_ids, true_labels_array=None)
        # Train per-class using *true labels* (original CIFAR-10 via indices)
        train_true_labels = np.array(train_loader.dataset.labels, dtype=int)
        train_true_acc = per_class_accuracy(model, train_loader, device, class_ids, true_labels_array=train_true_labels)
        # Test per-class (clean)
        test_acc_per_class = per_class_accuracy(model, test_loader, device, class_ids, true_labels_array=None)

        def _print_classwise(title, acc_dict):
            print(f"  {title}")
            for c in class_ids:
                print(f"    [{c}] {class_names[c]:>10}: {acc_dict[c]:6.2f}%")

        _print_classwise("Train (observed labels)", train_obs_acc)
        _print_classwise("Train (true labels)    ", train_true_acc)
        _print_classwise("Test (clean)           ", test_acc_per_class)

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
    probe_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN12_Model()
    train_model(model, train_loader, test_loader, probe_loader, NUM_EPOCHS, K_WINDOW, device)

if __name__ == '__main__':
    main()

