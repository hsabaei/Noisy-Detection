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
from typing import Optional

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
MIN_RUNS = 5 
STICKY_FLAG = True        # True = once flagged, stays True; False = can turn back to False
KAPPA = 1.0
ALPHA_FLOOR = 0.05
USE_CKL      = False        # True = CKL+α (NSES), False = plain CE baseline

EPS = 1e-12


import hashlib
from contextlib import contextmanager

def _is_bn(m):
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

def snapshot_bn_state(model):
    """
    Returns {module_name: {
        'running_mean': tensor(cpu).clone(),
        'running_var':  tensor(cpu).clone(),
        'num_batches_tracked': int or None,
        'track_running_stats': bool
    }}
    """
    snap = {}
    for name, m in model.named_modules():
        if _is_bn(m):
            d = {
                "running_mean": m.running_mean.detach().cpu().clone() if m.running_mean is not None else None,
                "running_var":  m.running_var.detach().cpu().clone()  if m.running_var  is not None else None,
                "num_batches_tracked": int(m.num_batches_tracked.item()) if hasattr(m, "num_batches_tracked") else None,
                "track_running_stats": bool(getattr(m, "track_running_stats", False)),
            }
            snap[name] = d
    return snap

def _max_abs_diff(a, b):
    if a is None or b is None:
        return None
    return float((a - b).abs().max().item())

def compare_bn_snapshots(s0, s1, atol=0.0, quiet_ok=True):
    """
    Prints any BN modules whose stats changed beyond atol.
    Returns True if all BN stats are unchanged within atol, else False.
    """
    ok = True
    for name in sorted(set(s0.keys()) | set(s1.keys())):
        if name not in s0 or name not in s1:
            print(f"[BN-CHECK] Module presence changed for '{name}' (before/after).")
            ok = False
            continue
        a, b = s0[name], s1[name]
        dm = _max_abs_diff(a["running_mean"], b["running_mean"])
        dv = _max_abs_diff(a["running_var"],  b["running_var"])
        dnbt = None
        if a["num_batches_tracked"] is not None and b["num_batches_tracked"] is not None:
            dnbt = b["num_batches_tracked"] - a["num_batches_tracked"]

        changed = False
        lines = []
        if dm is not None and (dm > atol):
            changed = True; lines.append(f"running_mean Δmax={dm:.3e}")
        if dv is not None and (dv > atol):
            changed = True; lines.append(f"running_var  Δmax={dv:.3e}")
        if dnbt is not None and dnbt != 0:
            changed = True; lines.append(f"num_batches_tracked Δ={dnbt}")

        if changed:
            ok = False
            print(f"[BN-CHECK] '{name}': " + ", ".join(lines))

    if ok and not quiet_ok:
        print("[BN-CHECK] All BN running stats unchanged (within atol).")
    return ok

def model_param_fingerprint(model):
    """Lightweight fingerprint to help ensure you're running the intended file/model."""
    h = hashlib.sha1()
    with torch.no_grad():
        for p in model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]

@contextmanager
def bn_guard(model, label="PROBE", atol=0.0):
    """
    Context that asserts BN stats don’t change inside the block.
    Use around your probe forward loop.
    """
    print(f"\n[BN-CHECK:{label}] entering… training={model.training}, grad_enabled={torch.is_grad_enabled()}")
    before = snapshot_bn_state(model)
    fp_before = model_param_fingerprint(model)
    try:
        yield
    finally:
        after = snapshot_bn_state(model)
        fp_after = model_param_fingerprint(model)
        print(f"[BN-CHECK:{label}] leaving…  training={model.training}, grad_enabled={torch.is_grad_enabled()}")
        same = compare_bn_snapshots(before, after, atol=atol, quiet_ok=False)
        print(f"[BN-CHECK:{label}] param fingerprint (before/after): {fp_before} → {fp_after}")
        if not same:
            print(f"[BN-CHECK:{label}] ❌ BN stats changed! Ensure model.eval() was set for the probe and no hidden training ops ran.")
        else:
            print(f"[BN-CHECK:{label}] ✅ BN stats unchanged.")



# =================== Determinism ======================
def set_all_seeds(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    import random as pyrand
    pyrand.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

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
    was_training = model.training

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    if was_training:
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


from typing import Optional

@torch.no_grad()
def subset_accuracy(
    model,
    data_loader,
    device,
    index_mask: np.ndarray,
    use_true_labels: bool = False,
    true_labels_array: Optional[np.ndarray] = None
) -> float:
    """
    Accuracy on a subset of the TRAIN set specified by a boolean index_mask
    (len == len(train_loader.dataset)). If use_true_labels=True, compare
    against the original CIFAR-10 labels (true_labels_array required).
    Otherwise, compare against the observed/noisy labels from the loader.
    """
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        inputs = batch[0].to(device)
        batch_indices = batch[2].cpu().numpy()  # global indices into the train dataset
        sel = index_mask[batch_indices]
        if not np.any(sel):
            continue

        if use_true_labels:
            assert true_labels_array is not None, "true_labels_array is required when use_true_labels=True"
            labels = torch.tensor(true_labels_array[batch_indices], device=device, dtype=torch.long)
        else:
            labels = batch[1].to(device)  # observed/noisy labels

        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)

        sel_t = torch.from_numpy(sel).to(device=device, dtype=torch.bool)
        correct += (preds[sel_t] == labels[sel_t]).sum().item()
        total   += int(sel_t.sum().item())

    model.train()
    return 100.0 * correct / max(total, 1)

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, num_epochs, k, device):
    model = model.to(device)

    # ---- Optimizer & losses ----
    optimizer      = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    ce_hard_vec    = nn.CrossEntropyLoss(reduction='none')   # per-sample (for queues & baseline loss)
    ce_hard_mean   = nn.CrossEntropyLoss()                   # scalar (baseline loss)
    ce_test_mean   = nn.CrossEntropyLoss()                   # scalar (test)

    # ---- Buffers & state ----
    distance_queues = {}   # idx -> deque of (hard-CE loss, epoch)
    epoch_ce_mean   = []   # store mean hard CE per epoch (for G-trace)
    train_loss_history, test_loss_history = [], []

    n_samples   = len(train_loader.dataset)
    num_classes = len(train_loader.dataset.base.classes)

    # For CKL path only
    above_streak  = defaultdict(int)                 # idx -> consecutive "above mean" count
    flagged_noisy = np.zeros(n_samples, dtype=bool)  # sticky or non-sticky depending on STICKY_FLAG
    alpha_buffer  = np.full(n_samples, np.nan, dtype=np.float32)  # α used for NEXT epoch
    lid = LIDEstimators(device=device)

    if USE_CKL:
        assert MIN_RUNS <= k, "MIN_RUNS should be <= K_WINDOW"
        assert num_epochs >= k, "Need at least K_WINDOW epochs to start detection"

    for epoch in range(num_epochs):
        # =========================================================
        # 1) TRAIN (single pass)
        #    - Always compute per-sample hard CE (for queues).
        #    - Loss:
        #         * Baseline or CKL with flagged-only α
        # =========================================================
        model.train()
        train_sum, train_cnt = 0.0, 0
        ce_sum_for_epoch, ce_cnt_for_epoch = 0.0, 0

        for inputs, labels, indices, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

            # (A) Per-sample hard CE for queues & baseline
            ce_vec = ce_hard_vec(logits, labels)  # shape [B]
            # enqueue for GIE history (use current epoch index)
            for i in range(len(indices)):
                idx = int(indices[i])
                if idx not in distance_queues:
                    distance_queues[idx] = deque(maxlen=k)
                distance_queues[idx].append((float(ce_vec[i].detach().cpu()), epoch))

            # accumulate epoch mean hard CE (for G trace)
            ce_sum_for_epoch += float(ce_vec.sum().detach().cpu())
            ce_cnt_for_epoch += int(labels.numel())

            # (B) Compute loss
            if not USE_CKL:
                # Plain baseline: mean hard CE
                loss = ce_vec.mean()
            else:
                # CKL mode, flagged-only α:
                with torch.no_grad():
                    # mask of currently flagged (α available from *previous* epoch’s detection)
                    idx_np = np.asarray(indices.cpu().numpy(), dtype=int)
                    a_np   = np.array([alpha_buffer[i] for i in idx_np])
                    flagged_mask = ~np.isnan(a_np) & (a_np < 1.0 - 1e-12)
                    a_flag = torch.tensor(a_np[flagged_mask], device=device, dtype=torch.float32).unsqueeze(1)

                if flagged_mask.any():
                    # Split batch into flagged / unflagged
                    flagged_idx   = torch.where(torch.tensor(flagged_mask, device=device))[0]
                    unflagged_idx = torch.where(~torch.tensor(flagged_mask, device=device))[0]

                    # Unflagged part: plain CE
                    loss_unflag = ce_vec[unflagged_idx].mean() if unflagged_idx.numel() > 0 else 0.0

                    # Flagged part: NSES with y* = α y + (1-α) ŷ
                    logits_flag = logits[flagged_idx]
                    labels_flag = labels[flagged_idx]
                    probs_flag  = F.softmax(logits_flag, dim=1).detach()
                    yhard_flag  = one_hot(labels_flag, num_classes=num_classes, device=device)
                    ystar_flag  = a_flag * yhard_flag + (1.0 - a_flag) * probs_flag
                    logp_flag   = F.log_softmax(logits_flag, dim=1)
                    loss_flag   = -(ystar_flag * logp_flag).sum(dim=1).mean()

                    # Combine
                    if unflagged_idx.numel() > 0:
                        #loss = 0.5 * loss_unflag + 0.5 * loss_flag  # or weighted by counts
                        n_flag = flagged_idx.numel()
                        n_unfl = unflagged_idx.numel()
                        loss = (loss_unflag * n_unfl + loss_flag * n_flag) / (n_unfl + n_flag)

                    else:
                        loss = loss_flag
                else:
                    # No flagged samples in this batch => plain CE
                    loss = ce_vec.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_sum += float(loss.detach().cpu()) * labels.size(0)
            train_cnt += int(labels.numel())

        # Finish mean CE for this epoch (for G-trace)
        epoch_ce_mean.append(ce_sum_for_epoch / max(ce_cnt_for_epoch, 1))

        train_epoch_loss = train_sum / max(train_cnt, 1)
        train_loss_history.append(train_epoch_loss)

        # =========================================================
        # 2) TEST (hard-label CE)
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
                  f"TrainLoss={train_loss_history[-1]:.4f}  "
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
        train_obs_acc = per_class_accuracy(model, train_loader, device, class_ids, true_labels_array=None)
        train_true_labels = np.array(train_loader.dataset.labels, dtype=int)
        train_true_acc = per_class_accuracy(model, train_loader, device, class_ids, true_labels_array=train_true_labels)
        test_acc_per_class = per_class_accuracy(model, test_loader, device, class_ids, true_labels_array=None)

        def _print_classwise(title, acc_dict):
            print(f"  {title}")
            for c in class_ids:
                print(f"    [{c}] {class_names[c]:>10}: {acc_dict[c]:6.2f}%")
        _print_classwise("Train (observed labels)", train_obs_acc)
        _print_classwise("Train (true labels)    ", train_true_acc)
        _print_classwise("Test (clean)           ", test_acc_per_class)
        
        # ---- Noisy-subset train accuracies ----
        noisy_mask = train_loader.dataset.noisy_mask
        true_labels_array = np.array(train_loader.dataset.labels, dtype=int)
        
        noisy_train_acc_observed = subset_accuracy(
            model, train_loader, device, noisy_mask, use_true_labels=False
        )
        noisy_train_acc_true = subset_accuracy(
            model, train_loader, device, noisy_mask, use_true_labels=True, true_labels_array=true_labels_array
        )
        
        print(f"  Train (noisy subset) — observed labels: {noisy_train_acc_observed:5.2f}%")
        print(f"  Train (noisy subset) — true labels    : {noisy_train_acc_true:5.2f}%")

        # =========================================================
        # 3) DETECTION (runs AFTER training; α applies NEXT epoch)
        #    Uses CE history up to *this* epoch (no extra forward).
        # =========================================================
        if USE_CKL and epoch >= k - 1:
            # Build refs per class from last k points in each deque
            per_cls_log_gie = {c: [] for c in range(num_classes)}
            per_cls_log_w   = {c: [] for c in range(num_classes)}
            sample_stats    = {}  # idx -> (gie_tr, w_tr, cls)

            for idx, dq in distance_queues.items():
                if len(dq) < k:
                    continue
                dists, epochs_ = zip(*dq)
                phi  = np.asarray(dists, dtype=float)
                G_tr = np.asarray([epoch_ce_mean[e] for e in epochs_], dtype=float)

                gie_tr, w_tr = lid.compute_GIE_LID(phi, G_tr)
                cls = int(train_loader.dataset.noisy_labels[idx])

                if np.isfinite(gie_tr) and gie_tr > 0.0:
                    per_cls_log_gie[cls].append(np.log(max(gie_tr, EPS)))
                if np.isfinite(w_tr) and w_tr > 0.0:
                    per_cls_log_w[cls].append(np.log(max(w_tr, EPS)))

                sample_stats[idx] = (gie_tr, w_tr, cls)

            cls_log_gie_huber = [huber_mean(per_cls_log_gie[c]) if per_cls_log_gie[c] else float('nan')
                                 for c in range(num_classes)]
            cls_log_w_huber   = [huber_mean(per_cls_log_w[c])   if per_cls_log_w[c]   else float('nan')
                                 for c in range(num_classes)]
            ref_d = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_gie_huber]
            ref_w = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_w_huber]

            per_cls_ckl = {c: [] for c in range(num_classes)}
            sample_ckl  = {}
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

            cls_ckl_mean = [huber_mean(per_cls_ckl[c]) if per_cls_ckl[c] else float('nan')
                            for c in range(num_classes)]
                            
            #cls_ckl_mean = [(float(np.mean(per_cls_ckl[c])) if per_cls_ckl[c] else float('nan'))
            #                for c in range(num_classes)]

            # Update streaks/flags (affects α for NEXT epoch)
            for idx, (val, cls) in sample_ckl.items():
                mean_c = cls_ckl_mean[cls]
                if not np.isfinite(mean_c):
                    continue
                if val > mean_c:
                    above_streak[idx] += 1
                else:
                    above_streak[idx] = 0

                if STICKY_FLAG:
                    if above_streak[idx] >= MIN_RUNS:
                        flagged_noisy[idx] = True
                else:
                    flagged_noisy[idx] = (above_streak[idx] >= MIN_RUNS)

            # Prepare α for NEXT epoch only
            new_alpha = np.full_like(alpha_buffer, np.nan, dtype=np.float32)
            for idx, (val, cls) in sample_ckl.items():
                mean_c = cls_ckl_mean[cls]
                if flagged_noisy[idx] and np.isfinite(mean_c):
                    new_alpha[idx] = float(ckl_to_alpha(val, mean_c, kappa=KAPPA, alpha_floor=ALPHA_FLOOR))

            if STICKY_FLAG:
                # keep existing α for already-flagged if new is NaN
                keep = np.isnan(new_alpha) & (alpha_buffer < 1.0)
                new_alpha[keep] = alpha_buffer[keep]

            alpha_buffer = new_alpha

            # Optional diagnostic: dog flagged counts (NEXT epoch α will apply)
            dog_idx = train_loader.dataset.dog_idx
            flagged_idxs = np.where(flagged_noisy)[0]
            dog_flagged = [i for i in flagged_idxs if int(train_loader.dataset.noisy_labels[i]) == dog_idx]
            correct = sum(1 for i in dog_flagged if train_loader.dataset.group_map.get(i) == 'noisy')
            incorrect = sum(1 for i in dog_flagged if train_loader.dataset.group_map.get(i) == 'dog')
            print(f"[Post-Epoch {epoch+1}] (for next epoch) Dog flagged: total={len(dog_flagged)}, "
                  f"correct={correct}, incorrect={incorrect}")


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
    train_model(model, train_loader, test_loader, NUM_EPOCHS, K_WINDOW, device)

if __name__ == '__main__':
    main()

