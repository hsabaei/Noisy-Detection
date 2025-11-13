import json
import math
import numpy as np
from collections import deque, defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# ======================= Config =======================
SEED = 12345
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS   = 120
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.0
WEIGHT_DECAY = 0.0

K_WINDOW     = 22          # sliding window for FIE/GIE
FLIP_SEED    = 777
NOISE_RATIO  = 0.10
USE_CKL      = True        # True = CKL pipeline (Step A/C); False = CE baseline only

# ---- Step C defaults (edit here) ----
USE_GLOBAL_TAU   = True    # False -> per-class quantiles
TAU_GLOBAL       = 5.0     # from Step-B FPR/TPR sweep (≈2–3% clean FPR mid/late)
ALPHA_PER_CLASS  = 0.02    # only used when USE_GLOBAL_TAU=False
M_RUNS           = 3       # min consecutive epochs above threshold
WARMUP_START_MIN = 45      # start detection no earlier than this epoch index (0-based)

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

class ControlledPairwiseSymmetricNoise(Dataset):
    """
    Wrap CIFAR-10 train set with controlled, pairwise-symmetric label noise.
    - Total flips = round(total_noise_frac * len(train)), e.g., 0.05 * 50000 = 2500.
    - Five symmetric pairs (names): 
        ('cat','dog'), ('deer','horse'), ('automobile','truck'),
        ('airplane','ship'), ('bird','frog')
    - For each pair (A <-> B), we flip the same count A->B and B->A.
    - Exposes:
        .noisy_mask[idx] -> True if idx is flipped
        .group_map[idx]  -> 'noisy_A->B' for flipped; otherwise class name
        .flipped_pairs   -> dict per pair: {'A->B': [...], 'B->A': [...]}
    """
    DEFAULT_PAIRS = [
        ('cat','dog'),
        ('deer','horse'),
        ('automobile','truck'),
        ('airplane','ship'),
        ('bird','frog'),
    ]

    def __init__(self, base_dataset, total_noise_frac=0.05, pairs=None, seed=777):
        self.base = base_dataset
        self.total_noise_frac = float(total_noise_frac)
        self.pairs = list(pairs) if pairs is not None else list(self.DEFAULT_PAIRS)
        self.seed = int(seed)

        # -- CIFAR-10 metadata
        self.class_names = list(self.base.classes)          # ['airplane', 'automobile', ..., 'truck']
        self.labels = list(self.base.targets)               # len=50000 for train split

        # Map pair names -> indices; validate names exist
        def idx_of(name):
            if name not in self.class_names:
                raise ValueError(f"Class '{name}' not found in dataset classes: {self.class_names}")
            return self.class_names.index(name)

        self.pair_indices = [(idx_of(a), idx_of(b)) for (a,b) in self.pairs]
        P = len(self.pair_indices)

        # Build per-class index lists
        class_to_ids = {i: [] for i in range(len(self.class_names))}
        for i, y in enumerate(self.labels):
            class_to_ids[y].append(i)

        n = len(self.labels)
        total_flips = int(round(self.total_noise_frac * n))
        if total_flips <= 0:
            raise ValueError("total_noise_frac too small; results in 0 flips.")
        # Ensure we can achieve pairwise symmetry: allocate per pair an even number
        # Base even allocation per pair
        base_per_pair = total_flips // P
        base_per_pair -= (base_per_pair % 2)  # make it even
        allocated = base_per_pair * P
        leftover = total_flips - allocated

        # Distribute leftover in chunks of 2 per pair (to keep symmetry per pair)
        per_pair_total = [base_per_pair for _ in range(P)]
        li = 0
        while leftover >= 2:
            per_pair_total[li % P] += 2
            leftover -= 2
            li += 1
        # Sanity: if any leftover remains (shouldn't), we can't keep symmetry
        if leftover != 0:
            raise RuntimeError("Cannot allocate flips symmetrically given total_noise_frac and number of pairs.")

        # Now each pair gets per_pair_total[k] flips, split equally A->B and B->A
        rng = np.random.default_rng(self.seed)
        self.noisy_labels = list(self.labels)
        self.noisy_mask = np.zeros(n, dtype=bool)
        self.group_map = {}
        self.flipped_pairs = {}  # {(a_name,b_name): {'A->B': [ids], 'B->A': [ids]}}

        # Initialize group_map with clean class names
        for i, y in enumerate(self.labels):
            self.group_map[i] = self.class_names[y]

        # Perform flips per pair
        for pair_idx, (a, b) in enumerate(self.pair_indices):
            a_name, b_name = self.class_names[a], self.class_names[b]
            pair_key = (a_name, b_name)

            total_for_pair = per_pair_total[pair_idx]
            per_side = total_for_pair // 2

            a_ids = class_to_ids[a]
            b_ids = class_to_ids[b]

            if per_side > len(a_ids) or per_side > len(b_ids):
                raise ValueError(
                    f"Requested {per_side} flips per side for pair ({a_name}<->{b_name}) "
                    f"but class sizes are a={len(a_ids)}, b={len(b_ids)}."
                )

            # Sample without replacement
            a2b_ids = rng.choice(a_ids, size=per_side, replace=False).tolist()
            b2a_ids = rng.choice(b_ids, size=per_side, replace=False).tolist()

            # Apply flips
            for idx in a2b_ids:
                self.noisy_labels[idx] = b
                self.noisy_mask[idx] = True
                self.group_map[idx] = f"noisy_{a_name}->{b_name}"
            for idx in b2a_ids:
                self.noisy_labels[idx] = a
                self.noisy_mask[idx] = True
                self.group_map[idx] = f"noisy_{b_name}->{a_name}"

            # Track selections
            self.flipped_pairs[pair_key] = {
                f"{a_name}->{b_name}": a2b_ids,
                f"{b_name}->{a_name}": b2a_ids,
            }

        # Summary
        total_noisy = int(self.noisy_mask.sum())
        print(f"[Noise] Total flips: {total_noisy} "
              f"({100.0 * total_noisy / n:.2f}% of {n}); "
              f"{P} pairs, symmetric per pair.")

    def __len__(self): 
        return len(self.base)

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

# =============== CKL (finite-boundary, ID form) =================

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
        else:
            # use ground-truth labels from array indexed by sample indices
            indices = batch[2].cpu().numpy()
            true_labels = torch.tensor(true_labels_array[indices], device=device, dtype=torch.long)
            labels = true_labels

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

# =================== Step A helpers & logging ===================

def mad_scaled(x: np.ndarray) -> float:
    """Scaled MAD to be comparable to std under normality."""
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * float(mad)

class StepALogger:
    """
    JSONL writers for per-sample and per-class Step A logs.
    - samples_path: rows (epoch, idx, y_obs, y_true, ckl, mu, s, z)
    - classes_path: rows (epoch, class, n, mu, s, q90, q95, q99)
    """
    def __init__(self, samples_path="logs_stepA_samples.jsonl", classes_path="logs_stepA_classes.jsonl"):
        self.samples_path = samples_path
        self.classes_path = classes_path
        # truncate if exist
        open(self.samples_path, "w").close()
        open(self.classes_path, "w").close()

    def log_samples(self, rows):
        # rows: list[dict]
        with open(self.samples_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def log_classes(self, rows):
        with open(self.classes_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, num_epochs, k, device):
    model = model.to(device)

    print("=== Mode:", "CKL+α(NSES)" if USE_CKL else "Baseline CE", "===")

    # ---- Optimizer & losses ----
    optimizer      = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    ce_hard_vec    = nn.CrossEntropyLoss(reduction='none')   # per-sample (for queues & baseline loss)
    ce_test_mean   = nn.CrossEntropyLoss()                   # scalar (test)

    # ---- Buffers & state ----
    distance_queues = {}   # idx -> deque of (hard-CE loss, epoch)
    epoch_ce_mean   = []   # store mean hard CE per epoch (for G-trace)
    train_loss_history, test_loss_history = [], []

    num_classes = len(train_loader.dataset.base.classes)

    # For CKL path only
    lid = LIDEstimators(device=device)

    # --- Step A logger ---
    stepA = StepALogger(
        samples_path="logs_stepA_samples.jsonl",
        classes_path="logs_stepA_classes.jsonl"
    )

    # --- Step C temporal state (persist across epochs) ---
    run_counter = defaultdict(int)
    final_flag  = defaultdict(int)

    if USE_CKL:
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
        
        print(f"  Train (noisy subset) — corrupted labels: {noisy_train_acc_observed:5.2f}%")
        print(f"  Train (noisy subset) — true labels    : {noisy_train_acc_true:5.2f}%")

        # =========================================================
        # 3) DETECTION (Step A + Step C)
        # =========================================================

        if USE_CKL and epoch >= k - 1:
            # --- Build per-sample CKL from last k points ---
            per_cls_log_gie = {c: [] for c in range(num_classes)}
            per_cls_log_w   = {c: [] for c in range(num_classes)}
            sample_stats    = {}  # idx -> (gie_tr, w_tr, cls_obs)

            for idx, dq in distance_queues.items():
                if len(dq) < k: continue
                dists, epochs_ = zip(*dq)
                phi  = np.asarray(dists, dtype=float)
                G_tr = np.asarray([epoch_ce_mean[e] for e in epochs_], dtype=float)
                gie_tr, w_tr = lid.compute_GIE_LID(phi, G_tr)
                cls_obs = int(train_loader.dataset.noisy_labels[idx])

                if np.isfinite(gie_tr) and gie_tr > 0.0: per_cls_log_gie[cls_obs].append(np.log(max(gie_tr, EPS)))
                if np.isfinite(w_tr)  and w_tr  > 0.0: per_cls_log_w[cls_obs].append(np.log(max(w_tr,  EPS)))
                sample_stats[idx] = (gie_tr, w_tr, cls_obs)

            # reference per class (Huber on logs)
            cls_log_gie_huber = [huber_mean(per_cls_log_gie[c]) if per_cls_log_gie[c] else float('nan')
                                 for c in range(num_classes)]
            cls_log_w_huber   = [huber_mean(per_cls_log_w[c])   if per_cls_log_w[c]   else float('nan')
                                 for c in range(num_classes)]
            ref_d = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_gie_huber]
            ref_w = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_w_huber]

            # compute CKL per sample
            per_cls_ckl = {c: [] for c in range(num_classes)}
            sample_ckl  = {}
            for idx, (gie_tr, w_tr, cls_obs) in sample_stats.items():
                if not (np.isfinite(gie_tr) and gie_tr > 0.0 and np.isfinite(w_tr) and w_tr > 0.0): continue
                d2, w2 = ref_d[cls_obs], ref_w[cls_obs]
                if not (np.isfinite(d2) and d2 > 0.0 and np.isfinite(w2) and w2 > 0.0): continue
                ckl_val = ckl_finite(w_tr, gie_tr, w2, d2)
                if np.isfinite(ckl_val):
                    per_cls_ckl[cls_obs].append(float(ckl_val))
                    sample_ckl[idx] = (float(ckl_val), cls_obs)

            # --- Step A: standardize within observed class, log ---
            y_obs_array  = np.array(train_loader.dataset.noisy_labels, dtype=int)
            y_true_array = np.array(train_loader.dataset.labels, dtype=int)

            valid_indices = sorted(sample_ckl.keys())
            if len(valid_indices) > 0:
                x = np.array([sample_ckl[i][0] for i in valid_indices], dtype=float)  # CKL values
                c_obs = np.array([sample_ckl[i][1] for i in valid_indices], dtype=int)
                y_true_valid = y_true_array[valid_indices]
                y_obs_valid  = y_obs_array[valid_indices]

                mu_c = np.full(num_classes, np.nan, dtype=float)
                s_c  = np.full(num_classes, np.nan, dtype=float)
                n_c  = np.zeros(num_classes, dtype=int)

                for c in range(num_classes):
                    xc = x[c_obs == c]; n_c[c] = int(xc.size)
                    if n_c[c] > 0:
                        mu = float(np.mean(xc)); s = mad_scaled(xc - mu)
                        if not np.isfinite(s) or s < 1e-12: s = 1.0
                        mu_c[c], s_c[c] = mu, s

                mu_vec = mu_c[c_obs]; s_vec = s_c[c_obs]
                s_vec = np.where((~np.isfinite(s_vec)) | (s_vec < 1e-12), 1.0, s_vec)
                z = (x - mu_vec) / (s_vec + 1e-8)

                # per-sample log
                stepA.log_samples([
                    {"epoch": int(epoch), "idx": int(idx), "y_obs": int(y_obs_valid[j]), "y_true": int(y_true_valid[j]),
                     "ckl": float(x[j]), "mu": float(mu_vec[j]), "s": float(s_vec[j]), "z": float(z[j])}
                    for j, idx in enumerate(valid_indices)
                ])
                # per-class log
                class_rows = []
                for c in range(num_classes):
                    zc = z[c_obs == c]
                    if zc.size == 0: continue
                    class_rows.append({
                        "epoch": int(epoch), "class": int(c), "n": int(n_c[c]),
                        "mu": float(mu_c[c]), "s": float(s_c[c]),
                        "q90": float(np.quantile(zc, 0.90)),
                        "q95": float(np.quantile(zc, 0.95)),
                        "q99": float(np.quantile(zc, 0.99)),
                    })
                stepA.log_classes(class_rows)

                # ---------------- Step C: thresholds + temporal gate ----------------
                warmup = (epoch >= max(k-1, WARMUP_START_MIN))

                # choose flags (global tau or per-class quantile)
                if USE_GLOBAL_TAU:
                    flags_epoch = (z > TAU_GLOBAL)
                    tau_meta = {"global_tau": TAU_GLOBAL}
                else:
                    taus_c = np.full(num_classes, TAU_GLOBAL, float)  # fallback to global
                    for c in range(num_classes):
                        zc = z[c_obs == c]
                        if zc.size >= 50:
                            taus_c[c] = float(np.quantile(zc, 1 - ALPHA_PER_CLASS))
                    flags_epoch = z > taus_c[c_obs]
                    tau_meta = {"taus_c": taus_c.tolist(), "alpha": ALPHA_PER_CLASS}

                if not warmup:
                    flags_epoch[:] = False

                # update run counters
                for j, idx in enumerate(valid_indices):
                    if flags_epoch[j]: run_counter[idx] += 1
                    else:              run_counter[idx] = 0
                    final_flag[idx] = 1 if run_counter[idx] >= M_RUNS else 0

                # quick stats for this epoch
                idx2cls = {int(i): int(c) for i, c in zip(valid_indices, c_obs)}
                flag_frac_epoch = float(flags_epoch.mean()) if flags_epoch.size else 0.0
                flag_frac_mruns = float(np.mean([final_flag[i] for i in valid_indices])) if valid_indices else 0.0
                print(f"    StepC: warmup={warmup} flags(epoch)={flag_frac_epoch:.4f} "
                      f"flags(m>={M_RUNS})={flag_frac_mruns:.4f} meta={tau_meta}")
            # else: nothing to log this epoch

# =================== Main =============================
def main():
    device = DEVICE
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])

    train_set_raw = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_set      = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # dataset with noise
    train_set = ControlledPairwiseSymmetricNoise(train_set_raw, total_noise_frac=NOISE_RATIO, seed=FLIP_SEED)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)
    probe_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN12_Model()
    train_model(model, train_loader, test_loader, NUM_EPOCHS, K_WINDOW, device)

if __name__ == '__main__':
    main()

