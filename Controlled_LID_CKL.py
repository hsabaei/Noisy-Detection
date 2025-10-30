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
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-3

K_WINDOW     = 22          # sliding window for FIE/GIE
FLIP_SEED    = 777         # which cats get flipped to dog
N_CATS_FLIP  = 100         # exactly 100 cats → dog
MIN_RUNS = 5 

EPS = 1e-12

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

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, num_epochs, k, device):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
    global_criterion     = nn.CrossEntropyLoss()

    distance_queues = {}  # idx -> deque of (loss, epoch)
    loss_history, test_loss_history = [], []
    lid = LIDEstimators(device=device)

    n_samples = len(train_loader.dataset)
    above_streak = defaultdict(int)                 # idx -> consecutive "above mean" count
    flagged_noisy = np.zeros(n_samples, dtype=bool) # persistent flags (True once met criterion)

  
    for epoch in range(num_epochs):
        model.train()
        running_loss_sum, running_count = 0.0, 0

        for inputs, labels, indices, is_noisy in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses  = per_sample_criterion(outputs, labels)

            # accumulate global train loss
            running_loss_sum += float(losses.sum().detach().cpu())
            running_count    += int(labels.numel())

            # track per-sample loss for all indices
            for i in range(len(indices)):
                idx = int(indices[i])
                if idx not in distance_queues:
                    distance_queues[idx] = deque(maxlen=k)
                distance_queues[idx].append((float(losses[i].detach().cpu()), epoch))

            optimizer.zero_grad()
            losses.mean().backward()
            optimizer.step()

        loss_history.append(running_loss_sum / max(running_count, 1))

        # test loss
        model.eval()
        t_sum, t_cnt = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                t_sum += float(global_criterion(logits, y).detach().cpu()) * y.size(0)
                t_cnt += int(y.numel())
        test_loss_history.append(t_sum / max(t_cnt, 1))

        num_classes = len(train_loader.dataset.base.classes)

        # ===== FIE/GIE + class refs + per-sample CKL (compute-only) =====
        if epoch >= k - 1:
            per_cls_log_gie = {c: [] for c in range(num_classes)}
            per_cls_log_w   = {c: [] for c in range(num_classes)}
            sample_stats = {}  # idx -> (gie_tr, w_tr, cls)

            # --- First pass: compute once, bucket logs, cache stats ---
            for idx, dq in distance_queues.items():
                if len(dq) < k:
                    continue
                dists, epochs_ = zip(*dq)
                phi  = np.array(dists, dtype=float)
                G_tr = np.array([loss_history[e] for e in epochs_], dtype=float)

                gie_tr, w_tr = lid.compute_GIE_LID(phi, G_tr)
                cls = int(train_loader.dataset.noisy_labels[idx])

                if np.isfinite(gie_tr) and gie_tr > 0.0:
                    per_cls_log_gie[cls].append(np.log(max(gie_tr, EPS)))
                if np.isfinite(w_tr) and w_tr > 0.0:
                    per_cls_log_w[cls].append(np.log(max(w_tr, EPS)))

                sample_stats[idx] = (gie_tr, w_tr, cls)  # cache even if later filtered

            # --- Class refs (Huber mean on logs, then exp back to original scale) ---
            cls_log_gie_huber = [huber_mean(per_cls_log_gie[c]) if per_cls_log_gie[c] else float('nan')
                                for c in range(num_classes)]
            cls_log_w_huber   = [huber_mean(per_cls_log_w[c])   if per_cls_log_w[c]   else float('nan')
                                for c in range(num_classes)]
            ref_d = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_gie_huber]
            ref_w = [math.exp(v) if np.isfinite(v) else np.nan for v in cls_log_w_huber]

            # --- Second pass: compute CKL using cached per-sample stats + refs ---
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

            # Compute the MEAN CKL per class for this epoch
            cls_ckl_mean = [
                (float(np.mean(per_cls_ckl[c])) if per_cls_ckl[c] else float('nan'))
                for c in range(num_classes)
            ]

            # Compare each sample CKL to its class mean (compute-only + streak/flags)
            for idx, (val, cls) in sample_ckl.items():
                mean_c = cls_ckl_mean[cls]
                if not np.isfinite(mean_c):
                    continue  # skip this sample this epoch (no penalty, no credit)

                if val > mean_c:                    # above class mean this epoch
                    above_streak[idx] += 1
                    if above_streak[idx] >= MIN_RUNS:
                        flagged_noisy[idx] = True   # sticky flag once threshold met
                else:
                    above_streak[idx] = 0           # reset consecutive streak

            # ---- Epoch print: dog flagged (correct vs incorrect) ----
            dog_idx = train_loader.dataset.dog_idx  # from ControlledCatDogNoise
            flagged_idxs = np.where(flagged_noisy)[0]

            dog_flagged = [i for i in flagged_idxs
                           if int(train_loader.dataset.noisy_labels[i]) == dog_idx]

            correct = sum(1 for i in dog_flagged
                          if train_loader.dataset.group_map.get(i) == 'noisy')  # flipped cat→dog
            incorrect = sum(1 for i in dog_flagged
                            if train_loader.dataset.group_map.get(i) == 'dog')   # true clean dog

            print(f"[Epoch {epoch+1}] Dog flagged: total={len(dog_flagged)}, "
                  f"correct={correct}, incorrect={incorrect}")


        # progress
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

