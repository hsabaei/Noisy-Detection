import os, json
import numpy as np
from collections import deque, defaultdict

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
FLIP_SEED    = 777         # which cats get flipped to dog
N_CATS_FLIP  = 100         # exactly 100 cats → dog

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
EPS = 1e-12  # tiny positive floor so logs never see 0
    
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

# =================== Training & logging ===============
def train_model(model, train_loader, test_loader, num_epochs, k, device):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
    global_criterion     = nn.CrossEntropyLoss()

    distance_queues = {}  # idx -> deque of (loss, epoch)
    loss_history, test_loss_history = [], []
    lid = LIDEstimators(device=device)

    # output dicts split by cohort
    w_fie_cat,  w_fie_dog,  w_fie_noisy  = {}, {}, {}
    w_gtr_cat,  w_gtr_dog,  w_gtr_noisy  = {}, {}, {}
    w_gte_cat,  w_gte_dog,  w_gte_noisy  = {}, {}, {}

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

        # ---- compute FIE/GIE once window is filled ----
        if epoch >= k - 1:
            for idx, dq in distance_queues.items():
                if len(dq) < k: continue
                dists, epochs = zip(*dq)
                phi = np.array(dists, dtype=float)
                G_tr = np.array([loss_history[e]     for e in epochs], dtype=float)
                G_te = np.array([test_loss_history[e] for e in epochs], dtype=float)

                # FIE* uses (phi[:-1], phi[1:])
                fie_star, w_fie = lid.compute_GIE_LID(phi[:-1], phi[1:])
                if np.isnan(fie_star) or abs(fie_star) > 20:
                    print(f"\n[Epoch {epoch}] Instability detected for idx {idx} (FIE)")
                    print(f"  phi = {phi[:-1]}")
                    print(f"  phi+1 = {phi[1:]}")
                    print(f"  FIE = {fie_star}")
                gie_tr,   w_tr  = lid.compute_GIE_LID(phi, G_tr)
                if np.isnan(gie_tr) or abs(gie_tr) > 20:
                    print(f"\n[Epoch {epoch}] Instability detected for idx {idx} (GIE)")
                    print(f"  phi = {phi}")
                    print(f"  G = {G_tr}")
                    print(f"  GIE = {gie_tr}")
                gie_te,   w_te  = lid.compute_GIE_LID(phi, G_te)

                # choose cohort
                cohort = train_loader.dataset.group_map.get(idx, 'other')
                def _push(dct, idx, w, val):
                    w = float(max(w, EPS))
                    v = float(val) if np.isfinite(val) else 0.0
                    dct.setdefault(idx, []).append((w, v))

                if cohort == 'cat':
                    _push(w_fie_cat, idx, w_fie, fie_star)
                    _push(w_gtr_cat, idx, w_tr,  gie_tr)
                    _push(w_gte_cat, idx, w_te,  gie_te)
                elif cohort == 'dog':
                    _push(w_fie_dog, idx, w_fie, fie_star)
                    _push(w_gtr_dog, idx, w_tr,  gie_tr)
                    _push(w_gte_dog, idx, w_te,  gie_te)
                elif cohort == 'noisy':
                    _push(w_fie_noisy, idx, w_fie, fie_star)
                    _push(w_gtr_noisy, idx, w_tr,  gie_tr)
                    _push(w_gte_noisy, idx, w_te,  gie_te)
                # ignore 'other'

        # progress
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc  = compute_accuracy(model, test_loader,  device)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"TrainLoss={loss_history[-1]:.4f} TestLoss={test_loss_history[-1]:.4f} "
              f"TrainAcc={train_acc:.2f}% TestAcc={test_acc:.2f}%")

    # ---- Save pairs (per cohort) ----
    save_pairs_to_file(w_fie_cat,   "cat_FIE_pairs.json")
    save_pairs_to_file(w_fie_dog,   "dog_FIE_pairs.json")
    save_pairs_to_file(w_fie_noisy, "noisy_FIE_pairs.json")

    save_pairs_to_file(w_gtr_cat,   "cat_GIE_train_pairs.json")
    save_pairs_to_file(w_gtr_dog,   "dog_GIE_train_pairs.json")
    save_pairs_to_file(w_gtr_noisy, "noisy_GIE_train_pairs.json")

    save_pairs_to_file(w_gte_cat,   "cat_GIE_test_pairs.json")
    save_pairs_to_file(w_gte_dog,   "dog_GIE_test_pairs.json")
    save_pairs_to_file(w_gte_noisy, "noisy_GIE_test_pairs.json")

    print("Saved pairs for cohorts: cat / dog / noisy.")

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

