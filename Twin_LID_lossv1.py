import os, json
import numpy as np
from copy import deepcopy
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

# ======================= Config =======================
SEED = 23456
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS   = 120
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.0
WEIGHT_DECAY = 0.0

K_WINDOW     = 22          # sliding window for FIE/GIE
FLIP_SEED    = 999         # which cats get flipped to dog
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
    - Flip exactly n_flip items with true class 'cat' → label 'dog'.
    - Exposes:
        .noisy_mask[idx] -> True if this idx is flipped cat→dog
        .group_map[idx]  -> 'cat' (clean cat), 'dog' (clean dog), 'noisy' (flipped cat), or 'other'
    """
    def __init__(self, base_dataset, n_flip=100, seed=777):
        self.base = base_dataset
        self.n_flip = int(n_flip)

        classes = list(self.base.classes)          # e.g., [..., 'cat', ..., 'dog', ...]
        self.cat_idx = classes.index('cat')
        self.dog_idx = classes.index('dog')

        self.labels = list(self.base.targets)      # len=50000
        self.cat_ids = [i for i,l in enumerate(self.labels) if l == self.cat_idx]
        self.dog_ids = [i for i,l in enumerate(self.labels) if l == self.dog_idx]

        rng = np.random.default_rng(seed)
        if self.n_flip > len(self.cat_ids):
            raise ValueError(f"Requested {self.n_flip} flips but only {len(self.cat_ids)} cats exist.")
        self.flipped_ids = sorted(rng.choice(self.cat_ids, size=self.n_flip, replace=False).tolist())

        # build noisy labels (or same as clean if n_flip=0)
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

        msg = f"[Noise] Flipped exactly {self.n_flip} cats → dog."
        print(msg)

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, int(self.noisy_labels[idx]), int(idx), bool(self.noisy_mask[idx])

# =================== LID Estimators ===================
EPS = 1e-12  # tiny positive floor so logs never see 0
    
class LIDEstimators:
    def __init__(self, device='cpu'):
        self.device = device

    def _hill(self, R, w):
        R = np.asarray(R, float)
        mask = R > EPS
        R = R[mask]
        if R.size <= 4: return 0.0
        k = R.size - 1
        denom = np.sum(np.log(np.abs(R/(w+1e-7))))
        if denom == 0.0 or not np.isfinite(denom): return 0.0
        return float(-k/denom)

    def compute_GIE_LID(self, phi, G):
        epsilon = 1e-7
        limit0 = np.mean(phi[-3:])
        R = np.abs(phi - limit0); w0 = float(np.max(R)) if R.size else 0.0

        limit1 = np.mean(G[-3:])
        FR = np.abs(G - limit1); w1 = float(np.max(FR)) if FR.size else 0.0

        mask = (R > EPS) & (FR > EPS)
        R_non_zero = R[mask]; FR_non_zero = FR[mask]
        k = R_non_zero.shape[0] - 1
        if k <= 4:
            return EPS, float(max(w0, EPS))

        hill_num = - (k / np.sum(np.log(np.abs(R_non_zero / (w0 + epsilon)))))
        hill_den = - (k / np.sum(np.log(np.abs(FR_non_zero / (w1 + epsilon)))))
        gie = hill_num / hill_den if hill_den != 0 else np.nan
        return float(gie), float(max(w0, EPS))
    
    def compute_Bayes_GIE(self, phi, G, Num0, Den0):
        epsilon = 1e-7
        limit0 = np.mean(phi[-3:])
        R = np.abs(phi - limit0); w0 = float(np.max(R)) if R.size else 0.0
        limit1 = np.mean(G[-3:])
        FR = np.abs(G - limit1); w1 = float(np.max(FR)) if FR.size else 0.0

        mask = (R > EPS) & (FR > EPS)
        R_non_zero = R[mask]; FR_non_zero = FR[mask]
        k = R_non_zero.shape[0] - 1
        if k <= 4:
            return EPS, float(max(w0, EPS)), 0.0, 0.0

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
        return float(LID_Bayes), float(max(w0, EPS)), float(Num1), float(Den1)

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

# ============ Epoch-synced training ===============
def train_model_with_epoch_samplers(
    model, dataset, test_loader, epoch_permutations, k, device, run_tag="run"
):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
    global_criterion     = nn.CrossEntropyLoss()

    distance_queues = {}  # idx -> deque of (loss, epoch)
    loss_history, test_loss_history = [], []
    lid = LIDEstimators(device=device)

    num_bfie, den_bfie = {}, {}
    num_bgie, den_bgie  = {}, {}
    
    # output dicts split by cohort
    w_fie_cat,  w_fie_dog,  w_fie_noisy  = {}, {}, {}
    w_gie_cat,  w_gie_dog,  w_gie_noisy  = {}, {}, {}

    w_bfie_cat,  w_bfie_dog,  w_bfie_noisy  = {}, {}, {}
    w_bgie_cat,  w_bgie_dog,  w_bgie_noisy  = {}, {}, {}

    for epoch in range(NUM_EPOCHS):
        # Build a DataLoader that uses the SAME permutation as the paired run
        perm = epoch_permutations[epoch]
        sampler = SubsetRandomSampler(perm)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)

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
                G_tr = np.array([loss_history[e] for e in epochs], dtype=float)

                # FIE* uses (phi[:-1], phi[1:])
                gie_fie, w_fie = lid.compute_GIE_LID(phi[:-1], phi[1:])
                NF0 = num_bfie.get(idx, 0.0); DF0 = den_bfie.get(idx, 0.0)
                bayes_fie, wb_fie, Num_fie, Den_fie = lid.compute_Bayes_GIE(phi[:-1], phi[1:], NF0, DF0)
                num_bfie[idx] = Num_fie; den_bfie[idx] = Den_fie
                
                gie_tr, w_tr  = lid.compute_GIE_LID(phi, G_tr)
                NG0 = num_bgie.get(idx, 0.0); DG0 = den_bgie.get(idx, 0.0)
                bayes_tr, wb_tr, Num_tr, Den_tr = lid.compute_Bayes_GIE(phi, G_tr, NG0, DG0)
                num_bgie[idx] = Num_tr; den_bgie[idx] = Den_tr

                cohort = dataset.group_map.get(idx, 'other')
                def _push(dct, idx, w, val):
                    w = float(max(w, EPS))
                    v = float(val) if np.isfinite(val) else 0.0
                    dct.setdefault(idx, []).append((w, v))

                if cohort == 'cat':
                    _push(w_fie_cat, idx, w_fie, gie_fie)
                    _push(w_gie_cat, idx, w_tr,  gie_tr)
                    _push(w_bfie_cat, idx, wb_fie, bayes_fie)
                    _push(w_bgie_cat, idx, wb_tr,  bayes_tr)
                elif cohort == 'dog':
                    _push(w_fie_dog, idx, w_fie, gie_fie)
                    _push(w_gie_dog, idx, w_tr,  gie_tr)
                    _push(w_bfie_dog, idx, wb_fie, bayes_fie)
                    _push(w_bgie_dog, idx, wb_tr,  bayes_tr)
                elif cohort == 'noisy':
                    _push(w_fie_noisy, idx, w_fie, gie_fie)
                    _push(w_gie_noisy, idx, w_tr,  gie_tr)
                    _push(w_bfie_noisy, idx, wb_fie, bayes_fie)
                    _push(w_bgie_noisy, idx, wb_tr,  bayes_tr)
                # ignore 'other'

        train_acc = compute_accuracy(model, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False), device)
        print(f"[{run_tag}] Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"TrainLoss={loss_history[-1]:.4f} TestLoss={test_loss_history[-1]:.4f} "
              f"TrainAcc={train_acc:.2f}%")

    # ---- Save pairs (per cohort), tagged ----
    def out(name): return f"{run_tag}_{name}.json"

    save_pairs_to_file(w_fie_cat,   out("cat_FIE_pairs"))
    save_pairs_to_file(w_fie_dog,   out("dog_FIE_pairs"))
    save_pairs_to_file(w_fie_noisy, out("noisy_FIE_pairs"))

    save_pairs_to_file(w_gie_cat,   out("cat_GIE_pairs"))
    save_pairs_to_file(w_gie_dog,   out("dog_GIE_pairs"))
    save_pairs_to_file(w_gie_noisy, out("noisy_GIE_pairs"))
    
    save_pairs_to_file(w_bfie_cat,   out("cat_BFIE_pairs"))
    save_pairs_to_file(w_bfie_dog,   out("dog_BFIE_pairs"))
    save_pairs_to_file(w_bfie_noisy, out("noisy_BFIE_pairs"))

    save_pairs_to_file(w_bgie_cat,   out("cat_BGIE_pairs"))
    save_pairs_to_file(w_bgie_dog,   out("dog_BGIE_pairs"))
    save_pairs_to_file(w_bgie_noisy, out("noisy_BGIE_pairs"))

    print(f"[{run_tag}] Saved pairs for cohorts: cat / dog / noisy.")
    return model

# =================== Main =============================
def main():
    device = DEVICE
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])

    train_set_raw = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_set      = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Two training views over the same base data:
    #   - clean: n_flip=0
    #   - noisy: n_flip=N_CATS_FLIP (controlled cat→dog)
    train_set_clean = ControlledCatDogNoise(train_set_raw, n_flip=0,            seed=FLIP_SEED)
    train_set_noisy = ControlledCatDogNoise(train_set_raw, n_flip=N_CATS_FLIP,  seed=FLIP_SEED)

    # Fixed, non-shuffled test loader
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    # Precompute identical per-epoch permutations to enforce same data order
    n_train = len(train_set_raw)
    epoch_permutations = []
    for ep in range(NUM_EPOCHS):
        gen = torch.Generator()
        gen.manual_seed(SEED + ep)   # identical across runs at the same epoch
        perm = torch.randperm(n_train, generator=gen).tolist()
        epoch_permutations.append(perm)

    # Build a single initialization, then clone into both models
    set_all_seeds(SEED)  # ensure layer init is deterministic
    init_model = ResNet32()
    init_state = deepcopy(init_model.state_dict())

    model_clean = ResNet32(); model_clean.load_state_dict(init_state)
    model_noisy = ResNet32(); model_noisy.load_state_dict(init_state)

    # Train both runs with the same epoch-by-epoch sampling order
    print("\n=== Training CLEAN run (no flips) with fixed initialization ===")
    train_model_with_epoch_samplers(
        model_clean, train_set_clean, test_loader, epoch_permutations, K_WINDOW, device, run_tag="clean"
    )

    print("\n=== Training NOISY run (cat→dog flips) with SAME initialization and SAME per-epoch order ===")
    train_model_with_epoch_samplers(
        model_noisy, train_set_noisy, test_loader, epoch_permutations, K_WINDOW, device, run_tag="noisy"
    )

if __name__ == '__main__':
    main()

