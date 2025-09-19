USE_DRIVE = True

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive'
else:
    BASE = '/content'

import os, time
DATA_ROOT  = f'{BASE}/data'                   # CIFAR-10/100 download here
POISON_DIR = f'{BASE}/synthetic_mislabeled'   # wolf/fox PNGs + CSV
RUNS_DIR   = f'{BASE}/runs'                   # each training run writes here
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(POISON_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Create a fresh run folder
RUN_DIR = f"{RUNS_DIR}/{time.strftime('%Y%m%d-%H%M%S')}-cifar-poison"
os.makedirs(RUN_DIR, exist_ok=True)
print("DATA_ROOT:", DATA_ROOT)
print("POISON_DIR:", POISON_DIR)
print("RUN_DIR:", RUN_DIR)

import os, json, csv, random, numpy as np
from collections import deque
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision, torchvision.transforms as transforms
from PIL import Image

# -------------------- determinism --------------------
EPS = 1e-12
def set_all_seeds(seed=12345):
    import random as pyr
    torch.manual_seed(seed); np.random.seed(seed); pyr.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------- model -----------------------
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
                nn.Conv2d(in_planes, self.expansion*planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)); out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1  = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.penultimate = None
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = F.avg_pool2d(out, 8); out = out.view(out.size(0), -1)
        self.penultimate = out
        return self.linear(out)

def ResNet32(): return ResNet(BasicBlock, [5,5,5])

# ---------------- poison add dataset ----------------
class CIFAR10PoisonAdd(Dataset):
    """
    Append k_add fake 'dog' images (paths from CSV) to a clean CIFAR-10 train set.
    Returns (x, y, global_idx, is_noisy).
    """
    def __init__(self, base_dataset, poison_csv, k_add=100, seed=777):
        self.base = base_dataset
        self.transform = getattr(self.base, "transform", transforms.ToTensor())
        self.classes = list(self.base.classes)
        self.dog_idx = self.classes.index('dog')

        with open(poison_csv, 'r') as f:
            rows = list(csv.DictReader(f))
        if not rows: raise ValueError(f"No rows in {poison_csv}")
        random.Random(seed).shuffle(rows)
        self.poison = rows[:int(k_add)]
        self.n_base, self.n_poison = len(self.base), len(self.poison)

        # cohort map
        self.group_map = {}
        base_targets = getattr(self.base, "targets", None)
        for i in range(self.n_base):
            y = int(base_targets[i]) if base_targets is not None else -1
            self.group_map[i] = ('dog' if y==self.dog_idx else 'other')
        for j in range(self.n_poison):
            self.group_map[self.n_base + j] = 'noisy'
        print(f"[poison-add] +{self.n_poison} fake dogs; total len={len(self)}")

    def __len__(self): return self.n_base + self.n_poison

    def __getitem__(self, idx):
        if idx < self.n_base:
            x, y = self.base[idx]
            return x, int(y), int(idx), False
        row = self.poison[idx - self.n_base]
        x = Image.open(row["path"]).convert("RGB")
        if self.transform: x = self.transform(x)
        return x, self.dog_idx, int(idx), True

# ------------------- LID estimators ------------------
class LIDEstimators:
    def compute_GIE_LID(self, phi, G):
        epsilon = 1e-7
        limit0 = np.mean(phi[-3:]); R = np.abs(phi - limit0); w0 = np.max(R) if R.size else EPS
        limit1 = np.mean(G[-3:]);   FR = np.abs(G - limit1);   w1 = np.max(FR) if FR.size else EPS
        mask = (R > EPS) & (FR > EPS); Rn, FRn = R[mask], FR[mask]
        k = Rn.shape[0] - 1
        if k <= 4: return EPS, float(w0)
        hn = - (k / np.sum(np.log(np.abs(Rn / (w0 + epsilon)))))
        hd = - (k / np.sum(np.log(np.abs(FRn / (w1 + epsilon)))))
        gie = hn / hd if hd != 0 else np.nan
        return float(gie), float(w0)

# ----------------------- helpers ---------------------
def save_pairs(pairs_dict, path):
    js = {str(i): [[float(w), float(v)] for (w, v) in pairs] for i, pairs in pairs_dict.items()}
    with open(path, "w") as f: json.dump(js, f, indent=2)

def compute_accuracy(model, loader, device):
    """Accepts (x,y) or (x,y,idx,is_noisy)."""
    model.eval(); correct=total=0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.numel()
    model.train(); return 100*correct/max(total,1)

def ensure_cifar10_100(root):
    torchvision.datasets.CIFAR10(root=root, train=True,  download=True)
    torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    torchvision.datasets.CIFAR100(root=root, train=False, download=True)

def build_wolf_fox_csv(poison_dir, data_root, k_add, rebuild=False):
    poison_dir = Path(poison_dir); poison_dir.mkdir(parents=True, exist_ok=True)
    csvp = poison_dir/"labels_external_c100_wolf_fox.csv"
    if csvp.exists() and not rebuild:
        print("[auto] using existing", csvp); return str(csvp)
    c100 = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
    c10  = torchvision.datasets.CIFAR10(root=data_root,  train=False, download=True)
    dog_idx = c10.classes.index('dog')
    rows=[]
    for i in range(len(c100)):
        img,y = c100[i]
        if c100.classes[y] in {"wolf","fox"}:
            p = poison_dir/f"external_c100_{c100.classes[y]}_{len(rows):05d}.png"
            img.save(p)
            rows.append({"path":str(p), "label":dog_idx, "is_outlier_gt":1,
                         "source":"external_c100","difficulty":"natural",
                         "p_dog_before":"","p_dog_after":"",
                         "alpha":0,"steps":0,"l2":0,"tv":0,
                         "src_idx_non_dog":-1,"src_idx_dog":-1,
                         "beta":0,"latent_side":0,"lr":0})
    if not rows: raise RuntimeError("No wolf/fox found (unexpected).")
    with open(csvp,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[auto] wrote {len(rows)} rows -> {csvp}")
    if len(rows)<k_add: print(f"[warn] have {len(rows)} < k_add={k_add} (will use all)")
    return str(csvp)

# -------------------- training loop ------------------
def train_model(model, train_loader, test_loader, epochs, k_window, device,
                lr=0.1, momentum=0.0, weight_decay=0.0, out_dir="."):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    per = nn.CrossEntropyLoss(reduction='none'); glob = nn.CrossEntropyLoss()

    queues = {}  # idx -> deque of (loss, epoch)
    loss_hist, test_hist = [], []
    lid = LIDEstimators()

    # per-cohort logs
    w_fie_dog, w_fie_noisy = {}, {}
    w_gie_dog, w_gie_noisy = {}, {}

    for epoch in range(epochs):
        model.train(); run_sum=0.0; run_cnt=0
        for x,y,idx,is_noisy in train_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            losses = per(logits,y)
            run_sum += float(losses.sum().detach().cpu()); run_cnt += int(y.numel())

            for i in range(len(idx)):
                ii=int(idx[i])
                dq = queues.get(ii)
                if dq is None: dq = queues[ii] = deque(maxlen=k_window)
                dq.append((float(losses[i].detach().cpu()), epoch))

            opt.zero_grad(); losses.mean().backward(); opt.step()

        # epoch losses
        loss_hist.append(run_sum/max(run_cnt,1))
        model.eval(); t_sum=t_cnt=0
        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device), y.to(device)
                t_sum += float(glob(model(x),y).detach().cpu()) * y.size(0)
                t_cnt += int(y.numel())
        test_hist.append(t_sum/max(t_cnt,1))

        # compute FIE/GIE once window is full
        if epoch >= k_window-1:
            for ii,dq in queues.items():
                if len(dq)<k_window: continue
                dists, eps = zip(*dq)
                phi = np.array(dists, float)
                G   = np.array([loss_hist[e] for e in eps], float)

                fie, wf = lid.compute_GIE_LID(phi[:-1], phi[1:])
                gie, wt = lid.compute_GIE_LID(phi, G)

                cohort = train_loader.dataset.group_map.get(ii,'other')
                def push(dic, w, v):
                    dic.setdefault(ii, []).append((float(max(w,EPS)), float(v if np.isfinite(v) else 0.0)))
                if cohort=='dog':   push(w_fie_dog, wf, fie); push(w_gie_dog, wt, gie)
                elif cohort=='noisy': push(w_fie_noisy, wf, fie); push(w_gie_noisy, wt, gie)

        tr_acc = compute_accuracy(model, train_loader, device)
        te_acc = compute_accuracy(model, test_loader,  device)
        print(f"Epoch [{epoch+1}/{epochs}] TrainLoss={loss_hist[-1]:.4f} TestLoss={test_hist[-1]:.4f} "
              f"TrainAcc={tr_acc:.2f}% TestAcc={te_acc:.2f}%")

    # save JSONs
    save_pairs(w_fie_dog,   os.path.join(out_dir, "dog_clean_FIE_pairs.json"))
    save_pairs(w_fie_noisy, os.path.join(out_dir, "dog_fake_FIE_pairs.json"))
    save_pairs(w_gie_dog,   os.path.join(out_dir, "dog_clean_GIE_pairs.json"))
    save_pairs(w_gie_noisy, os.path.join(out_dir, "dog_fake_GIE_pairs.json"))
    print("Saved JSONs to", out_dir)

set_all_seeds(12345)

ensure_cifar10_100(DATA_ROOT)

POISON_CSV = build_wolf_fox_csv(POISON_DIR, DATA_ROOT, k_add=100, rebuild=False)
print("POISON_CSV:", POISON_CSV)

# transforms and datasets
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])

train_base = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=train_tf)
test_set   = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tf)
train_set  = CIFAR10PoisonAdd(train_base, poison_csv=POISON_CSV, k_add=100, seed=777)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device, "| Train size:", len(train_set), "| Test size:", len(test_set))

model = ResNet32()

train_model(model, train_loader, test_loader,
            epochs=100, k_window=21, device=device,
            lr=0.1, momentum=0.0, weight_decay=0.0,
            out_dir=RUN_DIR)