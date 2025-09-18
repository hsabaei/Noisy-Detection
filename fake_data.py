#!/usr/bin/env python3
# Generate near-dog mislabeled images from CIFAR-100 (wolf/fox/lion/tiger/leopard/bear)
# All downloads/caches go to /tmp/$USER to avoid home-quota issues.

import os, csv, hashlib, urllib.request, shutil
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets.utils import extract_archive
from PIL import Image

# ----------------- Force ALL caches to /tmp/$USER -----------------
tmp_root = f"/tmp/{os.environ.get('USER','user')}"
data_root = os.path.join(tmp_root, "datasets")
os.makedirs(data_root, exist_ok=True)
os.environ["TORCH_HOME"]     = os.path.join(tmp_root, "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(tmp_root, "xdg")
os.environ["HF_HOME"]        = os.path.join(tmp_root, "hf")
print(">>> Using tmp_root:", tmp_root, "| data_root:", data_root)
# -----------------------------------------------------------------

# -------------------- tqdm shim (no pip needed) -------------------
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *a, **k): return x
# -----------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================= CONFIG =============================
OUT_DIR = Path("./fake_dogs")      # output folder for PNGs + manifest
N_TOTAL = 200                      # set 100 or 200
MIN_PER_CLASS = 20                 # per-class floor (clamped below)
BATCH_SIZE = 256
NUM_WORKERS = 2
SEED = 123
USE_PRETRAINED = True              # True => 1x 44MB ResNet18 to /tmp; False => no download
CANDIDATE_FINE_CLASSES = ["wolf", "fox", "lion", "tiger", "leopard", "bear"]
# =================================================================

torch.manual_seed(SEED)

# ------------------ Robust CIFAR fetch/extract --------------------
def _download_to(path: str, urls: list[str], desc: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for u in urls:
        try:
            print(f"[fetch] {desc} from {u}")
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
            print(f"[ok] saved to {path}")
            return
        except Exception as e:
            print(f"[warn] {u} failed: {e}")
    raise RuntimeError(f"All mirrors failed for {desc}")

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_cifar_archives(root: str):
    c10_name  = "cifar-10-python.tar.gz"
    c100_name = "cifar-100-python.tar.gz"
    c10_path  = os.path.join(root, c10_name)
    c100_path = os.path.join(root, c100_name)

    C10_URLS = [
        "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz",
        "https://download.pytorch.org/tutorial/cifar10_tutorial/cifar-10-python.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ]
    C100_URLS = [
        "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-100-python.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    ]

    if not os.path.exists(c10_path):
        _download_to(c10_path, C10_URLS, "CIFAR-10")
    if not os.path.exists(c100_path):
        _download_to(c100_path, C100_URLS, "CIFAR-100")

    OFFICIAL = {
        c10_path:  "c58f30108f718f92721af3b95e74349a",
        c100_path: "eb9058c3a382ffc7106e4002c42a8d85",
    }
    for p, md5 in OFFICIAL.items():
        cur = _md5(p)
        if cur != md5:
            try: os.remove(p)
            except: pass
            raise RuntimeError(f"MD5 mismatch for {os.path.basename(p)} (have {cur}, expected {md5}).")

def ensure_cifar_extracted(root: str):
    c10_dir  = os.path.join(root,  "cifar-10-batches-py")
    c100_dir = os.path.join(root, "cifar-100-python")
    if not os.path.isdir(c10_dir):
        extract_archive(os.path.join(root, "cifar-10-python.tar.gz"), root)
    if not os.path.isdir(c100_dir):
        extract_archive(os.path.join(root, "cifar-100-python.tar.gz"), root)

# Fetch/extract once to /tmp/$USER/datasets, then load with download=False
ensure_cifar_archives(data_root)
ensure_cifar_extracted(data_root)
# -----------------------------------------------------------------

# -------- Feature extractor (ImageNet ResNet-18; silent) ----------
if USE_PRETRAINED:
    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights, progress=False)
    embed_transform = weights.transforms()  # resize 224 + normalize
else:
    backbone = resnet18(weights=None, progress=False)
    embed_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
backbone.fc = nn.Identity()
backbone.eval().to(DEVICE)
# -----------------------------------------------------------------

# ------------- Datasets: raw (32x32) for saving ------------------
from torchvision.datasets import CIFAR10, CIFAR100
c10_raw  = CIFAR10(root=data_root, train=True,  download=False, transform=None)
c100_raw = CIFAR100(root=data_root, train=True, download=False, transform=None)

C10_CLASSES  = c10_raw.classes
C100_CLASSES = c100_raw.classes

DOG_IDX = C10_CLASSES.index("dog")
dog_indices = [i for i, y in enumerate(c10_raw.targets) if y == DOG_IDX]
c10_dogs_raw = Subset(c10_raw, dog_indices)

name_to_idx_100 = {name: i for i, name in enumerate(C100_CLASSES)}
cand_idx = [name_to_idx_100[name] for name in CANDIDATE_FINE_CLASSES if name in name_to_idx_100]
if not cand_idx:
    raise RuntimeError("None of the desired CIFAR-100 fine classes were found.")
cand_indices = [i for i, y in enumerate(c100_raw.targets) if y in cand_idx]
c100_cands_raw = Subset(c100_raw, cand_indices)
# -----------------------------------------------------------------

# -------- Wrapper to apply embedding transform lazily -------------
class EmbedWrap(Dataset):
    def __init__(self, base: Dataset, tfm):
        self.base = base
        self.tfm = tfm
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, y = self.base[i]  # PIL.Image, int
        return self.tfm(img), y, i, img  # tensor for model; keep original PIL for saving

# ---- custom collate: stack tensors, keep PILs as python lists ----
def collate_keep_pil(batch):
    xs, ys, idxs, pils = zip(*batch)
    xs   = torch.stack(xs, 0)
    ys   = torch.as_tensor(ys)
    idxs = list(idxs)
    pils = list(pils)
    return xs, ys, idxs, pils

def make_loader(ds, shuffle=False):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_keep_pil,  # <<< important to avoid PIL collate error
    )

dogs_embed = EmbedWrap(c10_dogs_raw, embed_transform)
cands_embed = EmbedWrap(c100_cands_raw, embed_transform)

dog_loader  = make_loader(dogs_embed,  shuffle=False)
cand_loader = make_loader(cands_embed, shuffle=False)
# -----------------------------------------------------------------

@torch.no_grad()
def embed(loader) -> Tuple[torch.Tensor, List[int], List[Image.Image]]:
    feats, labels, pils = [], [], []
    for xb, yb, ib, pilb in tqdm(loader, desc="Embedding"):
        xb = xb.to(DEVICE, non_blocking=True)
        fb = backbone(xb)
        fb = nn.functional.normalize(fb, dim=1)  # cosine-friendly
        feats.append(fb.cpu())
        labels.extend(yb.tolist() if isinstance(yb, torch.Tensor) else yb)
        pils.extend(pilb)  # list of PILs
    return torch.cat(feats, dim=0), labels, pils

# ---------- Compute dog centroid ----------
dog_feats, _, _ = embed(dog_loader)
dog_centroid = nn.functional.normalize(dog_feats.mean(dim=0, keepdim=True), dim=1)

# ---------- Embed candidates + score ----------
cand_feats, cand_labels_100, cand_pils = embed(cand_loader)
sims = (cand_feats @ dog_centroid.t()).squeeze(1)

idx_to_name_100 = {v: k for k, v in name_to_idx_100.items()}
records = []
for i, (sim, y) in enumerate(zip(sims.tolist(), cand_labels_100)):
    records.append({"i": i, "sim": sim, "true_name": idx_to_name_100[y]})

records.sort(key=lambda r: r["sim"], reverse=True)

# ---------- Select with per-class mins (clamped) ----------
num_classes_avail = max(1, len({r["true_name"] for r in records}))
per_class_min = min(MIN_PER_CLASS, N_TOTAL // num_classes_avail)

selected = []
taken = {name: 0 for name in CANDIDATE_FINE_CLASSES if name in name_to_idx_100}

# First: satisfy per-class minimums
for cname in taken:
    need = per_class_min
    if need <= 0: break
    for r in records:
        if r.get("_used"): continue
        if r["true_name"] == cname:
            selected.append(r); r["_used"] = True
            taken[cname] += 1
            need -= 1
            if need <= 0: break

# Then: fill globally by similarity
for r in records:
    if len(selected) >= N_TOTAL: break
    if not r.get("_used"):
        selected.append(r); r["_used"] = True

print(f"Selected {len(selected)} images "
      f"(per-class floor={per_class_min}, classes available={num_classes_avail}).")

# ---------------- Save outputs ----------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = OUT_DIR / "manifest.csv"
with open(manifest, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["filename", "assigned_label", "true_source", "similarity"])
    for k, r in enumerate(selected):
        pil = cand_pils[r["i"]]             # original 32x32 PIL
        fn = f"syn_dog_{k:05d}__src_{r['true_name']}.png"
        pil.save(OUT_DIR / fn)
        wr.writerow([fn, "dog", r["true_name"], f"{r['sim']:.6f}"])

print("Saved images to:", str(OUT_DIR))
print("Manifest:", str(manifest))
