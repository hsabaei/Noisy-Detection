# %% [markdown]
# CIFAR-10 "near-dog" synthetic mislabeled set from CIFAR-100 hard negatives
# - Mine wolf/fox/lion/tiger/leopard/bear that are closest to CIFAR-10 "dog"
# - Save as PNG and label as "dog" (noisy labels), keep true source in CSV.

import os, csv, math, json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# ======================= Config =======================
OUT_DIR = Path("./synthetic_mislabeled_dog")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV = OUT_DIR / "manifest.csv"

# How many synthetic mislabeled "dogs" to export total:
N_TOTAL = 200
# Guarantee a minimum per each candidate class (if available):
MIN_PER_CLASS = 20

# CIFAR-100 fine classes to consider as near-dog animals (all are 32x32)
CANDIDATE_FINE_CLASSES = ["wolf", "fox", "lion", "tiger", "leopard", "bear"]

BATCH_SIZE = 256
NUM_WORKERS = 2
SEED = 123

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ================== Feature Extractor ==================
# Use ImageNet-pretrained ResNet18; strip the final FC for embeddings.
weights = ResNet18_Weights.IMAGENET1K_V1
feat_model = resnet18(weights=weights)
feat_model.fc = nn.Identity()
feat_model.eval().to(DEVICE)

# Use the weights' recommended transforms (resizes to 224, normalizes)
embed_transform = weights.transforms()

# Also keep a simple 32x32 save transform (no norm) for writing PNGs
save_transform = transforms.Compose([
    # CIFAR-100 already 32x32; keep as-is to match CIFAR-10 scale
    # If you want light jitter to better match CIFAR-10 stats, you can add it here.
])

# =================== Datasets/Loaders ==================
root = "./data"

c10_train = CIFAR10(root=root, train=True, download=True)
c100_train = CIFAR100(root=root, train=True, download=True)

# CIFAR-10 class order includes dog=5
C10_CLASSES = c10_train.classes
DOG_IDX = C10_CLASSES.index("dog")  # should be 5

# Map CIFAR-100 fine label names -> indices
c100_name_to_idx = {name: i for i, name in enumerate(c100_train.classes)}
candidate_indices = [c100_name_to_idx[name] for name in CANDIDATE_FINE_CLASSES if name in c100_name_to_idx]
if len(candidate_indices) == 0:
    raise ValueError("None of the desired CIFAR-100 classes found. Check class names.")

# Build subsets:
c10_dog_idxs = [i for i, (_, y) in enumerate(c10_train) if y == DOG_IDX]
c10_dogs = Subset(c10_train, c10_dog_idxs)

c100_candidate_idxs = [i for i, (_, y) in enumerate(c100_train) if y in candidate_indices]
c100_candidates = Subset(c100_train, c100_candidate_idxs)

def make_loader(dataset, tfm, shuffle=False):
    class Wrapped(torch.utils.data.Dataset):
        def __init__(self, base, tfm):
            self.base = base
            self.tfm = tfm
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, y = self.base[i]
            return self.tfm(img), y, i  # keep original index
    return DataLoader(Wrapped(dataset, tfm), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)

dog_loader = make_loader(c10_dogs, embed_transform, shuffle=False)
cand_loader = make_loader(c100_candidates, embed_transform, shuffle=False)

# ==================== Embedding Helpers ====================
@torch.no_grad()
def embed_batches(loader) -> Tuple[torch.Tensor, List[int]]:
    embs = []
    ys = []
    idxs = []
    for xb, yb, ib in tqdm(loader, desc="Embedding"):
        xb = xb.to(DEVICE, non_blocking=True)
        feats = feat_model(xb)              # (B, 512)
        feats = nn.functional.normalize(feats, dim=1)
        embs.append(feats.cpu())
        ys += yb.tolist()
        idxs += ib.tolist()
    embs = torch.cat(embs, dim=0)
    return embs, ys, idxs

print("Embedding CIFAR-10 dogs…")
dog_embs, _, dog_subset_idxs = embed_batches(dog_loader)
dog_centroid = nn.functional.normalize(dog_embs.mean(dim=0, keepdim=True), dim=1)  # (1, D)

print("Embedding CIFAR-100 candidates…")
cand_embs, cand_ys, cand_subset_idxs = embed_batches(cand_loader)

# ==================== Similarity & Ranking ====================
# Cosine sim to dog centroid:
with torch.no_grad():
    sims = (cand_embs @ dog_centroid.t()).squeeze(1).numpy()  # (N,)

# Group by true CIFAR-100 class name for per-class quotas
idx_to_name = {v: k for k, v in c100_name_to_idx.items()}
cand_records = []
for sim, y, sub_i in zip(sims, cand_ys, cand_subset_idxs):
    true_name = idx_to_name[y]
    global_i = c100_candidate_idxs[sub_i]  # original index into CIFAR-100 train
    cand_records.append({"sim": float(sim), "true_idx": global_i, "true_name": true_name})

# Sort globally by similarity (desc)
cand_records.sort(key=lambda r: r["sim"], reverse=True)

# Enforce per-class minimums, then fill remaining by global rank
selected = []
per_class_taken = {name: 0 for name in CANDIDATE_FINE_CLASSES if name in c100_name_to_idx}

# First pass: satisfy per-class minimums
for name in per_class_taken.keys():
    needed = MIN_PER_CLASS
    for rec in cand_records:
        if rec.get("_used"): continue
        if rec["true_name"] == name:
            selected.append(rec)
            rec["_used"] = True
            per_class_taken[name] += 1
            needed -= 1
            if needed <= 0: break

# Second pass: fill up to N_TOTAL
for rec in cand_records:
    if rec.get("_used"): continue
    selected.append(rec)
    rec["_used"] = True
    if len(selected) >= N_TOTAL: break

print(f"Selected {len(selected)} synthetic mislabeled 'dog' images.")

# ==================== Save PNGs + Manifest ====================
# We need the raw 32x32 images to save; read directly from CIFAR-100 base dataset
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(MANIFEST_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "assigned_label", "true_source"])
    for k, rec in enumerate(selected):
        img_pil, true_y = c100_train[rec["true_idx"]]
        # Optional: apply any 32x32-level jitter here via save_transform
        img_pil = save_transform(img_pil) if save_transform else img_pil
        fname = f"syn_dog_{k:05d}__src_{rec['true_name']}.png"
        img_pil.save(OUT_DIR / fname)
        writer.writerow([fname, "dog", rec["true_name"]])

print(f"Saved images to: {OUT_DIR}")
print(f"Manifest: {MANIFEST_CSV}")

# ==================== (Optional) How to Load Mixed Data ====================
# You can now create a Dataset that wraps CIFAR-10 + these synthetic PNGs:
class SyntheticDogFolder(torch.utils.data.Dataset):
    def __init__(self, folder: Path, assigned_idx: int = DOG_IDX, tfm=None):
        self.folder = Path(folder)
        self.files = sorted([p for p in self.folder.iterdir() if p.suffix.lower()==".png"])
        self.assigned_idx = assigned_idx
        self.tfm = tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.tfm: img = self.tfm(img)
        return img, self.assigned_idx

# Example: mix 1:1 with CIFAR-10 training
mix_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),
])

c10_train_tfm = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=mix_transform)
syn_folder_ds = SyntheticDogFolder(OUT_DIR, assigned_idx=DOG_IDX, tfm=mix_transform)

mixed_train = torch.utils.data.ConcatDataset([c10_train_tfm, syn_folder_ds])
print("Mixed dataset size:", len(mixed_train))
