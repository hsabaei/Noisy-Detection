
import os, re
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/content/drive/MyDrive/results/100cat-class"
RUNS = {
    "Clean": os.path.join(BASE_DIR, "clean.txt"),
    "Baseline": os.path.join(BASE_DIR, "baseline.txt"),
    "Sticky": os.path.join(BASE_DIR, "sticky_huber_k20.txt"),
    "Non-sticky": os.path.join(BASE_DIR, "notsticky_huber_k20.txt"),

   # "κ = 2": os.path.join(BASE_DIR, "Kappa2.out"),
   # "κ = 3": os.path.join(BASE_DIR, "Kappa3.out"),
}

# Strict format: Epoch [e/N] ... TrainAcc=xx.xx% TestAcc=yy.yy%
PAT_STRICT = re.compile(
    r"Epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\].*?TrainAcc\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%.*?TestAcc\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE
)
# Fallback: looser pattern
PAT_LOOSE = re.compile(
    r"(?:Epoch[^0-9]*?(\d+)).*?TrainAcc\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%.*?TestAcc\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE
)

def parse_log(path: str) -> pd.DataFrame:
    rows = []
    if not os.path.isfile(path):
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame(columns=["epoch","train_acc","test_acc"])
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PAT_STRICT.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group(1)),
                    "train_acc": float(m.group(3)),
                    "test_acc":  float(m.group(4))
                })
    if not rows:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = PAT_LOOSE.search(line)
                if m:
                    rows.append({
                        "epoch": int(m.group(1)),
                        "train_acc": float(m.group(2)),
                        "test_acc":  float(m.group(3))
                    })
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    if df.empty:
        print(f"[WARN] Could not parse epochs from: {path}")
    return df

# Load runs
loaded = {name: parse_log(p) for name, p in RUNS.items()}
has_any = any(len(df) for df in loaded.values())
if not has_any:
    raise SystemExit("No data parsed from any file. Check paths/format.")

from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(7,5))

# Make a color map from the default cycle
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
run_names = list(loaded.keys())
color_map = {name: cycle_colors[i % len(cycle_colors)] for i, name in enumerate(run_names)}

for name, df in loaded.items():
    if df.empty:
        continue
    c = color_map[name]
    # If accuracies are in [0,1], multiply by 100
    # df = df.copy(); df[["train_acc","test_acc"]] *= 100.0

    ax.plot(df["epoch"], df["train_acc"], color=c, linestyle="-",  label=f"{name} – train")
    ax.plot(df["epoch"], df["test_acc"],  color=c, linestyle="--", label=f"{name} – test")

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Training & Test Accuracy vs Epoch (20% noise)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Optional: tighter y-lims with small margin
vals = [v for df in loaded.values() if not df.empty for v in df["train_acc"].tolist()+df["test_acc"].tolist()]
if vals:
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo - 0.02*max(1, hi-lo), hi + 0.02*max(1, hi-lo))

# Legend option A (simple, shows both lines):
ax.legend(ncol=2, fontsize=9)

fig.tight_layout()
plt.show()

# noisy_subset_train_true_only.py
# Plot ONLY the TRUE-label accuracy on the noisy subset during training.

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
BASE_DIR = "/content/drive/MyDrive/results/100cat-class"
FILES = {
    "Baseline":    os.path.join(BASE_DIR, "baseline.txt"),
    "Sticky":      os.path.join(BASE_DIR, "sticky_huber_k20.txt"),
    "Non-sticky":  os.path.join(BASE_DIR, "notsticky_huber_k20.txt"),
}
FIGSIZE = (7, 5)          # inches
LEGEND_OUTSIDE = False     # put legend outside the plot

# Accept em dash or hyphen in logs
EPOCH_RE = re.compile(r"Epoch\s*\[(\d+)\s*/\s*(\d+)\]")
TRUE_NOISY_RE = re.compile(
    r"Train\s*\(noisy subset\)\s*[—-]\s*true labels\s*:?\s*([0-9.]+)%"
)

def parse_true_noisy_series(path: Path):
    """Return (epochs, accuracies) for noisy-subset TRUE-label accuracy."""
    if not path.exists():
        print(f"[skip] Missing file: {path}")
        return [], []

    epochs, vals = [], []
    cur_epoch = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_ep = EPOCH_RE.search(line)
            if m_ep:
                cur_epoch = int(m_ep.group(1))
                continue

            m_true = TRUE_NOISY_RE.search(line)
            if m_true and cur_epoch is not None:
                epochs.append(cur_epoch)
                vals.append(float(m_true.group(1)))

    # Deduplicate by first-seen per epoch
    by_epoch = {}
    for e, v in zip(epochs, vals):
        if e not in by_epoch:
            by_epoch[e] = v

    es = sorted(by_epoch.keys())
    series = [by_epoch[e] for e in es]
    return es, series

def main():
    parsed = {}
    for name, p in FILES.items():
        es, series = parse_true_noisy_series(Path(p))
        if len(es) == 0:
            print(f"[skip] No true-label noisy-subset entries in: {name}")
            continue
        parsed[name] = (es, series)

    if not parsed:
        raise RuntimeError("No runs with true-label noisy-subset accuracy found.")

    # Consistent colors per run
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])
    names = list(parsed.keys())
    colors = {name: cycle[i % len(cycle)] for i, name in enumerate(names)}

    plt.figure(figsize=FIGSIZE)
    for name, (es, series) in parsed.items():
        plt.plot(es, series, label=name, linewidth=1.4, color=colors[name])

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Noisy-subset training accuracy (TRUE labels)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    if LEGEND_OUTSIDE:
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
    else:
        plt.legend()
        plt.tight_layout()

    # Uncomment to save:
    # plt.savefig("noisy_subset_train_true_only.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()

fig, ax = plt.subplots(figsize=(7,5))

# One color per run
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
run_names = list(loaded.keys())
color_map = {name: cycle_colors[i % len(cycle_colors)] for i, name in enumerate(run_names)}

for name, df in loaded.items():
    if df.empty:
        continue
    c = color_map[name]
    # If accuracies are in [0,1], uncomment:
    # df = df.copy(); df["test_acc"] *= 100.0

    ax.plot(df["epoch"], df["test_acc"], color=c, linestyle="-", label=name)

ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Test Accuracy vs Epoch (100 cat → dogs)")
ax.set_ylim(80, 90)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# ✅ Legend in bottom-right corner
ax.legend(fontsize=9, loc="lower right")

ax.margins(x=0.01)
fig.tight_layout()
plt.show()