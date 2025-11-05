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
ax.set_title("Training & Test Accuracy vs Epoch")
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

# --- Per-class accuracy visualizer (Test-clean) for CIFAR-10 logs ---
# Reads per-class lines under "Test (clean)" and plots cat/dog accuracies
# across runs: baseline, sticky_arith, nosticky_arith.
#
# Usage:
#   - Put/point the three files in FILES.
#   - Adjust LABELS_TO_PLOT if you want other classes.
#   - Run; it shows two tables and two charts (no saving, just display).


# ------------------ CONFIG ------------------
# Change these to your paths
FILES = {
    "Clean": Path("/content/drive/MyDrive/results/100cat-class/clean.txt"),
    "Baseline": Path("/content/drive/MyDrive/results/100cat-class/baseline.txt"),
    "Sticky": Path("/content/drive/MyDrive/results/100cat-class/sticky_huber_k20.txt"),
    "Not-sticky": Path("/content/drive/MyDrive/results/100cat-class/notsticky_huber_k20.txt"),
}

# Classes you want to visualize
LABELS_TO_PLOT = ["dog", "cat"]

# CIFAR-10 label order expected in your logs
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CIFAR10_LABELS)}

# ------------------ PARSER ------------------
epoch_re = re.compile(r"Epoch\s*\[(\d+)/\d+\]")
test_header_re = re.compile(r"\s*Test\s*\(clean\)\s*")
# Example line: "    [5]        dog:  84.40%"
class_line_re = re.compile(r"\s*\[(\d)\]\s+[A-Za-z ]+:\s+([\d\.]+)%")

def parse_test_clean_per_class(path: Path) -> Dict[int, List[float]]:
    """
    Returns a dict: class_index -> list of accuracies per epoch.
    """
    if not path.exists():
        print(f"WARNING: {path} not found.")
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    per_class: Dict[int, List[float]] = {i: [] for i in range(10)}
    i, N = 0, len(lines)
    while i < N:
        # Not strictly needed to know the epoch number, but keeps parsing aligned
        if epoch_re.search(lines[i]):
            i += 1
            continue
        if test_header_re.match(lines[i]):
            # Collect up to 10 class lines below this header
            collected, j = 0, i + 1
            while j < N and collected < 10:
                m_class = class_line_re.match(lines[j])
                if m_class:
                    cidx = int(m_class.group(1))
                    acc = float(m_class.group(2))
                    per_class[cidx].append(acc)
                    collected += 1
                j += 1
            i = j
            continue
        i += 1
    return per_class

# ------------------ BUILD DATAFRAMES ------------------
def build_df_for_label(label: str, runs: Dict[str, Dict[int, List[float]]]) -> pd.DataFrame:
    cidx = LABEL_TO_IDX[label]
    max_len = max((len(runs[r].get(cidx, [])) for r in runs), default=0)
    data = {"epoch": list(range(1, max_len + 1))}
    for run_name in runs:
        series = runs[run_name].get(cidx, [])
        padded = series + [float("nan")] * (max_len - len(series))
        data[run_name] = padded
    return pd.DataFrame(data)

def plot_label(df: pd.DataFrame, label: str):
    plt.figure(figsize=(7, 5))  # width=7 inches, height=5 inches

    for col in df.columns:
        if col == "epoch":
            continue
        plt.plot(df["epoch"], df[col], label=col, linewidth=1.0)
    plt.xlabel("Epoch")
    plt.ylabel(f"{label.title()} accuracy (%) on Test (clean)")
    plt.ylim(0, 100)
    plt.title(f"{label.title()} accuracy vs. epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ RUN ------------------
# Parse each run
run_data: Dict[str, Dict[int, List[float]]] = {name: parse_test_clean_per_class(p) for name, p in FILES.items()}

# For each requested label, show a small table and a plot
for lbl in LABELS_TO_PLOT:
    df = build_df_for_label(lbl, run_data)
    print(f"\n=== {lbl.title()} accuracies (Test clean) ===")
    display(df)  # Works in Colab/Jupyter
    plot_label(df, lbl)

# noisy_subset_true_vs_observed_one_figure.py
# Reads 3 logs and plots noisy-subset accuracy (TRUE solid, OBSERVED dashed) in one chart,
# using the same color for each mode across TRUE/OBSERVED.

LOG_FILES = {
    "Baseline": Path("/content/drive/MyDrive/results/100cat-class/baseline.txt"),
    "Sticky": Path("/content/drive/MyDrive/results/100cat-class/sticky_huber_k20.txt"),
    "Not-sticky": Path("/content/drive/MyDrive/results/100cat-class/notsticky_huber_k20.txt"),
}

# Regex patterns for your log format
EPOCH_RE = re.compile(r"Epoch\s*\[(\d+)/(\d+)\]")
TRUE_RE  = re.compile(r"Train\s*\(noisy subset\)\s*—\s*true labels\s*:?\s*([0-9.]+)%")
OBS_RE   = re.compile(r"Train\s*\(noisy subset\)\s*—\s*observed labels\s*:?\s*([0-9.]+)%")


def parse_log(path: Path):
    """Return (epochs, true_vals, obs_vals)."""
    by_epoch = {}
    cur_epoch = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                cur_epoch = int(m.group(1))
                by_epoch.setdefault(cur_epoch, {"true": None, "obs": None})
                continue

            m_true = TRUE_RE.search(line)
            if m_true and cur_epoch is not None:
                by_epoch.setdefault(cur_epoch, {"true": None, "obs": None})
                by_epoch[cur_epoch]["true"] = float(m_true.group(1))
                continue

            m_obs = OBS_RE.search(line)
            if m_obs and cur_epoch is not None:
                by_epoch.setdefault(cur_epoch, {"true": None, "obs": None})
                by_epoch[cur_epoch]["obs"] = float(m_obs.group(1))
                continue

    epochs = sorted(by_epoch.keys())
    true_vals = [by_epoch[e]["true"] for e in epochs]
    obs_vals  = [by_epoch[e]["obs"]  for e in epochs]
    return epochs, true_vals, obs_vals


def main():
    # Parse each file
    parsed = {}
    for name, path in LOG_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        parsed[name] = parse_log(path)

    # Build a consistent color map per mode from mpl's default color cycle
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    name_list = list(parsed.keys())
    color_map = {name: cycle_colors[i % len(cycle_colors)] for i, name in enumerate(name_list)}

    # One figure: TRUE as solid, OBSERVED as dashed, same color per mode
    plt.figure(figsize=(7, 5))  # width=7 inches, height=5 inches

    for name, (epochs, true_vals, obs_vals) in parsed.items():
        c = color_map[name]
        # TRUE (solid)
        plt.plot(epochs, true_vals, label=f"{name} — TRUE", linewidth=1.2, color=c)
        # OBSERVED (dashed) with same color
        plt.plot(epochs, obs_vals, label=f"{name} — OBSERVED", linewidth=1.0, linestyle="--", color=c, alpha=0.8)

    plt.xlabel("Epoch")
    plt.ylabel("Noisy subset accuracy (%)")
    plt.title("Noisy samples accuracy vs. epoch")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

