# === CKL per-epoch counts (threshold = mean CKL at each epoch) ================
BASE_DIR   = "/content/drive/MyDrive/runs/20250918-233203-cifar-poison"
DOG_FILE   = "dog_clean_GIE_pairs.json"
NOISY_FILE = "dog_fake_GIE_pairs.json"
OUT_PREFIX = "ckl_unionref_epoch_meanThr"

import os, json, math
import numpy as np
import pandas as pd

EPS = 1e-12

# ---------- Load JSON → {id: [(W, d), ...]} ----------
def load_pairs_from_id(path):
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for k, seq in raw.items():
        try: sid = int(k)
        except: sid = k
        cleaned = []
        for x in seq:
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                w, d = float(x[0]), float(x[1])
                if np.isfinite(w) and np.isfinite(d) and w > 0 and d > 0:
                    cleaned.append((w, d, math.log(d)))
        if cleaned:
            out[sid] = cleaned
    return out

# ---------- CKL (finite boundary, ID form) ----------
def ckl_equal_W(W, d1, d2):
    return W * ((d2 - d1)**2) / (((d1 + 1.0)**2) * (d2 + 1.0) + EPS)

def ckl_case_A(W1, d1, W2, d2):
    # W1 < W2 : CKL(F_{W1,d1} : F_{W2,d2})
    term = ( W1 * ((d2/(d1+1.0)) * math.log(max(W2/W1, EPS)) - (d1 - d2)/((d1+1.0)**2))
           + d2 * ((W2 - W1) + W1 * math.log(max(W1/W2, EPS))) )
    return term + (d1/(d1+1.0))*W1 - (d2/(d2+1.0))*W2

def ckl_case_B(W1, d1, W2, d2):
    # W2 < W1 : CKL(F_{W1,d1} : F_{W2,d2})
    ratio = max(W2/W1, EPS)
    term = (W1 / ((d1 + 1.0)**2)) * ( d2 * (ratio**(d1 + 1.0)) - d1 )
    return term + (d1/(d1+1.0))*W1 - (d2/(d2+1.0))*W2

def ckl_finite_boundary(W1, d1, W2, d2):
    if not all(np.isfinite([W1,W2,d1,d2])) or min(W1,W2,d1,d2) <= 0:
        return np.nan
    if abs(W1 - W2) < 1e-12:
        return ckl_equal_W(W1, d1, d2)
    return ckl_case_A(W1, d1, W2, d2) if W1 < W2 else ckl_case_B(W1, d1, W2, d2)

def ckl_sym(W1, d1, W2, d2):
    a = ckl_finite_boundary(W1, d1, W2, d2)
    b = ckl_finite_boundary(W2, d2, W1, d1)
    return 0.5 * (a + b)

# ---------- Build per-epoch union reference (ARITHMETIC MEAN in linear space) --
def build_union_reference_mean(dog, noisy):
    """
    For each epoch t, collect all (W_i,t, d_i,t) from dog ∪ noisy and set:
        W_ref(t) = mean_i W_i,t
        d_ref(t) = mean_i d_i,t
    Returns dict: epoch -> (W_ref, d_ref, n_used)
    """
    max_T = max((len(s) for s in list(dog.values()) + list(noisy.values())), default=0)
    refs = {}
    for t in range(max_T):
        Ws, Ds = [], []
        for seq in dog.values():
            if len(seq) > t:
                w, d, _ = seq[t]
                if np.isfinite(w) and np.isfinite(d) and w > 0 and d > 0:
                    Ws.append(w); Ds.append(d)
        for seq in noisy.values():
            if len(seq) > t:
                w, d, _ = seq[t]
                if np.isfinite(w) and np.isfinite(d) and w > 0 and d > 0:
                    Ws.append(w); Ds.append(d)
        if not Ws:
            continue
        W_ref = float(np.mean(Ws))
        d_ref = float(np.mean(Ds))
        refs[t] = (W_ref, d_ref, len(Ws))
    return refs

# ---------- Per-epoch metrics with gate + enrichment --------------------------
def per_epoch_metrics_with_gate(dog, noisy, refs, min_run=3):
    rows, counts = [], []
    runlen = {}  # sid -> current consecutive run length

    max_T = max((len(s) for s in list(dog.values()) + list(noisy.values())), default=0)
    for t in range(max_T):
        if t not in refs:
            continue
        W_ref, d_ref, n_used = refs[t]

        # Scores for all samples at epoch t
        epoch_rows = []  # (sid, cohort, w, d, logd, score)
        for sid, seq in dog.items():
            if len(seq) > t:
                w, d, logd = seq[t]
                score = ckl_sym(w, d, W_ref, d_ref) if USE_SYMMETRIC else ckl_finite_boundary(w, d, W_ref, d_ref)
                epoch_rows.append((sid, "dog_clean", w, d, logd, score))
        for sid, seq in noisy.items():
            if len(seq) > t:
                w, d, logd = seq[t]
                score = ckl_sym(w, d, W_ref, d_ref) if USE_SYMMETRIC else ckl_finite_boundary(w, d, W_ref, d_ref)
                epoch_rows.append((sid, "dog_noisy", w, d, logd, score))
        if not epoch_rows:
            continue

        scores = [r[5] for r in epoch_rows if np.isfinite(r[5])]
        thr = float(np.mean(scores)) if scores else np.nan

        # Raw flags + consecutive gate
        n_clean_avail = n_noisy_avail = 0
        hit_clean_raw = hit_noisy_raw = 0
        hit_clean_gate = hit_noisy_gate = 0

        for sid, cohort, w, d, logd, score in epoch_rows:
            raw_flag = np.isfinite(score) and np.isfinite(thr) and (score > thr)
            prev = runlen.get(sid, 0)
            curr = prev + 1 if raw_flag else 0
            runlen[sid] = curr
            gated_flag = (curr >= min_run)

            rows.append({
                "epoch": t, "sample_id": sid, "cohort": cohort,
                "W": w, "d": d, "log_d": logd,
                "W_ref": W_ref, "d_ref": d_ref,
                "CKL": score, "threshold": thr,
                "raw_flag": bool(raw_flag),
                "run_len": int(curr),
                "gated_flag": bool(gated_flag)
            })

            if cohort == "dog_clean":
                n_clean_avail += 1
                if raw_flag:   hit_clean_raw  += 1
                if gated_flag: hit_clean_gate += 1
            else:
                n_noisy_avail += 1
                if raw_flag:   hit_noisy_raw  += 1
                if gated_flag: hit_noisy_gate += 1

        # Raw metrics
        denom_raw = hit_clean_raw + hit_noisy_raw
        precision_raw = (hit_noisy_raw / denom_raw) if denom_raw > 0 else np.nan
        tpr_raw = (hit_noisy_raw / n_noisy_avail) if n_noisy_avail > 0 else np.nan
        fpr_raw = (hit_clean_raw / n_clean_avail) if n_clean_avail > 0 else np.nan
        enrichment_raw = (tpr_raw / fpr_raw) if (fpr_raw and np.isfinite(fpr_raw)) else np.nan

        # Gated metrics
        denom_gate = hit_clean_gate + hit_noisy_gate
        precision_gate = (hit_noisy_gate / denom_gate) if denom_gate > 0 else np.nan
        tpr_gate = (hit_noisy_gate / n_noisy_avail) if n_noisy_avail > 0 else np.nan
        fpr_gate = (hit_clean_gate / n_clean_avail) if n_clean_avail > 0 else np.nan
        enrichment_gate = (tpr_gate / fpr_gate) if (fpr_gate and np.isfinite(fpr_gate)) else np.nan

        counts.append({
            "epoch": t, "n_ref_used": n_used, "threshold": thr, "min_run": min_run,
            "n_clean_avail": n_clean_avail, "n_noisy_avail": n_noisy_avail,

            # raw
            "n_clean_hits_raw": hit_clean_raw, "n_noisy_hits_raw": hit_noisy_raw,
            "precision_hits_raw": precision_raw,
            "hit_rate_clean_raw": fpr_raw, "hit_rate_noisy_raw": tpr_raw,
            "enrichment_raw": enrichment_raw,

            # gated
            "n_clean_hits_gate": hit_clean_gate, "n_noisy_hits_gate": hit_noisy_gate,
            "precision_hits_gate": precision_gate,
            "hit_rate_clean_gate": fpr_gate, "hit_rate_noisy_gate": tpr_gate,
            "enrichment_gate": enrichment_gate
        })

    return pd.DataFrame(rows), pd.DataFrame(counts)

# ---------- Run ----------
dog   = load_pairs_from_id(os.path.join(BASE_DIR, DOG_FILE))
noisy = load_pairs_from_id(os.path.join(BASE_DIR, NOISY_FILE))

refs = build_union_reference_mean(dog, noisy)
df_samples, df_counts = per_epoch_metrics_with_gate(dog, noisy, refs, min_run=MIN_RUN)

# Save
out_samples = os.path.join(BASE_DIR, f"{OUT_PREFIX}_per_sample.csv")
out_counts  = os.path.join(BASE_DIR, f"{OUT_PREFIX}_per_epoch_counts.csv")
df_samples.to_csv(out_samples, index=False)
df_counts.to_csv(out_counts, index=False)

print("[Saved] per-sample →", out_samples)
print("[Saved] per-epoch  →", out_counts)
with pd.option_context("display.max_rows", 12, "display.width", 200):
    print(df_counts.head(12))

# === CKL per-epoch counts with Consecutive Gate + Enrichment (TPR/FPR) ========
BASE_DIR   = "/content/drive/MyDrive/runs/20250918-233203-cifar-poison"
DOG_FILE   = "dog_clean_GIE_pairs.json"
NOISY_FILE = "dog_fake_GIE_pairs.json"
OUT_PREFIX = "ckl_unionref_epoch_meanLogThr_runGate_enrichment"

# Gate settings
MIN_RUN = 5         # require >= MIN_RUN consecutive epochs above threshold to "count"
USE_SYMMETRIC = False  # set True to use symmetric CKL (avg of both directions)
