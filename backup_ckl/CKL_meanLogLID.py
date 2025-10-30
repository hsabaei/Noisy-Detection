BASE_DIR   = "/content/drive/MyDrive/runs/20250918-233203-cifar-poison"
DOG_FILE   = "dog_clean_GIE_pairs.json"
NOISY_FILE = "dog_fake_GIE_pairs.json"
OUT_PREFIX = "ckl_unionref_epoch_meanThr_runGate_enrichment"

# Gate settings
MIN_RUN = 5         # require >= MIN_RUN consecutive epochs above threshold to "count"
USE_SYMMETRIC = False  # set True to use symmetric CKL (avg of both directions)

import os, json, math
import numpy as np
import pandas as pd

EPS = 1e-12

# ---------- Huber mean ----------
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

# ---------- Load JSON → {id: [(W,d), ...]} ----------
def load_pairs(path):
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for k, seq in raw.items():
        try: sid = int(k)
        except: sid = k
        cleaned = []
        for x in seq:
            if isinstance(x,(list,tuple)) and len(x)>=2:
                w, d = float(x[0]), float(x[1])
                if np.isfinite(w) and np.isfinite(d) and w>0 and d>0:
                    cleaned.append((w, d, math.log(d)))
        if cleaned:
            out[sid] = cleaned
    return out

# ---------- CKL finite-boundary (ID form) ----------
def ckl_equal_W(W, d1, d2):
    return W * ((d2 - d1)**2) / (((d1 + 1.0)**2) * (d2 + 1.0) + EPS)

def ckl_case_A(W1, d1, W2, d2):
    term = ( W1 * ((d2/(d1+1.0)) * math.log(max(W2/W1, EPS)) - (d1 - d2)/((d1+1.0)**2))
           + d2 * ((W2 - W1) + W1 * math.log(max(W1/W2, EPS))) )
    return term + (d1/(d1+1.0))*W1 - (d2/(d2+1.0))*W2

def ckl_case_B(W1, d1, W2, d2):
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

# ---------- Build per-epoch UNION reference ----------
def build_union_reference(dog, noisy):
    max_T = max((len(s) for s in list(dog.values())+list(noisy.values())), default=0)
    refs = {}
    for t in range(max_T):
        logWs, logDs = [], []
        for seq in dog.values():
            if len(seq) > t:
                w, d, logd = seq[t]
                logWs.append(math.log(w)); logDs.append(logd)
        for seq in noisy.values():
            if len(seq) > t:
                w, d, logd = seq[t]
                logWs.append(math.log(w)); logDs.append(logd)
        if not logWs:
            continue
        refs[t] = ( math.exp(huber_mean(logWs)),
                    math.exp(huber_mean(logDs)),
                    len(logWs) )
    return refs  

# ---------- Per-epoch CKL + mean-threshold + consecutive gate + enrichment ----
def per_epoch_metrics_with_gate(dog, noisy, refs, min_run=3):
    rows = []      # per-sample per-epoch
    counts = []    # per-epoch summary

    runlen = {}    # sid -> current consecutive run length

    max_T = max((len(s) for s in list(dog.values())+list(noisy.values())), default=0)
    for t in range(max_T):
        if t not in refs:
            continue
        W_ref, d_ref, n_used = refs[t]

        # 1) compute CKL for all samples at epoch t
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

        # 2) threshold = mean score at epoch
        scores = [r[5] for r in epoch_rows if np.isfinite(r[5])]
        thr = float(np.mean(scores)) if scores else np.nan

        # 3) raw & gated flags; accumulate counts
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
                if raw_flag:  hit_clean_raw  += 1
                if gated_flag: hit_clean_gate += 1
            else:
                n_noisy_avail += 1
                if raw_flag:  hit_noisy_raw  += 1
                if gated_flag: hit_noisy_gate += 1

        # 4) precision (for reference) and enrichment metrics (your request)
        # Raw
        denom_raw  = hit_clean_raw  + hit_noisy_raw
        precision_raw  = (hit_noisy_raw  / denom_raw) if denom_raw  > 0 else np.nan
        hit_rate_noisy_raw = (hit_noisy_raw / n_noisy_avail) if n_noisy_avail > 0 else np.nan  # TPR
        hit_rate_clean_raw = (hit_clean_raw / n_clean_avail) if n_clean_avail > 0 else np.nan  # FPR
        enrichment_raw = (hit_rate_noisy_raw / hit_rate_clean_raw) if (hit_rate_clean_raw and np.isfinite(hit_rate_clean_raw)) else np.nan

        # Gated
        denom_gate = hit_clean_gate + hit_noisy_gate
        precision_gate = (hit_noisy_gate / denom_gate) if denom_gate > 0 else np.nan
        hit_rate_noisy_gate = (hit_noisy_gate / n_noisy_avail) if n_noisy_avail > 0 else np.nan
        hit_rate_clean_gate = (hit_clean_gate / n_clean_avail) if n_clean_avail > 0 else np.nan
        enrichment_gate = (hit_rate_noisy_gate / hit_rate_clean_gate) if (hit_rate_clean_gate and np.isfinite(hit_rate_clean_gate)) else np.nan

        counts.append({
            "epoch": t, "n_ref_used": n_used, "threshold": thr,
            "min_run": min_run,
            "n_clean_avail": n_clean_avail, "n_noisy_avail": n_noisy_avail,

            # raw flags
            "n_clean_hits_raw": hit_clean_raw, "n_noisy_hits_raw": hit_noisy_raw,
            "precision_hits_raw": precision_raw,
            "hit_rate_clean_raw": hit_rate_clean_raw,
            "hit_rate_noisy_raw": hit_rate_noisy_raw,
            "enrichment_raw": enrichment_raw,

            # gated flags
            "n_clean_hits_gate": hit_clean_gate, "n_noisy_hits_gate": hit_noisy_gate,
            "precision_hits_gate": precision_gate,
            "hit_rate_clean_gate": hit_rate_clean_gate,
            "hit_rate_noisy_gate": hit_rate_noisy_gate,
            "enrichment_gate": enrichment_gate
        })

    return pd.DataFrame(rows), pd.DataFrame(counts)

# ---------- Run ----------
dog   = load_pairs(os.path.join(BASE_DIR, DOG_FILE))
noisy = load_pairs(os.path.join(BASE_DIR, NOISY_FILE))

refs = build_union_reference(dog, noisy)
df_samples, df_counts = per_epoch_metrics_with_gate(dog, noisy, refs, min_run=MIN_RUN)

# Save
out_samples = os.path.join(BASE_DIR, f"{OUT_PREFIX}_per_sample.csv")
out_counts  = os.path.join(BASE_DIR, f"{OUT_PREFIX}_per_epoch_counts.csv")
df_samples.to_csv(out_samples, index=False)
df_counts.to_csv(out_counts, index=False)

print("[Saved] per-sample →", out_samples)
print("[Saved] per-epoch  →", out_counts)
print("\n[Head] per-epoch counts with gate & enrichment:")
with pd.option_context("display.max_rows", 12, "display.width", 200):
    print(df_counts.head(12))

