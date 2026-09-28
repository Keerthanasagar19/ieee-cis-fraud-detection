"""
concept_drift.py
-------------------
PROTOTYPE for Future Enhancement 2 (Concept Drift Detection).

Runs on REAL IEEE-CIS data. The original version of this script (referenced
in the dashboard's now-outdated blurb text) simulated a fabricated scenario:
"fraudsters switch to small, spread-out transactions". This version asks
the honest, harder question instead: across the REAL calendar time this
dataset actually spans, does the feature distribution genuinely shift, and
does the already-trained Mamba-KAN's performance genuinely degrade on the
most recent real transactions versus the earliest ones?

METHOD:
  1. Reuse preprocess_ieee.py's own loading/feature-engineering functions
     (same uid reconstruction, same causal features) -- consistent with the
     main pipeline and ring_fraud_gnn.py, not a third diverging definition
     of the data.
  2. Build sequences exactly as preprocess_ieee.py does, but additionally
     keep each window's real end-of-window TransactionDT.
  3. Sort ALL qualifying windows by that real timestamp and split
     chronologically: earliest 70% = "baseline" period, most recent 30% =
     "recent" period. This is a genuine temporal split of real transactions
     -- not a fabricated shift.
  4. PSI (Population Stability Index) per feature, computed on each
     window's last raw transaction value, comparing the recent period's
     distribution against the baseline period's (10 quantile bins from the
     baseline period). PSI > 0.25 is the conventional "significant drift"
     threshold; 0.1-0.25 is "moderate".
  5. Score both periods with the ALREADY-TRAINED Mamba-KAN model (same
     weights evaluate.py uses) at the default 0.5 threshold, and report the
     precision/recall/F1 gap between them.

CAVEAT TO STATE HONESTLY IN YOUR REPORT (same category of caveat as
ring_fraud_gnn.py's): Mamba-KAN's actual train/val/test split (in
preprocess_ieee.py) is ACCOUNT-level random, not time-based -- meaning
some uids contributing "recent"-period windows here were still in Mamba-
KAN's own training set (just at an earlier point in their own history).
This script is therefore a genuine but imperfect drift probe, not a
clean pre-registered before/after experiment. A stricter version would
retrain Mamba-KAN on a purely chronological train/test split; left as a
documented limitation, not fixed here, to keep this a prototype rather
than a second full training pipeline (same reasoning ring_fraud_gnn.py
documents for its own Mamba-KAN reuse).

IEEE-CIS's train set spans about 6 months of real transactions, so the
"drift" surfaced here is whatever real seasonal/behavioral movement
exists in that window -- it may be modest. That itself is a legitimate,
reportable finding: real drift over 6 months is not guaranteed to look
like a dramatic adversarial evasion story.

USAGE (run from inside fraud_project/src/enhancements/):
    python concept_drift.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..")
IEEE_DIR = os.path.join(SRC_DIR, "..", "data", "ieee_cis")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, IEEE_DIR)
from mamba_kan import MambaKAN
from preprocess_ieee import (load_and_merge, filter_by_uid_count,
                              engineer_features, FEATURE_COLUMNS, WINDOW_SIZE)

MODEL_DIR = os.path.join(SRC_DIR, "..", "models_saved")
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
RESULTS_DIR = os.path.join(SRC_DIR, "..", "results", "enhancements")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASELINE_FRAC = 0.7  # earliest 70% of real transaction time = "baseline"
N_PSI_BINS = 10


# ------------------------------------------------------------------
# 1. Load real data and build windows WITH their real end timestamp
# ------------------------------------------------------------------
def load_real_engineered_data():
    df = load_and_merge()
    df = filter_by_uid_count(df)
    df = engineer_features(df)
    return df


def build_sequences_with_time(df, window_size=WINDOW_SIZE):
    sequences, labels, end_dts, last_raw = [], [], [], []
    for uid, g in df.groupby("uid"):
        g = g.sort_values("TransactionDT").reset_index(drop=True)
        feats = g[FEATURE_COLUMNS].values.astype(np.float32)
        fraud = g["isFraud"].values
        dts = g["TransactionDT"].values
        if len(g) < window_size:
            continue
        for end in range(window_size - 1, len(g)):
            start = end - window_size + 1
            sequences.append(feats[start:end + 1])
            labels.append(fraud[end])
            end_dts.append(dts[end])
            last_raw.append(feats[end])  # raw (unnormalized) last-step values, for PSI
    return (np.stack(sequences), np.array(labels, dtype=np.int64),
            np.array(end_dts), np.stack(last_raw))


# ------------------------------------------------------------------
# 2. PSI (Population Stability Index)
# ------------------------------------------------------------------
def psi(baseline_vals, recent_vals, n_bins=N_PSI_BINS, eps=1e-6):
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(baseline_vals, quantiles))
    if len(edges) < 3:  # near-constant feature -- PSI undefined/meaningless
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline_vals, bins=edges)
    recent_counts, _ = np.histogram(recent_vals, bins=edges)
    base_pct = base_counts / max(base_counts.sum(), 1) + eps
    recent_pct = recent_counts / max(recent_counts.sum(), 1) + eps
    return float(np.sum((recent_pct - base_pct) * np.log(recent_pct / base_pct)))


# ------------------------------------------------------------------
# 3. Score a slice with the already-trained Mamba-KAN
# ------------------------------------------------------------------
def score_slice(model, X, norm_mean, norm_std, threshold=0.5):
    X_norm = (X - norm_mean) / norm_std
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_norm, dtype=torch.float32))).numpy()
    return probs


def metrics_at(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else float("nan"),
    }


def main():
    d = np.load(os.path.join(DATA_DIR, "sequences.npz"), allow_pickle=True)
    norm_mean, norm_std = d["norm_mean"], d["norm_std"]
    n_features = len(list(d["feature_names"]))

    print("Loading and engineering real IEEE-CIS data (same pipeline as preprocess_ieee.py)...")
    df = load_real_engineered_data()

    print("Building windows with real end-of-window timestamps...")
    X, y, end_dts, last_raw = build_sequences_with_time(df)
    print(f"Total windows: {len(X)}, spanning real TransactionDT "
          f"{end_dts.min()} to {end_dts.max()} ({(end_dts.max()-end_dts.min())/86400:.1f} real days)")

    order = np.argsort(end_dts)
    X, y, end_dts, last_raw = X[order], y[order], end_dts[order], last_raw[order]

    split_idx = int(len(X) * BASELINE_FRAC)
    X_base, y_base, raw_base = X[:split_idx], y[:split_idx], last_raw[:split_idx]
    X_recent, y_recent, raw_recent = X[split_idx:], y[split_idx:], last_raw[split_idx:]
    print(f"\nBaseline (earliest {BASELINE_FRAC*100:.0f}% real time): {len(X_base)} windows, "
          f"fraud rate {y_base.mean():.4f}")
    print(f"Recent   (most recent {(1-BASELINE_FRAC)*100:.0f}% real time): {len(X_recent)} windows, "
          f"fraud rate {y_recent.mean():.4f}")

    print("\nComputing PSI per feature (baseline vs. recent, real transaction time)...")
    psi_scores = {}
    for i, fname in enumerate(FEATURE_COLUMNS):
        psi_scores[fname] = round(psi(raw_base[:, i], raw_recent[:, i]), 4)
    top_feature = max(psi_scores, key=psi_scores.get)
    for fname, val in sorted(psi_scores.items(), key=lambda kv: -kv[1]):
        flag = "  <-- largest real drift" if fname == top_feature else ""
        print(f"  {fname:22s} PSI={val:.4f}{flag}")

    print("\nScoring both real periods with the already-trained Mamba-KAN "
          "(no retraining)...")
    model = MambaKAN(n_features=n_features)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mamba_kan_best.pt"), map_location="cpu"))
    model.eval()

    probs_base = score_slice(model, X_base, norm_mean, norm_std)
    probs_recent = score_slice(model, X_recent, norm_mean, norm_std)
    m_base = metrics_at(y_base, probs_base)
    m_recent = metrics_at(y_recent, probs_recent)

    print(f"\nBaseline period -> precision={m_base['precision']:.4f}  recall={m_base['recall']:.4f}  "
          f"f1={m_base['f1']:.4f}  pr_auc={m_base['pr_auc']:.4f}")
    print(f"Recent period   -> precision={m_recent['precision']:.4f}  recall={m_recent['recall']:.4f}  "
          f"f1={m_recent['f1']:.4f}  pr_auc={m_recent['pr_auc']:.4f}")
    print("\nNOTE: the recent period is not a purely held-out future -- Mamba-KAN's "
          "account-level train/val/test split means some 'recent'-period uids were "
          "still seen (at an earlier point in their history) during training. See "
          "this script's docstring for the full caveat.")

    # ---- Plot: PSI bar chart + baseline vs recent metric comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names = list(psi_scores.keys())
    vals = [psi_scores[n] for n in names]
    colors = ["#d64545" if v >= 0.25 else ("#e0a030" if v >= 0.1 else "#4a90d9") for v in vals]
    axes[0].barh(names, vals, color=colors)
    axes[0].axvline(0.25, color="#d64545", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].axvline(0.10, color="#e0a030", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("PSI (baseline vs. recent real transactions)")
    axes[0].set_title("Real feature drift by PSI\n(dashed lines: 0.10 moderate, 0.25 significant)")

    metric_names = ["precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    width = 0.35
    axes[1].bar(x - width/2, [m_base[m] for m in metric_names], width, label="Baseline (earlier)")
    axes[1].bar(x + width/2, [m_recent[m] for m in metric_names], width, label="Recent (later)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Mamba-KAN performance: earlier vs. later\nreal transactions (no retraining)")
    axes[1].legend()

    plt.suptitle("Real IEEE-CIS concept drift -- chronological split, no fabricated scenario", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "concept_drift_detection.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {out_path}")

    summary = {
        "data_source": "real_ieee_cis",
        "n_baseline_windows": int(len(X_base)),
        "n_recent_windows": int(len(X_recent)),
        "baseline_precision": round(m_base["precision"], 4),
        "baseline_recall": round(m_base["recall"], 4),
        "baseline_f1": round(m_base["f1"], 4),
        "baseline_pr_auc": round(m_base["pr_auc"], 4),
        "drifted_precision": round(m_recent["precision"], 4),
        "drifted_recall": round(m_recent["recall"], 4),
        "drifted_f1": round(m_recent["f1"], 4),
        "drifted_pr_auc": round(m_recent["pr_auc"], 4),
        "top_drifted_feature": top_feature,
        "top_drifted_feature_psi": psi_scores[top_feature],
        "psi_scores": psi_scores,
        "note": ("Recent period overlaps with Mamba-KAN's own account-level training "
                 "set at the uid level -- see script docstring for the full caveat."),
    }
    with open(os.path.join(RESULTS_DIR, "drift_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(RESULTS_DIR, 'drift_summary.json')}")


if __name__ == "__main__":
    main()
