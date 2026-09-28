"""
cold_start.py
----------------
PROTOTYPE for Phase 2 Enhancement 1 (Cold-Start Handling).

THE PROBLEM: Mamba-KAN requires a full 10-transaction window. A brand new
account's 1st transaction has zero history, its 2nd has one prior
transaction, and so on -- there is no way to hand Mamba-KAN a "complete"
window until the 10th transaction. preprocess_ieee.py's original
build_sequences() simply discarded these early transactions entirely,
which quietly means the model (and every metric in evaluate.py) was never
even tested on this case.

THE FIX (already built into preprocess_ieee.py's build_coldstart_raw /
normalize_and_pad_coldstart): every qualifying account's first 1-9
transactions are kept as real, labeled examples, normalized with the exact
same train-set statistics as the mature windows, and left-padded with a
sentinel value (-5.0) so the "current" transaction being scored is always
the last row of the window -- same convention every model already expects.

THE ROUTING DESIGN this script evaluates:

    New transaction
          |
    Account has >= 10 prior transactions?
       /                              \\
      NO                              YES
       |                               |
    Random Forest                 Mamba-KAN (+ SNN ensemble)
    (single-transaction            (uses full sequence context)
     features only)
       \\_____________  ____________/
                     |
              Fraud / Normal

Random Forest is a natural cold-start model here for a simple reason: it
was ALREADY trained using only the last transaction's features (see
baseline.py) -- it never depended on sequence history in the first place,
so it needs no retraining and no architecture change to be pointed at
these real cold examples.

WHAT THIS SCRIPT DOES:
  1. Loads the real cold-start TEST examples straight from sequences.npz
     (built by preprocess_ieee.py) -- no synthetic/simulated data.
  2. Scores them with the ALREADY-TRAINED, persisted Random Forest
     (models_saved/random_forest_baseline.joblib from baseline.py),
     using just the last (current) transaction's features.
  3. Scores the SAME examples with the ALREADY-TRAINED Mamba-KAN (full
     padded 10-step window, including the -5.0 sentinel rows) -- this is
     the "what if we forced the sequence model to handle this anyway"
     comparison that justifies why routing is worth doing at all.
  4. Breaks results down by real history length (1..9 prior transactions)
     to show whether/how performance improves as history accumulates.
  5. Saves a summary JSON + plot to results/enhancements/, in the same
     style as ring_fraud_gnn.py / concept_drift.py, ready for the
     dashboard.

HONEST CAVEAT to state in your report (same spirit as the other two
enhancement scripts' caveats): Random Forest's "generalizes fine to cold
examples" claim rests on the assumption that a first-time transaction's
engineered features (amount, dist1, C1-C14, D-features, etc.) are drawn
from the same distribution regardless of account maturity. That is
plausible -- these are IEEE-CIS's own per-transaction columns, not
history-derived -- but it is an assumption, not a proof, and is exactly
what the "RF on cold-start" numbers below are testing empirically.
"""

import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, accuracy_score)

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, SRC_DIR)
from mamba_kan import MambaKAN

MODEL_DIR = os.path.join(SRC_DIR, "..", "models_saved")
DATA_PATH = os.path.join(SRC_DIR, "..", "data", "sequences.npz")
RESULTS_DIR = os.path.join(SRC_DIR, "..", "results", "enhancements")
os.makedirs(RESULTS_DIR, exist_ok=True)

# History-length buckets for the breakdown chart -- 1-9 possible values,
# grouped into three bands so each bar has enough examples to be stable.
HISTORY_BUCKETS = [(1, 3, "1-3 prior tx"), (4, 6, "4-6 prior tx"), (7, 9, "7-9 prior tx")]


def load_coldstart_test():
    d = np.load(DATA_PATH, allow_pickle=True)
    required = ["X_test_coldstart", "y_test_coldstart", "history_len_test_coldstart"]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise RuntimeError(
            f"sequences.npz is missing {missing}. Re-run preprocess_ieee.py "
            "(the version that builds cold-start examples) before this script."
        )
    return d["X_test_coldstart"], d["y_test_coldstart"], d["history_len_test_coldstart"]


def load_mature_test():
    """The already-known Phase-1 test set (full-history windows), loaded
    purely so this script's printed table can show cold-vs-mature side by
    side without you having to cross-reference evaluate.py's output."""
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["X_test"], d["y_test"]


def score_rf(X_window):
    """RF only ever saw last-transaction features (see baseline.py) --
    apply it the same way here, regardless of how much padding precedes
    the current transaction."""
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest_baseline.joblib"))
    last_step = X_window[:, -1, :]
    return rf.predict_proba(last_step)[:, 1]


def score_mamba_kan(X_window, n_features):
    """Feeds the FULL padded window (including -5.0 sentinel rows for
    missing history) into the already-trained Mamba-KAN -- deliberately
    out-of-distribution input for a model that never saw padding like this
    during training, to demonstrate why routing to RF instead is the
    better design, not just an assumption."""
    model = MambaKAN(n_features=n_features)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mamba_kan_best.pt"), map_location="cpu"))
    model.eval()
    probs = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_window), batch_size):
            xb = torch.tensor(X_window[i:i + batch_size], dtype=torch.float32)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "fraud_rate": round(float(y_true.mean()), 4) if len(y_true) else None,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4) if len(set(y_true)) > 1 else None,
    }


def main():
    print("Loading real cold-start TEST examples (accounts' first 1-9 transactions)...")
    X_cold, y_cold, hist_len = load_coldstart_test()
    print(f"Cold-start test set: {X_cold.shape}, fraud rate {y_cold.mean():.4f}")

    X_mature, y_mature = load_mature_test()
    n_features = X_cold.shape[-1]

    print("\nScoring cold-start examples with the persisted Random Forest "
          "(last-transaction features only, no retraining)...")
    prob_rf_cold = score_rf(X_cold)

    print("Scoring cold-start examples with the already-trained Mamba-KAN "
          "(full padded window, including -5.0 sentinel rows)...")
    prob_mk_cold = score_mamba_kan(X_cold, n_features)

    print("\n=== Overall cold-start test performance (threshold=0.5) ===")
    rf_overall = compute_metrics(y_cold, prob_rf_cold)
    mk_overall = compute_metrics(y_cold, prob_mk_cold)
    print(f"Random Forest (cold-start) -> {rf_overall}")
    print(f"Mamba-KAN     (cold-start) -> {mk_overall}")

    # --- for reference: how each model does on MATURE (full-history) test data ---
    print("\nFor reference, re-scoring the MATURE test set the same way "
          "(RF: last-transaction features; Mamba-KAN: full real window)...")
    prob_rf_mature = score_rf(X_mature)
    prob_mk_mature = score_mamba_kan(X_mature, n_features)
    rf_mature = compute_metrics(y_mature, prob_rf_mature)
    mk_mature = compute_metrics(y_mature, prob_mk_mature)
    print(f"Random Forest (mature) -> {rf_mature}")
    print(f"Mamba-KAN     (mature) -> {mk_mature}")

    # --- breakdown by real history length ---
    print("\n=== Cold-start performance by history-length bucket ===")
    bucket_results = []
    for lo, hi, label in HISTORY_BUCKETS:
        mask = (hist_len >= lo) & (hist_len <= hi)
        n_bucket = int(mask.sum())
        if n_bucket == 0:
            continue
        rf_b = compute_metrics(y_cold[mask], prob_rf_cold[mask])
        mk_b = compute_metrics(y_cold[mask], prob_mk_cold[mask])
        print(f"\n[{label}] n={n_bucket}, fraud_rate={y_cold[mask].mean():.4f}")
        print(f"  Random Forest -> precision={rf_b['precision']:.4f}  recall={rf_b['recall']:.4f}  "
              f"f1={rf_b['f1']:.4f}  pr_auc={rf_b['pr_auc']}")
        print(f"  Mamba-KAN     -> precision={mk_b['precision']:.4f}  recall={mk_b['recall']:.4f}  "
              f"f1={mk_b['f1']:.4f}  pr_auc={mk_b['pr_auc']}")
        bucket_results.append({"bucket": label, "lo": lo, "hi": hi,
                                "random_forest": rf_b, "mamba_kan": mk_b})

    # --- plot: PR-AUC by history-length bucket, RF vs Mamba-KAN, with mature reference lines ---
    labels = [b["bucket"] for b in bucket_results]
    rf_prauc = [b["random_forest"]["pr_auc"] or 0 for b in bucket_results]
    mk_prauc = [b["mamba_kan"]["pr_auc"] or 0 for b in bucket_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, rf_prauc, width, label="Random Forest (cold-start route)")
    ax.bar(x + width / 2, mk_prauc, width, label="Mamba-KAN (forced on padded input)")
    ax.axhline(rf_mature["pr_auc"], color="tab:blue", linestyle="--", alpha=0.6,
               label=f"Random Forest, mature test (PR-AUC={rf_mature['pr_auc']})")
    ax.axhline(mk_mature["pr_auc"], color="tab:orange", linestyle="--", alpha=0.6,
               label=f"Mamba-KAN, mature test (PR-AUC={mk_mature['pr_auc']})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Cold-start PR-AUC by real history length\n(real IEEE-CIS accounts, no simulated data)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "cold_start_detection.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {out_path}")

    summary = {
        "data_source": "real_ieee_cis",
        "n_coldstart_test": int(len(y_cold)),
        "coldstart_fraud_rate": round(float(y_cold.mean()), 4),
        "overall": {
            "random_forest_coldstart": rf_overall,
            "mamba_kan_coldstart": mk_overall,
            "random_forest_mature_reference": rf_mature,
            "mamba_kan_mature_reference": mk_mature,
        },
        "by_history_length": bucket_results,
        "note": ("Cold-start examples are each qualifying account's real first 1-9 "
                 "transactions, previously discarded by preprocess_ieee.py. Mamba-KAN "
                 "is scored here on padded (-5.0 sentinel) input it was never trained "
                 "on, to demonstrate why routing to Random Forest -- which only ever "
                 "used last-transaction features -- is the better design rather than "
                 "an untested assumption."),
    }
    with open(os.path.join(RESULTS_DIR, "cold_start_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(RESULTS_DIR, 'cold_start_summary.json')}")


if __name__ == "__main__":
    main()