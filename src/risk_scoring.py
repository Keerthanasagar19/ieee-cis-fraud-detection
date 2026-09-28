"""
risk_scoring.py
-----------------
PROTOTYPE for Phase 2 Enhancement 2 (Risk Scoring).

THE IDEA (from the plan): instead of only "Fraud=1 / Normal=0", produce a
human-facing risk LEVEL on top of the raw probability:

    Risk score = 0.82
    Risk level = HIGH

METHOD -- thresholds are chosen from the VALIDATION set's own behavior,
not guessed round numbers like "0.3 / 0.7":

  - t_high = the F1-optimal decision threshold on validation for the
    Ensemble (exactly the same logic evaluate.py already uses for its
    "tuned threshold" table). Above this is the model's own best-
    calibrated guess that this transaction IS fraud -> HIGH risk.

  - t_low  = the highest threshold on validation that still achieves
    >= TARGET_RECALL (default 90%) recall. Below this, even a policy
    tuned to catch nearly all fraud wouldn't bother flagging the
    transaction -> LOW risk.

  - Anything between t_low and t_high -> MEDIUM risk: below the model's
    most confident cutoff, but too risky to wave through the same as
    something clearly LOW -> worth a human glance.

Both thresholds are computed ONCE, on validation only (never test), and
saved to results/enhancements/risk_scoring_summary.json. app.py loads this
file at startup rather than recomputing thresholds on every request.

CALIBRATION CHECK: on the untouched TEST set, the empirical fraud rate
within each bucket should increase monotonically LOW -> MEDIUM -> HIGH.
This script checks that explicitly and reports it honestly either way --
if the buckets aren't well-calibrated, that's exactly what this check is
for, not something to hide.

This reuses evaluate.py's own load_test_data / load_val_data / get_probs /
find_best_threshold functions directly (all plain module-level functions,
not locked inside evaluate.py's __main__ block) rather than re-implementing
them, so there is exactly one definition of "how the ensemble's
probability is computed" across the whole project.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from mamba_kan import MambaKAN
from snn_model import SNNFraudDetector
from evaluate import load_test_data, load_val_data, get_probs, find_best_threshold, MODEL_DIR

RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "enhancements")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_RECALL = 0.80  # the LOW/MEDIUM boundary must still catch this much fraud on validation


def pick_ensemble_partner(snn_rate, snn_latency, X_val, y_val):
    """Same adaptive selection evaluate.py uses: whichever SNN encoding has
    higher validation PR-AUC becomes the Ensemble's partner. Duplicated
    here (rather than imported) because it lives inside evaluate.py's
    __main__ block, not as a standalone function."""
    prob_rate_val = get_probs(snn_rate, X_val, use_last_step_only=True)
    prob_latency_val = get_probs(snn_latency, X_val, use_last_step_only=True)
    pr_rate = average_precision_score(y_val, prob_rate_val)
    pr_latency = average_precision_score(y_val, prob_latency_val)
    return "SNN-latency" if pr_latency >= pr_rate else "SNN-rate"


def compute_ensemble_probs(X, mamba_kan, snn_rate, snn_latency, partner_name):
    prob_mk = get_probs(mamba_kan, X, use_last_step_only=False)
    if partner_name == "SNN-latency":
        prob_partner = get_probs(snn_latency, X, use_last_step_only=True)
    else:
        prob_partner = get_probs(snn_rate, X, use_last_step_only=True)
    return 0.5 * prob_mk + 0.5 * prob_partner


def find_low_threshold(y_true, y_prob, target_recall=TARGET_RECALL):
    """Highest threshold that still achieves >= target_recall on y_true.
    Scanning from strict (high) to lenient (low): recall only grows as
    the threshold drops, so the FIRST threshold where recall crosses the
    target is exactly the highest one that satisfies it -- return
    immediately there instead of continuing to scan (continuing was the
    bug: it kept overwriting the answer with lower and lower thresholds
    all the way to the bottom, since recall stays >= target for every
    threshold below the crossover point too)."""
    thresholds = np.unique(y_prob)
    if len(thresholds) > 1000:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 1000))
    thresholds = np.sort(thresholds)[::-1]  # descending: strict -> lenient
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= target_recall:
            return float(t)
    return float(thresholds[-1]) if len(thresholds) else 0.0


def bucket_labels(probs, t_low, t_high):
    labels = np.full(len(probs), "LOW", dtype=object)
    labels[(probs >= t_low) & (probs < t_high)] = "MEDIUM"
    labels[probs >= t_high] = "HIGH"
    return labels


def main():
    X_test, y_test, _ = load_test_data()
    X_val, y_val = load_val_data()
    n_features = X_test.shape[-1]

    mamba_kan = MambaKAN(n_features=n_features)
    mamba_kan.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mamba_kan_best.pt"), map_location="cpu"))

    snn_rate = SNNFraudDetector(n_features=n_features, encoding="rate")
    snn_rate.load_state_dict(torch.load(os.path.join(MODEL_DIR, "snn_rate_best.pt"), map_location="cpu"))

    snn_latency = SNNFraudDetector(n_features=n_features, encoding="latency")
    snn_latency.load_state_dict(torch.load(os.path.join(MODEL_DIR, "snn_latency_best.pt"), map_location="cpu"))

    partner_name = pick_ensemble_partner(snn_rate, snn_latency, X_val, y_val)
    print(f"Ensemble partner (validation PR-AUC selection): {partner_name}")

    prob_val = compute_ensemble_probs(X_val, mamba_kan, snn_rate, snn_latency, partner_name)
    prob_test = compute_ensemble_probs(X_test, mamba_kan, snn_rate, snn_latency, partner_name)

    t_high, val_f1 = find_best_threshold(y_val, prob_val)
    t_low = find_low_threshold(y_val, prob_val, TARGET_RECALL)
    if t_low >= t_high:
        # guard against a degenerate case where the two thresholds cross
        # (can happen on a small/noisy validation set) -- keep MEDIUM
        # non-empty by pulling t_low down slightly.
        t_low = t_high * 0.5
    print(f"t_low  (>= {TARGET_RECALL:.0%} validation recall): {t_low:.4f}")
    print(f"t_high (F1-optimal on validation, F1={val_f1:.4f}):  {t_high:.4f}")

    # --- calibration check on the untouched TEST set ---
    test_labels = bucket_labels(prob_test, t_low, t_high)
    bucket_stats = {}
    print("\n=== Calibration check on TEST set ===")
    for level in ["LOW", "MEDIUM", "HIGH"]:
        mask = test_labels == level
        n = int(mask.sum())
        fraud_rate = float(y_test[mask].mean()) if n > 0 else None
        n_fraud_caught = int(y_test[mask].sum())
        bucket_stats[level] = {"n": n, "fraud_rate": round(fraud_rate, 4) if fraud_rate is not None else None,
                                "n_fraud_in_bucket": n_fraud_caught}
        print(f"{level:7s} -> n={n:6d}  empirical fraud rate={fraud_rate}  frauds_in_bucket={n_fraud_caught}")

    monotonic = (bucket_stats["LOW"]["fraud_rate"] or 0) <= (bucket_stats["MEDIUM"]["fraud_rate"] or 0) <= \
                (bucket_stats["HIGH"]["fraud_rate"] or 0)
    print(f"\nMonotonic (LOW <= MEDIUM <= HIGH fraud rate)? {monotonic}")

    total_fraud = int(y_test.sum())
    fraud_in_medium_or_high = bucket_stats["MEDIUM"]["n_fraud_in_bucket"] + bucket_stats["HIGH"]["n_fraud_in_bucket"]
    coverage = fraud_in_medium_or_high / total_fraud if total_fraud else None
    print(f"Fraud coverage if reviewing MEDIUM+HIGH only: {fraud_in_medium_or_high}/{total_fraud} "
          f"= {coverage:.4f}" if coverage is not None else "N/A")

    volume_medium_or_high = bucket_stats["MEDIUM"]["n"] + bucket_stats["HIGH"]["n"]
    volume_frac = volume_medium_or_high / len(y_test)
    print(f"Share of ALL test transactions that are MEDIUM+HIGH (i.e. would need review): "
          f"{volume_medium_or_high}/{len(y_test)} = {volume_frac:.4f}")

    # --- plot: empirical fraud rate by bucket ---
    fig, ax = plt.subplots(figsize=(6, 5))
    levels = ["LOW", "MEDIUM", "HIGH"]
    rates = [bucket_stats[l]["fraud_rate"] or 0 for l in levels]
    colors = ["#2DD4A7", "#FFB020", "#FF5D5D"]
    ax.bar(levels, rates, color=colors)
    ax.axhline(float(y_test.mean()), color="gray", linestyle="--", alpha=0.6,
               label=f"Overall test fraud rate ({y_test.mean():.4f})")
    ax.set_ylabel("Empirical fraud rate in bucket")
    ax.set_title("Risk bucket calibration (test set)\nEnsemble probability -> LOW / MEDIUM / HIGH")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "risk_scoring_calibration.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {out_path}")

    summary = {
        "method": {
            "t_low": round(t_low, 4),
            "t_high": round(t_high, 4),
            "t_low_definition": f"highest validation threshold achieving >= {TARGET_RECALL:.0%} recall",
            "t_high_definition": "F1-optimal threshold on validation",
            "ensemble_partner": partner_name,
        },
        "test_calibration": bucket_stats,
        "monotonic": bool(monotonic),
        "fraud_coverage_medium_plus_high": round(coverage, 4) if coverage is not None else None,
        "volume_share_medium_plus_high": round(volume_frac, 4),
        "note": ("Thresholds are computed on the VALIDATION set only, never test. "
                 "t_high is the same F1-optimal cutoff evaluate.py reports for the "
                 "Ensemble; t_low is a separate, lower recall-driven cutoff. "
                 "app.py loads this file at startup rather than recomputing per request."),
    }
    with open(os.path.join(RESULTS_DIR, "risk_scoring_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(RESULTS_DIR, 'risk_scoring_summary.json')}")


if __name__ == "__main__":
    main()