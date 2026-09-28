"""
baseline.py
------------
Trains classical baselines on the SAME data (flattened, last-transaction
features only) so you can honestly answer the examiner question:
"why not just use Random Forest / Logistic Regression?"

Two baselines:
  1. Logistic Regression - the simplest possible sane baseline
  2. Random Forest        - the standard strong tabular baseline

Both use ONLY the last transaction's engineered features (no sequence
context) -- this is the key comparison point. If Mamba-KAN clearly beats
these, it's evidence the sequence/context modelling is actually earning its
complexity, not just adding it for the sake of a fancier architecture.

FAIR-COMPARISON NOTE: earlier runs only reported baseline metrics at the
default threshold=0.5. That is not a fair comparison against evaluate.py's
Part B, which tunes each neural model's threshold on the validation set
before scoring on the test set. This version does the same thing for the
baselines: pick the F1-maximizing threshold on validation, then apply it,
unmodified, to the test set. Threshold-independent metrics (ROC-AUC, PR-AUC)
are unaffected either way and remain the primary numbers to compare.
"""

import os
import csv
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, accuracy_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sequences.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models_saved")

SEED = 42  # match train.py's SEED so the whole project is reproducible


def load_last_step_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    # use only the LAST transaction in each window -> flat feature vector,
    # exactly what a non-sequence classifier would see
    X_train = d["X_train"][:, -1, :]
    y_train = d["y_train"]
    X_val = d["X_val"][:, -1, :]
    y_val = d["y_val"]
    X_test = d["X_test"][:, -1, :]
    y_test = d["y_test"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_threshold(y_true, y_prob):
    """Same logic as evaluate.py's find_best_threshold -- scans thresholds
    on the VALIDATION set only (never test) and returns the F1-maximizing
    one. Kept as a duplicate here (rather than importing evaluate.py) so
    baseline.py has no dependency on torch."""
    thresholds = np.unique(y_prob)
    if len(thresholds) > 1000:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 1000))
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh), float(best_f1)


def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "name": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def print_result(r):
    print(f"\n--- {r['name']} (threshold={r['threshold']:.3f}) ---")
    print(f"Accuracy : {r['accuracy']:.4f}")
    print(f"Precision: {r['precision']:.4f}")
    print(f"Recall   : {r['recall']:.4f}")
    print(f"F1       : {r['f1']:.4f}")
    print(f"ROC-AUC  : {r['roc_auc']:.4f}")
    print(f"PR-AUC   : {r['pr_auc']:.4f}")


def write_summary(path, results, has_threshold_col=False):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = f.read().splitlines()
    header_cols = ["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] \
        if has_threshold_col else ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    header = rows[0] if rows else ",".join(header_cols)
    existing = rows[1:] if rows else []
    existing = [r for r in existing if "baseline" not in r]
    for r in results:
        if has_threshold_col:
            existing.append(f"{r['name']},{r['threshold']:.4f},{r['accuracy']:.4f},{r['precision']:.4f},"
                             f"{r['recall']:.4f},{r['f1']:.4f},{r['roc_auc']:.4f},{r['pr_auc']:.4f}")
        else:
            existing.append(f"{r['name']},{r['accuracy']:.4f},{r['precision']:.4f},"
                             f"{r['recall']:.4f},{r['f1']:.4f},{r['roc_auc']:.4f},{r['pr_auc']:.4f}")
    with open(path, "w", newline="") as f:
        f.write(header + "\n")
        for row in existing:
            f.write(row + "\n")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_last_step_data()

    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)
    prob_lr_val = lr.predict_proba(X_val)[:, 1]
    prob_lr_test = lr.predict_proba(X_test)[:, 1]

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    prob_rf_val = rf.predict_proba(X_val)[:, 1]
    prob_rf_test = rf.predict_proba(X_test)[:, 1]

    # Persist both baselines -- this is the model the cold-start pipeline
    # will reuse to score transactions with insufficient history, rather
    # than silently retraining a different (possibly non-identical) copy.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression_baseline.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_baseline.joblib"))
    print(f"\nSaved trained baseline models to {MODEL_DIR}")

    print("\n=== PART A: Baseline Results at default threshold=0.5 ===")
    results_default = [
        evaluate("Logistic Regression (baseline)", y_test, prob_lr_test, threshold=0.5),
        evaluate("Random Forest (baseline)", y_test, prob_rf_test, threshold=0.5),
    ]
    for r in results_default:
        print_result(r)

    print("\n=== PART B: Baseline Results at VALIDATION-tuned threshold ===")
    tuned_pairs = [
        ("Logistic Regression (baseline)", prob_lr_val, prob_lr_test),
        ("Random Forest (baseline)", prob_rf_val, prob_rf_test),
    ]
    results_tuned = []
    for name, val_probs, test_probs in tuned_pairs:
        best_t, val_f1 = find_best_threshold(y_val, val_probs)
        print(f"\n[{name}] best threshold on validation set: {best_t:.3f} (val F1={val_f1:.4f})")
        r = evaluate(f"{name} (tuned)", y_test, test_probs, threshold=best_t)
        results_tuned.append(r)
        print_result(r)

    # append/refresh baseline rows in both summary files, keeping the
    # neural models' rows (already written by evaluate.py) untouched
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    tuned_summary_path = os.path.join(RESULTS_DIR, "summary_metrics_tuned_threshold.csv")
    write_summary(summary_path, results_default, has_threshold_col=False)
    write_summary(tuned_summary_path, results_tuned, has_threshold_col=True)

    print(f"\nAppended baseline rows to {summary_path}")
    print(f"Appended tuned-threshold baseline rows to {tuned_summary_path}")