"""
evaluate.py
------------
Full evaluation of Mamba-KAN, both SNN variants, and the ensemble on the
held-out TEST set (never seen during training or model selection).

Metrics reported (why each matters for fraud, since accuracy alone lies
on imbalanced data):
  - Precision, Recall, F1        : the standard imbalanced-classification trio
  - ROC-AUC                       : threshold-independent ranking quality
  - PR-AUC (average precision)   : MORE informative than ROC-AUC when
                                    positives are rare (this is the number a
                                    reviewer will actually check first)
  - Confusion matrix              : shows the real cost trade-off (missed
                                    frauds vs false alarms)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                              average_precision_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              accuracy_score)

from mamba_kan import MambaKAN
from snn_model import SNNFraudDetector

import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "..", "data", "sequences.npz")
MODEL_DIR = os.path.join(_BASE_DIR, "..", "models_saved")
RESULTS_DIR = os.path.join(_BASE_DIR, "..", "results")


def load_test_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["X_test"], d["y_test"], list(d["feature_names"])


def load_val_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["X_val"], d["y_val"]


def find_best_threshold(y_true, y_prob):
    """Scans candidate thresholds and returns the one maximizing F1 on
    whatever set is passed in. IMPORTANT: this must be called on the
    VALIDATION set, never the test set -- picking a threshold using the
    test set's own labels is a form of leakage (you'd be "cheating" by
    letting the test set tell you how to score itself), the same category
    of bug this project's account-level train/val/test split was designed
    to avoid in the first place. The chosen threshold is then applied,
    unmodified, to the untouched test set."""
    thresholds = np.unique(y_prob)
    if len(thresholds) > 1000:  # subsample for speed on large val sets
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 1000))
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh), float(best_f1)


def get_probs(model, X, use_last_step_only, batch_size=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            if use_last_step_only:
                xb = xb[:, -1, :]
            logits = model(xb)
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs)


def report_metrics(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {name} (threshold={threshold:.3f}) ---")
    print(f"Accuracy : {acc:.4f}  (NOTE: misleading alone on imbalanced data, shown for completeness)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Confusion Matrix [ [TN FP] [FN TP] ]:\n{cm}")

    return {"name": name, "threshold": threshold, "accuracy": acc, "precision": prec,
             "recall": rec, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "cm": cm,
             "fpr": fpr, "tpr": tpr, "y_prob": y_prob}


def plot_roc_pr(results, out_path_prefix):
    # ROC
    plt.figure(figsize=(6, 5))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_roc.png", dpi=150)
    plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    for r in results:
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        plt.plot(rec, prec, label=f"{r['name']} (AP={r['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_pr.png", dpi=150)
    plt.close()


def plot_confusion(results, out_path_prefix):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = r["cm"]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(r["name"])
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_confusion.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, out_path):
    importance = model.get_feature_importance()  # (in_features=d_model, out=kan_hidden)
    # d_model is a learned projection, not raw features, so instead we do a
    # simple sensitivity analysis: perturb each RAW input feature and measure
    # output change -- more honest than pretending KAN edges map 1:1 to raw features.
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(feature_names)), sensitivity_scores(model, feature_names))
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.ylabel("Mean |output change| per unit perturbation")
    plt.title("Mamba-KAN Input Sensitivity (interpretability proxy)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def sensitivity_scores(model, feature_names, n_samples=256, window_size=10):
    """Perturbs each raw feature (across the whole window) by +1 std and
    measures the average absolute change in model output probability.
    This is the honest way to get feature-level interpretability out of a
    sequence model -- it directly measures what the report should claim,
    rather than over-interpreting internal KAN edge weights as raw feature
    importances."""
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X_test"][:n_samples]
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        base = torch.sigmoid(model(X_t)).numpy()

    scores = []
    for f_idx in range(len(feature_names)):
        X_pert = X.copy()
        X_pert[:, :, f_idx] += 1.0  # features are already z-normalized, so +1 = +1 std
        X_pert_t = torch.tensor(X_pert, dtype=torch.float32)
        with torch.no_grad():
            pert = torch.sigmoid(model(X_pert_t)).numpy()
        scores.append(np.mean(np.abs(pert - base)))
    return scores


if __name__ == "__main__":
    X_test, y_test, feature_names = load_test_data()
    n_features = X_test.shape[-1]

    mamba_kan = MambaKAN(n_features=n_features)
    mamba_kan.load_state_dict(torch.load(f"{MODEL_DIR}/mamba_kan_best.pt"))

    snn_rate = SNNFraudDetector(n_features=n_features, encoding="rate")
    snn_rate.load_state_dict(torch.load(f"{MODEL_DIR}/snn_rate_best.pt"))

    snn_latency = SNNFraudDetector(n_features=n_features, encoding="latency")
    snn_latency.load_state_dict(torch.load(f"{MODEL_DIR}/snn_latency_best.pt"))

    prob_mk = get_probs(mamba_kan, X_test, use_last_step_only=False)
    prob_snn_rate = get_probs(snn_rate, X_test, use_last_step_only=True)
    prob_snn_latency = get_probs(snn_latency, X_test, use_last_step_only=True)

    # Ensemble: simple average of Mamba-KAN and the BETTER SNN variant
    # (latency coding, per the training curves) -- justify this choice in
    # your report using the ablation results, don't just average everything blindly.
    prob_ensemble = 0.5 * prob_mk + 0.5 * prob_snn_latency

    print("\n################################################")
    print("# PART A: metrics at the default threshold=0.5   #")
    print("################################################")
    results = []
    results.append(report_metrics("Mamba-KAN", y_test, prob_mk))
    results.append(report_metrics("SNN (rate coding)", y_test, prob_snn_rate))
    results.append(report_metrics("SNN (latency coding)", y_test, prob_snn_latency))
    results.append(report_metrics("Ensemble (Mamba-KAN + SNN-latency)", y_test, prob_ensemble))

    plot_roc_pr(results, f"{RESULTS_DIR}/eval")
    plot_confusion(results, f"{RESULTS_DIR}/eval")
    plot_feature_importance(mamba_kan, feature_names, f"{RESULTS_DIR}/feature_importance.png")

    # Save a summary table (unchanged format/filename -- default threshold only)
    import csv
    with open(f"{RESULTS_DIR}/summary_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
        for r in results:
            writer.writerow([r["name"], f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                              f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}"])

    # --- PART B: threshold tuning -------------------------------------
    # 0.5 is an arbitrary default -- it's the natural cutoff for a raw
    # sigmoid, not a value chosen for THIS problem's class balance. Here
    # we pick, per model, the threshold that maximizes F1 on the VALIDATION
    # set (never the test set -- see find_best_threshold's docstring for
    # why that split matters), then report test-set metrics at that
    # threshold. ROC-AUC/PR-AUC don't change (they're threshold-independent
    # by definition) -- only precision/recall/F1/confusion matrix do.
    print("\n################################################")
    print("# PART B: metrics at a VALIDATION-tuned threshold #")
    print("################################################")
    X_val, y_val = load_val_data()
    prob_mk_val = get_probs(mamba_kan, X_val, use_last_step_only=False)
    prob_snn_rate_val = get_probs(snn_rate, X_val, use_last_step_only=True)
    prob_snn_latency_val = get_probs(snn_latency, X_val, use_last_step_only=True)
    prob_ensemble_val = 0.5 * prob_mk_val + 0.5 * prob_snn_latency_val

    tuned_pairs = [
        ("Mamba-KAN", prob_mk_val, prob_mk),
        ("SNN (rate coding)", prob_snn_rate_val, prob_snn_rate),
        ("SNN (latency coding)", prob_snn_latency_val, prob_snn_latency),
        ("Ensemble (Mamba-KAN + SNN-latency)", prob_ensemble_val, prob_ensemble),
    ]
    tuned_results = []
    for name, val_probs, test_probs in tuned_pairs:
        best_t, val_f1 = find_best_threshold(y_val, val_probs)
        print(f"\n[{name}] best threshold on validation set: {best_t:.3f} (val F1={val_f1:.4f})")
        tuned_results.append(report_metrics(f"{name} (tuned)", y_test, test_probs, threshold=best_t))

    with open(f"{RESULTS_DIR}/summary_metrics_tuned_threshold.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
        for r in tuned_results:
            writer.writerow([r["name"], f"{r['threshold']:.4f}", f"{r['accuracy']:.4f}",
                              f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                              f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}"])

    print("\nAll plots, summary_metrics.csv (threshold=0.5), and")
    print("summary_metrics_tuned_threshold.csv (validation-tuned per model)")
    print("saved to", RESULTS_DIR)
