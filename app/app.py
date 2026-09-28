"""
app.py
-------
A simple local web dashboard for the fraud detection project.

Run with:  python app.py
Then open: http://127.0.0.1:5000 in your browser

What it does:
  - Loads the three trained models (Mamba-KAN, SNN-rate, SNN-latency)
  - Lets you pull a random transaction window from the TEST set and see
    what each model predicts, side by side, with the true label
  - Displays your evaluation plots (ROC, PR curve, confusion matrices,
    feature importance) from the results/ folder

This turns the project from "a script that prints numbers" into an actual
interactive demo you can show in your review/viva.
"""

import os
import json
import numpy as np
import torch
from flask import Flask, render_template, jsonify, send_from_directory

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mamba_kan import MambaKAN
from snn_model import SNNFraudDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sequences.npz")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models_saved")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

app = Flask(__name__)

# ---- Load data and models once at startup ----
_d = np.load(DATA_PATH, allow_pickle=True)
X_TEST, Y_TEST = _d["X_test"], _d["y_test"]
FEATURE_NAMES = list(_d["feature_names"])
N_FEATURES = X_TEST.shape[-1]

# norm_mean/norm_std let us turn a z-normalized feature value back into a
# real-world number (real dollars, real seconds, a real yes/no flag) for
# the plain-English explanation panel. Without this, every flag/category
# feature (device_changed, card4_code, etc.) can only be read in its
# normalized form, which doesn't map cleanly back to "yes" vs "no" once
# it's been centered/scaled -- this is the actual bug fix, not just wording.
NORM_MEAN = _d["norm_mean"]
NORM_STD = _d["norm_std"]
_FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

mamba_kan = MambaKAN(n_features=N_FEATURES)
mamba_kan.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mamba_kan_best.pt"), map_location="cpu"))
mamba_kan.eval()

snn_rate = SNNFraudDetector(n_features=N_FEATURES, encoding="rate")
snn_rate.load_state_dict(torch.load(os.path.join(MODEL_DIR, "snn_rate_best.pt"), map_location="cpu"))
snn_rate.eval()

snn_latency = SNNFraudDetector(n_features=N_FEATURES, encoding="latency")
snn_latency.load_state_dict(torch.load(os.path.join(MODEL_DIR, "snn_latency_best.pt"), map_location="cpu"))
snn_latency.eval()

# --- Risk scoring thresholds (computed once by src/risk_scoring.py on the
# validation set) -- loaded here, never recomputed per-request. ---
_risk_summary_path = os.path.join(RESULTS_DIR, "enhancements", "risk_scoring_summary.json")
RISK_THRESHOLDS = None
if os.path.exists(_risk_summary_path):
    with open(_risk_summary_path) as _f:
        _risk_summary = json.load(_f)
        RISK_THRESHOLDS = (_risk_summary["method"]["t_low"], _risk_summary["method"]["t_high"])

RISK_EXPLAIN = {
    "LOW": "This score is low enough to auto-clear without review.",
    "MEDIUM": "This score is in the review range — not clearly safe, not clearly fraud.",
    "HIGH": "This score is above the cutoff we use to catch the large majority of real fraud — flagged for review.",
}


def risk_level_for(prob):
    if RISK_THRESHOLDS is None:
        return None
    t_low, t_high = RISK_THRESHOLDS
    if prob >= t_high:
        return "HIGH"
    if prob >= t_low:
        return "MEDIUM"
    return "LOW"


# ============================================================
# Plain-English feature descriptions
# ------------------------------------------------------------
# These reverse-lookup tables MUST stay in sync with the fixed
# category maps in data/ieee_cis/preprocess_ieee.py -- they are
# copied here (not imported) so this dashboard doesn't depend on
# the ieee_cis folder existing at runtime. If you ever change a
# category list in preprocess_ieee.py, update it here too.
# ============================================================
PRODUCT_CODES_FIXED = ["W", "C", "R", "H", "S"]
CARD4_FIXED = ["discover", "mastercard", "visa", "american express"]
CARD6_FIXED = ["credit", "debit", "debit or credit", "charge card"]
M4_FIXED = ["M0", "M1", "M2"]

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "amount_zscore": "How unusual the amount was for this account",
    "hour": "Time of day",
    "day_of_week": "Weekly timing pattern",
    "inter_tx_seconds": "Time since this account's last transaction",
    "device_changed": "Different device than last time",
    "location_changed": "Different email domain than last time",
    "merchant_code": "Product category",
    "card4_code": "Card network",
    "card6_code": "Card type",
    "dist1": "Distance between billing & transaction address",
    "C1": "Card-activity signal (C1)",
    "C2": "Card-activity signal (C2)",
    "C13": "Card-activity signal (C13)",
    "C14": "Card-activity signal (C14)",
    "D2": "Time-since-card-use signal (D2)",
    "D4": "Time-since-card-use signal (D4)",
    "D15": "Time-since-card-use signal (D15)",
    "M4_code": "Address/name match flag",
    "email_domain_match": "Buyer & recipient email domains",
}

# Kaggle's C1-C14 / D-family columns have withheld exact definitions (the
# competition organizers never published what they count), so we can't
# translate them into a fully specific sentence -- only "higher/lower
# than typical". Stating that limitation honestly beats inventing a false
# specific meaning for an anonymized column.
OPAQUE_KAGGLE_FEATURES = {"amount_zscore", "dist1", "C1", "C2", "C13", "C14", "D2", "D4", "D15"}


def denorm(fname, z_value):
    idx = _FEATURE_IDX[fname]
    return z_value * NORM_STD[idx] + NORM_MEAN[idx]


# These specific opaque columns use -1 as an explicit "missing" sentinel in
# preprocess_ieee.py (not a real value) -- amount_zscore is excluded here
# since it's a genuine running z-score that can legitimately sit near -1.
MISSING_SENTINEL_FEATURES = {"dist1", "C1", "C2", "C13", "C14", "D2", "D4", "D15"}


def _magnitude_phrase(z):
    az = abs(z)
    direction = "higher" if z > 0 else "lower"
    if az < 0.3:
        return "about typical"
    elif az < 1.0:
        return f"somewhat {direction} than typical"
    else:
        return f"much {direction} than typical"


def _format_duration(seconds):
    seconds = max(seconds, 0)
    if seconds < 120:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60
    if minutes < 120:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours/24:.1f} days"


def humanize_feature(fname, z_value):
    """Returns (plain_value_string, is_flag_style) for ONE feature, using
    the real-world (denormalized) value where that's meaningful, and the
    normalized value's magnitude where the raw units are Kaggle-anonymized
    and not independently interpretable."""
    raw = denorm(fname, z_value)

    if fname == "amount":
        return f"${max(raw, 0):,.2f}"
    if fname == "inter_tx_seconds":
        return _format_duration(raw)
    if fname == "hour":
        h = int(round(raw)) % 24
        return f"around {h:02d}:00"
    if fname == "day_of_week":
        # NOTE: IEEE-CIS's time reference point is anonymized, so we know
        # the 7-day CYCLE is real but not which position is a real Monday
        # -- naming an actual weekday here would be a fabricated precision
        # this data can't support.
        d = int(round(raw)) % 7
        return f"position {d} in the account's weekly cycle"
    if fname == "device_changed":
        return "Yes" if round(raw) >= 1 else "No"
    if fname == "location_changed":
        return "Yes" if round(raw) >= 1 else "No"
    if fname == "email_domain_match":
        r = int(round(raw))
        return {1: "Matched", 0: "Different", -1: "Not available"}.get(r, "Not available")
    if fname == "merchant_code":
        r = int(round(raw))
        return PRODUCT_CODES_FIXED[r] if 0 <= r < len(PRODUCT_CODES_FIXED) else "Not available"
    if fname == "card4_code":
        r = int(round(raw))
        return CARD4_FIXED[r].title() if 0 <= r < len(CARD4_FIXED) else "Not available"
    if fname == "card6_code":
        r = int(round(raw))
        return CARD6_FIXED[r].title() if 0 <= r < len(CARD6_FIXED) else "Not available"
    if fname == "M4_code":
        r = int(round(raw))
        return M4_FIXED[r] if 0 <= r < len(M4_FIXED) else "Not available"
    # Opaque Kaggle count/time-delta columns that use -1 as a "missing"
    # sentinel: check that FIRST, before falling back to magnitude phrasing,
    # or a genuinely missing value could get mislabeled as "much lower than
    # typical" instead of "Not available".
    if fname in MISSING_SENTINEL_FEATURES and abs(raw - (-1)) < 0.5:
        return "Not available"
    # Remaining opaque Kaggle count/time-delta columns: magnitude-only, honestly.
    return _magnitude_phrase(z_value)


def explanation_sentence(fname, z_value, contribution):
    label = FEATURE_LABELS.get(fname, fname)
    plain_value = humanize_feature(fname, z_value)
    direction = "made this look MORE like fraud" if contribution > 0 else "made this look LESS like fraud"
    return f"{label}: {plain_value} — {direction}"


# ------------------------------------------------------------
def explain_prediction(x, top_k=6):
    """Local (per-transaction) explainability for Mamba-KAN, using the same
    perturb-and-measure family of method as evaluate.py's sensitivity_scores
    (used for the global "Mamba-KAN Input Sensitivity" chart) -- but here
    applied to ONE specific transaction, and using zero-ablation instead of
    a +1 std nudge, which is the more natural question for "why did THIS
    transaction get this score": for each feature, replace its value with
    the population mean (0.0, since features are z-normalized) across the
    whole 10-step window, and measure how much the predicted probability
    drops (or rises).

    contribution = base_prob - ablated_prob
      positive -> this feature's actual value was PUSHING the prediction
                  toward fraud (removing it lowers the score)
      negative -> this feature's actual value was PUSHING the prediction
                  toward normal / suppressing the fraud signal (removing
                  it raises the score)

    This is a standard local explanation technique (occlusion / feature
    ablation), not a novel method invented for this project -- worth
    saying exactly that in the report and viva.
    """
    with torch.no_grad():
        base_prob = torch.sigmoid(mamba_kan(x)).item()

    contributions = []
    for f_idx, fname in enumerate(FEATURE_NAMES):
        x_ablated = x.clone()
        x_ablated[:, :, f_idx] = 0.0  # 0.0 = the z-normalized population mean
        with torch.no_grad():
            ablated_prob = torch.sigmoid(mamba_kan(x_ablated)).item()
        contribution = base_prob - ablated_prob
        current_value = x[0, -1, f_idx].item()  # this transaction's actual (current, z-normalized) value
        contributions.append({
            "feature": fname,
            "label": FEATURE_LABELS.get(fname, fname),
            "contribution": round(contribution, 4),
            "value": round(current_value, 3),
            "plain_value": humanize_feature(fname, current_value),
            "text": explanation_sentence(fname, current_value, contribution),
        })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    # A genuine finding worth surfacing directly, not just implying from a
    # table: is this a case where many small signals add up, or did one
    # feature dominate? Threshold chosen so a single-feature contribution
    # has to be clearly larger than the rest of a typical top-6 spread
    # before we call it "dominant" -- avoids overclaiming on borderline cases.
    max_abs = max(abs(c["contribution"]) for c in contributions) if contributions else 0.0
    if max_abs < 0.08:
        summary = ("No single factor dominated this decision — the model combined "
                   "several small signals rather than relying on one obvious red flag.")
    else:
        top = contributions[0]
        direction = "pushed this toward FRAUD" if top["contribution"] > 0 else "pushed this toward NORMAL"
        summary = f"The single biggest factor was: {top['label']} ({top['plain_value']}) — it {direction}."

    return contributions[:top_k], summary


def predict_one(idx):
    x = torch.tensor(X_TEST[idx:idx + 1], dtype=torch.float32)
    x_last = x[:, -1, :]

    with torch.no_grad():
        p_mk = torch.sigmoid(mamba_kan(x)).item()
        p_rate = torch.sigmoid(snn_rate(x_last)).item()
        p_latency = torch.sigmoid(snn_latency(x_last)).item()
    p_ensemble = 0.5 * p_mk + 0.5 * p_latency

    window = X_TEST[idx].tolist()  # (10, n_features) normalized values
    last_row_z = window[-1]
    plain_features = [
        {"feature": fname, "label": FEATURE_LABELS.get(fname, fname),
         "plain_value": humanize_feature(fname, last_row_z[i])}
        for i, fname in enumerate(FEATURE_NAMES)
    ]

    explanation, explanation_summary = explain_prediction(x)
    risk_level = risk_level_for(p_ensemble)

    return {
        "index": int(idx),
        "true_label": int(Y_TEST[idx]),
        "feature_names": FEATURE_NAMES,
        "window": window,
        "plain_features": plain_features,
        "predictions": {
            "Mamba-KAN": round(p_mk, 4),
            "SNN (rate coding)": round(p_rate, 4),
            "SNN (latency coding)": round(p_latency, 4),
            "Ensemble": round(p_ensemble, 4),
        },
        "risk_level": risk_level,  # None if risk_scoring.py hasn't been run yet
        "risk_explain": RISK_EXPLAIN.get(risk_level),
        "explanation": explanation,  # top contributing features for THIS transaction
        "explanation_summary": explanation_summary,
    }


def load_json_safe(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@app.route("/")
def index():
    metrics_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    metrics_rows = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            lines = f.read().splitlines()
            header = lines[0].split(",")
            for line in lines[1:]:
                metrics_rows.append(dict(zip(header, line.split(","))))

    gnn_summary = load_json_safe(os.path.join(RESULTS_DIR, "enhancements", "gnn_summary.json"))
    drift_summary = load_json_safe(os.path.join(RESULTS_DIR, "enhancements", "drift_summary.json"))
    cold_start_summary = load_json_safe(os.path.join(RESULTS_DIR, "enhancements", "cold_start_summary.json"))
    risk_scoring_summary = load_json_safe(os.path.join(RESULTS_DIR, "enhancements", "risk_scoring_summary.json"))

    return render_template("index.html", metrics_rows=metrics_rows,
                            gnn_summary=gnn_summary, drift_summary=drift_summary,
                            cold_start_summary=cold_start_summary,
                            risk_scoring_summary=risk_scoring_summary)

@app.route("/api/random_sample")
def random_sample():
    idx = np.random.randint(0, len(X_TEST))
    return jsonify(predict_one(idx))


@app.route("/api/fraud_sample")
def fraud_sample():
    """Specifically pull a known-fraud test example, so the demo isn't
    always showing the (much more common) normal case."""
    fraud_indices = np.where(Y_TEST == 1)[0]
    if len(fraud_indices) == 0:
        return random_sample()
    idx = np.random.choice(fraud_indices)
    return jsonify(predict_one(idx))


@app.route("/results/<path:filename>")
def result_files(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    print("\nModels loaded. Starting dashboard... open http://127.0.0.1:5000 in your browser")
    print("(the first request after startup may take a moment while PyTorch warms up)\n")
    app.run(debug=False, port=5000)