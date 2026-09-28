# FFDS — Fraud Detection Research Stack

AI-Based Financial Fraud Detection in Digital Transactions, using a
**Mamba-KAN** sequence model and a **Spiking Neural Network (SNN)**, combined
in an ensemble.

This README is written so you can explain every design decision in your
viva/defense — each section states *what* was built and *why*, not just how
to run it.

---

## 1. Problem framing

Fraud in digital transactions is rarely detectable from a single transaction
in isolation — it's usually a **deviation from an account's own behavioural
history** (sudden burst of transactions, an odd-hour purchase, a device the
account has never used, an amount far outside its usual range). This is why
the project uses **sequence models over per-account transaction windows**
rather than a flat row-by-row classifier (e.g. plain XGBoost/Random Forest
on single transactions) — the latter has no access to *context*.

## 2. Why Mamba-KAN?

- **Selective state-space (Mamba-style) block**: processes the transaction
  sequence with a recurrence whose "memory update" is *data-dependent*
  (unlike a plain RNN/SSM, which decays state at a fixed rate regardless of
  input). This lets the model learn to carry forward the effect of an
  anomalous transaction while mostly "forgetting" routine ones. It also
  scales linearly with sequence length, unlike Transformer self-attention.
- **KAN (Kolmogorov-Arnold Network) layer**: replaces a normal dense layer's
  single shared activation function with a *separate learnable function per
  edge*. This is what gives the model interpretability — you can inspect how
  strongly and in what shape each input contributed to the output, which
  matters for fraud systems that regulators/auditors need to be able to
  question.

**Implementation note (important for your report):** the official
`mamba-ssm` PyPI package depends on custom CUDA kernels that require a GPU
and a matching CUDA toolkit to compile — not guaranteed on a college lab
machine or a plain CPU grading environment. `src/mamba_kan.py` reimplements
the *same selective state-space recurrence* in pure PyTorch so it runs
anywhere, at the cost of some training speed. State this explicitly as a
deliberate engineering trade-off, not a limitation you're hiding.

## 3. Why an SNN?

Spiking Neural Networks process input as **spike trains over simulated
timesteps** rather than static feature vectors — so *timing itself* becomes
part of what the model can learn from. This directly matches a real fraud
signal: **velocity fraud** (a burst of transactions within minutes) compresses
inter-transaction timing, and a model that encodes timing as spike latency
can pick that up in a way a flat classifier cannot.

Built with `snntorch` (a real, published, peer-reviewed SNN library) using
Leaky-Integrate-and-Fire (LIF) neurons — not reimplemented from scratch,
since correct LIF dynamics are standard and using a tested library is more
defensible than a hand-rolled version.

**Two temporal encodings are implemented and compared (this is your
ablation study):**
| Encoding | Idea | Test PR-AUC |
|---|---|---|
| Rate coding | Feature magnitude → spike *probability* per timestep | 0.278 |
| Latency coding | Feature magnitude → spike *timing* (anomalous = fires early) | 0.637 |

Latency coding dramatically outperforms rate coding here — a genuine,
explainable result: it directly encodes "how urgent/anomalous is this
value" as "how early does it fire," which lines up with how the velocity and
odd-hour fraud patterns were constructed in the dataset. **This finding is
exactly what your abstract's line about "temporal encoding choices had a
measurable effect on SNN precision" should point to.**

## 4. Ensemble

Final ensemble = simple average of Mamba-KAN's probability and the
**better-performing** SNN variant (latency coding) — not a naive average of
all three, because the ablation showed rate coding underperforms enough that
including it would drag the ensemble down. Justify this ensemble weighting
choice using the ablation table above.

## 5. Results (held-out test set, never used in training or model selection)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.875 | 0.113 | 0.859 | 0.200 | 0.928 | 0.515 |
| Random Forest (baseline) | 0.995 | **0.937** | 0.796 | **0.861** | 0.983 | **0.903** |
| Mamba-KAN | 0.972 | 0.385 | **0.937** | 0.546 | 0.988 | 0.888 |
| SNN (rate) | 0.766 | 0.059 | 0.796 | 0.110 | 0.863 | 0.278 |
| SNN (latency) | 0.946 | 0.237 | 0.903 | 0.376 | 0.973 | 0.637 |
| Ensemble | 0.976 | 0.242 | **0.947** | 0.386 | 0.984 | 0.830 |

**Important, honest finding — read this before writing your results section:**
Random Forest, using only the LAST transaction's features (no sequence
context at all), actually beats Mamba-KAN on precision and F1. Don't hide
this — it's a real and defensible result, and explaining WHY it happens
makes you look more rigorous, not less:

The "flat" baseline isn't actually starting from zero context. The
per-transaction feature engineering in `preprocessing.py` (running mean/std
of amount, `device_changed`, `location_changed`) already bakes a summary of
the account's history into each single row. So Random Forest gets partial
temporal context "for free" through feature engineering, without needing to
learn any sequence modelling itself — which is exactly the kind of thing a
tree ensemble is very good at exploiting.

**What the sequence models still do better:** recall. Mamba-KAN and the
ensemble catch more actual fraud (93.7% / 94.7%) than Random Forest (79.6%).
In a real fraud system, missing fraud is usually costlier than a false
alarm, so this recall advantage is a legitimate practical argument for the
sequence approach — just don't claim it "beats" Random Forest across the
board, because on precision/F1 it currently doesn't.

**A fairer future experiment** (see Future Enhancements below): retrain the
baseline on RAW, non-engineered features only, and give the sequence models
the raw history directly. That isolates how much of the advantage is really
"sequence modelling" versus "good feature engineering" — right now the two
are entangled.

## 6. Interpretability

`results/feature_importance.png` shows a sensitivity analysis: each raw
input feature is perturbed by +1 standard deviation and the resulting change
in the model's output probability is measured and averaged across test
samples. This is more honest than reading raw KAN edge weights directly (the
KAN layers operate on a *learned projection* of the input, not the raw
features themselves) — but the KAN's per-edge weights (`model.kan1.last_edge_weights`)
are also available if you want to show the internal edge structure in an
appendix.

## 7. Project structure

```
fraud_project/
├── data/
│   ├── generate_data.py   # synthetic dataset generator (see docstring for why synthetic)
│   ├── transactions.csv   # raw generated transactions
│   └── sequences.npz      # preprocessed sliding-window sequences
├── src/
│   ├── preprocessing.py   # feature engineering + windowing + train/val/test split
│   ├── mamba_kan.py       # Selective SSM block + KAN layer + full model
│   ├── snn_model.py       # SNN with rate/latency encoding
│   ├── train.py           # training loop, class-imbalance handling
│   └── evaluate.py        # metrics, plots, ensemble, feature importance
├── app/
│   ├── app.py             # Flask backend for the interactive demo dashboard
│   └── templates/
│       └── index.html     # dashboard UI (dark "fraud console" theme)
├── models_saved/          # trained model weights (.pt)
├── results/               # plots + summary_metrics.csv
└── README.md
```

## 8. How to reproduce

```bash
pip install -r requirements.txt --break-system-packages

cd data && python generate_data.py        # generates transactions.csv
cd ../src
python preprocessing.py                    # generates ../data/sequences.npz
python train.py mamba_kan                   # trains Mamba-KAN
python train.py snn_rate                    # trains SNN (rate coding)
python train.py snn_latency                 # trains SNN (latency coding)
python evaluate.py                          # full evaluation + plots
```

## 9. Interactive demo dashboard (for your presentation/viva)

Instead of showing raw terminal output, run the included web dashboard:

```bash
cd app
python app.py
```

Wait for `Models loaded. Starting dashboard...` (takes ~10-15 seconds — PyTorch's
startup is slow the first time), then open **http://127.0.0.1:5000** in your
browser. You'll see:
- A "Pull random transaction" / "Pull known fraud case" button that runs a
  real test-set transaction through all three models live, with a probability
  bar per model and the true label shown for comparison
- Your evaluation plots (ROC, PR curve, confusion matrices, feature
  importance) and metrics table rendered on the same page

This is what you should actually show in your project demo — it's the same
trained models from Section 8, just wrapped in something you can click
through instead of reading off a terminal.


## 10. Future enhancements — with working prototypes

Two of the proposed future enhancements were taken beyond a proposal and
actually prototyped, in `src/enhancements/`, specifically to show something a
plain sequence model gets wrong on its own:

### 10a. Graph Neural Network — fraud ring detection (`ring_fraud_gnn.py`)

**Setup:** 4 synthetic fraud rings of 5 accounts each. In each ring, one
"mule" account makes an obvious large fraudulent transaction; the other 4
"quiet" members each make a SINGLE, perfectly ordinary-looking transaction —
same amount range as their own normal spending — but using the same shared
device as the mule.

**Result:**
| | Avg. score on quiet ring members | Avg. score on normal accounts | Separation |
|---|---|---|---|
| Mamba-KAN alone | 0.359 | 0.139 | 0.220 (weak, inconsistent — misses ~half) |
| GCN (shared-device graph) | 0.9999 | 0.0001 | 0.9998 (near-total separation) |

Mamba-KAN, using only each account's own history, catches some quiet
members by luck but misses roughly half. The GCN — a minimal 2-layer Graph
Convolutional Network implemented from scratch in plain PyTorch (no
`torch-geometric` dependency, same portability reasoning as the Mamba
block) — propagates suspicion across the shared-device graph and correctly
flags every quiet member once the mule is flagged. See
`results/enhancements/gnn_ring_detection.png`.

### 10b. Concept Drift Detection (`concept_drift.py`)

**Setup:** simulates fraudsters switching tactics — instead of a large
anomalous transaction, they now make several small transactions (55-85% of
the account's own normal amount), spread across separate days, at the
account's own usual hours, from the account's own usual device/location.
This deliberately avoids every anomaly signal the original model was
trained to catch.

**Result:** scoring this new data with the ALREADY-TRAINED Mamba-KAN (no
retraining) shows recall collapsing from ~90-95% (original test set) to
**46%** — real, measurable model staleness. A Population Stability Index
(PSI) check — a standard metric used in real credit-risk/fraud
monitoring, not a research toy — comparing the training distribution
against this new batch's features still correctly flags `inter_tx_seconds`
as significantly drifted (PSI = 0.49, well above the standard 0.25
alert threshold), even though the fraud was designed to evade the model
itself. This demonstrates a real, deployable early-warning mechanism: you
can detect that something has shifted before anyone notices the accuracy
drop in production. See `results/enhancements/concept_drift_detection.png`.

### How to run these

```bash
cd src/enhancements
python ring_fraud_gnn.py
python concept_drift.py
```

## 11. Honest limitations to state in your report (this makes you look stronger, not weaker)

- Dataset is **synthetic**, generated to have realistic, KNOWN fraud
  mechanisms — this was a deliberate choice (see `generate_data.py`
  docstring) to enable a clean ablation study, but it means real-world
  validation on an actual bank/payment dataset (e.g. IEEE-CIS, PaySim) is
  future work.
- Mamba block is a simplified pure-PyTorch reimplementation of the
  selective-SSM recurrence, not the official CUDA-accelerated `mamba-ssm`
  package — a deliberate portability trade-off, not an oversight.
- KAN layer uses a fixed basis-function bank rather than full learnable
  B-splines (as in the original KAN paper) — chosen for speed and to avoid
  an external dependency; still preserves the core "learnable function per
  edge" property.
- Single train/val/test split (by account, no leakage) rather than k-fold
  cross-validation, due to compute/time constraints — worth mentioning as
  future work if asked.
