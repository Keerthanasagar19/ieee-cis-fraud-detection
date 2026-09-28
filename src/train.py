"""
train.py
---------
Trains both models on the preprocessed sequence data.

Class imbalance handling: fraud is ~2% of the data. We use a weighted
BCEWithLogitsLoss (pos_weight = n_negative / n_positive) rather than naive
oversampling/undersampling, because:
  - Oversampling the minority class in SEQUENCE data risks duplicating
    near-identical windows and inflating validation metrics artificially.
  - pos_weight keeps every real example, just reweights the loss, which is
    the more defensible choice to explain in a viva.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mamba_kan import MambaKAN
from snn_model import SNNFraudDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "..", "data", "sequences.npz")
SAVE_DIR = os.path.join(_BASE_DIR, "..", "models_saved")

SEED = 42
# Softening pos_weight: the raw imbalance ratio (~20.5x) pushes every model
# to predict "fraud-leaning" for almost everything, which is why default
# threshold=0.5 precision is so low (~0.10-0.12) across the board. Taking
# the square root gives a gentler push -- still compensates for imbalance,
# but doesn't force the model to treat "very unsure" as "predict fraud".
# Set to False to reproduce the original raw-ratio behaviour for comparison.
SOFTEN_POS_WEIGHT = True


def set_seed(seed=SEED):
    """Fixes every source of randomness so consecutive runs are comparable.
    Without this, two runs of the same script differ in weight init and
    data shuffling, making it impossible to tell whether a change (like
    adding dropout) actually helped or if you're just looking at noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"])


def get_pos_weight(y, soften=SOFTEN_POS_WEIGHT):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ratio = n_neg / max(n_pos, 1)
    raw_ratio = ratio
    if soften:
        ratio = ratio ** 0.5  # see SOFTEN_POS_WEIGHT note above
    print(f"pos_weight: raw imbalance ratio={raw_ratio:.4f}  "
          f"{'(softened to sqrt) -> using ' + f'{ratio:.4f}' if soften else '(using raw, unsoftened)'}")
    return torch.tensor(ratio, dtype=torch.float32)


def train_model(model, train_loader, val_loader, pos_weight, epochs=12, lr=1e-3, use_last_step_only=False, name="model"):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if use_last_step_only:
                xb = xb[:, -1, :]  # SNN takes only the latest transaction's features
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if use_last_step_only:
                    xb = xb[:, -1, :]
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[{name}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/{name}_best.pt")

    return history


def main(which="all"):
    set_seed(SEED)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    n_features = X_train.shape[-1]

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    pos_weight = get_pos_weight(y_train)

    histories = {}

    if which in ("all", "mamba_kan"):
        print("\n=== Training Mamba-KAN ===")
        mamba_kan = MambaKAN(n_features=n_features)
        histories["mamba_kan"] = train_model(mamba_kan, train_loader, val_loader, pos_weight,
                                              epochs=12, lr=1e-3, use_last_step_only=False, name="mamba_kan")

    if which in ("all", "snn_rate"):
        print("\n=== Training SNN (rate coding) ===")
        snn_rate = SNNFraudDetector(n_features=n_features, encoding="rate")
        histories["snn_rate"] = train_model(snn_rate, train_loader, val_loader, pos_weight,
                                             epochs=12, lr=1e-3, use_last_step_only=True, name="snn_rate")

    if which in ("all", "snn_latency"):
        print("\n=== Training SNN (latency coding) ===")
        snn_latency = SNNFraudDetector(n_features=n_features, encoding="latency")
        histories["snn_latency"] = train_model(snn_latency, train_loader, val_loader, pos_weight,
                                                epochs=12, lr=1e-3, use_last_step_only=True, name="snn_latency")

    # merge with any existing histories file so partial runs don't clobber each other
    import os
    hist_path = f"{SAVE_DIR}/histories.npz"
    merged = {}
    if os.path.exists(hist_path):
        old = np.load(hist_path, allow_pickle=True)
        for k in old.files:
            merged[k] = old[k].item()
    merged.update(histories)
    np.savez(hist_path, **merged)
    print(f"\nDone with '{which}'. Saved to", SAVE_DIR)


if __name__ == "__main__":
    import sys
    which_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(which_arg)