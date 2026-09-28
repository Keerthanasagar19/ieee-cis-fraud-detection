"""
preprocessing.py
------------------
Turns the raw transaction table into sliding-window sequences per account.

Key design decisions (defend these in your report):
  1. Running statistics (expanding mean/std of amount) are computed using
     ONLY past transactions for that account -> no leakage from future to past.
  2. inter_tx_seconds (time since the account's previous transaction) is kept
     as its own feature specifically because it becomes the input to the SNN's
     temporal / spike encoding -- this is the direct link between your
     preprocessing and your "SNN treats transactions as spike trains" claim.
  3. Splitting is done by ACCOUNT, not by row, so the same account never
     appears in both train and test -- this avoids the single most common
     data leakage bug in fraud detection papers.
"""

import numpy as np
import pandas as pd

WINDOW_SIZE = 10  # number of past transactions considered as context


# Fixed category mapping (must match generate_data.py's MERCHANT_CATEGORIES).
# IMPORTANT: this is deliberately hardcoded rather than recomputed from
# whatever categories happen to appear in a given dataframe. Recomputing
# per-dataframe silently breaks any new/incoming data that doesn't contain
# the full original category set (a real bug caught while testing the GNN
# enhancement script on new synthetic accounts) -- category codes would
# shift and the model would see meaningless feature values with no error
# or warning. A fixed mapping is what any real deployment needs anyway.
MERCHANT_CATEGORIES_FIXED = ["grocery", "electronics", "travel", "dining", "utilities",
                             "entertainment", "fashion", "fuel", "healthcare", "online_retail"]
_CAT_CODES = {c: i for i, c in enumerate(sorted(MERCHANT_CATEGORIES_FIXED))}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Per-account expanding (causal) statistics -- shift(1) so current row's
    # own amount is never included in its own baseline (no leakage)
    grp = df.groupby("account_id")["amount"]
    df["running_mean_amount"] = grp.transform(lambda s: s.shift(1).expanding().mean())
    df["running_std_amount"] = grp.transform(lambda s: s.shift(1).expanding().std())
    df["running_mean_amount"] = df["running_mean_amount"].fillna(df["amount"])
    df["running_std_amount"] = df["running_std_amount"].fillna(1.0).replace(0, 1.0)
    df["amount_zscore"] = (df["amount"] - df["running_mean_amount"]) / df["running_std_amount"]

    # Time since previous transaction for this account, in seconds (spike-train input)
    df["prev_timestamp"] = df.groupby("account_id")["timestamp"].shift(1)
    df["inter_tx_seconds"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    median_gap = df["inter_tx_seconds"].median()
    df["inter_tx_seconds"] = df["inter_tx_seconds"].fillna(median_gap)
    df["inter_tx_seconds"] = np.clip(df["inter_tx_seconds"], 0, 30 * 24 * 3600)  # cap at 30 days

    # Device / location change flags vs previous transaction of same account
    df["prev_device"] = df.groupby("account_id")["device_id"].shift(1)
    df["prev_location"] = df.groupby("account_id")["location_code"].shift(1)
    df["device_changed"] = (df["device_id"] != df["prev_device"]).astype(int)
    df["location_changed"] = (df["location_code"] != df["prev_location"]).astype(int)
    df.loc[df["prev_device"].isna(), "device_changed"] = 0
    df.loc[df["prev_location"].isna(), "location_changed"] = 0

    # Merchant category -> integer code (FIXED mapping, see _CAT_CODES above)
    df["merchant_code"] = df["merchant_category"].map(_CAT_CODES)
    df["merchant_code"] = df["merchant_code"].fillna(-1)  # unseen category -> distinct sentinel value

    return df


FEATURE_COLUMNS = [
    "amount", "amount_zscore", "hour", "day_of_week",
    "inter_tx_seconds", "device_changed", "location_changed", "merchant_code",
]


def build_sequences(df: pd.DataFrame, window_size: int = WINDOW_SIZE):
    """
    For each account, build overlapping windows of `window_size` consecutive
    transactions. The label of a window = is_fraud of the LAST transaction in
    the window (i.e. "given this account's recent history, is the newest
    transaction fraudulent?").
    """
    sequences, labels, account_ids = [], [], []

    for acc_id, g in df.groupby("account_id"):
        g = g.reset_index(drop=True)
        feats = g[FEATURE_COLUMNS].values.astype(np.float32)
        fraud = g["is_fraud"].values

        if len(g) < window_size:
            continue

        for end in range(window_size - 1, len(g)):
            start = end - window_size + 1
            seq = feats[start:end + 1]
            label = fraud[end]
            sequences.append(seq)
            labels.append(label)
            account_ids.append(acc_id)

    X = np.stack(sequences)               # (N, window_size, n_features)
    y = np.array(labels, dtype=np.int64)  # (N,)
    accs = np.array(account_ids)
    return X, y, accs


def split_by_account(df, seed=42, train_frac=0.7, val_frac=0.15):
    accounts = df["account_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(accounts)
    n = len(accounts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_acc = set(accounts[:n_train])
    val_acc = set(accounts[n_train:n_train + n_val])
    test_acc = set(accounts[n_train + n_val:])
    return train_acc, val_acc, test_acc


def normalize_features(X_train, X_val, X_test):
    """Z-normalize continuous features using train-set statistics only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0

    def apply(X):
        return (X - mean) / std

    return apply(X_train), apply(X_val), apply(X_test), (mean, std)


if __name__ == "__main__":
    import os
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    df = pd.read_csv(os.path.join(_DATA_DIR, "transactions.csv"), parse_dates=["timestamp"])
    df = engineer_features(df)

    train_acc, val_acc, test_acc = split_by_account(df)
    df_train = df[df["account_id"].isin(train_acc)]
    df_val = df[df["account_id"].isin(val_acc)]
    df_test = df[df["account_id"].isin(test_acc)]

    X_train, y_train, _ = build_sequences(df_train)
    X_val, y_val, _ = build_sequences(df_val)
    X_test, y_test, _ = build_sequences(df_test)

    X_train, X_val, X_test, norm_stats = normalize_features(X_train, X_val, X_test)

    print("Train:", X_train.shape, "fraud rate:", y_train.mean())
    print("Val:  ", X_val.shape, "fraud rate:", y_val.mean())
    print("Test: ", X_test.shape, "fraud rate:", y_test.mean())

    np.savez(os.path.join(_DATA_DIR, "sequences.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             feature_names=np.array(FEATURE_COLUMNS),
             norm_mean=norm_stats[0], norm_std=norm_stats[1])
    print("Saved to data/sequences.npz")
