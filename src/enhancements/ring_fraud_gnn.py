"""
ring_fraud_gnn.py
-------------------
PROTOTYPE for Future Enhancement 1 (Graph Neural Networks).

Runs on REAL IEEE-CIS data and REAL fraud labels. The original version of
this script (referenced in the dashboard's now-outdated blurb text) built
a fully fabricated "ring" scenario (fake accounts, a fake shared device,
fake labels) to illustrate the concept cleanly. This version instead asks
the honest, harder question: in the REAL data, does the device-sharing
graph actually help catch real fraud accounts whose own transaction
history looks unremarkable to Mamba-KAN? There's no guarantee the answer
is as clean as the synthetic demo's -- that's the point of testing on
real data instead of a constructed example.

METHOD:
  1. Reuse preprocess_ieee.py's own loading/feature-engineering functions
     (same uid reconstruction, same causal features), consistent with the
     main pipeline and concept_drift.py -- not a third diverging
     definition of the data.
  2. Build a graph where nodes = real uids (pseudo-accounts) and edges =
     "these two uids were seen using the same real DeviceInfo value".
     IEEE-CIS's identity table only covers a minority of transactions, so
     this graph is sparse by construction -- that's a real property of
     the data, not a bug to work around.
  3. Score every uid's OWN last transaction window with the already-trained
     Mamba-KAN (same as evaluate.py does) -- this is "no graph knowledge".
  4. Train a small GCN (from-scratch implementation, no torch-geometric
     dependency) on [uid summary features + Mamba-KAN's own score]
     propagated across the real device graph, using REAL isFraud labels.
  5. The interesting comparison: among uids device-graph-CONNECTED to at
     least one other uid, how does the GCN score real fraud uids that
     Mamba-KAN itself scored LOW (i.e. real "quiet" cases Mamba-KAN missed
     on its own), versus real normal uids with no device connections at
     all (a genuine control group, not fabricated)?

CAVEATS TO STATE HONESTLY IN YOUR REPORT:
  - Mamba-KAN here is the SAME model trained on the main train/val/test
    split (which is account-level random, not graph-aware) -- some of
    these uids may have been in its own training set, which can make its
    "no graph knowledge" score look slightly better than a true held-out
    evaluation would. A stricter version would retrain excluding any uid
    that appears in this graph; left as a documented limitation, not
    fixed here, in the interest of keeping this a prototype rather than
    a second full training pipeline.
  - Whether real IEEE-CIS device-sharing structure resembles organized
    "rings" at all is an empirical question this script answers, not an
    assumption baked into it.

USAGE (run from inside fraud_project/src/enhancements/):
    python ring_fraud_gnn.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

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

MAX_CONTROL_ACCOUNTS = 300  # cap on isolated normal uids, purely for a readable plot/runtime
MAX_CONNECTED_ACCOUNTS = 4000  # safety cap on graph size for memory/runtime


# ------------------------------------------------------------------
# 1. Load real data and build the REAL device-sharing graph
# ------------------------------------------------------------------
def load_real_engineered_data():
    df = load_and_merge()
    df = filter_by_uid_count(df)
    df = engineer_features(df)
    return df


def build_real_device_graph(df):
    """Edges between uids that share a real, known DeviceInfo/DeviceType
    value. 'unknown_device' (no identity match) is deliberately excluded --
    it's not a real shared device, just missing data, and connecting every
    identity-less uid through it would create a meaningless giant component."""
    known = df[df["device_id"] != "unknown_device"]
    device_to_uids = known.groupby("device_id")["uid"].unique().to_dict()

    G = nx.Graph()
    for device, uids in device_to_uids.items():
        uids = list(uids)
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                G.add_edge(uids[i], uids[j], device=device)
    return G


# ------------------------------------------------------------------
# 2. Score each uid with Mamba-KAN using its OWN real transaction window
#    (identical logic to how evaluate.py/preprocess_ieee.py build windows)
# ------------------------------------------------------------------
def mamba_kan_uid_scores(df, uid_list, norm_mean, norm_std, n_features):
    model = MambaKAN(n_features=n_features)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mamba_kan_best.pt"), map_location="cpu"))
    model.eval()

    scores, agg_features, fraud_label = {}, {}, {}
    uid_set = set(uid_list)
    for uid, g in df.groupby("uid"):
        if uid not in uid_set:
            continue
        g = g.sort_values("TransactionDT").reset_index(drop=True)
        feats = g[FEATURE_COLUMNS].values.astype(np.float32)
        window = feats[-WINDOW_SIZE:]  # every uid here already has >= WINDOW_SIZE rows (filter_by_uid_count)
        window_norm = (window - norm_mean) / norm_std
        x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        scores[uid] = prob
        agg_features[uid] = [len(g), float(g["amount"].mean()), float(g["amount_zscore"].max())]
        fraud_label[uid] = int(g["isFraud"].max())  # real ground truth: did this uid EVER commit fraud
    return scores, agg_features, fraud_label


# ------------------------------------------------------------------
# 3. Minimal from-scratch GCN
# ------------------------------------------------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden=8):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden)
        self.w2 = nn.Linear(hidden, 1)

    def forward(self, X, A_hat):
        h = F.relu(A_hat @ self.w1(X))
        out = A_hat @ self.w2(h)
        return out.squeeze(-1)


def normalized_adjacency(G, node_order):
    n = len(node_order)
    idx = {a: i for i, a in enumerate(node_order)}
    A = np.eye(n)
    for u, v in G.edges():
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1
            A[idx[v], idx[u]] = 1
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_hat, dtype=torch.float32)


# ------------------------------------------------------------------
# 4. Run the full real-data comparison
# ------------------------------------------------------------------
def main():
    d = np.load(os.path.join(DATA_DIR, "sequences.npz"), allow_pickle=True)
    norm_mean, norm_std = d["norm_mean"], d["norm_std"]
    n_features = len(list(d["feature_names"]))

    print("Loading and engineering real IEEE-CIS data (same pipeline as preprocess_ieee.py)...")
    df = load_real_engineered_data()

    print("Building real device-sharing graph...")
    G_full = build_real_device_graph(df)
    connected_uids = [u for u in G_full.nodes() if G_full.degree(u) > 0]
    print(f"Real device-sharing graph: {len(connected_uids)} uids connected via a shared device "
          f"(out of {df['uid'].nunique()} total qualifying uids).")

    if len(connected_uids) == 0:
        print("\nNo uids share a device in this dataset (identity coverage may be too low, or "
              "train_identity.csv wasn't found by preprocess_ieee.py). Nothing to compare -- "
              "skipping the GNN demo rather than fabricating a fallback graph.")
        return

    if len(connected_uids) > MAX_CONNECTED_ACCOUNTS:
        rng = np.random.default_rng(7)
        connected_uids = list(rng.choice(connected_uids, size=MAX_CONNECTED_ACCOUNTS, replace=False))
        print(f"(capped to a random {MAX_CONNECTED_ACCOUNTS} connected uids for runtime/memory)")

    all_uids = set(df["uid"].unique())
    isolated_uids = list(all_uids - set(G_full.nodes()))
    fraud_isolated = df[df["uid"].isin(isolated_uids)].groupby("uid")["isFraud"].max()
    normal_isolated = list(fraud_isolated[fraud_isolated == 0].index)  # real, non-fraud, no device link
    rng = np.random.default_rng(7)
    if len(normal_isolated) > MAX_CONTROL_ACCOUNTS:
        normal_isolated = list(rng.choice(normal_isolated, size=MAX_CONTROL_ACCOUNTS, replace=False))
    print(f"Control group: {len(normal_isolated)} real uids with no device connection and no fraud, ever.")

    node_order = connected_uids + normal_isolated
    print("\nScoring every uid with Mamba-KAN (its own real transaction history, no graph knowledge)...")
    mk_scores, agg_features, fraud_label = mamba_kan_uid_scores(df, node_order, norm_mean, norm_std, n_features)

    G = build_real_device_graph(df[df["uid"].isin(node_order)])
    G.add_nodes_from(node_order)  # keep isolated control uids as valid (self-loop-only) nodes

    X = np.array([agg_features[u] + [mk_scores[u]] for u in node_order], dtype=np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    X_t = torch.tensor(X, dtype=torch.float32)
    A_hat = normalized_adjacency(G, node_order)

    y = np.array([fraud_label[u] for u in node_order], dtype=np.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    print(f"Real fraud rate in this node set: {y.mean():.4f}")

    gcn = SimpleGCN(in_features=X.shape[1])
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.05)
    pos_weight = torch.tensor([(1 - y.mean()) / max(y.mean(), 1e-4)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training GCN on REAL isFraud labels (device-sharing graph)...")
    for epoch in range(150):
        gcn.train()
        optimizer.zero_grad()
        logits = gcn(X_t, A_hat)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: loss={loss.item():.4f}")

    gcn.eval()
    with torch.no_grad():
        gcn_scores_arr = torch.sigmoid(gcn(X_t, A_hat)).numpy()
    gcn_scores = {u: gcn_scores_arr[i] for i, u in enumerate(node_order)}

    # "Quiet, real" cases: uids that REALLY committed fraud, are graph-connected
    # to at least one other uid, but whose own Mamba-KAN score is LOW (missed
    # on its own merits) -- the real-data analog of the synthetic "quiet ring member"
    quiet_ids = [u for u in connected_uids if fraud_label[u] == 1 and mk_scores[u] < 0.5]
    normal_ids = normal_isolated

    print(f"\nReal 'quiet' fraud cases found (fraud=1, device-connected, own Mamba-KAN score < 0.5): {len(quiet_ids)}")

    if len(quiet_ids) == 0:
        print("None found -- on this data, every device-connected fraud uid was ALREADY caught by "
              "Mamba-KAN alone (score >= 0.5). That's a genuine, useful negative result: it means "
              "graph structure isn't adding value for the specific 'quiet member' failure mode here, "
              "even though the graph and real fraud labels are both real. Report this honestly rather "
              "than searching for a different cutoff until a nonzero count appears.")
        mk_quiet_avg = gcn_quiet_avg = float("nan")
    else:
        mk_quiet_avg = float(np.mean([mk_scores[u] for u in quiet_ids]))
        gcn_quiet_avg = float(np.mean([gcn_scores[u] for u in quiet_ids]))

    mk_normal_avg = float(np.mean([mk_scores[u] for u in normal_ids])) if normal_ids else float("nan")
    gcn_normal_avg = float(np.mean([gcn_scores[u] for u in normal_ids])) if normal_ids else float("nan")

    print(f"\nAverage score on QUIET real fraud uids   -> Mamba-KAN: {mk_quiet_avg:.4f}   GCN: {gcn_quiet_avg:.4f}")
    print(f"Average score on real NORMAL control uids -> Mamba-KAN: {mk_normal_avg:.4f}   GCN: {gcn_normal_avg:.4f}")
    if quiet_ids:
        print(f"\nSeparation (quiet - normal): Mamba-KAN gap = {mk_quiet_avg - mk_normal_avg:.4f}   "
              f"GCN gap = {gcn_quiet_avg - gcn_normal_avg:.4f}")
        print("(Bigger gap = better at telling real quiet fraud uids apart from real normal accounts. "
              "This gap is whatever the real data actually shows, not a constructed demo.)")

    # ---- Plot: only the connected subgraph, colored by GCN score, shaped by real role ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_uids = connected_uids[:500] if len(connected_uids) > 500 else connected_uids  # cap for a readable layout
    G_plot = G.subgraph(plot_uids)
    pos = nx.spring_layout(G_plot, seed=7, k=0.6)

    for ax, scores, title in [
        (axes[0], mk_scores, "Mamba-KAN alone (no graph knowledge)"),
        (axes[1], gcn_scores, "GCN (uses real device-sharing graph)"),
    ]:
        fraud_nodes = [u for u in G_plot.nodes() if fraud_label[u] == 1]
        normal_nodes = [u for u in G_plot.nodes() if fraud_label[u] == 0]
        nx.draw_networkx_edges(G_plot, pos, ax=ax, alpha=0.25)
        if normal_nodes:
            nx.draw_networkx_nodes(G_plot, pos, nodelist=normal_nodes, node_shape='o',
                                    node_color=[scores[u] for u in normal_nodes],
                                    cmap="Reds", vmin=0, vmax=1, node_size=100, ax=ax, label="real normal")
        if fraud_nodes:
            nx.draw_networkx_nodes(G_plot, pos, nodelist=fraud_nodes, node_shape='^',
                                    node_color=[scores[u] for u in fraud_nodes],
                                    cmap="Reds", vmin=0, vmax=1, node_size=220, ax=ax, label="real fraud")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[1].legend(loc="upper right", fontsize=8)
    plt.suptitle("Real IEEE-CIS device-sharing graph — darker red = higher predicted risk\n"
                  "(triangles = uids with a real fraudulent transaction)", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "gnn_ring_detection.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved graph visualization to {out_path}")

    summary = {
        "data_source": "real_ieee_cis",
        "n_connected_uids": len(connected_uids),
        "n_control_uids": len(normal_isolated),
        "n_quiet_fraud_found": len(quiet_ids),
        "mamba_kan_quiet_avg": None if not quiet_ids else round(mk_quiet_avg, 4),
        "mamba_kan_normal_avg": round(mk_normal_avg, 4) if normal_ids else None,
        "gcn_quiet_avg": None if not quiet_ids else round(gcn_quiet_avg, 4),
        "gcn_normal_avg": round(gcn_normal_avg, 4) if normal_ids else None,
        "mamba_kan_separation": None if not quiet_ids else round(mk_quiet_avg - mk_normal_avg, 4),
        "gcn_separation": None if not quiet_ids else round(gcn_quiet_avg - gcn_normal_avg, 4),
    }
    with open(os.path.join(RESULTS_DIR, "gnn_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(RESULTS_DIR, 'gnn_summary.json')}")


if __name__ == "__main__":
    main()
