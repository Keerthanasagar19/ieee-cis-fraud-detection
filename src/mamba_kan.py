"""
mamba_kan.py
-------------
A simplified, from-scratch implementation of:
  1. A selective state-space (Mamba-style) sequence block
  2. A Kolmogorov-Arnold Network (KAN) layer for interpretable feature weighting

WHY "simplified"? The official `mamba-ssm` package depends on custom CUDA
kernels (causal-conv1d, selective-scan-cuda) that require a GPU + matching
CUDA toolkit to compile. That is not something you can guarantee on a
college lab machine or a CPU-only evaluation setup. So this file implements
the SAME mathematical idea -- a selective state-space recurrence with
input-dependent (data-dependent) gating -- in plain PyTorch, which runs
anywhere. In your report, say exactly this: "we reimplemented the selective
SSM recurrence in pure PyTorch rather than depending on mamba-ssm's fused
CUDA kernels, trading some training speed for portability and for the
ability to inspect intermediate states during ablation."

KAN layer: instead of `y = activation(Wx + b)` with one FIXED activation
function shared by the whole layer (ReLU/Tanh/etc, as in an MLP), a KAN
layer learns a SEPARATE small univariate function (approximated here by a
short basis expansion of learnable weights) on every edge, then sums them.
That is what "learnable activation on the edge, not the node" means, and it
is what gives KAN layers their interpretability: you can plot the learned
function on any single edge and see exactly how that one feature influences
the output.

REGULARIZATION NOTE: dropout was added to the SSM block output and to the
KAN hidden representation after observing validation loss climb every
epoch during training on the full real IEEE-CIS dataset (train_loss fell
from 1.09 to 0.48 while val_loss rose from 1.39 to 3.62) -- a clear
overfitting signature with zero regularization anywhere in the original
architecture. This is a small, targeted fix, not an architecture change.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Selective State-Space block (Mamba-style, simplified, pure PyTorch)
# ----------------------------------------------------------------------
class SelectiveSSMBlock(nn.Module):
    """
    Implements a simplified selective state-space recurrence:

        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        y_t = C_t * h_t

    where A_bar_t, B_bar_t are DATA-DEPENDENT (computed from x_t itself),
    which is exactly what makes Mamba "selective" -- unlike a plain linear
    RNN/SSM where A and B are fixed for every timestep, here the model can
    choose to "remember" or "forget" based on the content of each
    transaction. This is the mechanism that should let the model down-weight
    routine transactions and up-weight anomalous ones in the hidden state.
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent projections that produce per-timestep A, B, C, and a gate
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)
        # Base (learnable) decay rates, one per state dimension, per channel
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.9 + 0.1))
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection (like Mamba's D)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.2)  # regularization -- see module docstring

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        b, l, d = x.shape
        x = self.act(self.in_proj(x))

        proj = self.x_proj(x)  # (b, l, 2*d_state + d_model)
        B, C, gate = torch.split(proj, [self.d_state, self.d_state, self.d_model], dim=-1)
        gate = torch.sigmoid(gate)  # data-dependent "how much to update" gate

        A = -torch.exp(self.A_log)  # (d_model, d_state), negative => stable decay

        h = torch.zeros(b, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            # discretize with a per-timestep step size derived from the gate
            dt = gate[:, t, :].mean(dim=-1, keepdim=True).unsqueeze(-1)  # (b,1,1)
            A_bar = torch.exp(A.unsqueeze(0) * dt)                       # (b, d_model, d_state)
            Bx = torch.einsum("bd,bs->bds", x[:, t, :], B[:, t, :])      # (b, d_model, d_state)
            h = A_bar * h + Bx
            y_t = torch.einsum("bds,bs->bd", h, C[:, t, :])              # (b, d_model)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)          # (b, l, d_model)
        y = y + x * self.D                  # skip / residual term
        y = self.out_proj(y)
        y = self.dropout(y)                 # regularization -- see module docstring
        return y


# ----------------------------------------------------------------------
# 2. KAN layer (spline-free, basis-expansion approximation, lightweight)
# ----------------------------------------------------------------------
class KANLayer(nn.Module):
    """
    Approximates a Kolmogorov-Arnold layer using a fixed set of basis
    functions (here: a small fixed bank of sinusoids + a linear + a SiLU
    term) with LEARNABLE per-edge coefficients. This keeps things fast and
    dependency-free (no external `pykan` package, no B-spline library)
    while preserving the core KAN idea: each (input, output) edge gets its
    own learned univariate function rather than sharing one activation
    across the whole layer.

    interpretability hook: `self.last_edge_weights` stores, per forward
    pass, how much each input feature contributed to each output unit --
    use this in your report's "feature importance" plots.
    """

    N_BASIS = 6  # number of basis functions per edge

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # one coefficient vector (length N_BASIS) per (in, out) edge
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, self.N_BASIS) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.last_edge_weights = None  # populated on forward for interpretability

    def _basis(self, x):
        # x: (..., in_features) -> (..., in_features, N_BASIS)
        terms = [
            x,
            torch.sin(x),
            torch.sin(2 * x),
            torch.cos(x),
            F.silu(x),
            x ** 2,
        ]
        return torch.stack(terms, dim=-1)

    def forward(self, x):
        # x: (batch, in_features)
        basis = self._basis(x)  # (batch, in_features, N_BASIS)
        # edge_out[b, i, o] = sum_k basis[b,i,k] * coeffs[i,o,k]
        edge_out = torch.einsum("bik,iok->bio", basis, self.coeffs)
        self.last_edge_weights = edge_out.detach().abs().mean(dim=0)  # (in, out) importance map
        out = edge_out.sum(dim=1) + self.bias
        return out


# ----------------------------------------------------------------------
# 3. Full Mamba-KAN model for fraud sequence classification
# ----------------------------------------------------------------------
class MambaKAN(nn.Module):
    def __init__(self, n_features, d_model=32, d_state=16, n_ssm_layers=2, kan_hidden=16):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.ssm_layers = nn.ModuleList([SelectiveSSMBlock(d_model, d_state) for _ in range(n_ssm_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])

        self.kan1 = KANLayer(d_model, kan_hidden)
        self.kan2 = KANLayer(kan_hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        h = self.input_proj(x)
        for ssm, norm in zip(self.ssm_layers, self.norms):
            h = norm(h + ssm(h))  # residual + pre-ish norm

        pooled = h[:, -1, :]  # representation of the most recent transaction, with full sequence context
        z = F.silu(self.kan1(pooled))
        z = F.dropout(z, p=0.2, training=self.training)  # regularization -- see module docstring
        logit = self.kan2(z).squeeze(-1)
        return logit  # raw logit; apply sigmoid outside for probability

    def get_feature_importance(self):
        """Returns the KAN's first-layer edge importance map (interpretability)."""
        if self.kan1.last_edge_weights is None:
            return None
        return self.kan1.last_edge_weights.cpu().numpy()


if __name__ == "__main__":
    # quick smoke test
    model = MambaKAN(n_features=8)
    x = torch.randn(4, 10, 8)
    out = model(x)
    print("Output shape:", out.shape)  # expect (4,)
    print("Feature importance shape:", model.get_feature_importance().shape)