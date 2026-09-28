"""
snn_model.py
-------------
Spiking Neural Network for fraud detection using the `snntorch` library
(a real, published SNN library -- https://snntorch.readthedocs.io -- not
reimplemented from scratch, since LIF neuron dynamics are standard and a
correct, well-tested implementation is more defensible for a project report
than a hand-rolled one).

CORE IDEA (this is the "temporal encoding" your abstract refers to):
Instead of feeding raw feature vectors straight into a normal MLP, we
encode each transaction's features as a SPIKE TRAIN over T simulated
timesteps. Two encoding schemes are implemented so the project can run
an ablation on which one matters more (matches your abstract's claim that
"temporal encoding choices had a measurable effect on SNN precision"):

  1. RATE CODING       - a feature's magnitude controls the PROBABILITY of
                          a spike at each timestep (classic Poisson/rate coding).
  2. LATENCY CODING     - a feature's magnitude controls WHEN the (single)
                          spike fires: large/anomalous values fire early,
                          normal values fire late or not at all. This is the
                          encoding that should be most sensitive to the
                          `inter_tx_seconds` feature, since fast bursts
                          (velocity fraud) directly compress spike timing.

Both encodings operate on the already-engineered per-transaction feature
vector (the last transaction in each window), so the SNN, unlike the
Mamba-KAN, focuses on fine-grained TIMING of a single decision point rather
than the full sequence -- this asymmetry is deliberate and is what your
"where each architecture has a practical edge" discussion should compare.

REGULARIZATION NOTE: dropout was added on the second hidden layer's spike
output after the same overfitting signature (climbing val_loss) showed up
across all three models when trained on the full real IEEE-CIS dataset.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

N_STEPS = 25  # number of simulated spiking timesteps


def rate_encode(x, n_steps=N_STEPS):
    """x: (batch, features) in roughly [-3,3] after normalization.
    Squash to [0,1] spike probability and sample a Bernoulli spike train."""
    p = torch.sigmoid(x)  # (batch, features)
    p = p.unsqueeze(0).repeat(n_steps, 1, 1)  # (T, batch, features)
    spikes = torch.bernoulli(p)
    return spikes


def latency_encode(x, n_steps=N_STEPS):
    """Larger (more anomalous) values fire EARLIER. Implemented via the
    classic inverse-time latency trick: latency = (n_steps-1) * (1 - sigmoid(x)).
    Produces one spike per feature per sample, at the computed timestep."""
    p = torch.sigmoid(x)  # (batch, features), higher => more urgent => earlier spike
    latency = ((1 - p) * (n_steps - 1)).long()  # (batch, features)
    spikes = torch.zeros(n_steps, x.shape[0], x.shape[1], device=x.device)
    b_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(latency)
    f_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand_as(latency)
    spikes[latency, b_idx, f_idx] = 1.0
    return spikes


class SNNFraudDetector(nn.Module):
    def __init__(self, n_features, hidden=32, n_steps=N_STEPS, encoding="rate"):
        super().__init__()
        assert encoding in ("rate", "latency")
        self.encoding = encoding
        self.n_steps = n_steps

        beta = 0.9  # membrane decay rate (leak) for LIF neurons
        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(n_features, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(hidden // 2, 1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout = nn.Dropout(0.2)  # regularization -- see module docstring

    def forward(self, x):
        # x: (batch, features) -- the LAST transaction's engineered features
        if self.encoding == "rate":
            spk_in = rate_encode(x, self.n_steps)
        else:
            spk_in = latency_encode(x, self.n_steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        mem3_rec = []
        for t in range(self.n_steps):
            cur1 = self.fc1(spk_in[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout(spk2)  # regularization -- see module docstring
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            mem3_rec.append(mem3)

        # use the membrane potential trace of the output neuron, averaged
        # over time, as the fraud logit (a common SNN "readout" strategy)
        out = torch.stack(mem3_rec, dim=0).mean(dim=0).squeeze(-1)
        return out


if __name__ == "__main__":
    model_rate = SNNFraudDetector(n_features=8, encoding="rate")
    model_latency = SNNFraudDetector(n_features=8, encoding="latency")
    x = torch.randn(4, 8)
    print("Rate-coded output:", model_rate(x).shape)
    print("Latency-coded output:", model_latency(x).shape)