"""Linear / MLP board probes: 512 -> 64 squares x 13 classes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import EMPTY


class LinearProbe(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.w = nn.Linear(d, 64 * 13)

    def forward(self, x):
        return self.w(x).view(-1, 64, 13)


class MLPProbe(nn.Module):
    def __init__(self, d=512, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 64 * 13))

    def forward(self, x):
        return self.net(x).view(-1, 64, 13)


def fit(X: torch.Tensor, Y: np.ndarray, kind="linear", epochs=20, lr=1e-3, wd=0.01, bs=256,
        seed=0, device="cpu", log=None, tag=""):
    torch.manual_seed(seed)
    p = (LinearProbe if kind == "linear" else MLPProbe)(X.shape[1]).to(device)
    opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))
    Xd, Yd = X.float().to(device), torch.as_tensor(Y).to(device)
    n = len(Xd)
    g = torch.Generator(device="cpu").manual_seed(seed)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(p(Xd[idx]).reshape(-1, 13), Yd[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if log:
            log({f"probe/{tag}/train_loss": tot / n, "probe/epoch": ep})
    return p.eval()


@torch.no_grad()
def predict(p, X: torch.Tensor, device="cpu", bs=4096):
    out = []
    for i in range(0, len(X), bs):
        out.append(p(X[i:i + bs].float().to(device)).softmax(-1).cpu())
    return torch.cat(out).numpy()  # (N,64,13)


def errors(pred_probs: np.ndarray, Y: np.ndarray):
    """Per-position counts: occupied squares, occupied wrong, all-square wrong."""
    pred = pred_probs.argmax(-1)
    occ = Y != EMPTY
    wrong = pred != Y
    return dict(n_occ=occ.sum(1), occ_wrong=(wrong & occ).sum(1), all_wrong=wrong.sum(1))


def summarize(e):
    return dict(occ_err=float(e["occ_wrong"].sum() / e["n_occ"].sum()),
                all_acc=float(1 - e["all_wrong"].sum() / (64 * len(e["all_wrong"]))))
