"""Batch activation extraction at White-to-move dots, all residual layers."""
from __future__ import annotations

import numpy as np
import torch

from .labels import white_dots
from .model import encode


@torch.no_grad()
def collect(model, games, min_move=2, layers=None, batch=16, dtype=torch.float16):
    """Returns dict: acts (L,N,512) fp16 tensor on CPU, labels (N,64), meta arrays
    gid_idx, move_no, ply. layers: list of resid indices (0=embed, k=after block k-1)."""
    dev = next(model.parameters()).device
    layers = layers if layers is not None else list(range(model.config.n_layer + 1))
    A, Y, G, MV, PL = [], [], [], [], []
    for bi in range(0, len(games), batch):
        chunk = games[bi:bi + batch]
        items = [(gi + bi, list(white_dots(g.text, g.sans, min_move))) for gi, g in enumerate(chunk)]
        T = min(1023, max(len(g.text) for g in chunk))
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, g in enumerate(chunk):
            e = encode(g.text[:T]); ids[j, :len(e)] = torch.tensor(e)
        _, resid = model(ids.to(dev), return_resid=True)
        R = torch.stack([resid[l] for l in layers], 0)  # (L,B,T,C)
        for j, (gidx, dots) in enumerate(items):
            if not dots:
                continue
            pos = torch.tensor([d[0] for d in dots], device=dev)
            A.append(R[:, j, pos].to(dtype).cpu())
            Y.append(np.stack([d[3] for d in dots]))
            G += [gidx] * len(dots); MV += [d[1] for d in dots]; PL += [d[2] for d in dots]
    return dict(acts=torch.cat(A, 1), labels=np.concatenate(Y), gid_idx=np.array(G),
                move_no=np.array(MV), ply=np.array(PL), layers=layers)
