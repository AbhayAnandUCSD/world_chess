"""Move-level log-probs and per-char target masks.

A move's score = sum of log P(char) over its SAN chars AND the terminating
space (e.g. 'e5 '). Game-ending (last) moves are excluded everywhere so the
delimiter rule is uniform. Causal masking means one forward pass over the
whole game equals scoring each move with its exact history prefix.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import encode

CTX = 1023


def player_plies(g, player: str, min_ply: int = 1):
    """0-based ply indices of `player`'s non-final moves with 1-based ply >= min_ply
    whose target (incl. space) fits in context."""
    colors = (g.meta or {}).get("colors")  # explicit {player: 'w'/'b'} (Lichess data, h2h)
    if colors and player in colors:
        par = 0 if colors[player] == "w" else 1
    elif g.player == player:
        par = 0 if g.color == "w" else 1
    elif g.player == "both":  # head-to-head (PGN Mentor): figure color from names
        par = 0 if player in g.white.lower() else 1
    else:
        return []
    skip = set((g.meta or {}).get("skip_plies", ()))  # e.g. time-scramble moves (clock filter)
    out = []
    for p in range(par, len(g.sans) - 1, 2):
        s, e = g.spans[p]
        if p + 1 >= min_ply and e <= CTX and p not in skip:  # space at index e must be a target <= CTX
            out.append(p)
    return out


def target_mask(g, player: str, T: int) -> np.ndarray:
    """mask over target positions t (predicting text[t+1]) for input length T."""
    m = np.zeros(T, dtype=bool)
    for p in player_plies(g, player, 1):
        s, e = g.spans[p]
        for c in range(s, e + 1):          # chars s..e-1 plus space at e
            if 1 <= c <= T:
                m[c - 1] = True
    return m


@torch.no_grad()
def move_logprobs(model, games, player: str, min_ply: int = 31, batch: int = 16):
    """Returns list of dicts (gid, ply, color, logp) for every eligible move."""
    dev = next(model.parameters()).device
    rows = []
    for bi in range(0, len(games), batch):
        chunk = games[bi:bi + batch]
        T = min(CTX, max(len(g.text) for g in chunk) - 1)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, g in enumerate(chunk):
            e = encode(g.text[:T]); ids[j, :len(e)] = torch.tensor(e)
        lp = model(ids.to(dev)).float().log_softmax(-1).cpu()
        for j, g in enumerate(chunk):
            full = encode(g.text[:T + 1])
            for p in player_plies(g, player, min_ply):
                s, e = g.spans[p]
                tot = sum(lp[j, c - 1, full[c]].item() for c in range(s, e + 1))
                rows.append(dict(gid=g.gid, ply=p + 1, color="w" if p % 2 == 0 else "b",
                                 eco=g.eco, logp=tot, split=g.split))
    return rows
