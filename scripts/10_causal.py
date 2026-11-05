"""Causal check: does editing the layer-6 residual along a probe direction change moves
the way the probe says it should?

For White-to-move test positions (move >= 16) pick a White N/B/R/Q that has legal moves.
Edit resid[6] at the '.' position so the probe reads that square as EMPTY (logit margin M),
then score every legal move (exact log P(SAN + ' ')). Metric: change in total probability
mass on the removed piece's moves. Control: random direction with the same norm.
Run with the frozen base probe on every model, and with each copy's refit probe if given.

Usage: python scripts/10_causal.py OUT_NAME CKPT [CKPT ...]
"""
import json
import sys

import chess
import numpy as np
import torch

from wc import config
from wc.data import load_games
from wc.labels import EMPTY, white_dots
from wc.model import ROOT, encode, load_model, pick_device
from wc.probe import LinearProbe

C = config.load(); dev = pick_device(); L = C["probe"]["layer_resid"]
import os
MARGINS, N_POS = (3.0, 10.0, 25.0), int(os.environ.get("N_POS", 400))
out_name, ckpts = sys.argv[1], sys.argv[2:]
probe = LinearProbe(); probe.load_state_dict(torch.load(ROOT / "ckpts" / "probe_base_frozen.pt", map_location="cpu"))
W = probe.w.weight.detach().view(64, 13, 512); Bb = probe.w.bias.detach().view(64, 13)

# build fixed position set
rng = np.random.default_rng(0)
games = [g for g in load_games("test") if g.player in C["players"]]
cands = []
for g in games:
    for dot, mv, ply, lab in white_dots(g.text, g.sans, 16):
        cands.append((g, dot, ply))
rng.shuffle(cands)
positions = []
for g, dot, ply in cands:
    b = chess.Board()
    for s in g.sans[:ply]:
        b.push_san(s)
    pcs = [(sq, pc) for sq, pc in b.piece_map().items() if pc.color == chess.WHITE and pc.piece_type in (2, 3, 4, 5)]
    pcs = [(sq, pc) for sq, pc in pcs if any(m.from_square == sq for m in b.legal_moves)]
    if not pcs:
        continue
    sq, pc = pcs[rng.integers(len(pcs))]
    legal = list(b.legal_moves)
    positions.append(dict(prefix=g.text[:dot + 1], sq=sq, cls=EMPTY + pc.piece_type,
                          sans=[b.san(m) for m in legal], is_piece=[m.from_square == sq for m in legal]))
    if len(positions) >= N_POS:
        break


@torch.no_grad()
def move_probs(m, pos, vec=None):
    seqs = [encode(pos["prefix"] + s + " ") for s in pos["sans"]]
    T = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
    x = x.to(dev); n0 = len(pos["prefix"])
    edit = None
    if vec is not None:
        edit = (L, torch.full((len(seqs),), n0 - 1, device=dev), vec.to(dev).expand(len(seqs), -1))
    lp = m(x, edit=edit).float().log_softmax(-1).cpu()
    sc = torch.tensor([sum(lp[i, t - 1, s[t]].item() for t in range(n0, len(s))) for i, s in enumerate(seqs)])
    return sc.softmax(0).numpy()  # renormalised over legal moves


@torch.no_grad()
def resid_at_dot(m, prefix):
    _, res = m(torch.tensor([encode(prefix)], device=dev), return_resid=True)
    return res[L][0, -1].float().cpu()


out = []
for ck in ckpts:
    m = load_model(ck, dev).eval()
    for MARGIN in MARGINS:
        rows = []
        for pos in positions:
            h = resid_at_dot(m, pos["prefix"])
            d = W[pos["sq"], EMPTY] - W[pos["sq"], pos["cls"]]
            gap = float(d @ h + Bb[pos["sq"], EMPTY] - Bb[pos["sq"], pos["cls"]])
            alpha = (MARGIN - gap) / float(d.norm() ** 2)
            vec = alpha * d if alpha > 0 else torch.zeros_like(d)
            rnd = torch.randn(512, generator=torch.Generator().manual_seed(len(rows))); rnd = rnd / rnd.norm() * vec.norm()
            mask = np.array(pos["is_piece"])
            p0, p1, pr = move_probs(m, pos), move_probs(m, pos, vec), move_probs(m, pos, rnd)
            rows.append(dict(mass0=float(p0[mask].sum()), mass_edit=float(p1[mask].sum()), mass_rand=float(pr[mask].sum()),
                             norm=float(vec.norm()), resid_norm=float(h.norm()),
                             top_piece0=bool(mask[p0.argmax()]), top_piece_edit=bool(mask[p1.argmax()])))
        M0 = np.array([r["mass0"] for r in rows]); ME = np.array([r["mass_edit"] for r in rows]); MR = np.array([r["mass_rand"] for r in rows])
        boot = np.random.default_rng(0).integers(0, len(rows), (1000, len(rows)))
        de = 1 - ME[boot].sum(1) / M0[boot].sum(1); dr = 1 - MR[boot].sum(1) / M0[boot].sum(1)
        res = dict(ckpt=ck, margin=MARGIN, n=len(rows), mass0=float(M0.mean()),
                   frac_drop_edit=float(1 - ME.sum() / M0.sum()), ci_edit=[float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
                   frac_drop_rand=float(1 - MR.sum() / M0.sum()), ci_rand=[float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
                   top1_piece_before=float(np.mean([r["top_piece0"] for r in rows])),
                   top1_piece_after_edit=float(np.mean([r["top_piece_edit"] for r in rows])),
                   edit_norm_over_resid=float(np.mean([r["norm"] / r["resid_norm"] for r in rows])))
        print(json.dumps(res), flush=True); out.append(res)
        json.dump(out, open(ROOT / "results" / f"10_causal_{out_name}.json", "w"), indent=1)
