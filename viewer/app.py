from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chess
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse

from wc import config
from wc.data import load_games
from wc.labels import board_to_labels, white_dots
from wc.model import ROOT, encode, load_model, pick_device
from wc.probe import LinearProbe

C = config.load(); DEV = pick_device(); L = C["probe"]["layer_resid"]
app = FastAPI()
HERE = Path(__file__).parent


def model_paths():
    out = {"base": C["base_ckpt"]}
    for p in C["players"]:
        d = ROOT / "ckpts" / f"ft_{p}"
        cks = sorted(d.glob("step*.pt"), key=lambda x: int(x.stem[4:])) if d.exists() else []
        if cks:
            out[f"{p}_ft"] = str(cks[-1])
    return out


@lru_cache(maxsize=4)
def get_model(ck):
    return load_model(ck, DEV)


@lru_cache(maxsize=1)
def get_probe():
    p = ROOT / "ckpts" / "probe_base_frozen.pt"
    if not p.exists():
        return None
    pr = LinearProbe(); pr.load_state_dict(torch.load(p, map_location="cpu")); return pr.to(DEV).eval()


@lru_cache(maxsize=8)
def games_by_split(split):
    return {g.gid: g for g in load_games(split)}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/games")
def games(split: str = "test", player: str = ""):
    gs = [g for g in games_by_split(split).values() if not player or g.player == player]
    return [dict(gid=g.gid, white=g.white, black=g.black, n=len(g.sans), eco=g.eco, date=g.date, player=g.player)
            for g in gs[:400]]


@torch.no_grad()
def legal_move_scores(m, prefix: str, board: chess.Board, k=5):
    """Exact log P(SAN + ' ') for every legal move after prefix (ending in '.')."""
    cands = [board.san(mv) for mv in board.legal_moves]
    seqs = [encode(prefix + s + " ") for s in cands]
    T = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
    lp = m(x.to(DEV)).float().log_softmax(-1).cpu()
    n0 = len(prefix)
    scores = [sum(lp[i, t - 1, s[t]].item() for t in range(n0, len(s))) for i, s in enumerate(seqs)]
    ranked = sorted(zip(cands, scores), key=lambda z: -z[1])
    return ranked[:k]


@app.get("/api/position")
def position(gid: str, move: int, split: str = "test"):
    g = games_by_split(split)[gid]
    dots = {mv: (dot, ply, lab) for dot, mv, ply, lab in white_dots(g.text, g.sans, 2)}
    move = max(2, min(move, max(dots)))
    dot, ply, lab = dots[move]
    board = chess.Board()
    for s in g.sans[:ply]:
        board.push_san(s)
    prefix = g.text[:dot + 1]
    probe = get_probe()
    res = {"prefix": prefix[-160:], "true": lab.tolist(), "played": g.sans[ply] if ply < len(g.sans) else None,
           "move": move, "max_move": max(dots), "fen": board.fen(), "models": {}}
    for name, ck in model_paths().items():
        m = get_model(ck)
        x = torch.tensor([encode(prefix)], device=DEV)
        _, resid = m(x, return_resid=True)
        entry = {"top": legal_move_scores(m, prefix, board)}
        if probe is not None:
            pr = probe(resid[L][:, -1]).softmax(-1)[0].cpu()
            entry["probe_pred"] = pr.argmax(-1).tolist()
            entry["probe_conf"] = pr.max(-1).values.tolist()
        res["models"][name] = entry
    return res


@app.get("/api/runs")
def runs():
    out = {}
    for f in (ROOT / "runs").glob("finetune_*.jsonl"):
        out[f.stem] = [json.loads(l) for l in open(f) if l.strip()]
    return out


@app.get("/api/results")
def results():
    return {f.stem: json.load(open(f)) for f in (ROOT / "results").glob("*.json")}
