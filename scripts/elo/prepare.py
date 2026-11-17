"""Reuse the exact GM neutral boards, then sample 1,200 held-out boards per Elo band."""
import json
import random
from pathlib import Path
import chess
import numpy as np
from wc.data import load_games
from wc.model import CODE_ROOT, ROOT
from wc.score import CTX, player_plies

out = ROOT / "results/18"
out.mkdir(parents=True, exist_ok=True)
original = [json.loads(line) for line in (CODE_ROOT / "results/18/positions.jsonl").open()]
rows = [r for r in original if r["set"] in ("neutral_otb", "h2h_p2")]
assert [r["i"] for r in rows] == list(range(2500))
rng = random.Random(0)
for band in ("low", "high"):
    games = load_games("test", band)
    candidates = [(g, p) for g in games for p in player_plies(g, band, 31)]
    rng.shuffle(candidates)
    count = 0
    for g, p in candidates:
        board = chess.Board()
        for san in g.sans[:p]:
            board.push_san(san)
        legal = sorted(board.san(move) for move in board.legal_moves)
        prefix = g.text[:g.spans[p][0]]
        if len(prefix) + max(map(len, legal)) + 1 > CTX:
            continue
        rows.append(dict(i=len(rows), set=f"own_{band}", player=band, gid=g.gid,
                         ply=p + 1, color="w" if board.turn else "b",
                         fen=board.fen(), prefix=prefix, legal=legal, played=g.sans[p]))
        count += 1
        if count == 1200:
            break
    assert count == 1200
path = out / "positions.jsonl"
if path.exists():
    assert [json.loads(line) for line in path.open()] == rows
else:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
# Existing engine evaluations are identical for the shared boards.
engine = {}
if (out / "engine_cached.jsonl").exists():
    engine.update({r["i"]: r for r in map(json.loads, (out / "engine_cached.jsonl").open())})
for path in (CODE_ROOT / "results/18").glob("engine_*.jsonl"):
    for row in map(json.loads, path.open()):
        if row["i"] < 2500:
            engine[row["i"]] = row
assert len(engine) == 2500
(out / "engine_cached.jsonl").write_text("".join(json.dumps(engine[i]) + "\n" for i in sorted(engine)))
print("Prepared", len(rows), "boards, with 2500 cached engine evaluations.")
