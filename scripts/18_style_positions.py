"""Style analysis, step 1: sample positions and list their legal moves.

Sets (all plies >= 31, prefix + longest legal move must fit the 1023-char context):
  neutral_otb  : 1500 positions from OTB champion games with none of the four players (P2 probe_fit_other)
  h2h_p2       : 1000 positions from Carlsen-Tang games (Lichess positions both reach)
  own_<player> : 1200 positions from each player's test games where that player is to move
Writes results/18/positions.jsonl: one row per position with prefix text, fen, legal SANs, played SAN.
"""
import json
import random

import chess

from wc.data import load_games
from wc.model import CODE_ROOT
from wc.score import CTX, player_plies

OUT = CODE_ROOT / "results/18"; OUT.mkdir(parents=True, exist_ok=True)
P1D, P2D = CODE_ROOT / "data/games.jsonl", CODE_ROOT / "exp_p2/data/games.jsonl"
rng = random.Random(0)


def pos_row(g, p, set_name, player=None):
    b = chess.Board()
    for s in g.sans[:p]:
        b.push_san(s)
    legal = sorted(b.san(m) for m in b.legal_moves)
    s0, _ = g.spans[p]
    prefix = g.text[:s0]
    if s0 + max(len(x) for x in legal) + 1 > CTX:
        return None
    return dict(set=set_name, player=player, gid=g.gid, ply=p + 1, color="w" if p % 2 == 0 else "b",
                fen=b.fen(), prefix=prefix, legal=legal, played=g.sans[p])


def sample(cands, n, set_name):
    rng.shuffle(cands); rows = []
    for g, p, pl in cands:
        r = pos_row(g, p, set_name, pl)
        if r: rows.append(r)
        if len(rows) == n: break
    return rows


rows = []
other = load_games("probe_fit_other", path=P2D)
rows += sample([(g, p, None) for g in other for p in range(30, min(len(g.sans) - 1, 100))], 1500, "neutral_otb")
h2h = load_games("h2h", path=P2D)
rows += sample([(g, p, None) for g in h2h for p in range(30, min(len(g.sans) - 1, 100))], 1000, "h2h_p2")
for path, pls in ((P1D, ["karpov", "kasparov"]), (P2D, ["carlsen", "tang"])):
    test = load_games("test", path=path)
    for pl in pls:
        rows += sample([(g, p, pl) for g in test if g.player == pl for p in player_plies(g, pl, 31)], 1200, f"own_{pl}")
with open(OUT / "positions.jsonl", "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"i": i, **r}) + "\n")
from collections import Counter
print(Counter(r["set"] for r in rows), "legal moves total", sum(len(r["legal"]) for r in rows))
