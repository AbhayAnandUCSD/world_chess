"""Style analysis, step 3 (CPU): Stockfish evaluation of every legal move, and how sharp the resulting position is.

For each position: MultiPV = all legal moves at depth D -> eval of each move (mover's view, centipawns,
mate clipped to +-1000). For each legal move, the resulting position is analysed with MultiPV 2 at depth
D2 -> gap between the opponent's best and second-best reply ("only-move" pressure on the opponent).

Usage: python scripts/18c_style_engine.py SHARD NSHARD STOCKFISH_PATH
Writes results/18/engine_<SHARD>.jsonl
"""
import json
import sys

import chess
import chess.engine

from wc.model import CODE_ROOT

shard, nshard, sf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
D, D2, CLIP = 12, 8, 1000
OUT = CODE_ROOT / "results/18"
rows = [json.loads(l) for l in open(OUT / "positions.jsonl")][shard::nshard]
eng = chess.engine.SimpleEngine.popen_uci(sf)
eng.configure({"Threads": 1, "Hash": 32})


def cp(score, pov):
    return max(-CLIP, min(CLIP, score.pov(pov).score(mate_score=10 * CLIP)))


with open(OUT / f"engine_{shard}.jsonl", "w") as f:
    for r in rows:
        b = chess.Board(r["fen"]); me = b.turn
        infos = eng.analyse(b, chess.engine.Limit(depth=D), multipv=len(r["legal"]))
        ev = {b.san(i["pv"][0]): cp(i["score"], me) for i in infos if "pv" in i}
        evals, gaps = [], []
        for s in r["legal"]:
            b2 = b.copy(); b2.push_san(s)
            if b2.is_game_over():
                v = CLIP if b2.is_checkmate() else 0
                evals.append(ev.get(s, v)); gaps.append(0); continue
            inf2 = eng.analyse(b2, chess.engine.Limit(depth=D2), multipv=2)
            opp = [cp(i["score"], not me) for i in inf2 if "score" in i]
            evals.append(ev.get(s, -opp[0]))
            gaps.append(opp[0] - opp[1] if len(opp) > 1 else CLIP)
        f.write(json.dumps({"i": r["i"], "eval": evals, "opp_gap": gaps}) + "\n"); f.flush()
eng.quit()
