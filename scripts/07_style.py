"""Descriptive (engine-free) style stats + where the two fine-tunes disagree most."""
import collections
import json

import numpy as np
import pandas as pd

from wc.data import load_games
from wc import config
from wc.model import ROOT

PL = config.load()["players"]; A, B = PL

out = {}
for p in PL:
    gs = [g for g in load_games(player=p)]
    own = [s for g in gs for i, s in enumerate(g.sans) if (i % 2 == 0) == (g.color == "w")]
    late = [s for g in gs for i, s in enumerate(g.sans) if (i % 2 == 0) == (g.color == "w") and i >= 30]
    kind = collections.Counter("O" if s.startswith("O") else (s[0] if s[0] in "NBRQK" else "P") for s in late)
    tot = sum(kind.values())
    out[p] = dict(
        games=len(gs), white_share=float(np.mean([g.color == "w" for g in gs])),
        median_plies=float(np.median([len(g.sans) for g in gs])),
        eco_family={k: v / len(gs) for k, v in sorted(collections.Counter(g.eco[:1] for g in gs if g.eco).items())},
        top_eco_white=collections.Counter(g.eco for g in gs if g.color == "w").most_common(6),
        top_eco_black=collections.Counter(g.eco for g in gs if g.color == "b").most_common(6),
        first_move_white={k: v for k, v in collections.Counter(g.sans[0] for g in gs if g.color == "w").most_common(4)},
        capture_rate=float(np.mean(["x" in s for s in own])), check_rate=float(np.mean(["+" in s for s in own])),
        late_capture_rate=float(np.mean(["x" in s for s in late])), late_check_rate=float(np.mean(["+" in s for s in late])),
        late_piece_mix={k: v / tot for k, v in sorted(kind.items())},
    )
# where fine-tunes disagree (needs 04 output)
csvs = sorted((ROOT / "results").glob("04_moves_test_s*.csv"))
if csvs:
    df = pd.read_csv(csvs[-1])
    piv = df.pivot_table(index=["player", "gid", "ply"], columns="model", values="logp").reset_index()
    piv["kp_minus_ks"] = piv[f"{A}_ft"] - piv[f"{B}_ft"]   # A-copy minus B-copy
    games = {g.gid: g for g in load_games("test")}
    def row(r):
        g = games[r.gid]; return dict(player=r.player, gid=r.gid, ply=int(r.ply), move=g.sans[int(r.ply) - 1],
                                      context=g.text[:g.spans[int(r.ply) - 1][0]][-60:], kp_minus_ks=float(r.kp_minus_ks))
    out[f"disagree_top_{A}ish"] = [row(r) for r in piv.nlargest(10, "kp_minus_ks").itertuples()]
    out[f"disagree_top_{B}ish"] = [row(r) for r in piv.nsmallest(10, "kp_minus_ks").itertuples()]
    # does the specialist gap correlate with move type?
    piv["mv"] = [games[g].sans[int(p) - 1] for g, p in zip(piv.gid, piv.ply)]
    piv["capture"] = piv.mv.str.contains("x"); piv["check"] = piv.mv.str.contains(r"\+")
    out["gap_by_movetype"] = {f"{c}={v}": float(piv[piv[c] == v].kp_minus_ks.mean()) for c in ("capture", "check") for v in (True, False)}
json.dump(out, open(ROOT / "results" / "07_style.json", "w"), indent=1, default=str)
print(json.dumps({p: {k: v for k, v in out[p].items() if "eco" not in k} for p in PL}, indent=1))
