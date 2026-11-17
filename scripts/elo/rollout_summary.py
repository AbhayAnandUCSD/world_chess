"""Compare Elo rollouts and GM rollouts on the same starts and stopping rules."""
import json
from pathlib import Path
import numpy as np
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc.model import ROOT, CODE_ROOT

D = ROOT / "results/19"
F = ROOT / "results/figs"
names = ["base", "low_ft", "high_ft", "low_s1", "high_s1"]
games = {name:{r["i"]:r for r in map(json.loads,(D/f"rollout_{name}.jsonl").open())}
         for name in names}
assert all(len(g)==600 for g in games.values())
assert all(set(g)==set(games["base"]) for g in games.values())
pairs = [("low_ft","high_ft"),("low_s1","high_s1"),("low_ft","low_s1"),
         ("high_ft","high_s1"),("low_ft","base"),("high_ft","base")]
out = {}
for a,b in pairs:
    rows = []
    for i in sorted(games[a]):
        ga,gb = games[a][i],games[b][i]
        n = min(len(ga["moves"]),len(gb["moves"]))
        split = next((t//2+1 for t in range(n) if ga["moves"][t]!=gb["moves"][t]),None)
        squares = None
        if n >= 30:
            ba = chess.Board(ga["fens"][29]+" w - - 0 1")
            bb = chess.Board(gb["fens"][29]+" w - - 0 1")
            squares = sum(ba.piece_at(s)!=bb.piece_at(s) for s in chess.SQUARES)
        rows.append(dict(i=i,set=ga["set"],split_at=split,own_reached=(n+1)//2,squares=squares))
    def survival(selected):
        rates,counts = [],[]
        for k in range(1,16):
            same=sum((r["split_at"] is None and r["own_reached"]>=k) or
                     (r["split_at"] is not None and r["split_at"]>k) for r in selected)
            eligible=sum((r["split_at"] is not None and r["split_at"]<=k) or
                         r["own_reached"]>=k for r in selected)
            rates.append(same/eligible if eligible else None)
            counts.append(eligible)
        return dict(survival=rates,eligible=counts)
    squares=[r["squares"] for r in rows if r["squares"] is not None]
    out[f"{a}|{b}"]=dict(n=len(rows),**survival(rows),
        by_set={st:survival([r for r in rows if r["set"]==st]) for st in ["neutral_otb","h2h_p2"]},
        squares_mean=float(np.mean(squares)),n_end=len(squares),
        first_move_differs=float(np.mean([r["split_at"]==1 for r in rows])))
(D/"rollouts.json").write_text(json.dumps(out,indent=2))
gm=json.loads((CODE_ROOT/"results/19/rollouts.json").read_text())["pairs"]
fig,ax=plt.subplots(figsize=(8,4.8))
for key,label in [("karpov_ft|kasparov_ft","Karpov / Kasparov"),
                  ("carlsen_ft|tang_ft","Carlsen / Tang")]:
    ax.plot(range(1,16),100*np.array(gm[key]["survival"]),label=label)
for key,label,ls in [("low_ft|high_ft","Low / high Elo","-"),
                    ("low_ft|low_s1","Low Elo, two seeds",":"),
                    ("high_ft|high_s1","High Elo, two seeds",":")]:
    ax.plot(range(1,16),100*np.array(out[key]["survival"]),ls,label=label)
ax.set(xlabel="Own moves played",ylabel="Eligible games still identical (%)",ylim=(0,100),
       title="Same 600 starts and greedy base opponent")
ax.legend()
fig.tight_layout()
fig.savefig(F/"rollouts_comparison.png",dpi=170)
print(json.dumps(out,indent=2))
