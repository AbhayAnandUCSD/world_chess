"""Style detectability: can the two copies tell who played a game, from that player's moves alone?

Per test game: s = sum over the player's scored moves (plies >= 31) of log P_copyA(move) - log P_copyB(move).
Predict A if s > 0. Report AUC and accuracy using the first k moves (k = 1, 2, 5, 10, 20, all),
for P1 (Karpov vs Kasparov) and P2 (Carlsen vs Tang). Uses 04_moves_test_*.csv.
"""
import json
import numpy as np
import pandas as pd
from wc.model import CODE_ROOT

SETS = {"p1": (CODE_ROOT / "results/04_moves_test_s60.csv", "karpov", "kasparov"),
        "p2": (CODE_ROOT / "exp_p2/results/04_moves_test_s720.csv", "carlsen", "tang")}
KS = [1, 2, 5, 10, 20, 40, None]


def auc(pos, neg):
    s = np.r_[pos, neg]; r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


out = {}
for name, (path, A, B) in SETS.items():
    df = pd.read_csv(path)
    print(name, "models:", sorted(df.model.unique()))
    mA = next(m for m in df.model.unique() if m.startswith(A)); mB = next(m for m in df.model.unique() if m.startswith(B))
    w = df[df.model.isin([mA, mB])].pivot_table(index=["player", "gid", "ply"], columns="model", values="logp").dropna()
    w["d"] = w[mA] - w[mB]
    w = w.reset_index().sort_values(["player", "gid", "ply"])
    res = {"n_games": {p: int(w[w.player == p].gid.nunique()) for p in (A, B)},
           "per_move_mean_d": {p: float(w[w.player == p].d.mean()) for p in (A, B)},
           "per_move_acc": {p: float(((w[w.player == p].d > 0) == (p == A)).mean()) for p in (A, B)}, "by_k": {}}
    for k in KS:
        g = (w.groupby(["player", "gid"]).head(k) if k else w).groupby(["player", "gid"]).d.sum().reset_index()
        g = g[g.groupby(["player", "gid"]).d.transform("size") > 0]
        sA, sB = g[g.player == A].d.values, g[g.player == B].d.values
        acc = (np.r_[(sA > 0), (sB < 0)]).mean()
        res["by_k"][str(k or "all")] = {"auc": auc(sA, sB), "acc": float(acc)}
    out[name] = res
    print(name, json.dumps(res, indent=None))
json.dump(out, open(CODE_ROOT / "results/17_identify.json", "w"), indent=1)
