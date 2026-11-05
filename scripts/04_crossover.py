"""Crossover table: mean log P(played move), plies >= 31, 3 models x 2 players.
Usage: python scripts/04_crossover.py [split] [ckpt_step]   (default: test, final)"""
import json
import sys

import numpy as np
import pandas as pd

from wc import config, logging as wlog
from wc.data import load_games
from wc.model import ROOT, load_model
from wc.score import move_logprobs
from wc.stats import mean_ci

C = config.load()
split = sys.argv[1] if len(sys.argv) > 1 else "test"
step = sys.argv[2] if len(sys.argv) > 2 else str(C["finetune"]["steps"])
PL = C["players"]; A, B = PL
models = {"base": C["base_ckpt"], **{f"{p}_ft": str(ROOT / "ckpts" / f"ft_{p}" / f"step{step}.pt") for p in PL}}
run = wlog.init(f"04_crossover_{split}_s{step}", C, job_type="eval")
rows = []
for mname, ck in models.items():
    m = load_model(ck)
    for player in PL:
        gs = load_games(split, player) if split != "h2h" else load_games("h2h")
        for r in move_logprobs(m, gs, player, C["policy"]["min_ply_scored"]):
            rows.append({**r, "model": mname, "player": player})
    del m
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / f"04_moves_{split}_s{step}.csv", index=False)

out = {"split": split, "step": step, "cells": {}, "diffs": {}}
for player in PL:
    sub = df[df.player == player]
    piv = sub.pivot_table(index=["gid", "ply"], columns="model", values="logp")
    assert piv.notna().all().all(), "models must share identical eligible positions"
    g = piv.index.get_level_values("gid").to_numpy()
    out["cells"][player] = {m: float(piv[m].mean()) for m in models} | {"n_moves": len(piv), "n_games": len(set(g))}
    col = sub.drop_duplicates(["gid", "ply"]).set_index(["gid", "ply"]).loc[piv.index, "color"].to_numpy()
    for a, b in [("base", f"{A}_ft"), ("base", f"{B}_ft"), (f"{B}_ft", f"{A}_ft")]:
        for c in ("all", "w", "b"):
            sel = np.ones(len(piv), bool) if c == "all" else col == c
            out["diffs"][f"{player}/{b}-{a}/{c}"] = mean_ci(g[sel], (piv[b] - piv[a]).to_numpy()[sel])
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / "results" / f"04_crossover_{split}_s{step}.json", "w"), indent=1)
run.log({"crossover": out}); run.finish()
