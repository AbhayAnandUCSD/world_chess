"""Frozen base probe applied to base / Karpov-FT / Kasparov-FT at every saved ckpt.
Usage: python scripts/05_probe_transfer.py [split]  (default test)"""
import json
import sys

import numpy as np
import torch

from wc import config, logging as wlog
from wc.acts import collect
from wc.data import load_games
from wc.model import ROOT, load_model, pick_device
from wc.probe import LinearProbe, errors, predict
from wc.stats import paired_bootstrap

C = config.load(); P = C["probe"]; dev = pick_device(); L = P["layer_resid"]
split = sys.argv[1] if len(sys.argv) > 1 else "test"
PL = C["players"]
games = [g for g in load_games(split) if g.player in PL]
probe = LinearProbe(); probe.load_state_dict(torch.load(ROOT / "ckpts" / "probe_base_frozen.pt", map_location="cpu")); probe.to(dev).eval()
steps = [int(round(f * C["finetune"]["steps"])) for f in C["finetune"]["ckpt_fracs"]]
run = wlog.init(f"05_probe_transfer_{split}", C, job_type="eval")

def evaluate(ck):
    m = load_model(ck, dev)
    D = collect(m, games, P["eval_min_move"], layers=[L])
    e = errors(predict(probe, D["acts"][0], dev), D["labels"]); del m
    return e, D["gid_idx"]

base_e, gidx = evaluate(C["base_ckpt"])
base_err = base_e["occ_wrong"].sum() / base_e["n_occ"].sum()
out = {"split": split, "n_positions": int(len(gidx)), "n_games": int(len(set(gidx))),
       "base_occ_err": float(base_err), "tolerance_pp": C["preservation"]["tolerance_pp"], "rows": []}
print("base occ err", base_err, out["n_positions"], "positions")
for player in PL:
    for s in steps:
        e, g2 = evaluate(str(ROOT / "ckpts" / f"ft_{player}" / f"step{s}.pt"))
        assert (g2 == gidx).all()
        pt, lo, hi = paired_bootstrap(gidx, base_e["occ_wrong"].astype(float), e["occ_wrong"].astype(float),
                                      base_e["n_occ"].astype(float), C["preservation"]["bootstrap"])
        r = dict(model=f"{player}_ft", step=s, occ_err=float(e["occ_wrong"].sum() / e["n_occ"].sum()),
                 delta_pp=100 * pt, ci_pp=[100 * lo, 100 * hi],
                 preserved=bool(100 * hi < C["preservation"]["tolerance_pp"]))
        out["rows"].append(r); print(r); run.log({f"transfer/{player}/delta_pp": r["delta_pp"], "ckpt_step": s})
json.dump(out, open(ROOT / "results" / f"05_probe_transfer_{split}.json", "w"), indent=1)
run.finish()
