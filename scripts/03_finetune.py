"""Fine-tune Karpov and Kasparov copies with matched settings + target-char budget.

Usage: python scripts/03_finetune.py [karpov|kasparov|both]
Policy pool -> 90% train / 10% policy_dev (dev used for live curves + setting choice).
Karpov train pool is subsampled (color-stratified) so its total target chars
match Kasparov's, so both copies see the same target-char budget per epoch.
"""
import json
import random
import sys

from wc import config, logging as wlog
from wc.data import load_games
from wc.finetune import finetune, n_targets
from wc.model import ROOT

C = config.load(); F = C["finetune"]; seed = C["seed"]
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("which", nargs="?", default="both")
ap.add_argument("--lr", type=float); ap.add_argument("--steps", type=int); ap.add_argument("--dev_every", type=int)
ap.add_argument("--dropout", type=float); ap.add_argument("--schedule_steps", type=int); ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, help="training seed (batch order, torch)")
ap.add_argument("--data_seed", type=int, default=0, help="which games fill the matched budget (0 = original)")
ap.add_argument("--ckpt_every", type=int, help="save a checkpoint every N steps (else ckpt_fracs)")
a = ap.parse_args(); which = a.which
for k in ("lr", "steps", "dev_every", "dropout", "schedule_steps"):
    if getattr(a, k) is not None:
        F[k] = getattr(a, k)
train_seed = a.seed if a.seed is not None else seed
if a.ckpt_every:
    F["ckpt_steps"] = list(range(0, F["steps"] + 1, a.ckpt_every)) + [F["steps"]]
rng = random.Random(seed)  # dev holdout fixed across seeds

pools, devs = {}, {}
PL = C["players"]
for p in PL:
    gs = sorted(load_games("policy", p), key=lambda g: g.gid); rng.shuffle(gs)
    nd = len(gs) // 10
    devs[p], pools[p] = gs[:nd], gs[nd:]

# match target-char budget: subsample larger pool per color to the smaller's chars
for col in ("w", "b"):
    tot = {p: sum(n_targets(g, p) for g in pools[p] if g.color == col) for p in pools}
    big = max(tot, key=tot.get); small = min(tot, key=tot.get)
    keep, acc = [], 0
    cand = [g for g in pools[big] if g.color == col]
    if a.data_seed:
        random.Random(1000 + a.data_seed).shuffle(cand)
    for g in cand:
        if acc >= tot[small]:
            break
        keep.append(g); acc += n_targets(g, big)
    pools[big] = [g for g in pools[big] if g.color != col] + keep
budget = {p: dict(games=len(pools[p]), target_chars=sum(n_targets(g, p) for g in pools[p]),
                  white=sum(g.color == "w" for g in pools[p])) for p in pools}
print("budget", budget)
json.dump({p: [g.gid for g in devs[p]] for p in devs}, open(ROOT / "results" / "policy_dev_gids.json", "w"))

for p in (PL if which == "both" else [which]):
    run = wlog.init(f"03_finetune_{p}{a.tag}", {**C, "player": p, "budget": budget[p], "train_seed": train_seed, "data_seed": a.data_seed}, group="finetune", job_type="finetune")
    finetune(p, pools[p], devs, F, C["base_ckpt"], ROOT / "ckpts" / f"ft_{p}{a.tag}", run=run, log_name=f"{p}{a.tag}", seed=train_seed)
    run.finish()
