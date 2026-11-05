"""Can the metric detect a known break? Swap a fraction of activations with another
game's activation at the same move number; labels + probe fixed. Dev games."""
import json

import numpy as np
import torch

from wc import config, logging as wlog
from wc.acts import collect
from wc.data import load_games
from wc.model import ROOT, load_model, pick_device
from wc.probe import LinearProbe, errors, predict
from wc.stats import paired_bootstrap

C = config.load(); P = C["probe"]; dev = pick_device()
games = [g for g in load_games("probe_dev") if g.player != "both"]
probe = LinearProbe(); probe.load_state_dict(torch.load(ROOT / "ckpts" / "probe_base_frozen.pt", map_location="cpu")); probe.to(dev).eval()
D = collect(load_model(C["base_ckpt"], dev), games, P["eval_min_move"], layers=[P["layer_resid"]])
X, Y, G, MV = D["acts"][0], D["labels"], D["gid_idx"], D["move_no"]
e0 = errors(predict(probe, X, dev), Y)
rng = np.random.default_rng(0); out = []
run = wlog.init("06_sensitivity", C, job_type="eval")
for f in C["sensitivity"]["swap_fracs"]:
    Xs = X.clone(); chosen = rng.random(len(X)) < f
    for i in np.where(chosen)[0]:
        cand = np.where((MV == MV[i]) & (G != G[i]))[0]
        if len(cand):
            Xs[i] = X[rng.choice(cand)]
    e = errors(predict(probe, Xs, dev), Y)
    pt, lo, hi = paired_bootstrap(G, e0["occ_wrong"].astype(float), e["occ_wrong"].astype(float), e0["n_occ"].astype(float))
    r = dict(frac=f, occ_err=float(e["occ_wrong"].sum() / e["n_occ"].sum()), delta_pp=100 * pt, ci_pp=[100 * lo, 100 * hi],
             detected=bool(100 * lo > C["preservation"]["tolerance_pp"]))
    out.append(r); print(r); run.log({"sens/delta_pp": r["delta_pp"], "sens/frac": f})
json.dump(out, open(ROOT / "results" / "06_sensitivity.json", "w"), indent=1); run.finish()
