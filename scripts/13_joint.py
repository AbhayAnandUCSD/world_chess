"""Joint comparison: base + 4 copies (P1 Karpov/Kasparov, P2 Carlsen/Tang) on both test sets.
Run from the code root (WC_ROOT unset). Paths: P1 = ., P2 = exp_p2.
  - cross-player move prediction: every model x every player's test moves (plies >= 31, same positions)
  - frozen probes: P1 probe and P2 probe applied to every model, on each test set
  - layer-6 geometry vs base: relative distance, CKA, and cosine between every pair of copies' mean changes
"""
import json
from pathlib import Path

import numpy as np
import torch

from wc.acts import collect
from wc.data import load_games
from wc.model import CODE_ROOT, load_model, pick_device
from wc.probe import LinearProbe, errors, predict
from wc.score import move_logprobs

dev = pick_device(); L = 6
P1, P2 = CODE_ROOT, CODE_ROOT / "exp_p2"
import yaml
st1 = yaml.safe_load(open(P1 / "configs/p1.yaml"))["finetune"]["steps"]; st2 = yaml.safe_load(open(P2 / "configs/p2.yaml"))["finetune"]["steps"]
MODELS = {"base": "lichess_8layers_ckpt_no_optimizer.pt",
          "karpov_ft": str(P1 / f"ckpts/ft_karpov/step{st1}.pt"), "kasparov_ft": str(P1 / f"ckpts/ft_kasparov/step{st1}.pt"),
          "carlsen_ft": str(P2 / f"ckpts/ft_carlsen/step{st2}.pt"), "tang_ft": str(P2 / f"ckpts/ft_tang/step{st2}.pt")}
SETS = {"p1_test": (P1 / "data/games.jsonl", ["karpov", "kasparov"]), "p2_test": (P2 / "data/games.jsonl", ["carlsen", "tang"])}
games = {k: [g for g in load_games("test", path=p) if g.player in pl] for k, (p, pl) in SETS.items()}
probes = {}
for name, root in (("p1_probe", P1), ("p2_probe", P2)):
    pr = LinearProbe(); pr.load_state_dict(torch.load(root / "ckpts/probe_base_frozen.pt", map_location="cpu")); probes[name] = pr.to(dev).eval()

out = {"steps": {"p1": st1, "p2": st2}, "moves": {}, "probe": {}, "geometry": {}}
acts = {}
for mname, ck in MODELS.items():
    m = load_model(ck, dev).eval()
    for sname, gs in games.items():
        for p in SETS[sname][1]:
            rows = move_logprobs(m, [g for g in gs if g.player == p], p, 31)
            out["moves"][f"{mname}|{p}"] = float(np.mean([r["logp"] for r in rows]))
        D = collect(m, gs, 16, layers=[L])
        acts[(mname, sname)] = (D["acts"][0].float().numpy(), D["labels"])
        for pn, pr in probes.items():
            e = errors(predict(pr, D["acts"][0], dev), D["labels"])
            out["probe"][f"{mname}|{sname}|{pn}"] = float(e["occ_wrong"].sum() / e["n_occ"].sum())
    print(mname, "done", flush=True); del m

def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
rng = np.random.default_rng(0)
for sname in games:
    X0, _ = acts[("base", sname)]; sub = rng.choice(len(X0), min(4000, len(X0)), replace=False)
    deltas = {m: acts[(m, sname)][0] - X0 for m in MODELS if m != "base"}
    g = {m: dict(rel_dist=float(np.median(np.linalg.norm(d, axis=1) / np.linalg.norm(X0, axis=1))),
                 cka=cka(X0[sub], acts[(m, sname)][0][sub])) for m, d in deltas.items()}
    names = list(deltas)
    g["mean_delta_cos"] = {f"{a}|{b}": float(deltas[a].mean(0) @ deltas[b].mean(0) / (np.linalg.norm(deltas[a].mean(0)) * np.linalg.norm(deltas[b].mean(0))))
                           for i, a in enumerate(names) for b in names[i + 1:]}
    out["geometry"][sname] = g
json.dump(out, open(CODE_ROOT / "results" / "13_joint.json", "w"), indent=1)
print(json.dumps(out, indent=1))
