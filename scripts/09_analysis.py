"""Per-checkpoint decomposition of probe drift + policy drift.

For every checkpoint given:
  frozen      : frozen base probe on the copy's layer-6 activations (test)        -> coordinate stability
  refit       : linear probe refit on the copy's own activations (same recipe)    -> information present
  procrustes  : frozen probe after orthogonal alignment copy->base (fit on probe-fit pool)
  affine      : frozen probe after least-squares affine map copy->base             (upper bound on 'just moved')
  cka         : linear CKA base vs copy at every residual layer (test subset)
  probe_cos   : cosine(refit W row, frozen W row) averaged over (square, piece) rows
  weight_delta: ||theta - theta0|| / ||theta0|| per block
  policy      : mean log P(played move) plies>=31 per player, and KL(base || copy) on target-move chars

Usage: python scripts/09_analysis.py OUT_NAME CKPT [CKPT ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from wc import config
from wc.acts import collect
from wc.data import load_games
from wc.model import ROOT, encode, load_model, pick_device
from wc.probe import LinearProbe, errors, fit, predict, summarize
from wc.score import CTX, move_logprobs, target_mask
from wc.stats import paired_bootstrap

C = config.load(); P = C["probe"]; dev = pick_device(); L = P["layer_resid"]
out_name, ckpts = sys.argv[1], sys.argv[2:]
OUT = ROOT / "results" / "09"; OUT.mkdir(parents=True, exist_ok=True)

fit_games = [g for sp in P["fit_splits"] for g in load_games(sp)]
PL = C["players"]
test_games = [g for g in load_games("test") if g.player in PL]
frozen = LinearProbe(); frozen.load_state_dict(torch.load(ROOT / "ckpts" / "probe_base_frozen.pt", map_location="cpu"))
frozen.to(dev).eval()
rng = np.random.default_rng(0)


def acts(m, games, min_move, layers):
    return collect(m, games, min_move, layers=layers)


def cka(X, Y):
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


@torch.no_grad()
def kl_on_targets(base, m, games, batch=16):
    tot, n = 0.0, 0
    for bi in range(0, len(games), batch):
        chunk = games[bi:bi + batch]
        T = min(CTX, max(len(g.text) for g in chunk) - 1)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.bool)
        for j, g in enumerate(chunk):
            e = encode(g.text[:T]); ids[j, :len(e)] = torch.tensor(e)
            mask[j] = torch.from_numpy(target_mask(g, g.player, T))
        ids = ids.to(dev); mask = mask.to(dev)
        lp = base(ids).float().log_softmax(-1); lq = m(ids).float().log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        tot += kl[mask].sum().item(); n += int(mask.sum())
    return tot / n


base = load_model(C["base_ckpt"], dev)
theta0 = {k: v.detach().clone() for k, v in base.state_dict().items()}
B_fit = acts(base, fit_games, P["train_min_move"], [L])
B_test = acts(base, test_games, P["eval_min_move"], list(range(9)))
sub = rng.choice(B_test["acts"].shape[1], size=min(4096, B_test["acts"].shape[1]), replace=False)
Yt = B_test["labels"]; G = B_test["gid_idx"]
e_base = errors(predict(frozen, B_test["acts"][L], dev), Yt)
Xb_fit = B_fit["acts"][0].float()
mu_b = Xb_fit.mean(0)

results = []
for ck in ckpts:
    name = str(Path(ck).relative_to(ROOT)) if Path(ck).is_absolute() else ck
    m = load_model(ck, dev)
    F_fit = acts(m, fit_games, P["train_min_move"], [L])
    F_test = acts(m, test_games, P["eval_min_move"], list(range(9)))
    Xf_fit = F_fit["acts"][0].float(); Xf_test = F_test["acts"][L].float()
    assert (F_test["gid_idx"] == G).all()

    def score(e):
        pt, lo, hi = paired_bootstrap(G, e_base["occ_wrong"].astype(float), e["occ_wrong"].astype(float),
                                      e_base["n_occ"].astype(float), 1000)
        return dict(occ_err=float(e["occ_wrong"].sum() / e["n_occ"].sum()), delta_pp=100 * pt, ci_pp=[100 * lo, 100 * hi])

    r = {"ckpt": name}
    r["frozen"] = score(errors(predict(frozen, Xf_test, dev), Yt))
    refit = fit(Xf_fit, F_fit["labels"], "linear", P["epochs"], P["lr"], P["wd"], bs=P["batch"], device=dev)
    r["refit"] = score(errors(predict(refit, Xf_test, dev), Yt))
    if ck == C["base_ckpt"]:  # probe-seed noise floor: refit base with other probe seeds
        r["refit_seed_noise"] = [score(errors(predict(fit(Xf_fit, F_fit["labels"], "linear", P["epochs"], P["lr"], P["wd"],
                                                         bs=P["batch"], seed=s_, device=dev), Xf_test, dev), Yt)) for s_ in (1, 2)]
    # orthogonal Procrustes (centered) copy -> base
    mu_f = Xf_fit.mean(0)
    U, _, Vt = torch.linalg.svd((Xf_fit - mu_f).T @ (Xb_fit - mu_b), full_matrices=False)
    R = U @ Vt
    I = torch.eye(R.shape[0])
    r["map"] = {"rot_dist_from_identity": float((R - I).norm() / I.norm()),           # 0 = no rotation
                "rot_mean_angle_deg": float(torch.rad2deg(torch.arccos(torch.diagonal(R).clamp(-1, 1))).mean()),
                "mean_shift_over_norm": float((mu_f - mu_b).norm() / mu_b.norm()),
                "fit_resid_identity": float(((Xf_fit - Xb_fit).norm() / (Xb_fit - mu_b).norm())),
                "fit_resid_procrustes": float((((Xf_fit - mu_f) @ R + mu_b - Xb_fit).norm() / (Xb_fit - mu_b).norm()))}
    r["procrustes"] = score(errors(predict(frozen, (Xf_test - mu_f) @ R + mu_b, dev), Yt))
    # affine least squares copy -> base (ridge 1e-3)
    Xa = torch.cat([Xf_fit, torch.ones(len(Xf_fit), 1)], 1).double()
    W = torch.linalg.solve(Xa.T @ Xa + 1e-3 * torch.eye(Xa.shape[1], dtype=torch.double), Xa.T @ Xb_fit.double()).float()
    r["map"]["fit_resid_affine"] = float(((Xa.float() @ W - Xb_fit).norm() / (Xb_fit - mu_b).norm()))
    r["map"]["affine_dist_from_identity"] = float((W[:-1] - torch.eye(W.shape[1])).norm() / W.shape[1] ** 0.5)
    r["affine"] = score(errors(predict(frozen, torch.cat([Xf_test, torch.ones(len(Xf_test), 1)], 1) @ W, dev), Yt))
    # probe direction similarity (rows = square x class)
    wf, wr = frozen.w.weight.detach().cpu(), refit.w.weight.detach().cpu()
    r["probe_cos"] = float(torch.nn.functional.cosine_similarity(wf, wr, dim=1).mean())
    r["cka"] = [cka(B_test["acts"][l][sub].float().numpy(), F_test["acts"][l][sub].float().numpy()) for l in range(9)]
    sd = m.state_dict()
    blocks = {}
    for k, v in sd.items():
        grp = k.split(".")[2] if k.startswith("transformer.h.") else k.split(".")[1]
        a_, b_ = blocks.get(grp, (0.0, 0.0))
        blocks[grp] = (a_ + (v.float() - theta0[k].float()).norm().item() ** 2, b_ + theta0[k].float().norm().item() ** 2)
    r["weight_delta"] = {g: (a_ / b_) ** 0.5 for g, (a_, b_) in blocks.items()}
    m.eval()
    r["policy"] = {f"{p}_logp": float(np.mean([x["logp"] for x in move_logprobs(m, [g for g in test_games if g.player == p], p, 31)]))
                   for p in PL}
    r["policy"]["kl_base_copy"] = kl_on_targets(base, m, test_games)
    print(json.dumps({k: (v if k in ("ckpt", "probe_cos") else v) for k, v in r.items() if k != "cka"}), flush=True)
    results.append(r)
    json.dump({"base_occ_err": float(e_base["occ_wrong"].sum() / e_base["n_occ"].sum()), "rows": results},
              open(OUT / f"{out_name}.json", "w"), indent=1)
    del m
