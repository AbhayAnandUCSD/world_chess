"""Geometry of the latent spaces: base vs fine-tuned copies, same positions.

Per residual layer (0..8), on White-to-move test positions (move >= 16):
  dist       : median ||h_ft - h_base|| / ||h_base||, mean cosine(h_ft, h_base)
  pca        : explained-variance spectrum (top 50), principal angles between top-k PCA subspaces (k=10,50)
  rsa        : Spearman corr of pairwise-distance matrices (2k-position subsample)
  delta      : PCA of the change D = h_ft - h_base -> effective rank, top-k variance share
  delta_probe: share of ||D||^2 inside the frozen probe's row space (layer 6 only) -> does the change hit board directions?
  shared     : cosine between mean deltas of the two copies; share of Karpov delta explained by Kasparov delta subspace
Also saves 2-D joint PCA coordinates (layer 6) for plotting.

Usage: python scripts/11_geometry.py OUT_NAME KARPOV_CKPT KASPAROV_CKPT [EXTRA_CKPT ...]
"""
import json
import sys

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from wc import config
from wc.acts import collect
from wc.data import load_games
from wc.model import ROOT, load_model, pick_device
from wc.probe import LinearProbe

C = config.load(); dev = pick_device(); L = C["probe"]["layer_resid"]
out_name, ckpts = sys.argv[1], sys.argv[2:]
games = [g for g in load_games("test") if g.player in C["players"]]
rng = np.random.default_rng(0)


def get(ck):
    m = load_model(ck, dev)
    D = collect(m, games, 16, layers=list(range(9)))
    del m
    return D["acts"].float().numpy(), D


A0, D0 = get(C["base_ckpt"])
N = A0.shape[1]
sub = rng.choice(N, size=min(2000, N), replace=False)
probe = LinearProbe(); probe.load_state_dict(torch.load(ROOT / "ckpts" / "probe_base_frozen.pt", map_location="cpu"))
Wp = probe.w.weight.detach().numpy()                       # (832, 512)
# probe row space is full-rank (832 rows > 512 dims), so use its dominant directions instead:
# top-k right singular vectors of W holding 90% of the probe's weight energy; random expectation = k/512.
_, Sp, Vtp = np.linalg.svd(Wp, full_matrices=False)
rank_p = int(np.searchsorted(np.cumsum(Sp ** 2) / (Sp ** 2).sum(), 0.90) + 1)
Qp = Vtp[:rank_p].T


def pca(X, k=50):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return Vt[:k], var[:k]


def angles(V1, V2):
    s = np.linalg.svd(V1 @ V2.T, compute_uv=False).clip(-1, 1)
    return np.degrees(np.arccos(s))


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


res = {"n_positions": int(N), "probe_top_dirs_90pct": int(rank_p), "models": {}}
deltas = {}
for ck in ckpts:
    A1, D1 = get(ck)
    assert (D1["gid_idx"] == D0["gid_idx"]).all()
    per = []
    for l in range(9):
        X0, X1 = A0[l], A1[l]
        n0 = np.linalg.norm(X0, axis=1)
        rel = np.linalg.norm(X1 - X0, axis=1) / n0
        cos = (X0 * X1).sum(1) / (n0 * np.linalg.norm(X1, axis=1))
        V0, var0 = pca(X0); V1, var1 = pca(X1)
        d0, d1 = pdist(X0[sub]), pdist(X1[sub])
        Dl = X1 - X0
        Vd, vard = pca(Dl, 50)
        row = dict(layer=l, rel_dist_median=float(np.median(rel)), cos_mean=float(cos.mean()),
                   pca_var_top10_base=float(var0[:10].sum()), pca_var_top10_ft=float(var1[:10].sum()),
                   angles_top10=angles(V0[:10], V1[:10]).tolist(), max_angle_top50=float(angles(V0, V1).max()),
                   rsa_spearman=float(spearmanr(d0, d1).correlation),
                   delta_eff_rank=eff_rank(Dl), delta_var_top1=float(vard[0]), delta_var_top10=float(vard[:10].sum()),
                   delta_mean_over_norm=float(np.linalg.norm(Dl.mean(0)) / np.linalg.norm(Dl, axis=1).mean()))
        if l == L:
            proj = Dl @ Qp
            row["delta_energy_in_probe_space"] = float((proj ** 2).sum() / (Dl ** 2).sum())
            row["random_subspace_expectation"] = rank_p / 512
        per.append(row)
        deltas.setdefault(ck, {})[l] = Dl
    res["models"][ck] = per
    print(ck, json.dumps({k: v for k, v in per[L].items() if k != "angles_top10"}), flush=True)

# shared vs player-specific change (first two ckpts = Karpov, Kasparov)
kp, ks = ckpts[0], ckpts[1]
shared = []
for l in range(9):
    a, b = deltas[kp][l], deltas[ks][l]
    ma, mb = a.mean(0), b.mean(0)
    Vb, _ = pca(b, 20)
    ac = a - a.mean(0)
    shared.append(dict(layer=l, cos_mean_delta=float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb))),
                       per_pos_cos=float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).mean()),
                       A_delta_var_in_B_top20=float(((ac @ Vb.T) ** 2).sum() / (ac ** 2).sum()),
                       ft_ft_rel_dist=float(np.median(np.linalg.norm(a - b, axis=1) / np.linalg.norm(A0[l], axis=1)))))
res["shared_change"] = shared
print("shared L6", json.dumps(shared[L]), flush=True)

# joint 2-D PCA at layer L for plotting (base + first two copies), coloured later by move number / material
Xj = np.concatenate([A0[L][sub], (A0[L] + deltas[kp][L])[sub], (A0[L] + deltas[ks][L])[sub]])
Vj, varj = pca(Xj, 2); Z = (Xj - Xj.mean(0)) @ Vj.T
lab = D0["labels"][sub]
res["joint_pca_L"] = dict(var=varj.tolist(), z=Z.round(3).tolist(), model=["base"] * len(sub) + [f"{C['players'][0]}_ft"] * len(sub) + [f"{C['players'][1]}_ft"] * len(sub),
                          move_no=np.tile(D0["move_no"][sub], 3).tolist(), n_pieces=np.tile((lab != 6).sum(1), 3).tolist())
json.dump(res, open(ROOT / "results" / f"11_geometry_{out_name}.json", "w"))
print("done")
