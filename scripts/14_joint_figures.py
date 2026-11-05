"""Joint figure: 4 copies (P1 + P2) vs base on all 4 players' test moves, frozen-probe drift, and change-direction overlap."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc.model import CODE_ROOT

R = CODE_ROOT / "results"; F = R / "figs"; F.mkdir(exist_ok=True)
J = json.load(open(R / "13_joint.json"))
PL = ["karpov", "kasparov", "carlsen", "tang"]; CP = [f"{p}_ft" for p in PL]
NM = {p: p.capitalize() for p in PL}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})

fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1]})

# 1. Δ log P vs base, copy (rows) × whose moves (cols)
M = np.array([[J["moves"][f"{c}|{p}"] - J["moves"][f"base|{p}"] for p in PL] for c in CP])
ax = axs[0]; v = np.abs(M).max()
im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold" if i == j else None)
ax.set_xticks(range(4), [f"{NM[p]}'s\nmoves" for p in PL]); ax.set_yticks(range(4), [f"{NM[p]} copy" for p in PL])
ax.set_title("Δ mean log P(move) vs base, test sets")
fig.colorbar(im, ax=ax, fraction=0.046)

# 2. frozen-probe drift on each domain (own-experiment probe)
ax = axs[1]; x = np.arange(4); w = 0.38
for k, (dom, prb, lab, col) in enumerate([("p1_test", "p1_probe", "OTB positions (P1 test, P1 probe)", "#8a5a2b"),
                                          ("p2_test", "p2_probe", "Lichess bullet positions (P2 test, P2 probe)", "#2d6cdf")]):
    d = [100 * (J["probe"][f"{c}|{dom}|{prb}"] - J["probe"][f"base|{dom}|{prb}"]) for c in CP]
    ax.bar(x + (k - 0.5) * w, d, w, color=col, label=lab)
ax.axhline(0, color="k", lw=0.7); ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xticks(x, [f"{NM[p]}\n(step {J['steps']['p1' if i < 2 else 'p2']})" for i, p in enumerate(PL)])
ax.set_ylabel("Δ occupied-square error (pp)"); ax.set_title("Frozen base probe drift, by position domain")
ax.set_ylim(-0.25, 1.35); ax.legend(fontsize=7.5, frameon=False, loc="upper right")

# 3. cosine between mean residual changes (layer 6)
G = J["geometry"]["p2_test"]["mean_delta_cos"]
C = np.eye(4)
for i, a in enumerate(CP):
    for j, b in enumerate(CP):
        if i != j: C[i, j] = G.get(f"{a}|{b}", G.get(f"{b}|{a}"))
ax = axs[2]
im = ax.imshow(C, cmap="viridis", vmin=0.4, vmax=1)
for i in range(4):
    for j in range(4):
        if i != j: ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="w" if C[i, j] < 0.7 else "k")
ax.set_xticks(range(4), [NM[p] for p in PL]); ax.set_yticks(range(4), [NM[p] for p in PL])
ax.set_title("Cosine of mean resid-6 change (copy − base)")
fig.colorbar(im, ax=ax, fraction=0.046)

fig.tight_layout(); fig.savefig(F / "joint_comparison.png", dpi=160)
print("wrote", F / "joint_comparison.png")
