"""Figures for WRITEUP.md from results/*.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc import config
from wc.model import ROOT

PL = config.load()["players"]; A, B = PL; NM = {p: p.capitalize() for p in PL}

R = ROOT / "results"; F = R / "figs"; F.mkdir(exist_ok=True)
C1, C2, C3 = "#2d6cdf", "#c8372d", "#8a8a8a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. layer sweep
pb = json.load(open(R / "02_probe_base.json"))
fig, ax = plt.subplots(figsize=(6, 3.4))
for (m, k, t), c, ls, lab in [(("base", "linear", "real"), C1, "-", "trained, linear"), (("base", "mlp", "real"), C1, "--", "trained, MLP"),
                              (("randinit", "linear", "real"), C3, "-", "random-init, linear"), (("randinit", "mlp", "real"), C3, "--", "random-init, MLP"),
                              (("base", "linear", "control"), C2, ":", "trained, linear, control task")]:
    xs = [x for x in pb if (x["model"], x["kind"], x["task"]) == (m, k, t)]
    ax.plot([x["layer"] for x in xs], [100 * x["occ_err"] for x in xs], ls, color=c, marker="o", ms=3, label=lab)
ax.set_xlabel("residual stream (0 = embedding, k = after block k-1)"); ax.set_ylabel("occupied-square error (%)")
ax.set_title("Board state becomes linearly decodable after block 5"); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(F / "probe_layer_sweep.png", dpi=160)

# 2. crossover
cx = json.load(open(sorted(R.glob("04_crossover_test_s*.json"))[-1]))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for i, p in enumerate(PL):
    for j, (m, c) in enumerate(((f"{A}_ft", C1), (f"{B}_ft", C2))):
        pt, lo, hi = cx["diffs"][f"{p}/{m}-base/all"]
        ax.bar(i + (j - 0.5) * 0.36, pt, 0.34, color=c, yerr=[[pt - lo], [hi - pt]], capsize=3,
               label=(f"{NM[A]} FT" if j == 0 else f"{NM[B]} FT") if i == 0 else None)
ax.set_xticks([0, 1], [f"{NM[p]}'s moves\n(n={cx['cells'][p]['n_moves']})" for p in PL])
ax.set_ylabel("Δ mean log P(played move) vs base"); ax.axhline(0, color="k", lw=0.6)
ax.set_title("Test set, plies ≥ 31"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F / "crossover_test.png", dpi=160)

# 3. probe drift over checkpoints + sensitivity
pt_ = json.load(open(R / "05_probe_transfer_test.json")); se = json.load(open(R / "06_sensitivity.json"))
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
for m, c in ((f"{A}_ft", C1), (f"{B}_ft", C2)):
    rows = [r for r in pt_["rows"] if r["model"] == m]
    axs[0].errorbar([r["step"] for r in rows], [r["delta_pp"] for r in rows],
                    yerr=[[r["delta_pp"] - r["ci_pp"][0] for r in rows], [r["ci_pp"][1] - r["delta_pp"] for r in rows]],
                    color=c, marker="o", ms=3, capsize=3, label=f"{NM[m[:-3]]} FT")
axs[0].axhline(pt_["tolerance_pp"], color="k", ls="--", lw=0.8, label="locked tolerance (1 pp)")
axs[0].set_xlabel("fine-tuning step"); axs[0].set_ylabel("Δ occupied-square error (pp)")
axs[0].set_title(f"Frozen base probe (base error {100*pt_['base_occ_err']:.1f}%)"); axs[0].legend(frameon=False, fontsize=8)
axs[1].errorbar([100 * r["frac"] for r in se], [r["delta_pp"] for r in se],
                yerr=[[r["delta_pp"] - r["ci_pp"][0] for r in se], [r["ci_pp"][1] - r["delta_pp"] for r in se]], color=C3, marker="o", ms=3, capsize=3)
axs[1].axhline(1.0, color="k", ls="--", lw=0.8); axs[1].set_xlabel("% activations swapped with another game, same ply")
axs[1].set_ylabel("Δ occupied-square error (pp)"); axs[1].set_title("Sensitivity: metric sees a known break")
fig.tight_layout(); fig.savefig(F / "probe_preservation.png", dpi=160)
print("wrote", sorted(p.name for p in F.iterdir()))
