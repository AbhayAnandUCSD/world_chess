"""P1.5 figures: probe-drift decomposition vs policy drift, seed robustness, geometry."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc import config
from wc.model import ROOT

CFG = config.load(); PL = CFG["players"]; LR = float(CFG["finetune"]["lr"]); A, B = PL; NM = {p: p.capitalize() for p in PL}
R = ROOT / "results"; F = R / "figs"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
COL = {"frozen": "#c8372d", "procrustes": "#e08a2c", "affine": "#2d6cdf", "refit": "#2f7d4f"}
LAB = {"frozen": "frozen base probe (coordinates fixed)", "procrustes": "frozen probe + rotation",
       "affine": "frozen probe + affine map", "refit": "probe refit on copy (information)"}

# 1. decomposition vs policy KL along trajectories
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, p in zip(axs, PL):
    rows = json.load(open(R / "09" / f"traj_{p}.json"))["rows"]
    for run, mk in (("dose", "o"), ("hot", "^")):
        rr = sorted([r for r in rows if f"_p15_{run}/" in r["ckpt"]], key=lambda r: r["policy"]["kl_base_copy"])
        x = [r["policy"]["kl_base_copy"] for r in rr]
        for k in COL:
            ax.plot(x, [r[k]["delta_pp"] for r in rr], marker=mk, ms=3.5, color=COL[k], ls="-" if run == "dose" else "--",
                    label=LAB[k] if run == "dose" else None)
    ax.axhline(1.0, color="k", lw=0.7, ls=":"); ax.set_xscale("log")
    ax.set_xlabel("policy drift: KL(base ‖ copy) per target char"); ax.set_title(f"{NM[p]} copy  (solid lr {LR:.0e}, dashed lr 3e-5)")
axs[0].set_ylabel("Δ occupied-square error vs base (pp)"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(F / "p15_drift_decomposition.png", dpi=160)

# 2. seeds at the locked checkpoint (step 60): crossover + frozen/refit drift
S = json.load(open(R / "09" / "seeds.json"))["rows"]
T = {p: json.load(open(R / "09" / f"traj_{p}.json"))["rows"] for p in PL}
step = config.load()["finetune"]["steps"]
fin = {p: [r for r in S if f"ft_{p}_" in r["ckpt"]] + [r for r in T[p] if f"_dose/step{step}.pt" in r["ckpt"]] for p in PL}
def uniq(rs):  # identical runs (data_seed does not change the smaller pool) counted once
    seen, out = set(), []
    for r in rs:
        k = round(r["policy"][f"{A}_logp"], 6)
        if k not in seen: seen.add(k); out.append(r)
    return out
fin = {p: uniq(v) for p, v in fin.items()}
xA = [a["policy"][f"{A}_logp"] - b["policy"][f"{A}_logp"] for a in fin[A] for b in fin[B]]
xB = [b["policy"][f"{B}_logp"] - a["policy"][f"{B}_logp"] for a in fin[A] for b in fin[B]]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
axs[0].boxplot([xA, xB], tick_labels=[f"{NM[A]}'s moves\n{NM[A]} − {NM[B]} copy", f"{NM[B]}'s moves\n{NM[B]} − {NM[A]} copy"])
axs[0].axhline(0, color="k", lw=0.7); axs[0].set_ylabel("Δ mean log P (test)")
axs[0].set_title(f"Crossover across all copy pairs ({len(fin[A])}×{len(fin[B])} runs)")
for i, p in enumerate(PL):
    for j, k in enumerate(("frozen", "refit")):
        v = [r[k]["delta_pp"] for r in fin[p]]
        axs[1].scatter(np.full(len(v), i * 3 + j) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v, color=COL[k], s=14,
                       label=LAB[k] if i == 0 else None)
axs[1].set_xticks([0.5, 3.5], [f"{NM[A]} copies", f"{NM[B]} copies"]); axs[1].axhline(1.0, color="k", lw=0.7, ls=":")
axs[1].set_ylabel("Δ occupied-square error (pp)"); axs[1].set_title(f"Board readout at step {step}, all seeds"); axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(F / "p15_seeds.png", dpi=160)
summary = dict(n_runs={p: len(fin[p]) for p in PL}, crossover_A=[min(xA), float(np.mean(xA)), max(xA)], crossover_B=[min(xB), float(np.mean(xB)), max(xB)],
               crossover_positive_frac=[float(np.mean(np.array(xA) > 0)), float(np.mean(np.array(xB) > 0))],
               frozen={p: [r["frozen"]["delta_pp"] for r in fin[p]] for p in PL}, refit={p: [r["refit"]["delta_pp"] for r in fin[p]] for p in PL},
               procrustes={p: [r["procrustes"]["delta_pp"] for r in fin[p]] for p in PL})
json.dump(summary, open(R / "12_p15_seed_summary.json", "w"), indent=1); print(json.dumps(summary, indent=1))

# 3. geometry (if present)
gp = R / "11_geometry_p15.json"
if gp.exists():
    G = json.load(open(gp)); ks = list(G["models"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
    for ck in ks:
        rows = G["models"][ck]; lab = ck.replace("ckpts/ft_", "").replace(".pt", "")
        axs[0].plot([r["layer"] for r in rows], [r["rel_dist_median"] for r in rows], marker="o", ms=3, label=lab)
        axs[1].plot([r["layer"] for r in rows], [r["delta_eff_rank"] for r in rows], marker="o", ms=3, label=lab)
    axs[0].set_xlabel("residual layer"); axs[0].set_ylabel("median ‖h_ft − h_base‖ / ‖h_base‖"); axs[0].legend(frameon=False, fontsize=6.5)
    axs[1].set_xlabel("residual layer"); axs[1].set_ylabel("effective rank of change"); axs[1].set_title("Is the change low-rank?")
    jp = G["joint_pca_L"]; z = np.array(jp["z"]); md = np.array(jp["model"]); mv = np.array(jp["move_no"])
    for m_, c in zip(dict.fromkeys(md), ("#8a8a8a", "#2d6cdf", "#c8372d")):
        s = md == m_; axs[2].scatter(z[s, 0], z[s, 1], s=3, alpha=0.35, color=c, label=m_)
    axs[2].set_title("Joint PCA of layer-6 activations"); axs[2].legend(frameon=False, fontsize=7, markerscale=3)
    fig.tight_layout(); fig.savefig(F / "p15_geometry.png", dpi=160)
print("figs written")
