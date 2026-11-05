"""Matched-KL comparison of P1 (Karpov/Kasparov) and P2 (Carlsen/Tang) probe drift, on BOTH position domains.

Each trajectory checkpoint was scored by 09_analysis.py twice:
  own domain  : results/09/traj_{karpov,kasparov}.json (OTB, P1 probe) | exp_p2/results/09/traj_{carlsen,tang}.json (Lichess, P2 probe)
  other domain: exp_p2/results/09/xdom_{karpov,kasparov}.json           | results/09/xdom_{carlsen,tang}.json
In each panel x = KL(base || copy) measured on THAT domain's positions, so copies are compared at equal policy shift.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc.model import CODE_ROOT

R1, R2 = CODE_ROOT / "results", CODE_ROOT / "exp_p2" / "results"
SRC = {"otb": {"karpov": R1 / "09/traj_karpov.json", "kasparov": R1 / "09/traj_kasparov.json",
               "carlsen": R1 / "09/xdom_carlsen.json", "tang": R1 / "09/xdom_tang.json"},
       "lichess": {"karpov": R2 / "09/xdom_karpov.json", "kasparov": R2 / "09/xdom_kasparov.json",
                   "carlsen": R2 / "09/traj_carlsen.json", "tang": R2 / "09/traj_tang.json"}}
DOM = {"otb": "OTB classical positions (P1 probe)", "lichess": "Lichess bullet positions (P2 probe)"}
COL = {"karpov": "#8a5a2b", "kasparov": "#d08a3c", "carlsen": "#2d6cdf", "tang": "#6aa5f0"}
GRID = [0.005, 0.01, 0.02, 0.04]


def runs(path):
    out = {"dose": [(0.0, 0.0, 0.0)], "hot": [(0.0, 0.0, 0.0)]}
    for r in json.load(open(path))["rows"]:
        if "policy" not in r or "lichess_8layers" in r["ckpt"]:
            continue
        kind = "hot" if "_hot/" in r["ckpt"] else "dose"
        out[kind].append((r["policy"]["kl_base_copy"], r["frozen"]["delta_pp"], r["refit"]["delta_pp"]))
    return {k: sorted(v) for k, v in out.items()}


D = {d: {p: runs(f) for p, f in s.items()} for d, s in SRC.items()}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex="col")
for j, d in enumerate(DOM):
    for i, (idx, lab) in enumerate([(1, "frozen base probe (coordinates fixed)"), (2, "probe refit on copy (information)")]):
        ax = axs[i, j]
        for p, rr in D[d].items():
            for kind, ls in (("dose", "-"), ("hot", "--")):
                a = np.array(rr[kind]); a = a[a[:, 0] > 0] if kind == "hot" else a
                ax.plot(a[:, 0], a[:, idx], ls, marker="o" if kind == "dose" else "^", ms=3.5, color=COL[p],
                        label=f"{p.capitalize()} copy" if kind == "dose" else None)
        ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_xlim(0, 0.35)
        ax.set_title(f"{lab}\non {DOM[d]}", fontsize=9.5)
        if j == 0: ax.set_ylabel("Δ occupied-square error (pp)")
        if i == 1: ax.set_xlabel("policy shift: KL(base ‖ copy) per target char, on these positions")
axs[0, 0].legend(fontsize=8, frameon=False, title="solid = selected lr, dashed = hot lr", title_fontsize=7.5)
fig.tight_layout(); fig.savefig(R1 / "figs" / "matched_kl.png", dpi=160)

# interpolated values at matched KL (dose runs only; hot runs reported separately)
summ = {}
for d in DOM:
    for p, rr in D[d].items():
        a = np.array(rr["dose"]); kmax = a[:, 0].max()
        summ[f"{d}|{p}"] = {"kl_max": float(kmax), **{f"{m}@{g}": (float(np.interp(g, a[:, 0], a[:, k])) if g <= kmax else None)
                                                     for g in GRID for m, k in (("frozen", 1), ("refit", 2))}}
        h = np.array(rr["hot"]); summ[f"{d}|{p}"]["hot_last"] = dict(kl=float(h[-1, 0]), frozen=float(h[-1, 1]), refit=float(h[-1, 2]))
json.dump(summ, open(R1 / "16_matched_kl.json", "w"), indent=1)
for k, v in summ.items():
    print(f"{k:18s} klmax {v['kl_max']:.3f} " + " ".join(f"{g}: f{v[f'frozen@{g}']:+.2f}/r{v[f'refit@{g}']:+.2f}" if v[f"frozen@{g}"] is not None else f"{g}: -" for g in GRID)
          + f" | hot kl {v['hot_last']['kl']:.3f} f{v['hot_last']['frozen']:+.2f} r{v['hot_last']['refit']:+.2f}")
