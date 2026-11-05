"""Rollout divergence: how fast do two models' games split from the same start, and how different are the boards after?

The opponent (base model, greedy) is deterministic, so two games can only split on the tested model's own
move. For each pair of models and each start position:
  split_at   : own-move number of the first different move (censored if both games end first)
  squares    : number of the 64 squares whose contents differ after ply 30 (both games must reach it)
Pairs: same player different seed (noise floor), the two players of a pair, each copy vs the base model.
Outputs results/19/rollouts.json and results/figs/rollouts.png
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chess

from wc.model import CODE_ROOT

D = CODE_ROOT / "results/19"; F = CODE_ROOT / "results/figs"
NAMES = ["base", "karpov_ft", "kasparov_ft", "carlsen_ft", "tang_ft", "karpov_s1", "kasparov_s1", "carlsen_s1", "tang_s1"]
R = {}
for n in NAMES:
    p = D / f"rollout_{n}.jsonl"
    if p.exists():
        R[n] = {d["i"]: d for d in map(json.loads, open(p))}
CAP = {"karpov": "Karpov", "kasparov": "Kasparov", "carlsen": "Carlsen", "tang": "Tang"}
PAIRS = {"same player, two seeds": [(f"{p}_ft", f"{p}_s1") for p in CAP],
         "different players": [("karpov_ft", "kasparov_ft"), ("carlsen_ft", "tang_ft"), ("karpov_s1", "kasparov_s1"), ("carlsen_s1", "tang_s1")],
         "copy vs base": [(f"{p}_ft", "base") for p in CAP]}
K = 15; END_PLY = 30


def sqdiff(f1, f2):
    a, b = chess.Board(f1 + " w - - 0 1"), chess.Board(f2 + " w - - 0 1")
    return sum(a.piece_at(s) != b.piece_at(s) for s in chess.SQUARES)


def compare(a, b):
    out = []
    for i in sorted(set(R[a]) & set(R[b])):
        ga, gb = R[a][i], R[b][i]; ma, mb = ga["moves"], gb["moves"]
        n = min(len(ma), len(mb))
        j = next((t for t in range(n) if ma[t] != mb[t]), None)
        own_reached = (n + 1) // 2
        sq = sqdiff(ga["fens"][END_PLY - 1], gb["fens"][END_PLY - 1]) if len(ma) >= END_PLY and len(mb) >= END_PLY else None
        out.append(dict(i=i, set=ga["set"], split_at=None if j is None else j // 2 + 1, own_reached=own_reached, squares=sq))
    return out


def survival(rows):
    s = []
    for k in range(1, K + 1):
        same = sum(1 for r in rows if (r["split_at"] is None and r["own_reached"] >= k) or (r["split_at"] is not None and r["split_at"] > k))
        at_risk = sum(1 for r in rows if (r["split_at"] is not None and r["split_at"] <= k) or r["own_reached"] >= k)
        s.append(same / at_risk if at_risk else np.nan)
    return s


res = {"pairs": {}}
for grp, prs in PAIRS.items():
    for a, b in prs:
        if a not in R or b not in R:
            continue
        rows = compare(a, b)
        sq = [r["squares"] for r in rows if r["squares"] is not None]
        sq_div = [r["squares"] for r in rows if r["squares"] is not None and r["split_at"] is not None]
        res["pairs"][f"{a}|{b}"] = dict(
            group=grp, n=len(rows), survival=survival(rows),
            by_set={st: survival([r for r in rows if r["set"] == st]) for st in ("neutral_otb", "h2h_p2")},
            first_move_differs=float(np.mean([r["split_at"] == 1 for r in rows])),
            median_split=float(np.median([r["split_at"] if r["split_at"] else 99 for r in rows])),
            squares_mean=float(np.mean(sq)) if sq else None, squares_mean_if_split=float(np.mean(sq_div)) if sq_div else None,
            identical_at_end=float(np.mean([s == 0 for s in sq])) if sq else None, n_end=len(sq), squares=sq)
json.dump(res, open(D / "rollouts.json", "w"), indent=1)

# ---- figure ----
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
STY = {"same player, two seeds": dict(ls=":", lw=1.6), "different players": dict(ls="-", lw=2.2),
       "across pairs": dict(ls="-.", lw=1.2), "copy vs base": dict(ls="--", lw=1.2)}
COL = {"karpov": "#8a5a2b", "kasparov": "#d08a3c", "carlsen": "#2d6cdf", "tang": "#6aa5f0"}
ax = axs[0]; ks = np.arange(1, K + 1)
for key, v in res["pairs"].items():
    a, b = key.split("|"); g = v["group"]
    if g == "different players":
        col = "#8a5a2b" if "karpov" in a else "#2d6cdf"; lab = f"{CAP[a.split('_')[0]]} vs {CAP[b.split('_')[0]]}" + (" (seed 1)" if "s1" in a else "")
        alpha = 1.0 if "_ft" in a else 0.45
    elif g == "same player, two seeds":
        col = COL[a.split("_")[0]]; lab = f"{CAP[a.split('_')[0]]}: two seeds"; alpha = 1.0
    elif g == "across pairs":
        col = "#8e5bb5"; lab = "across pairs (P1 copy vs P2 copy)" if key == "karpov_ft|carlsen_ft" else None; alpha = 0.8
    else:
        col = "#999999"; lab = "copy vs base" if a == "karpov_ft" else None; alpha = 0.8
    ax.plot(ks, 100 * np.array(v["survival"]), color=col, alpha=alpha, label=lab, **STY[g])
ax.set_xlabel("own moves played from the start position"); ax.set_ylabel("games still identical (%)")
ax.set_ylim(0, 100); ax.set_xticks(ks[::2])
ax.set_title("A. How long do two models' games stay identical?\n(same start, same greedy base opponent)")
ax.legend(frameon=False, fontsize=7.5, ncol=2)

ax = axs[1]
order = [k for g in PAIRS for k in res["pairs"] if res["pairs"][k]["group"] == g]
data = [res["pairs"][k]["squares"] for k in order]
labs = []
for k in order:
    a, b = k.split("|")
    labs.append(f"{CAP[a.split('_')[0]]} 2 seeds" if res["pairs"][k]["group"].startswith("same") else
                f"{CAP[a.split('_')[0]]} copy vs base" if b == "base" else
                f"{CAP[a.split('_')[0]]} vs {CAP[b.split('_')[0]]}" + (" (s1)" if "s1" in a else ""))
bp = ax.boxplot(data, orientation="horizontal", tick_labels=labs, showfliers=False, patch_artist=True, widths=0.6)
for patch, k in zip(bp["boxes"], order):
    g = res["pairs"][k]["group"]
    patch.set_facecolor({"same player, two seeds": "#e8e3d8", "different players": "#9fc0ee", "across pairs": "#d7c4ea", "copy vs base": "#dddddd"}[g])
for j, k in enumerate(order):
    ax.plot(res["pairs"][k]["squares_mean"], j + 1, "kD", ms=3.5)
ax.invert_yaxis(); ax.set_xlabel(f"squares that differ after {END_PLY} plies (of 64; ◆ = mean)")
ax.set_title("B. How different are the boards by then?")
fig.tight_layout(); fig.savefig(F / "rollouts.png", dpi=160)

for k, v in res["pairs"].items():
    print(f"{k:24s} {v['group']:22s} n={v['n']} first-move-differs {100*v['first_move_differs']:.1f}%  identical@5 {100*v['survival'][4]:.0f}% @10 {100*v['survival'][9]:.0f}% @15 {100*v['survival'][14]:.0f}%  "
          f"median split {v['median_split']:.0f}  squares {v['squares_mean']:.1f} (if split {v['squares_mean_if_split']:.1f}) identical-end {100*v['identical_at_end']:.0f}% n_end={v['n_end']}")
