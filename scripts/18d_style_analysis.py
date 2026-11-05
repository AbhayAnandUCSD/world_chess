"""Style analysis, step 4: style profile of base + 4 copies on the SAME positions, and a check against the real players.

For each position and model, the policy p over legal moves gives expected move features:
  habits : capture, check, pawn move, king move, queen move, castle, move into opponent's half, equal trade
  engine : centipawn loss vs Stockfish best (clipped 1000), P(engine top move), P(blunder >= 200 cp),
           P(move leaves the opponent an only move: best reply >= 150 cp better than the 2nd)
Outputs results/18/style.json and results/figs/style.png:
  A  identification AUC vs number of moves (from 17_identify.json)
  B  copy - base shift of each feature on neutral positions (OTB champions; Carlsen-Tang h2h)
  C  on each player's own positions: real tendency (played move - base expectation) vs copy shift (copy - base)
  D  top-1 agreement between the pair's copies on neutral positions, and move-matching accuracy on own positions
"""
import json
from collections import defaultdict

import chess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wc.model import CODE_ROOT

R = CODE_ROOT / "results"; D = R / "18"
MODELS = ["base", "karpov_ft", "kasparov_ft", "carlsen_ft", "tang_ft"]
COPY = {"karpov": "karpov_ft", "kasparov": "kasparov_ft", "carlsen": "carlsen_ft", "tang": "tang_ft"}
PAIR = {"karpov": "kasparov", "kasparov": "karpov", "carlsen": "tang", "tang": "carlsen"}
COL = {"karpov_ft": "#8a5a2b", "kasparov_ft": "#d08a3c", "carlsen_ft": "#2d6cdf", "tang_ft": "#6aa5f0"}
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

rows = [json.loads(l) for l in open(D / "positions.jsonl")]
eng = {}
for f in sorted(D.glob("engine_*.jsonl")):
    for l in open(f):
        e = json.loads(l); eng[e["i"]] = e
rows = [r for r in rows if r["i"] in eng]
allrows = [json.loads(l) for l in open(D / "positions.jsonl")]
starts = dict(zip([r["i"] for r in allrows], np.cumsum([0] + [len(r["legal"]) for r in allrows])[:-1]))
POL = {m: np.load(D / f"policy_{m}.npy") for m in MODELS}
SEEDS = {c: c.replace("_ft", "_s1") for c in MODELS[1:] if (D / f"policy_{c.replace('_ft', '_s1')}.npy").exists()}
for s_ in SEEDS.values():
    POL[s_] = np.load(D / f"policy_{s_}.npy")
HAB = 8  # first 8 features are move habits; the rest are engine-based

FEATS = ["capture", "check", "pawn", "king", "queen", "castle", "opp_half", "trade",
         "cp_loss", "engine_top", "blunder", "only_move_for_opp"]
LABEL = {"capture": "capture", "check": "check", "pawn": "pawn move", "king": "king move", "queen": "queen move",
         "castle": "castle", "opp_half": "into opp. half", "trade": "equal trade", "cp_loss": "cp loss (/100)",
         "engine_top": "engine top move", "blunder": "blunder ≥200cp", "only_move_for_opp": "leaves opp. one move"}


def move_feats(r, e):
    b = chess.Board(r["fen"]); me = b.turn
    ev = np.array(e["eval"], float); best = ev.max()
    F = np.zeros((len(r["legal"]), len(FEATS)))
    for j, s in enumerate(r["legal"]):
        mv = b.parse_san(s); pc = b.piece_at(mv.from_square); cap = b.piece_at(mv.to_square)
        rank = chess.square_rank(mv.to_square)
        F[j] = [b.is_capture(mv), "+" in s or "#" in s, pc.piece_type == chess.PAWN, pc.piece_type == chess.KING,
                pc.piece_type == chess.QUEEN, b.is_castling(mv), (rank >= 4) if me == chess.WHITE else (rank <= 3),
                cap is not None and VAL[cap.piece_type] == VAL[pc.piece_type],
                (best - ev[j]) / 100, ev[j] == best, best - ev[j] >= 200, e["opp_gap"][j] >= 150]
    return F


per = defaultdict(lambda: defaultdict(list))   # set -> key -> list of per-position values
top1 = defaultdict(lambda: defaultdict(list))
js_rows = []
for r in rows:
    F = move_feats(r, eng[r["i"]]); a = starts[r["i"]]; n = len(r["legal"])
    P = {m: np.exp(POL[m][a:a + n].astype(float)) for m in POL}
    st = r["set"]
    for m in MODELS:
        per[st][m].append(P[m] @ F)
    pj = r["legal"].index(r["played"])
    per[st]["played"].append(F[pj])
    for m in MODELS:
        top1[st][f"match_{m}"].append(float(P[m].argmax() == pj))
    for c_, s_ in SEEDS.items():
        top1[st][f"agree_{c_}|{s_}"].append(float(P[c_].argmax() == P[s_].argmax()))
    for c_ in MODELS[1:]:
        top1[st][f"agree_{c_}|base"].append(float(P[c_].argmax() == P["base"].argmax()))
    for a_, b_ in (("karpov_ft", "kasparov_ft"), ("carlsen_ft", "tang_ft")):
        top1[st][f"agree_{a_}|{b_}"].append(float(P[a_].argmax() == P[b_].argmax()))
        if st in ("neutral_otb", "h2h_p2"):
            M = 0.5 * (P[a_] + P[b_])
            js = 0.5 * (P[a_] @ np.log(P[a_] / M + 1e-12) + P[b_] @ np.log(P[b_] / M + 1e-12))
            js_rows.append(dict(set=st, pair=f"{a_}|{b_}", js=float(js), fen=r["fen"], played=r["played"],
                                top_a=[(r["legal"][k], round(float(P[a_][k]), 3)) for k in np.argsort(-P[a_])[:3]],
                                top_b=[(r["legal"][k], round(float(P[b_][k]), 3)) for k in np.argsort(-P[b_])[:3]]))

rng = np.random.default_rng(0)


def diff_ci(x, y, n=1000):
    d = np.asarray(x) - np.asarray(y); idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return d.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


out = {"n": {s: len(v["base"]) for s, v in per.items()}, "features": FEATS, "neutral": {}, "own": {}, "top1": {}}
for st in ("neutral_otb", "h2h_p2"):
    out["neutral"][st] = {"base": np.mean(per[st]["base"], 0).tolist()}
    for m in MODELS[1:]:
        mu, lo, hi = diff_ci(per[st][m], per[st]["base"])
        out["neutral"][st][m] = dict(shift=mu.tolist(), lo=lo.tolist(), hi=hi.tolist())
for p, c in COPY.items():
    st = f"own_{p}"
    real = np.mean(per[st]["played"], 0) - np.mean(per[st]["base"], 0)
    own = np.mean(per[st][c], 0) - np.mean(per[st]["base"], 0)
    other = np.mean(per[st][COPY[PAIR[p]]], 0) - np.mean(per[st]["base"], 0)
    out["own"][p] = dict(real=real.tolist(), copy=own.tolist(), other_copy=other.tolist(),
                         corr_copy=float(np.corrcoef(real, own)[0, 1]), corr_other=float(np.corrcoef(real, other)[0, 1]),
                         sign_agree=float(np.mean(np.sign(real) == np.sign(own))))
out["contrast"] = {}
for A, B in (("karpov", "kasparov"), ("carlsen", "tang")):
    ra = np.mean(per[f"own_{A}"]["played"], 0) - np.mean(per[f"own_{A}"]["base"], 0)
    rb = np.mean(per[f"own_{B}"]["played"], 0) - np.mean(per[f"own_{B}"]["base"], 0)
    real = (ra - rb)[:HAB]
    cc = np.mean([np.mean(per[f"own_{x}"][COPY[A]], 0) - np.mean(per[f"own_{x}"][COPY[B]], 0) for x in (A, B)], 0)[:HAB]
    out["contrast"][f"{A}|{B}"] = dict(real=real.tolist(), copy=cc.tolist(), corr=float(np.corrcoef(real, cc)[0, 1]),
                                      sign_agree=float(np.mean(np.sign(real) == np.sign(cc))))
for st, d in top1.items():
    out["top1"][st] = {k: float(np.mean(v)) for k, v in d.items()}
js_rows.sort(key=lambda x: -x["js"])
out["most_divergent"] = {pr: [x for x in js_rows if x["pair"] == pr][:8] for pr in ("karpov_ft|kasparov_ft", "carlsen_ft|tang_ft")}
out["mean_js"] = {f"{st}|{pr}": float(np.mean([x["js"] for x in js_rows if x["pair"] == pr and x["set"] == st]))
                  for st in ("neutral_otb", "h2h_p2") for pr in ("karpov_ft|kasparov_ft", "carlsen_ft|tang_ft")}
json.dump(out, open(D / "style.json", "w"), indent=1)

# ---------- figure ----------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(14, 8.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
ident = json.load(open(R / "17_identify.json"))
ax = fig.add_subplot(gs[0, 0])
for pr, lab, col in (("p1", "Karpov vs Kasparov", "#8a5a2b"), ("p2", "Carlsen vs Tang", "#2d6cdf")):
    ks = [k for k in ident[pr]["by_k"] if k != "all"]
    ax.plot([int(k) for k in ks], [ident[pr]["by_k"][k]["auc"] for k in ks], "o-", color=col, label=lab)
ax.axhline(0.5, color="k", lw=0.6); ax.set_xscale("log"); ax.set_ylim(0.45, 0.85)
ax.set_xlabel("player's moves seen (plies ≥ 31)"); ax.set_ylabel("AUC: who played this game?")
ax.set_title("A. Copies tell the two players apart"); ax.legend(frameon=False)

for k, st in enumerate(("neutral_otb", "h2h_p2")):
    ax = fig.add_subplot(gs[0, 1 + k]); y = np.arange(len(FEATS)); h = 0.2
    for j, m in enumerate(MODELS[1:]):
        s = out["neutral"][st][m]; sh = 100 * np.array(s["shift"]); lo = 100 * np.array(s["lo"]); hi = 100 * np.array(s["hi"])
        ax.barh(y + (j - 1.5) * h, sh, h, color=COL[m], label=m.replace("_ft", " copy").capitalize(),
                xerr=[sh - lo, hi - sh], error_kw=dict(lw=0.6))
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(y, [LABEL[f] for f in FEATS]); ax.invert_yaxis()
    ax.set_xlabel("copy − base (pp; cp loss in cp)")
    ax.set_title("B. Style shift on neutral positions:\n" + ("OTB champions (none of the 4)" if st == "neutral_otb" else "Carlsen–Tang head-to-head"))
    if k == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower right")

ax = fig.add_subplot(gs[1, 0:2])
for (pr, o), col in zip(out["contrast"].items(), ("#8a5a2b", "#2d6cdf")):
    A, B = pr.split("|"); x = 100 * np.array(o["real"]); y = 100 * np.array(o["copy"])
    ax.scatter(x, y, color=col, s=26, label=f"{A.capitalize()} − {B.capitalize()}  (r = {o['corr']:.2f}, same sign {int(round(8 * o['sign_agree']))}/8)")
    for f, xi, yi in zip(FEATS[:HAB], x, y):
        ax.annotate(LABEL[f], (xi, yi), fontsize=7, color=col, xytext=(3, 2), textcoords="offset points")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("real players: (A played − base) − (B played − base), on their own positions (pp)")
ax.set_ylabel("copies: A copy − B copy, same positions (pp)")
ax.set_title("C. Do the copies differ from each other the way the players do? (move habits)")
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[1, 2]); t = out["top1"]["neutral_otb"]
items = [("Karpov copy vs base", t["agree_karpov_ft|base"], "#cccccc"), ("Kasparov copy vs base", t["agree_kasparov_ft|base"], "#cccccc"),
         ("Karpov vs Kasparov copy", t["agree_karpov_ft|kasparov_ft"], "#8a5a2b")]
items += [(f"Karpov copy, 2 seeds", t["agree_karpov_ft|karpov_s1"], "#e3c9a8")] if "agree_karpov_ft|karpov_s1" in t else []
items += [("Carlsen copy vs base", t["agree_carlsen_ft|base"], "#cccccc"), ("Tang copy vs base", t["agree_tang_ft|base"], "#cccccc"),
          ("Carlsen vs Tang copy", t["agree_carlsen_ft|tang_ft"], "#2d6cdf")]
items += [(f"Carlsen copy, 2 seeds", t["agree_carlsen_ft|carlsen_s1"], "#b9d2f5")] if "agree_carlsen_ft|carlsen_s1" in t else []
ax.barh(range(len(items)), [100 * v for _, v, _ in items], color=[c for *_, c in items])
ax.set_yticks(range(len(items)), [l for l, *_ in items]); ax.invert_yaxis(); ax.set_xlim(70, 100)
for i, (_, v, _) in enumerate(items): ax.text(100 * v + 0.2, i, f"{100 * v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("same top-1 move on neutral OTB positions (%)")
ax.set_title("D. Same top-1 move, two models")
fig.tight_layout(); fig.savefig(R / "figs/style.png", dpi=160)

print(json.dumps({k: out[k] for k in ("n", "top1", "mean_js")}, indent=1))
for st in ("neutral_otb", "h2h_p2"):
    print(st, "base", dict(zip(FEATS, np.round(out["neutral"][st]["base"], 4))))
    for m in MODELS[1:]:
        print("  ", m, " ".join(f"{f}:{100*v:+.2f}" for f, v in zip(FEATS, out["neutral"][st][m]["shift"])))
for k, o in out["contrast"].items():
    print("CONTRAST", k, "corr %.2f sign %.2f" % (o["corr"], o["sign_agree"]))
    print("   real", " ".join(f"{f}:{100*v:+.2f}" for f, v in zip(FEATS, o["real"])))
    print("   copy", " ".join(f"{f}:{100*v:+.2f}" for f, v in zip(FEATS, o["copy"])))
for p in COPY:
    o = out["own"][p]; print(p, "corr copy %.2f other %.2f sign %.2f" % (o["corr_copy"], o["corr_other"], o["sign_agree"]))
    print("   real", " ".join(f"{f}:{100*v:+.2f}" for f, v in zip(FEATS, o["real"])))
    print("   copy", " ".join(f"{f}:{100*v:+.2f}" for f, v in zip(FEATS, o["copy"])))
