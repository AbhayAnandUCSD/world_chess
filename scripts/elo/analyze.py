"""Summarize Elo results with the same definitions and boards as the GM experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc.model import CODE_ROOT, ROOT

R = ROOT / "results"
F = R / "figs"
F.mkdir(exist_ok=True)
def save(name, value):
    (R / name).write_text(json.dumps(value, indent=2, allow_nan=False))

# Rank-based AUC with average ranks for tied scores.
df = pd.read_csv(R / "04_moves_test_s6000.csv")
w = df[df.model.isin(["low_ft", "high_ft"])].pivot(
    index=["player", "gid", "ply"], columns="model", values="logp").dropna().reset_index()
w["difference"] = w.low_ft - w.high_ft
w = w.sort_values(["player", "gid", "ply"])
ident = {}
for k in [1, 2, 5, 10, 20, 40, None]:
    selected = w.groupby(["player", "gid"]).head(k) if k else w
    games = selected.groupby(["player", "gid"]).difference.sum().reset_index()
    positive = games.player.eq("low")
    ranks = games.difference.rank(method="average")
    npos, nneg = positive.sum(), (~positive).sum()
    auc = (ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    ident[str(k or "all")] = dict(auc=float(auc), accuracy=float(
        ((games.difference > 0) == positive).mean()), n_games=len(games))
save("17_identify.json", ident)

fig, ax = plt.subplots(figsize=(6, 4))
for label, data in [("Elo", ident)] + [
    ({"p1": "Karpov / Kasparov", "p2": "Carlsen / Tang"}[key], value["by_k"]) for key, value in json.loads(
        (CODE_ROOT / "results/17_identify.json").read_text()).items()
]:
    keys = [key for key in data if key != "all"]
    ax.plot(list(map(int, keys)), [data[key]["auc"] for key in keys], "o-", label=label)
ax.axhline(.5, color="gray", linewidth=.7)
ax.set(xlabel="Maximum scored moves per game", ylabel="Game identification AUC", xscale="log")
ax.legend()
fig.tight_layout()
fig.savefig(F / "identification.png", dpi=170)
plt.close(fig)

# Own-domain trajectories. Cross-domain comparisons require a shared evaluation probe.
trajectories = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, band in zip(axes, ["low", "high"]):
    rows = json.loads((R / f"09/traj_{band}.json").read_text())["rows"]
    trajectories[band] = {}
    for kind, linestyle in [("dose", "-"), ("hot", "--")]:
        selected = sorted([r for r in rows if f"_p15_{kind}/" in r["ckpt"]],
                          key=lambda r: r["policy"]["kl_base_copy"])
        xs = [r["policy"]["kl_base_copy"] for r in selected]
        trajectories[band][kind] = [
            dict(kl=r["policy"]["kl_base_copy"], frozen=r["frozen"]["delta_pp"],
                 refit=r["refit"]["delta_pp"], rotation=r["procrustes"]["delta_pp"])
            for r in selected]
        for metric, color in [("frozen", "#c8372d"), ("refit", "#2f7d4f"),
                              ("procrustes", "#d08a3c"), ("affine", "#2d6cdf")]:
            ax.plot(xs, [r[metric]["delta_pp"] for r in selected], linestyle,
                    color=color, marker="o", markersize=3, label=metric if kind == "dose" else None)
    ax.axhline(1, color="gray", linestyle=":")
    ax.set(title=f"{band.capitalize()} Elo", xlabel="KL from base per target character",
           ylabel="Extra occupied-square error (percentage points)", xscale="log")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(F / "probe_drift.png", dpi=170)
plt.close(fig)
save("16_own_domain_trajectories.json", trajectories)

D = R / "18"
models = ["base", "low_ft", "high_ft", "low_s1", "high_s1"]
if not all((D / f"policy_{model}.npy").exists() for model in models):
    print("Identification and trajectory figures complete; waiting for policy arrays.")
    raise SystemExit(0)
rows = list(map(json.loads, (D / "positions.jsonl").open()))
policies = {m: np.load(D / f"policy_{m}.npy") for m in models}
total = sum(len(r["legal"]) for r in rows)
assert all(len(p) == total for p in policies.values())
engines = {}
for path in D.glob("engine_*.jsonl"):
    for row in map(json.loads, path.open()):
        engines[row["i"]] = row
pairs = [("low_ft", "high_ft"), ("low_s1", "high_s1"),
         ("low_ft", "low_s1"), ("high_ft", "high_s1"),
         ("low_ft", "base"), ("high_ft", "base")]
features = ["capture", "check", "pawn", "king", "queen", "castle", "opp_half", "trade"]
values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
records = []
offset = 0
for row in rows:
    n = len(row["legal"])
    ps = {m: np.exp(p[offset:offset+n].astype(float)) for m, p in policies.items()}
    offset += n
    assert all(abs(p.sum()-1) < 1e-5 for p in ps.values())
    b = chess.Board(row["fen"])
    fs = []
    for san in row["legal"]:
        move = b.parse_san(san)
        piece, captured = b.piece_at(move.from_square), b.piece_at(move.to_square)
        rank = chess.square_rank(move.to_square)
        fs.append([b.is_capture(move), "+" in san or "#" in san, piece.piece_type == chess.PAWN,
                   piece.piece_type == chess.KING, piece.piece_type == chess.QUEEN, b.is_castling(move),
                   rank >= 4 if b.turn else rank <= 3,
                   captured is not None and values[captured.piece_type] == values[piece.piece_type]])
    fs = np.array(fs, dtype=float)
    tops = {m: int(p.argmax()) for m,p in ps.items()}
    item = dict(i=row["i"], gid=row["gid"], set=row["set"], tops=tops,
                habits={m:(p @ fs).tolist() for m,p in ps.items()},
                played=fs[row["legal"].index(row["played"])].tolist(),
                played_index=row["legal"].index(row["played"]))
    mixture = (ps["low_ft"] + ps["high_ft"])/2
    item["js"] = float(sum(.5*np.sum(ps[m]*np.log((ps[m]+1e-30)/(mixture+1e-30)))
                          for m in ["low_ft","high_ft"]))
    if row["i"] in engines:
        ev = np.array(engines[row["i"]]["eval"])
        assert len(ev) == n
        best = ev.max()
        item["engine"] = {
            m: dict(top_loss_cp=float(best-ev[tops[m]]),
                    expected_loss_cp=float(p @ (best-ev)),
                    blunder_probability=float(p @ (best-ev >= 200)),
                    best_probability=float(p @ (ev == best)),
                    only_move_for_opponent=float(p @ (np.array(engines[row["i"]]["opp_gap"]) >= 150)))
            for m,p in ps.items()}
        item["sole_best"] = bool(n > 1 and np.sort(ev)[-1]-np.sort(ev)[-2] > 50)
        item["hits"] = {m:bool(ev[tops[m]] == best) for m in models}
    records.append(item)
summary = {}
for st in sorted({r["set"] for r in rows}):
    rr = [r for r in records if r["set"] == st]
    s = dict(n=len(rr), disagreement={
        f"{a}|{b}":float(np.mean([r["tops"][a] != r["tops"][b] for r in rr])) for a,b in pairs},
        mean_js=float(np.mean([r["js"] for r in rr])),
        played_move_match={m:float(np.mean([r["tops"][m]==r["played_index"] for r in rr])) for m in models},
        habits={m:np.mean([r["habits"][m] for r in rr],axis=0).tolist() for m in models})
    measured = [r for r in rr if "engine" in r]
    if measured:
        sole = [r for r in measured if r["sole_best"]]
        s["engine_n"] = len(measured)
        s["sole_best_n"] = len(sole)
        s["sole_best_hits"] = {m:sum(r["hits"][m] for r in sole) for m in models}
        s["quality"] = {m:{metric:float(np.mean([r["engine"][m][metric] for r in measured]))
                            for metric in measured[0]["engine"][m]} for m in models}
    summary[st] = s
real, learned = [], []
for band in ["low", "high"]:
    rr = [r for r in records if r["set"] == f"own_{band}"]
    real.append(np.mean([np.array(r["played"])-r["habits"]["base"] for r in rr],axis=0))
    learned.append(np.mean([np.array(r["habits"]["low_ft"])-r["habits"]["high_ft"] for r in rr],axis=0))
contrast_real = real[0]-real[1]
contrast_copy = np.mean(learned,axis=0)
save("18/style.json", dict(sets=summary, features=features,
    contrast=dict(real=contrast_real.tolist(), copy=contrast_copy.tolist(),
                  correlation=float(np.corrcoef(contrast_real,contrast_copy)[0,1]))))
save("18/per_position.json", records)
fig, axes = plt.subplots(1,2,figsize=(11,4))
gm = json.loads((CODE_ROOT/"results/18/style.json").read_text())["top1"]["neutral_otb"]
labels = ["Karpov / Kasparov", "Carlsen / Tang", "Low / High Elo", "Low Elo, two seeds", "High Elo, two seeds"]
rates = [1-gm["agree_karpov_ft|kasparov_ft"],1-gm["agree_carlsen_ft|tang_ft"]] + [
    summary["neutral_otb"]["disagreement"][key] for key in
    ["low_ft|high_ft","low_ft|low_s1","high_ft|high_s1"]]
axes[0].barh(labels,100*np.array(rates))
axes[0].set(xlabel="Different top move (%)",title="Exact same 1,500 neutral boards")
axes[0].invert_yaxis()
axes[1].scatter(100*contrast_real,100*contrast_copy)
for label,x,y in zip(features,100*contrast_real,100*contrast_copy):
    axes[1].annotate(label,(x,y),fontsize=8)
axes[1].axhline(0,color="gray",lw=.5)
axes[1].axvline(0,color="gray",lw=.5)
axes[1].set(xlabel="Low − high player habits relative to base (pp)",
            ylabel="Low − high model preferences (pp)",title="Eight move habits")
fig.tight_layout()
fig.savefig(F/"policy_comparison.png",dpi=170)
plt.close(fig)
print(json.dumps(summary["neutral_otb"],indent=2))
