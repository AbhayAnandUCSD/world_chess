"""Paired game-bootstrap uncertainty and selected chess illustrations."""
import json
from collections import defaultdict
import numpy as np
import chess
import chess.svg
from wc.model import ROOT

R=ROOT/"results"
rows=json.loads((R/"18/per_position.json").read_text())
positions={r["i"]:r for r in map(json.loads,(R/"18/positions.jsonl").open())}
rng=np.random.default_rng(2026)
def interval(records, value):
    groups=defaultdict(list)
    for r in records:
        groups[r["gid"]].append(value(r))
    counts=np.array([len(v) for v in groups.values()])
    sums=np.array([sum(v) for v in groups.values()])
    draws=rng.multinomial(len(counts), np.full(len(counts),1/len(counts)),size=2000)
    estimates=(draws@sums)/(draws@counts)
    return dict(mean=float(sums.sum()/counts.sum()),ci=np.percentile(estimates,[2.5,97.5]).tolist(),
                n_positions=int(counts.sum()),n_games=len(counts))
out={}
for st in ["neutral_otb","h2h_p2","own_low","own_high"]:
    selected=[r for r in rows if r["set"]==st]
    assert all("engine" in r for r in selected)
    res={}
    for a,b in [("low_ft","high_ft"),("low_ft","low_s1"),("high_ft","high_s1"),
                ("low_ft","base"),("high_ft","base")]:
        res[f"disagreement_{a}|{b}"]=interval(selected,lambda r:float(r["tops"][a]!=r["tops"][b]))
    for model in ["low_ft","high_ft"]:
        for metric in ["top_loss_cp","expected_loss_cp","blunder_probability","best_probability","only_move_for_opponent"]:
            res[f"{model}-base/{metric}"]=interval(selected,lambda r:r["engine"][model][metric]-r["engine"]["base"][metric])
        sole=[r for r in selected if r["sole_best"]]
        res[f"{model}-base/sole_best"]=interval(sole,lambda r:float(r["hits"][model])-float(r["hits"]["base"]))
    out[st]=res
(R/"18/quality_intervals.json").write_text(json.dumps(out,indent=2))

# Select reproducible illustrations after aggregate statistics, with both gains and losses.
examples=[]
for band in ["low","high"]:
    model=f"{band}_ft"
    candidates=[r for r in rows if r["set"]==f"own_{band}" and r["tops"][model]==r["tops"][f"{band}_s1"]
                and r["tops"][model]!=r["tops"]["base"]]
    for kind,sign in [("gain",-1),("loss",1)]:
        candidates_kind=[r for r in candidates if sign*(r["engine"][model]["top_loss_cp"]-r["engine"]["base"]["top_loss_cp"])>=100]
        if not candidates_kind:
            continue
        selected=max(candidates_kind,key=lambda r:sign*(r["engine"][model]["top_loss_cp"]-r["engine"]["base"]["top_loss_cp"]))
        p=positions[selected["i"]]
        board=chess.Board(p["fen"])
        a=board.parse_san(p["legal"][selected["tops"]["base"]])
        b=board.parse_san(p["legal"][selected["tops"][model]])
        path=f"elo_{band}_{kind}.svg"
        (R/"figs"/path).write_text(chess.svg.board(board=board,orientation=board.turn,size=420,
             arrows=[chess.svg.Arrow(a.from_square,a.to_square,color="#c47732"),
                     chess.svg.Arrow(b.from_square,b.to_square,color="#276b9a")]))
        examples.append(dict(band=band,kind=kind,gid=p["gid"],ply=p["ply"],figure=path,
            base_move=p["legal"][selected["tops"]["base"]],copy_move=p["legal"][selected["tops"][model]],
            base_loss_cp=selected["engine"]["base"]["top_loss_cp"],
            copy_loss_cp=selected["engine"][model]["top_loss_cp"],
            actual_move=p["played"],note="Selected depth-12 engine example; both training seeds agree."))
(R/"18/examples.json").write_text(json.dumps(examples,indent=2))
print(json.dumps(out["neutral_otb"],indent=2))
