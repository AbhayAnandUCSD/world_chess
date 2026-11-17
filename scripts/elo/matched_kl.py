"""Compare all six specialists using the same probe and positions in each domain."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wc.model import CODE_ROOT, ROOT

domains = {
    "otb": (CODE_ROOT / "results", ["karpov","kasparov"]),
    "bullet": (CODE_ROOT / "exp_p2/results", ["carlsen","tang"]),
}
players = ["karpov","kasparov","carlsen","tang","low","high"]
colors = ["#8a5a2b","#d08a3c","#2d6cdf","#6aa5f0","#993399","#359b79"]
out = {}
fig, axes = plt.subplots(2,2,figsize=(12,8))
for j,(domain,(root,own)) in enumerate(domains.items()):
    for player,color in zip(players,colors):
        source = root / "09" / f"{'traj' if player in own else 'xdom'}_{player}.json"
        rows=json.loads(source.read_text())["rows"]
        if player in ("low", "high"):
            assert len(rows) == 12, f"Incomplete cross-domain evaluation: {source}"
        curves={}
        for kind,style in [("dose","-"),("hot","--")]:
            selected=sorted([r for r in rows if f"_p15_{kind}/" in r["ckpt"]],
                            key=lambda r:r["policy"]["kl_base_copy"])
            assert selected
            points=([(0.,0.,0.)] if kind=="dose" else [])+[(r["policy"]["kl_base_copy"],r["frozen"]["delta_pp"],
                                 r["refit"]["delta_pp"]) for r in selected]
            points=np.array(points)
            curves[kind]=dict(points=points.tolist(),matched={
                str(k):dict(frozen=float(np.interp(k,points[:,0],points[:,1])),
                            refit=float(np.interp(k,points[:,0],points[:,2])))
                for k in [.005,.01,.02,.04] if points[:,0].min()<=k<=points[:,0].max()})
            for i,metric in enumerate(["Frozen probe","Refitted probe"]):
                axes[i,j].plot(points[:,0],points[:,i+1],style,color=color,marker="o",ms=3,
                               label=player.capitalize() if kind=="dose" else None)
                axes[i,j].set(xlabel="KL(base || copy) per target character",
                              ylabel="Extra occupied-square error (pp)",title=f"{metric}, {domain} positions",
                              xscale="symlog")
                axes[i,j].set_xscale("symlog", linthresh=0.001)
        out[f"{domain}|{player}"]=curves
    for i in range(2):
        axes[i,j].axhline(0,color="gray",lw=.6)
        axes[i,j].axhline(1,color="gray",lw=.6,ls=":")
axes[0,0].legend(fontsize=8,ncol=2)
fig.tight_layout()
fig.savefig(ROOT/"results/figs/matched_kl_all_six.png",dpi=170)
(ROOT/"results/16_matched_kl_all_six.json").write_text(json.dumps(out,indent=2))
