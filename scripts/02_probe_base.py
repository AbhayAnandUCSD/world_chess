"""Base-model board probes: layer sweep x {linear, MLP} x {real, control task}
x {trained base, random-init}. Saves frozen layer-6 linear probe for transfer."""
import json

import numpy as np
import torch

from wc import config, logging as wlog
from wc.acts import collect
from wc.data import load_games
from wc.labels import control_labels
from wc.model import ROOT, load_model, pick_device
from wc.probe import errors, fit, predict, summarize

C = config.load(); P = C["probe"]; dev = pick_device()
fit_games = [g for sp in P["fit_splits"] for g in load_games(sp)]
dev_games = load_games("probe_dev")
run = wlog.init("02_probe_base", C, job_type="probe")

def control(games, D):
    marg = np.stack([np.bincount(D["labels"][:, s], minlength=13) for s in range(64)]).astype(float)
    marg /= marg.sum(1, keepdims=True)
    return np.stack([control_labels(games[g].sans, int(p), marg) for g, p in zip(D["gid_idx"], D["ply"])]), marg

results = []
for mname, ck in [("base", C["base_ckpt"]), ("randinit", C["randinit_ckpt"])]:
    m = load_model(ck, dev)
    Dtr = collect(m, fit_games, P["train_min_move"])
    Dev = collect(m, dev_games, P["eval_min_move"])
    Ytr_c, marg = control(fit_games, Dtr)
    Yev_c = np.stack([control_labels(dev_games[g].sans, int(p), marg) for g, p in zip(Dev["gid_idx"], Dev["ply"])])
    print(mname, "train positions", Dtr["acts"].shape, "dev positions", Dev["acts"].shape)
    for li, L in enumerate(Dtr["layers"]):
        for kind in ("linear", "mlp"):
            for task, ytr, yev in (("real", Dtr["labels"], Dev["labels"]), ("control", Ytr_c, Yev_c)):
                tag = f"{mname}/L{L}/{kind}/{task}"
                p = fit(Dtr["acts"][li], ytr, kind, P["epochs"], P["lr"], P["wd"], bs=P["batch"], device=dev, log=run.log, tag=tag)
                s = summarize(errors(predict(p, Dev["acts"][li], dev), yev))
                # train acc on control task shows memorisation capacity
                s_tr = summarize(errors(predict(p, Dtr["acts"][li], dev), ytr))
                r = dict(model=mname, layer=L, kind=kind, task=task, **s, train_all_acc=s_tr["all_acc"])
                results.append(r); print(r); run.log({f"{tag}/occ_err": s["occ_err"], f"{tag}/all_acc": s["all_acc"]})
                if mname == "base" and kind == "linear" and task == "real" and L == P["layer_resid"]:
                    torch.save(p.state_dict(), ROOT / "ckpts" / "probe_base_frozen.pt")
                    print("saved frozen probe", tag)
    del m

(ROOT / "results").mkdir(exist_ok=True)
json.dump(results, open(ROOT / "results" / "02_probe_base.json", "w"), indent=1)
import wandb
run.log({"probe_sweep": wandb.Table(dataframe=__import__("pandas").DataFrame(results))})
run.finish()
