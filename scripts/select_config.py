"""Pre-registered selection: config+step maximizing mean over players of own-player
dev Δlogp vs base, same (lr, step) for both players. Prints top configs + dev crossover."""
import glob, json, collections, sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "s3"
import os
from wc import config
from wc.model import ROOT
PL = config.load()["players"]; A, B = PL
CFG = ROOT / "configs" / f"{os.environ.get('WC_CONFIG', 'p1')}.yaml"
curves = collections.defaultdict(dict)
for f in glob.glob(str(ROOT / "runs" / f"finetune_*_{TAG}_*.jsonl")):
    name = os.path.basename(f).split("finetune_")[1][:-6]; p, cfg = name.split("_", 1)
    curves[cfg][p] = {e["step"]: e for e in (json.loads(l) for l in open(f)) if f"dev/{A}_logp" in e}
rows = []
for cfg, d in curves.items():
    if set(d) != {A, B}: continue
    BASE = {A: d[A][0][f"dev/{A}_logp"], B: d[A][0][f"dev/{B}_logp"]}   # step 0 = untouched base
    for s in sorted(set(d[A]) & set(d[B])):
        kp, ks = d[A][s], d[B][s]
        own = ((kp[f"dev/{A}_logp"] - BASE[A]) + (ks[f"dev/{B}_logp"] - BASE[B])) / 2
        x_kp = kp[f"dev/{A}_logp"] - ks[f"dev/{A}_logp"]      # A-copy advantage on A's moves
        x_ks = ks[f"dev/{B}_logp"] - kp[f"dev/{B}_logp"]      # B-copy advantage on B's moves
        rows.append((own, cfg, s, x_kp, x_ks))
rows.sort(reverse=True)
print(f"mean_own_Δ  config                 step  crossover({A} moves)  crossover({B} moves)")
for r in rows[:8]:
    print(f"{r[0]:+.4f}   {r[1]:22s} {r[2]:4d}  {r[3]:+.4f}                  {r[4]:+.4f}")
json.dump(dict(zip(["mean_own", "config", "step", "x_A", "x_B"], rows[0])), open(ROOT / "results" / "selected_config.json", "w"), indent=1)
# lock: write selected lr/steps into configs/p1.yaml
import re
sel = rows[0]; lr = float(re.search(r"lr([0-9.e-]+)_dp", sel[1]).group(1)); steps = int(sel[2])
y = open(CFG).read()
y = re.sub(r"(finetune:\n  lr: )\S+", rf"\g<1>{lr:.1e}", y)
y = re.sub(r"(\n  steps: )\d+.*", rf"\g<1>{steps}           # LOCKED by select_config.py ({sel[1]}) on dev", y)
open(CFG, "w").write(y); print("locked lr", lr, "steps", steps)
