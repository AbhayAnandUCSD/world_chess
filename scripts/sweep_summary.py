"""Summarize dev curves of fine-tune runs: best step per run, crossover margin."""
import glob, json, sys
pat = sys.argv[1] if len(sys.argv) > 1 else "runs/finetune_*_*.jsonl"
BASE = {"karpov": -1.9608, "kasparov": -1.9131}
for f in sorted(glob.glob(pat)):
    name = f.split("finetune_")[1][:-6]; own = name.split("_")[0]; other = "kasparov" if own == "karpov" else "karpov"
    ev = [json.loads(l) for l in open(f) if "dev/karpov_logp" in l]
    if not ev: continue
    best = max(ev, key=lambda e: e[f"dev/{own}_logp"])
    curve = " ".join(f"{e['step']}:{e[f'dev/{own}_logp'] - BASE[own]:+.3f}" for e in ev[::3])
    print(f"{name:28s} best step {best['step']:4d}  own Δ {best[f'dev/{own}_logp'] - BASE[own]:+.4f}  other Δ {best[f'dev/{other}_logp'] - BASE[other]:+.4f} | own Δ curve {curve}")
