"""Style analysis, step 2 (GPU): each model's probability over every legal move of every position.

log P(move) = sum of char log-probs of SAN + trailing space after the exact game prefix, then
renormalised over the legal moves. Output: results/18/policy_<name>.npy, a flat float32 array of
log-probs aligned with the concatenated `legal` lists of positions.jsonl.

Usage: python scripts/18b_style_policy.py NAME CKPT
"""
import json
import sys

import numpy as np
import torch

from wc.model import CODE_ROOT, encode, load_model, pick_device

name, ck = sys.argv[1], sys.argv[2]
OUT = CODE_ROOT / "results/18"
dev = pick_device()
m = load_model(ck, dev).eval()
rows = [json.loads(l) for l in open(OUT / "positions.jsonl")]
res = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
    for k, r in enumerate(rows):
        pre = encode(r["prefix"]); seqs = [pre + encode(s + " ") for s in r["legal"]]
        T = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
        lp = m(ids[:, :-1].to(dev)).float().log_softmax(-1)
        tgt = ids[:, 1:].to(dev)
        tok = lp.gather(-1, tgt[..., None])[..., 0]
        mask = torch.zeros_like(tok, dtype=torch.bool)
        for j, s in enumerate(seqs):
            mask[j, len(pre) - 1:len(s) - 1] = True
        sc = (tok * mask).sum(1)
        res.append((sc - sc.logsumexp(0)).cpu().numpy())
        if k % 1000 == 0:
            print(name, k, flush=True)
np.save(OUT / f"policy_{name}.npy", np.concatenate(res).astype(np.float32))
print(name, "done")
