"""Full-weight fine-tune with loss masked to one player's move chars (+ space)."""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ROOT, encode, load_model, save_model, set_dropout
from .score import CTX, move_logprobs, target_mask


def make_batch(games, player, dev):
    T = min(CTX, max(len(g.text) for g in games) - 1)
    x = torch.zeros(len(games), T, dtype=torch.long)
    y = torch.zeros(len(games), T, dtype=torch.long)
    m = torch.zeros(len(games), T, dtype=torch.bool)
    for j, g in enumerate(games):
        e = encode(g.text[:T + 1])
        n = len(e) - 1
        x[j, :n] = torch.tensor(e[:-1]); y[j, :n] = torch.tensor(e[1:])
        m[j, :n] = torch.from_numpy(target_mask(g, player, T)[:n])
    return x.to(dev), y.to(dev), m.to(dev)


def n_targets(g, player):
    return int(target_mask(g, player, min(CTX, len(g.text) - 1)).sum())


def finetune(player, train_games, dev_sets: dict, cfg: dict, base_ckpt: str, out_dir, run=None, seed=0, log_name=None):
    """dev_sets: {'karpov': games, 'kasparov': games} for live crossover curves."""
    torch.manual_seed(seed); rng = random.Random(seed)
    model = load_model(base_ckpt); set_dropout(model, cfg.get("dropout", 0.0)); model.train()
    dev = next(model.parameters()).device
    cfg = {**cfg, "lr": float(cfg["lr"])}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95))
    steps, warm = cfg["steps"], cfg["warmup_steps"]
    horizon = cfg.get("schedule_steps") or steps  # LR schedule length (can exceed steps: early stop)
    lr_at = lambda s: cfg["lr"] * min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / horizon)))
    ck_steps = ({s_: s_ / steps for s_ in cfg["ckpt_steps"]} if cfg.get("ckpt_steps")
                else {int(round(f * steps)): f for f in cfg["ckpt_fracs"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(ROOT / "runs" / f"finetune_{log_name or player}.jsonl", "w")
    order, oi, chars_seen = [], 0, 0

    def dev_eval(step):
        model.eval()
        r = {"step": step, "chars_seen": chars_seen}
        for p, gs in dev_sets.items():
            rows = move_logprobs(model, gs, p, min_ply=31)
            r[f"dev/{p}_logp"] = float(np.mean([x["logp"] for x in rows]))
        model.train()
        return r

    for step in range(steps + 1):
        if step in ck_steps:
            save_model(model, out_dir / f"step{step}.pt", {"frac": ck_steps[step], "player": player})
        if step % cfg["dev_every"] == 0 or step == steps:
            r = dev_eval(step); print(player, r, flush=True)
            log_f.write(json.dumps(r) + "\n"); log_f.flush()
            if run: run.log(r, step=step)
        if step == steps:
            break
        if oi + cfg["batch_games"] > len(order):
            order = list(range(len(train_games))); rng.shuffle(order); oi = 0
        batch = [train_games[i] for i in order[oi:oi + cfg["batch_games"]]]; oi += cfg["batch_games"]
        x, y, m = make_batch(batch, player, dev)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        logits = model(x)
        loss = F.cross_entropy(logits[m].float(), y[m])
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
        chars_seen += int(m.sum())
        r = {"step": step, "train/loss": loss.item(), "train/lr": lr_at(step), "train/grad_norm": gn.item(),
             "train/target_chars": chars_seen}
        log_f.write(json.dumps(r) + "\n"); log_f.flush()
        if run and step % 10 == 0:
            run.log(r, step=step)
    log_f.close()
    model.eval()
    return model
