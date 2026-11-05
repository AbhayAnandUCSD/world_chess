"""Rollouts: let a model play on from shared start positions against a fixed opponent (the base model).

Both sides play greedily (top move among the legal moves, scored as in 18b: sum of char log-probs of
SAN + trailing space). With a deterministic opponent, each model's game from a start position is fixed,
so two models' games can be compared move by move: where they first split, and how different the boards
are by the end.

Start positions: the first N of each of neutral_otb and h2h_p2 in results/18/positions.jsonl.
The model under test plays the side to move at the start; the base plays the other side.
Stops after PLIES plies, at game end, or when the text would exceed the 1023-char context.

Usage: python scripts/19_rollouts.py NAME CKPT [N_PER_SET=300] [PLIES=30]
Writes results/19/rollout_<NAME>.jsonl
"""
import json
import os
import sys

import chess
import torch

from wc.model import CODE_ROOT, encode, load_model, pick_device
from wc.score import CTX

name, ck = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
PLIES = int(sys.argv[4]) if len(sys.argv) > 4 else 30
OUT = CODE_ROOT / "results/19"; OUT.mkdir(parents=True, exist_ok=True)
dev = pick_device()
player = load_model(ck, dev).eval()
opp = load_model("lichess_8layers_ckpt_no_optimizer.pt", dev).eval()

rows = [json.loads(l) for l in open(CODE_ROOT / "results/18/positions.jsonl")]
starts = [r for r in rows if r["set"] == "neutral_otb"][:N] + [r for r in rows if r["set"] == "h2h_p2"][:N]


@torch.no_grad()
def best_moves(model, texts, boards, max_seqs=int(os.environ.get("MAXSEQ", 768))):
    """Greedy legal move for each (text, board)."""
    cands = []  # (game index, san, token list, prefix len)
    for gi, (t, b) in enumerate(zip(texts, boards)):
        pre = encode(t)
        for m in b.legal_moves:
            s = b.san(m)
            cands.append((gi, s, pre + encode(s + " "), len(pre)))
    best = {}
    for k in range(0, len(cands), max_seqs):
        chunk = cands[k:k + max_seqs]
        T = max(len(c[2]) for c in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T - 1, dtype=torch.bool)
        for j, (_, _, seq, lp) in enumerate(chunk):
            ids[j, :len(seq)] = torch.tensor(seq)
            mask[j, lp - 1:len(seq) - 1] = True
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(ids[:, :-1].to(dev))
        tok = logits.float().log_softmax(-1).gather(-1, ids[:, 1:].to(dev)[..., None])[..., 0]
        sc = (tok * mask.to(dev)).sum(1).cpu()
        for j, (gi, s, _, _) in enumerate(chunk):
            if gi not in best or sc[j] > best[gi][1]:
                best[gi] = (s, float(sc[j]))
    return [best[gi][0] for gi in range(len(texts))]


B = 24  # games advanced together
with open(OUT / f"rollout_{name}.jsonl", "w") as f:
    for bi in range(0, len(starts), B):
        chunk = starts[bi:bi + B]
        texts = [r["prefix"] for r in chunk]
        boards = [chess.Board(r["fen"]) for r in chunk]
        moves = [[] for _ in chunk]; fens = [[] for _ in chunk]; end = [None] * len(chunk)
        for ply in range(PLIES):
            live = [i for i in range(len(chunk)) if end[i] is None]
            for i in list(live):
                if boards[i].is_game_over():
                    end[i] = "game_over"; live.remove(i)
                elif len(texts[i]) + 9 > CTX:
                    end[i] = "context"; live.remove(i)
            if not live:
                break
            model = player if ply % 2 == 0 else opp
            picks = best_moves(model, [texts[i] for i in live], [boards[i] for i in live])
            for i, s in zip(live, picks):
                white_moved = boards[i].turn == chess.WHITE
                boards[i].push_san(s)
                texts[i] += s + " " + ("" if white_moved else f"{boards[i].fullmove_number}.")
                moves[i].append(s); fens[i].append(boards[i].board_fen())
        for i, r in enumerate(chunk):
            f.write(json.dumps({"i": r["i"], "set": r["set"], "moves": moves[i], "fens": fens[i],
                                "end": end[i] or "plies"}) + "\n")
        f.flush()
        print(name, bi + len(chunk), "/", len(starts), flush=True)
print(name, "done")
