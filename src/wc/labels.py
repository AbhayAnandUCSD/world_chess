"""Board labels at White-to-move '.' boundaries (Karvonen convention).

In ';1.e4 e5 2.' the final '.' (move N=2) is read; label = board after
Black's move N-1. Classes: 0..12 = piece value + 6, value in -6..6
(white +1..+6 = P N B R Q K, black negative, 0 = empty -> class 6).
Square order: a1, b1, ..., h1, a2, ..., h8 (python-chess index).
"""
from __future__ import annotations

import hashlib

import chess
import numpy as np

EMPTY = 6
PIECE_NAMES = ["k", "q", "r", "b", "n", "p", ".", "P", "N", "B", "R", "Q", "K"]


def board_to_labels(b: chess.Board) -> np.ndarray:
    out = np.full(64, EMPTY, dtype=np.int64)
    for sq, pc in b.piece_map().items():
        out[sq] = EMPTY + (pc.piece_type if pc.color == chess.WHITE else -pc.piece_type)
    return out


def white_dots(text: str, sans: list[str], min_move: int = 2):
    """Yield (char_index_of_dot, move_number, ply_before, labels) for each
    White-to-move dot with move_number >= min_move, while the prefix up to and
    including the dot fits in the 1023 context."""
    b = chess.Board()
    i = 0
    n_ply = len(sans)
    for ply in range(0, n_ply, 2):
        mv_no = ply // 2 + 1
        tok = f"{mv_no}."
        i = text.index(tok, i if ply == 0 else i)
        dot = i + len(tok) - 1
        if mv_no >= min_move and dot < 1023:
            yield dot, mv_no, ply, board_to_labels(b)
        # advance board by white + black
        for p in (ply, ply + 1):
            if p < n_ply:
                b.push_san(sans[p])
        i = dot + 1


def control_labels(sans: list[str], ply: int, real_marginals: np.ndarray) -> np.ndarray:
    """Hewitt&Liang-style control task: label of square s is a fixed random
    function of (last two SAN moves, s), sampled from the real per-square class
    marginal. Only memorisation of surface context can solve it."""
    ctx = "|".join(sans[max(0, ply - 2):ply])
    out = np.empty(64, dtype=np.int64)
    for s in range(64):
        h = int(hashlib.md5(f"{ctx}#{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[s] = int(np.searchsorted(np.cumsum(real_marginals[s]), h, side="right").clip(0, 12))
    return out


def render(labels) -> str:
    rows = []
    for r in range(7, -1, -1):
        rows.append(" ".join(PIECE_NAMES[int(labels[r * 8 + f])] for f in range(8)))
    return "\n".join(rows)
