"""PGN -> Karvonen-format strings, per-ply char spans, deterministic splits."""
from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.pgn

from .model import ROOT

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "games.jsonl"
PLAYERS = {"karpov": "Karpov", "kasparov": "Kasparov"}


@dataclass
class Game:
    gid: str
    text: str                     # ";1.e4 e5 2.Nf3 ..." (no headers/result)
    sans: list[str]
    spans: list[tuple[int, int]]  # per ply: [start, end) of SAN chars in text
    white: str
    black: str
    eco: str
    date: str
    event: str
    player: str = ""              # 'karpov' / 'kasparov' / 'both' (head-to-head)
    color: str = ""               # player's color 'w'/'b' ('' for head-to-head)
    split: str = ""
    meta: dict = field(default_factory=dict)


def to_text(sans: list[str]) -> tuple[str, list[tuple[int, int]]]:
    out, spans = [";"], []
    pos = 1
    for i, san in enumerate(sans):
        if i % 2 == 0:
            tok = f"{i // 2 + 1}."
            out.append(tok); pos += len(tok)
        spans.append((pos, pos + len(san)))
        out.append(san); pos += len(san)
        if i != len(sans) - 1:
            out.append(" "); pos += 1
    return "".join(out), spans


import re

# Exact name variants present in PGN Mentor files (audit: notes/data_audit.md).
# Substring matching wrongly caught 'Kasparov,Georgy' (junior) and 'Karpov, Anantoly2'.
NAME_RE = {
    "karpov": re.compile(r"^Karpov,\s*(Anatoly|Ana|An)$"),
    "kasparov": re.compile(r"^Kasparov,\s*(Gary|Garry|G\.?)$"),
}
LOOSE = re.compile(r"karpov|kasparov", re.I)

# Event classifier (heuristic, on Event + Site headers) from the data audit.
EVENT_RULES = [
    ("vs_computer", r"x3d|man-?machine|man vs machine|man-comp|computer|philadelphia m|deep blue|deep junior|fritz"),
    ("advanced", r"man\+comp|advanced"),
    ("online", r"internet|\bicc\b|kasparovchess|chess\.net|wireplay|\bmsn\b|online|chess\.com|lichess|playchess"),
    ("simul", r"\bsim\b|simul|sim1|handicap|ger-kasparov|swz-kasparov|isr-kasparov|hsk-kasparov|consultation sim"),
    ("blindfold", r"blind"),
    ("blitz", r"blitz|5'|g/5\b|g/10\b|kosmos m|bullet"),
    ("rapid", r"rapid|rap\b|rap m|25'|30'|g/\d\d|g25|tv|karpov trophy|trophee|cap d.agde|corsica masters|giants|chess classics|immopar|credit suisse|munich express|intel express|pca/intel|moscow pca-gp|chess@iceland|world cup of rapid|tiebreak|\btb\b|playoff|disney|champions rapid|amber|eur-asia|legends rap|fide-wch k\.o\. f 25'"),
    ("consultation", r"consult|:"),
    ("match", r"exhib|exh\b|exh\d|\bmatch\b|\bm\b|\bm\d|vip game|clash of the titans|casual|telex|telechess"),
]
KEEP_EVENTS = {"classical", "match", "rapid", "blitz"}  # blitz added for data volume; tagged in meta.event_class


def event_class(h) -> str:
    if ":" in h.get("White", "") + h.get("Black", ""):
        return "consultation"
    s = (h.get("Event", "") + " | " + h.get("Site", "")).lower()
    for name, pat in EVENT_RULES:
        if re.search(pat, s):
            return name
    return "classical"


def who(name: str) -> str | None:
    for k, r in NAME_RE.items():
        if r.match(name.strip()):
            return k
    return None


def parse_file(path: Path):
    with open(path, encoding="latin-1") as f:
        while (g := chess.pgn.read_game(f)) is not None:
            if g.headers.get("SetUp") == "1" or "FEN" in g.headers or g.errors:
                continue
            b = g.board(); sans = []
            for mv in g.mainline_moves():
                sans.append(b.san(mv)); b.push(mv)
            if len(sans) < 2:
                continue
            yield g.headers, sans


def build(seed: int = 0, n_other: int = 3000, frac=(("policy", 0.70), ("probe_fit", 0.10), ("probe_dev", 0.05), ("test", 0.15))):
    seen, games = set(), []
    for fn in ("Karpov.pgn", "Kasparov.pgn"):
        for h, sans in parse_file(RAW / fn):
            text, spans = to_text(sans)
            w, bl = who(h.get("White", "")), who(h.get("Black", ""))
            ev = event_class(h)
            if ev not in KEEP_EVENTS:
                continue
            key = hashlib.sha1((text + "|" + h.get("White", "") + h.get("Black", "")).encode()).hexdigest()[:12]
            mkey = hashlib.sha1(text.encode()).hexdigest()  # dedupe on move sequence
            if mkey in seen:
                continue
            seen.add(mkey)
            if w and bl and w != bl:
                player, color = "both", ""
            elif w:
                player, color = w, "w"
            elif bl:
                player, color = bl, "b"
            else:
                continue
            games.append(Game(key, text, sans, spans, h.get("White", ""), h.get("Black", ""),
                              h.get("ECO", ""), h.get("Date", ""), h.get("Event", ""), player, color,
                              meta={"event_class": ev}))
    # split by whole game, stratified by (player, color)
    rng = random.Random(seed)
    for p in ("karpov", "kasparov"):
        for c in ("w", "b"):
            grp = sorted([g for g in games if g.player == p and g.color == c], key=lambda g: g.gid)
            rng.shuffle(grp)
            n, i = len(grp), 0
            for k, (name, fr) in enumerate(frac):
                j = n if k == len(frac) - 1 else i + round(fr * n)
                for g in grp[i:j]:
                    g.split = name
                i = j
    for g in games:
        if g.player == "both":
            g.split = "h2h"
    # extra player-neutral probe-fit games from other world champions
    orng = random.Random(seed + 1)
    other = []
    for fn in sorted((ROOT / "data" / "raw_other").glob("*.pgn")):
        for h, sans in parse_file(fn):
            if LOOSE.search(h.get("White", "") + h.get("Black", "")):  # keep other pool free of any K/K
                continue
            text, spans = to_text(sans)
            mkey = hashlib.sha1(text.encode()).hexdigest()
            if mkey in seen or len(sans) < 32:
                continue
            seen.add(mkey)
            other.append(Game(mkey[:12], text, sans, spans, h.get("White", ""), h.get("Black", ""),
                              h.get("ECO", ""), h.get("Date", ""), h.get("Event", ""), "other", "", "probe_fit_other"))
    other.sort(key=lambda g: g.gid); orng.shuffle(other)
    games += other[:n_other]
    PROC.parent.mkdir(parents=True, exist_ok=True)
    with open(PROC, "w") as f:
        for g in games:
            f.write(json.dumps(asdict(g)) + "\n")
    return games


def load_games(split: str | None = None, player: str | None = None, path=None) -> list[Game]:
    out = []
    with open(path or PROC) as f:
        for line in f:
            d = json.loads(line)
            d["spans"] = [tuple(s) for s in d["spans"]]
            g = Game(**d)
            if (split is None or g.split == split) and (player is None or g.player == player):
                out.append(g)
    return out
