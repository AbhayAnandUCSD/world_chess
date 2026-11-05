"""Lichess monthly-dump games, split into rating bands -> the same Game records / splits as P1/P2.

Input: per-band PGN files from scripts/01c_lichess_db_filter.py (high_<month>.pgn, low_<month>.pgn).
Each band is one "player" (key = band name). Per game, ONE side is the target (chosen by a hash of the game id,
so ~50/50 colours); its username is kept in meta.target_user.

Filters (counted in the report), same recipe as wc.data_lichess (P2) unless noted:
  - standard variant, no setup FEN, Termination Normal / Time forfeit, legal SAN replay, clocks on every ply
  - clock anomalies: a clock above base time, or a side's clock rising by more than increment + 2 s
  - moves made with < clock_min_s on the mover's clock (berserk-aware start) -> meta.skip_plies
  - exact-duplicate move sequences dropped
  - NEW: at most max_per_user games per target account per band (stops a few prolific accounts dominating)
  - NEW: time controls with base < 60 s (hyperbullet, mostly premoves) are dropped
  - NEW: the low band is sampled to the high band's exact time-control histogram (same size, same TC mix)
Splits: 70/10/5/15 of games, grouped by target account (no account in two splits). No head-to-head set (bands never meet).
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import chess
import chess.pgn

from .data import PROC, Game, parse_file, to_text

FRAC = (("policy", 0.70), ("probe_fit", 0.10), ("probe_dev", 0.05), ("test", 0.15))


def _raw_games(path):
    """Split a PGN file into raw per-game strings (cheap); parse lazily with _pg()."""
    txt = Path(path).read_text(encoding="utf-8")
    return [("[Event" + x) if not x.startswith("[Event") else x for x in re.split(r"\n\n(?=\[Event )", txt.strip()) if x.strip()]


_TC = re.compile(r'\[TimeControl "([^"]*)"\]')


def _pg(raw):
    return chess.pgn.read_game(io.StringIO(raw))


def _parse(g, rep, band, clock_min_s, min_plies, min_base_s=60):
    h = g.headers
    if h.get("Variant", "Standard") != "Standard" or "FEN" in h or h.get("SetUp") == "1":
        rep[f"{band}_drop_variant_or_setup"] += 1; return None
    if h.get("Termination") not in ("Normal", "Time forfeit"):
        rep[f"{band}_drop_termination"] += 1; return None
    if g.errors:
        rep[f"{band}_drop_parse_error"] += 1; return None
    try:
        b0, inc = (float(x) for x in h["TimeControl"].split("+"))
    except Exception:
        rep[f"{band}_drop_timecontrol"] += 1; return None
    if b0 < min_base_s:  # hyperbullet (15+0, 30+0): mostly premoves
        rep[f"{band}_drop_tc_too_fast"] += 1; return None
    b = g.board(); sans, clocks = [], []
    for node in g.mainline():
        try:
            sans.append(b.san(node.move)); b.push(node.move)
        except Exception:
            rep[f"{band}_drop_illegal"] += 1; return None
        c = node.clock(); clocks.append(float(c) if c is not None else None)
    if len(sans) < min_plies:
        rep[f"{band}_drop_short"] += 1; return None
    if any(c is None for c in clocks):
        rep[f"{band}_drop_missing_clock"] += 1; return None
    if any(c > b0 + 0.5 for c in clocks) or any(clocks[i] > clocks[i - 2] + inc + 2 for i in range(2, len(clocks))):
        rep[f"{band}_drop_clock_anomaly"] += 1; return None
    half = b0 / 2 + 1
    start = {0: clocks[0] if clocks[0] <= half else b0, 1: clocks[1] if len(clocks) > 1 and clocks[1] <= half else b0}
    skip = [p for p in range(len(sans)) if (start[p % 2] if p < 2 else clocks[p - 2]) < clock_min_s]
    gid = h.get("Site", "").rsplit("/", 1)[-1] or hashlib.sha1(" ".join(sans).encode()).hexdigest()[:12]
    color = "w" if int(hashlib.sha1(gid.encode()).hexdigest()[:8], 16) % 2 == 0 else "b"
    text, spans = to_text(sans)
    return Game(gid, text, sans, spans, h["White"], h["Black"], h.get("ECO", ""), h.get("UTCDate", ""), h.get("Event", ""),
                band, color,
                meta=dict(event_class=f"lichess_{h['TimeControl']}", tc=h["TimeControl"], colors={band: color},
                          skip_plies=skip, clocks=clocks, elo={"w": h.get("WhiteElo"), "b": h.get("BlackElo")},
                          target_user=(h["White"] if color == "w" else h["Black"]).lower(),
                          termination=h.get("Termination"), opening=h.get("Opening", ""),
                          berserk={"w": clocks[0] <= half, "b": len(clocks) > 1 and clocks[1] <= half},
                          arena="arena" in h.get("Event", "").lower()))


def build(raw_dir: Path, months, n_per_band=100_000, clock_min_s=10.0, min_plies=20, max_per_user=1000,
          seed=0, other_from=None):
    rep = collections.Counter(); seen = set()
    rng = random.Random(seed)

    def accept(g, band, per_user):
        key = hashlib.sha1(g.text.encode()).hexdigest()
        if key in seen:
            rep[f"{band}_drop_duplicate"] += 1; return False
        u = g.meta["target_user"]
        if per_user[u] >= max_per_user:
            rep[f"{band}_drop_user_cap"] += 1; return False
        seen.add(key); per_user[u] += 1; return True

    # ---- high band: parse everything, shuffle, cap per user, take n_per_band
    high, pu = [], collections.Counter()
    cand = [r for m in months for r in _raw_games(raw_dir / f"high_{m}.pgn")]
    rep["high_read"] = len(cand); rng.shuffle(cand)
    for raw in cand:
        g = _parse(_pg(raw), rep, "high", clock_min_s, min_plies)
        if g and accept(g, "high", pu):
            high.append(g)
            if len(high) == n_per_band:
                break
    rep["high_kept"] = len(high)
    # ---- low band: same size, same time-control histogram
    quota = collections.Counter(g.meta["tc"] for g in high)
    by_tc = collections.defaultdict(list)
    lows = [r for m in months for r in _raw_games(raw_dir / f"low_{m}.pgn")]
    rep["low_read"] = len(lows); rng.shuffle(lows)
    for raw in lows:
        mt = _TC.search(raw); by_tc[mt.group(1) if mt else ""].append(raw)
    low, pl = [], collections.Counter()
    for tc, need in sorted(quota.items()):
        got = 0
        for raw in by_tc.get(tc, []):
            if got == need:
                break
            g = _parse(_pg(raw), rep, "low", clock_min_s, min_plies)
            if g and accept(g, "low", pl):
                low.append(g); got += 1
        if got < need:
            rep["low_tc_shortfall"] += need - got
    rep["low_kept"] = len(low)
    games = high + low
    # Splits are grouped by TARGET ACCOUNT: an account's games all land in one split, so test players are never
    # trained on (the high band has only ~3k accounts; a per-game split put 99% of test games on training accounts).
    for band in ("low", "high"):
        by_user = collections.defaultdict(list)
        for g in games:
            if g.player == band:
                by_user[g.meta["target_user"]].append(g)
        users = sorted(by_user); random.Random(seed).shuffle(users)
        n = sum(len(v) for v in by_user.values())
        bounds = [sum(fr for _, fr in FRAC[:k + 1]) * n for k in range(len(FRAC))]
        done = 0
        for u in users:
            k = next(k for k, b in enumerate(bounds) if done < b or k == len(bounds) - 1)
            for g in by_user[u]:
                g.split = FRAC[k][0]
            done += len(by_user[u])
        rep[f"{band}_accounts"] = len(users)
    if other_from is not None:  # same player-neutral OTB probe pool recipe as P1/P2
        excl = re.compile(other_from["exclude_names"], re.I)
        pool, seen_o = [], set(seen)
        for fn in sorted(Path(other_from["raw_dir"]).glob("*.pgn")):
            for hh, sans in parse_file(fn):
                if excl.search(hh.get("White", "") + "|" + hh.get("Black", "")):
                    continue
                text, spans = to_text(sans)
                kk = hashlib.sha1(text.encode()).hexdigest()
                if kk in seen_o or len(sans) < 32:
                    continue
                seen_o.add(kk)
                pool.append(Game(kk[:12], text, sans, spans, hh.get("White", ""), hh.get("Black", ""), hh.get("ECO", ""),
                                 hh.get("Date", ""), hh.get("Event", ""), "other", "", "probe_fit_other"))
        pool.sort(key=lambda g: g.gid); random.Random(seed + 1).shuffle(pool)
        games += pool[:3000]; rep["other_pool"] = min(3000, len(pool))
    PROC.parent.mkdir(parents=True, exist_ok=True)
    with open(PROC, "w") as f:
        for g in games:
            f.write(json.dumps(asdict(g)) + "\n")
    return games, rep
