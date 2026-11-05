"""Lichess API PGNs -> the same Game records / splits as the PGN Mentor pipeline.

Preprocessing (each step counted in the build report):
  - keep: Variant Standard, rated, no setup FEN, chosen TimeControl(s), Termination in {Normal, Time forfeit}
  - drop: games vs BOT accounts, games where neither/both sides are a target player (both -> h2h reserve)
  - moves re-serialised to SAN by python-chess from the mainline -> ';1.e4 e5 2.Nf3 ...' (clock comments,
    '1...' black numbers, annotations, results all removed by construction)
  - per-ply clocks kept in meta; plies where the MOVER had < clock_min_s seconds BEFORE the move are put in
    meta.skip_plies -> excluded from training targets and from scoring (premove / time-scramble noise)
  - dedupe on exact move sequence; splits by whole game stratified by (player, colour), same fractions as P1
  - player identity = exact Lichess username (case-insensitive); colour stored explicitly in meta.colors
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import chess
import chess.pgn

from .data import PROC, Game, to_text
from .model import CODE_ROOT, ROOT

RAW = ROOT / "data" / "raw_lichess"
FRAC = (("policy", 0.70), ("probe_fit", 0.10), ("probe_dev", 0.05), ("test", 0.15))


def _clock_before(clocks, ply, base_s):
    """Mover's remaining time before playing ply (0-based); clocks[i] = mover's clock after ply i."""
    prev = ply - 2
    return base_s if prev < 0 else clocks[prev]


def build(players: dict[str, str], time_controls: tuple[str, ...], clock_min_s: float = 10.0,
          min_plies: int = 2, seed: int = 0, other_from: Path | None = None):
    """players: {key: lichess_username}. Returns (games, report)."""
    user2key = {u.lower(): k for k, u in players.items()}
    rep = collections.Counter()
    seen, games = set(), []
    for k, u in players.items():
        with open(RAW / f"{u}.pgn", encoding="utf-8") as f:
            while (g := chess.pgn.read_game(f)) is not None:
                h = g.headers; rep["read"] += 1
                if h.get("Variant", "Standard") != "Standard" or "FEN" in h or h.get("SetUp") == "1":
                    rep["drop_variant_or_setup"] += 1; continue
                # rated-ness is guaranteed by the API query (rated=true); arena/league games are rated too
                if h.get("TimeControl") not in time_controls:
                    rep["drop_timecontrol"] += 1; continue
                if h.get("Termination") not in ("Normal", "Time forfeit"):
                    rep["drop_termination"] += 1; continue
                if "BOT" in (h.get("WhiteTitle", ""), h.get("BlackTitle", "")):
                    rep["drop_bot"] += 1; continue
                if g.errors:
                    rep["drop_parse_error"] += 1; continue
                b = g.board(); sans, clocks = [], []
                ok = True
                for node in g.mainline():
                    try:
                        sans.append(b.san(node.move)); b.push(node.move)
                    except Exception:
                        ok = False; break
                    c = node.clock()
                    clocks.append(float(c) if c is not None else None)
                if not ok:
                    rep["drop_illegal"] += 1; continue
                if len(sans) < min_plies:
                    rep["drop_short"] += 1; continue
                if any(c is None for c in clocks):
                    rep["drop_missing_clock"] += 1; continue
                # clock anomalies (e.g. Lichess 'give +15 s'): any clock above the base time, or a side's
                # clock rising by > 2 s between its own moves, means the time control was not really 60+0
                b0 = float(h["TimeControl"].split("+")[0])
                if any(c > b0 + 0.5 for c in clocks) or any(clocks[i] > clocks[i - 2] + 2 for i in range(2, len(clocks))):
                    rep["drop_clock_anomaly"] += 1; continue
                text, spans = to_text(sans)
                key = hashlib.sha1(text.encode()).hexdigest()
                if key in seen:
                    rep["drop_duplicate"] += 1; continue  # also removes h2h games fetched from both accounts
                seen.add(key)
                w, bl = user2key.get(h["White"].lower()), user2key.get(h["Black"].lower())
                base_s = float(h["TimeControl"].split("+")[0])
                berserk = {"w": clocks[0] <= base_s / 2 + 1, "b": len(clocks) > 1 and clocks[1] <= base_s / 2 + 1}
                # a berserked side starts with half the clock: use its real starting time for its first move
                start = {0: clocks[0] if clocks[0] <= base_s / 2 + 1 else base_s,
                         1: (clocks[1] if len(clocks) > 1 and clocks[1] <= base_s / 2 + 1 else base_s)}
                skip = [p for p in range(len(sans)) if (start[p % 2] if p < 2 else clocks[p - 2]) < clock_min_s]
                colors = {}
                if w: colors[w] = "w"
                if bl: colors[bl] = "b"
                if w and bl:
                    player, color = "both", ""
                elif w or bl:
                    player, color = (w, "w") if w else (bl, "b")
                else:
                    rep["drop_no_target_player"] += 1; continue
                games.append(Game(h.get("GameId", key[:12]), text, sans, spans, h["White"], h["Black"], h.get("ECO", ""),
                                  h.get("UTCDate", ""), h.get("Event", ""), player, color,
                                  meta=dict(event_class=f"bullet_{h['TimeControl']}", colors=colors, skip_plies=skip,
                                            clocks=clocks, elo={"w": h.get("WhiteElo"), "b": h.get("BlackElo")},
                                            termination=h.get("Termination"), opening=h.get("Opening", ""), berserk=berserk,
                                            arena="arena" in h.get("Event", "").lower())))
                rep["kept"] += 1
    rng = random.Random(seed)
    for p in players:
        for c in ("w", "b"):
            grp = sorted([g for g in games if g.player == p and g.color == c], key=lambda g: g.gid)
            rng.shuffle(grp)
            n, i = len(grp), 0
            for kk, (name, fr) in enumerate(FRAC):
                j = n if kk == len(FRAC) - 1 else i + round(fr * n)
                for g in grp[i:j]:
                    g.split = name
                i = j
    for g in games:
        if g.player == "both":
            g.split = "h2h"
    # player-neutral probe-fit pool, same recipe as P1 (3000 OTB games of other champions, PGN Mentor files),
    # excluding ANY game involving a target player of this experiment or of P1 (name regex on both sides)
    if other_from is not None:
        import re
        from .data import parse_file
        excl = re.compile(other_from["exclude_names"], re.I)
        pool, seen_o = [], set(seen)
        for fn in sorted(Path(other_from["raw_dir"]).glob("*.pgn")):
            for hh, sans in parse_file(fn):
                if excl.search(hh.get("White", "") + "|" + hh.get("Black", "")):
                    rep["other_excluded_by_name"] += 1; continue
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
