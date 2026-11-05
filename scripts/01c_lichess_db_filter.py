"""Header-level filter of a monthly Lichess database dump (.pgn.zst) into small per-band PGN files.

Keeps a game only if: rated standard game, both players' Elo inside the band (no provisional '?'),
no BOT title, Termination in {Normal, Time forfeit}, and the movetext carries [%clk] clocks.
  high : both players >= HIGH_MIN                       -> every matching game
  low  : both players in [LOW_MIN, LOW_MAX)             -> deterministic sample (hash of the game URL < LOW_P),
                                                           enough to later match the high band's time-control mix
Full move parsing / clock filters / dedupe happen later in wc.data_lichess_db.build (python-chess).

Usage: python scripts/01c_lichess_db_filter.py DUMP.pgn.zst OUT_DIR [HIGH_MIN=2300] [LOW_MIN=1500] [LOW_MAX=2000] [LOW_P=0.05]
"""
import collections
import hashlib
import subprocess
import sys
from pathlib import Path

dump, out = Path(sys.argv[1]), Path(sys.argv[2])
HIGH_MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 2300
LOW_MIN = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
LOW_MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
LOW_P = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
out.mkdir(parents=True, exist_ok=True)
month = dump.name.split("_")[-1].split(".")[0]
fh = open(out / f"high_{month}.pgn", "w"); fl = open(out / f"low_{month}.pgn", "w")
rep = collections.Counter()


def header(lines, key):
    p = f"[{key} "
    for l in lines:
        if l.startswith(p):
            return l[len(p) + 1:-2] if l.endswith('"]') else l[len(p):].strip('"]')
    return ""


def elo(s):
    return int(s) if s.isdigit() else None


proc = subprocess.Popen(["zstdcat", str(dump)], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
game, in_moves = [], False
for line in proc.stdout:
    line = line.rstrip("\n")
    game.append(line)
    if line.startswith("1.") or (in_moves is False and line and not line.startswith("[")):
        in_moves = True
    if in_moves and line == "":
        rep["games"] += 1
        hd = [l for l in game if l.startswith("[")]
        mv = " ".join(l for l in game if l and not l.startswith("["))
        we, be = elo(header(hd, "WhiteElo")), elo(header(hd, "BlackElo"))
        ev = header(hd, "Event")
        band = None
        if we is not None and be is not None and "rated" in ev.lower():
            if we >= HIGH_MIN and be >= HIGH_MIN:
                band = "high"
            elif LOW_MIN <= we < LOW_MAX and LOW_MIN <= be < LOW_MAX:
                site = header(hd, "Site")
                if int(hashlib.sha1(site.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < LOW_P:
                    band = "low"
        if band:
            rep[f"{band}_elo_match"] += 1
            if "BOT" in (header(hd, "WhiteTitle"), header(hd, "BlackTitle")):
                rep[f"{band}_drop_bot"] += 1
            elif header(hd, "Termination") not in ("Normal", "Time forfeit"):
                rep[f"{band}_drop_termination"] += 1
            elif "%clk" not in mv:
                rep[f"{band}_drop_no_clock"] += 1
            else:
                (fh if band == "high" else fl).write("\n".join(game) + "\n")
                rep[f"{band}_kept"] += 1
        game, in_moves = [], False
        if rep["games"] % 1_000_000 == 0:
            print(month, dict(rep), flush=True)
proc.wait()
print(month, "FINAL", dict(rep), flush=True)
