"""Build WC_ROOT/data/games.jsonl from Lichess API PGNs (see wc.data_lichess). Run with WC_ROOT=exp_p2 WC_CONFIG=p2."""
import collections
import runpy, pathlib

from wc import config
from wc.data_lichess import build
from wc.model import CODE_ROOT
from wc.score import player_plies

C = config.load(); LC = C["lichess"]
games, rep = build(LC["usernames"], tuple(LC["time_controls"]), LC["clock_min_s"], other_from={"raw_dir": CODE_ROOT / "data" / "raw_other",
                                 "exclude_names": LC["exclude_from_probe_pool"]})
print("report", dict(rep))
cnt = collections.Counter((g.player, g.color, g.split) for g in games)
for k in sorted(cnt): print(k, cnt[k])
for p in C["players"]:
    gs = [g for g in games if g.player == p and g.split == "policy"]
    tot = sum(len(player_plies(g, p, 1)) for g in gs)
    raw = sum(len(range(0 if g.color == "w" else 1, len(g.sans) - 1, 2)) for g in gs)
    print(p, "policy games", len(gs), "own moves kept", tot, "of", raw, f"({1 - tot / raw:.1%} skipped by clock/context)")
L = [len(g.text) for g in games if g.player != "other"]
print("median chars", sorted(L)[len(L) // 2], "frac fits 1023:", sum(l <= 1023 for l in L) / len(L))
runpy.run_path(str(pathlib.Path(__file__).with_name("01b_dedupe_test.py")))
