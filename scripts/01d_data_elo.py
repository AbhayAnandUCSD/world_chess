"""Build WC_ROOT/data/games.jsonl for the Elo-band experiment (see wc.data_lichess_db).
Run with WC_ROOT=exp_elo WC_CONFIG=elo after scripts/01c_lichess_db_filter.py has written
WC_ROOT/data/raw_db/{high,low}_<month>.pgn."""
import collections
import json
import pathlib
import runpy

from wc import config
from wc.data_lichess_db import build
from wc.model import CODE_ROOT, ROOT
from wc.score import player_plies

C = config.load(); L = C["lichess_db"]
games, rep = build(ROOT / "data" / "raw_db", L["months"], n_per_band=L["n_per_band"], clock_min_s=L["clock_min_s"],
                   max_per_user=L["max_per_user"],
                   other_from={"raw_dir": CODE_ROOT / "data" / "raw_other", "exclude_names": L["exclude_from_probe_pool"]})
print("report", dict(rep))
cnt = collections.Counter((g.player, g.color, g.split) for g in games)
for k in sorted(cnt):
    print(k, cnt[k])
summary = {"report": dict(rep)}
for p in C["players"]:
    gs = [g for g in games if g.player == p]
    pol = [g for g in gs if g.split == "policy"]
    tot = sum(len(player_plies(g, p, 1)) for g in pol)
    raw = sum(len(range(0 if g.color == "w" else 1, len(g.sans) - 1, 2)) for g in pol)
    users = collections.Counter(g.meta["target_user"] for g in gs)
    elos = sorted(int(e) for g in gs for e in g.meta["elo"].values() if e and e.isdigit())
    summary[p] = dict(games=len(gs), policy_games=len(pol), target_moves=tot, skipped_frac=1 - tot / raw,
                      tc=collections.Counter(g.meta["tc"] for g in gs).most_common(8), n_users=len(users),
                      top_users=users.most_common(10), elo_median=elos[len(elos) // 2],
                      elo_p5_p95=[elos[len(elos) // 20], elos[-len(elos) // 20]])
    print(p, json.dumps(summary[p]))
L_ = [len(g.text) for g in games if g.player != "other"]
print("median chars", sorted(L_)[len(L_) // 2], "frac fits 1023:", sum(l <= 1023 for l in L_) / len(L_))
(ROOT / "results").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(ROOT / "results" / "01_data_summary.json", "w"), indent=1, default=str)
runpy.run_path(str(pathlib.Path(__file__).with_name("01b_dedupe_test.py")))
