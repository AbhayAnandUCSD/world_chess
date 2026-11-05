"""Build data/games.jsonl and print split counts + context coverage."""
import collections
from wc.data import build

games = build()
cnt = collections.Counter((g.player, g.color, g.split) for g in games)
for k in sorted(cnt):
    print(k, cnt[k])
L = [len(g.text) for g in games]
print("games", len(games), "median chars", sorted(L)[len(L)//2], "frac fits 1023:", sum(l <= 1023 for l in L) / len(L))
g = next(x for x in games if x.player == "karpov")
print(g.text[:120]); print(g.sans[:4], [g.text[a:b] for a, b in g.spans[:4]])

# quarantine near-duplicate test/h2h games (see scripts/01b_dedupe_test.py)
import runpy, pathlib
runpy.run_path(str(pathlib.Path(__file__).with_name("01b_dedupe_test.py")))
