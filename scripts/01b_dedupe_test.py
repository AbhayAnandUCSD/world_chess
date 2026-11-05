"""Move test/h2h games that near-duplicate any game in another split into split 'test_leak'.
Near-duplicate: same player surnames (order-free) and identical first 30 plies, or one
game's move list is a prefix (>=20 plies) of the other's. Training splits are untouched."""
import collections, json, re
from wc.data import PROC

rows = [json.loads(l) for l in open(PROC)]
sur = lambda n: re.split(r"[,\s]", n.strip())[0].lower()
key30 = lambda r: (frozenset((sur(r["white"]), sur(r["black"]))), tuple(r["sans"][:30]))
EVAL = {"test", "h2h"}
others = [r for r in rows if r["split"] not in EVAL]
by30 = collections.defaultdict(list)
for r in others:
    if len(r["sans"]) >= 20:
        by30[key30(r)].append(r)
by20 = collections.defaultdict(list)
for r in others:
    if len(r["sans"]) >= 20:
        by20[tuple(r["sans"][:20])].append(r)
moved = collections.Counter()
for r in rows:
    if r["split"] not in EVAL or len(r["sans"]) < 20:
        continue
    hit = bool(by30.get(key30(r))) if len(r["sans"]) >= 30 else False
    if not hit:
        for o in by20.get(tuple(r["sans"][:20]), []):
            a, b = r["sans"], o["sans"]; n = min(len(a), len(b))
            if a[:n] == b[:n]:
                hit = True; break
    if hit:
        moved[(r["player"], r["split"])] += 1; r["split"] = "test_leak"
with open(PROC, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("moved to test_leak:", dict(moved))
print("remaining:", collections.Counter((r["player"], r["split"]) for r in rows if r["split"] in EVAL))
