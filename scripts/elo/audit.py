"""Verify target-account separation and save exact split counts for the trained data."""
import json
from collections import defaultdict
from wc.model import ROOT
accounts=defaultdict(set)
counts=defaultdict(int)
for game in map(json.loads,(ROOT/"data/games.jsonl").open()):
    if game["player"] in ("low","high"):
        key=(game["player"],game["split"])
        accounts[key].add(game["meta"]["target_user"])
        counts[key]+=1
out={}
for band in ["low","high"]:
    overlap=accounts[(band,"test")] & accounts[(band,"policy")]
    assert not overlap
    out[band]=dict(splits={split:count for (player,split),count in counts.items() if player==band},
                   test_policy_target_account_overlap=len(overlap))
(ROOT/"results/data_audit_final.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
