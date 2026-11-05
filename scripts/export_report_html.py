"""Build a self-contained report.html from a Claude Docs HTML export.

The export references figures as blob/<id>; this swaps each for the matching local PNG (base64-embedded),
replaces the author chip with a name, and wraps the fragment in a styled standalone page.

Usage: python scripts/export_report_html.py EXPORT_JSON_OR_TXT [OUT=report.html]
"""
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]); out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "report.html")
d = json.loads(src.read_text())
d = d.get("data", d)
body = base64.b64decode(d["bytes_b64"]).decode()

FIGS = {"f806cdfb-32c2": "results/figs/probe_layer_sweep.png", "c61efe19-1705": "results/figs/crossover_test.png",
        "b819426c-4cc0": "results/figs/p15_seeds.png", "83226d3c-00ec": "results/figs/probe_preservation.png",
        "c6ce03cd-ff10": "results/figs/p15_drift_decomposition.png", "dde150c4-a514": "results/figs/p15_geometry.png",
        "69c4b9a0-9822": "results/figs/p2/p2_crossover_test.png", "1dcac570-443a": "results/figs/p2/p2_p15_seeds.png",
        "522d8aa0-f356": "results/figs/p2/p2_p15_drift_decomposition.png", "5bfdd9c3-e07d": "results/figs/joint_comparison.png",
        "2697c12f-3c79": "results/figs/matched_kl.png", "41d13ff9-9c4a": "results/figs/style.png",
        "ae0b9161-d12c": "results/figs/rollouts.png"}
used = re.findall(r'src="blob/([0-9a-f-]+)"', body)
missing = [b for b in used if b not in FIGS]
if missing:
    sys.exit(f"no local figure for blobs {missing}")
for b, p in FIGS.items():
    body = body.replace(f'src="blob/{b}"', 'src="data:image/png;base64,' + base64.b64encode((ROOT / p).read_bytes()).decode() + '"')
body = re.sub(r'<span data-atom="mention"[^>]*>[^<]*</span>', "Abhay Anand", body)

CSS = """
:root{--bg:#fbfaf7;--fg:#1f1d1a;--muted:#5f5b54;--rule:#e4e0d8;--link:#2d5fb3;--code:#f1eee8}
@media (prefers-color-scheme: dark){:root{--bg:#1b1a18;--fg:#ebe7df;--muted:#a8a296;--rule:#3a3732;--link:#8db4f0;--code:#2a2825}}
body{background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0}
main{max-width:880px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:2rem;line-height:1.2;margin:0 0 .4rem}
h2{font-size:1.35rem;margin:2.4rem 0 .6rem;padding-top:.8rem;border-top:1px solid var(--rule)}
h3{font-size:1.1rem;margin:1.6rem 0 .4rem}
a{color:var(--link)} code{background:var(--code);padding:.1em .3em;border-radius:3px;font-size:.9em}
img{max-width:100%;height:auto;display:block;margin:1rem auto;background:#fff;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:.4rem .6rem;text-align:left;vertical-align:top} th{background:var(--code)}
li p{margin:.2rem 0} time{color:var(--muted)}
"""
out.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess Fine-Tuning Board Probes Report</title><style>{CSS}</style></head>
<body><main>
{body}
</main></body></html>
""")
print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(used)} figures, doc rev {json.loads(src.read_text()).get('rev')})")
