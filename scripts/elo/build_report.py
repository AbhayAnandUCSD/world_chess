"""Build the Elo chapter, insert it into WRITEUP.md and the full HTML report."""
import base64
import html
import json
import re
import sys
from pathlib import Path
from wc.model import ROOT, CODE_ROOT

R=ROOT/"results"
target=Path(sys.argv[1]) if len(sys.argv)>1 else CODE_ROOT
style=json.loads((R/"18/style.json").read_text())
quality=json.loads((R/"18/quality_intervals.json").read_text())
ident=json.loads((R/"17_identify.json").read_text())
roll=json.loads((R/"19/rollouts.json").read_text())
matched=json.loads((R/"16_matched_kl_all_six.json").read_text())
neutral=style["sets"]["neutral_otb"]
s=roll["low_ft|high_ft"]
seed_low=roll["low_ft|low_s1"]
seed_high=roll["high_ft|high_s1"]
chapter=f"""### 3j. Third pair (P3) — fine-tuning across Elo bands

I expected a wider rating gap to produce a larger policy difference than the grandmaster pairs. The low- and high-Elo copies chose different moves on **12.3% of the same 1,500 neutral boards**, compared with **14.2% for Karpov–Kasparov and 14.6% for Carlsen–Tang**. This is well above the **3.3–3.7%** difference between two seeds trained on the same Elo band. The frozen probe continued to read the board within the original one-percentage-point tolerance.

#### Data and controls

The low band contains games with both players rated 1500–1999. The high band requires both players to be rated at least 2300. Each pool has 100,000 Lichess games from August–October 2017, with the time-control distribution matched across bands. After account-grouped splitting, the policy pools contain 70,000 low-band and 70,148 high-band games. The audit confirmed zero target-player account overlap between policy training and test in both bands. Eighteen games labelled test_leak are excluded from the test split.

The inherited 3,000-game extra probe-fit pool includes some online events and is treated as mixed-source.

The shared setting was chosen on development games after learning-rate and batch-size/weight-decay sweeps. Both copies used learning rate 3e-6, batch size 16, weight decay 0.1 and 6,000 updates. This matches sampled-game updates; target-character totals can still differ because game lengths differ. The study also saved five training seeds per band, two data-subsample runs per band, and higher-learning-rate trajectories. Identical low-band subsamples were deduplicated in the robustness comparison, leaving five distinct low runs and seven high runs.

For the behavioural comparison, I reused the exact grandmaster evaluation positions and engine scores: 1,500 neutral champion-game boards and 1,000 Carlsen–Tang positions. I additionally sampled 1,200 held-out positions from each Elo band. Legal moves were scored with the same character-level likelihood and trailing-space convention. Stockfish used depth 12 for move scores and depth 8 for reply pressure. Confidence intervals below resample whole games.

These rating bands label the training data. The experiment did not measure a playing Elo for the fine-tuned models. A shared pretrained model and the overlap in reasonable moves across these rating bands could explain the modest separation, but this experiment does not isolate either explanation.

#### Policy difference on identical boards

| Comparison | Different top move on 1,500 neutral boards |
|---|---:|
| Karpov vs Kasparov | 14.2% |
| Carlsen vs Tang | 14.6% |
| Low vs high Elo, seed 0 | 12.3% |
| Low vs high Elo, seed 1 | 12.5% |
| Low Elo, two seeds | 3.7% |
| High Elo, two seeds | 3.3% |
| Low Elo vs base | 8.8% |
| High Elo vs base | 13.0% |

On the 1,000 shared bullet positions, the Elo copies disagree on 11.5% of moves, versus 2.6–2.7% across seeds. On their respective held-out Elo positions, disagreement is 13.0% for the low-band set and 12.4% for the high-band set.

The low-band copy gave its own band’s held-out moves **+0.0297 nats per move** more probability in log space than the high-band copy (95% interval +0.0286 to +0.0309). The high-band copy’s own-band advantage was **+0.0261** (+0.0251 to +0.0270). Both directions were positive in all 35 pairings of the distinct training runs.

The model scores identify the Elo band of a held-out game with **AUC {ident['all']['auc']:.3f}**, using the target player’s moves after move 15. The corresponding earlier results were approximately 0.63 for Karpov–Kasparov and 0.76 for Carlsen–Tang. Using up to ten scored moves, Elo AUC was {ident['10']['auc']:.3f}. This establishes rating-specific prediction preferences.

![Policy disagreement on the same boards and the eight-habit comparison](results/figs/policy_comparison.png)

![Game identification as more moves are observed](results/figs/identification.png)

The Elo-minus-Karpov–Kasparov disagreement difference is −1.87 percentage points (paired game-bootstrap 95% interval −3.99 to +0.14); Elo-minus-Carlsen–Tang is −2.27 points (−4.36 to −0.20).

The eight-habit comparison gives a different result. The correlation between the models’ low-minus-high preferences and the human low-minus-high tendencies relative to the base is **r = {style['contrast']['correlation']:.3f}**, compared with 0.66 and 0.85 in the grandmaster pairs. Thus, this particular habit summary does not reproduce the grandmaster result. Elo specialization is supported by move prediction and game identification; broad claims about learned human playing style would exceed this evidence.

#### Move quality

| Model | Sole near-best move found, out of 380 | Mean top-move loss, centipawns | Expected loss under legal-move probabilities, centipawns |
|---|---:|---:|---:|
| Base | 244 | 65.79 | 103.81 |
| Low Elo | 245 | 66.04 | 101.79 |
| High Elo | 251 | 63.61 | 98.90 |

The high-Elo copy assigned less probability to poor moves. Its expected centipawn loss was 4.91 lower than the base (95% interval 4.04 to 5.84 lower), and its probability of choosing a move at least 200 centipawns below the engine’s best was 0.93 percentage points lower.

For the greedy top move, the high-Elo change was −2.18 centipawns (95% interval −5.71 to +1.23). Its seven extra successes on the 380 sole-near-best positions correspond to +1.84 percentage points (−0.52 to +4.01). Those intervals include zero. The probability distribution improves more clearly than these top-choice measures. These are depth-limited engine evaluations on the shared board set, rather than a measured playing rating.

![Robustness across training seeds and data subsamples](results/figs/p15_seeds.png)

#### Frozen probe and representation

The Elo experiment fitted its own base probe, using the same layer-6 linear recipe with its larger probe-fit pool. On 741,106 held-out positions from 29,001 games, occupied-square error was **4.143% for the base, 4.250% for the low-Elo copy and 4.018% for the high-Elo copy**. Their changes were +0.107 points (95% interval +0.100 to +0.114) and −0.125 points (−0.133 to −0.118). Both stayed within the prespecified one-point tolerance, including the checked intermediate checkpoints at the selected learning rate.

Absolute error rates here use a different probe-fit pool and evaluation set from P1 and P2. The shared-domain comparison below applies the original grandmaster probes to the Elo checkpoints on the original grandmaster test positions. At the selected Elo checkpoints, the original P1 probe changes by −0.171 points for low Elo and −0.228 for high Elo; the original P2 probe changes by −0.134 and −0.341 points. Preservation therefore also holds with the earlier probes and domains.

The higher-learning-rate runs do not always preserve the frozen probe: peak extra error reaches 1.64 points for low Elo and 1.09 for high Elo on the Elo test set. Refitting reduces those peaks to 0.64 and 0.55 points, respectively. Preservation is therefore a result for the selected training regime, not a guarantee under further training.

The random-label and untrained-model controls were rerun. Replacing activations for 5% of validation positions with another game’s activations raised error by 3.74 points, so the preservation test remained sensitive to wrong boards.

At the selected checkpoints, refitting the probe left changes of +0.070 points for low Elo and −0.070 for high Elo. Rotating activations back to the base increased error to +0.328 and +0.219 points. A rotation therefore offered no improvement at these selected checkpoints. Their board representations were already readable using the frozen probe.

At layer 6, median relative activation changes were 8.8% for low Elo and 11.5% for high Elo. Pairwise geometry stayed similar to the base (rank correlations 0.993 and 0.988). The fractions of activation-change energy in the probe subspace were 46.4% and 39.3%, against a 44.9% random-subspace expectation. The earlier pattern of changes avoiding the probe subspace does not hold uniformly for this pair.

![Elo probe error during training](results/figs/probe_preservation.png)

![Frozen, refitted, rotated and affine-mapped probe drift](results/figs/probe_drift.png)

![Layerwise changes and the joint layer-6 PCA projection](results/figs/p15_geometry.png)

#### Matched policy shift across all six copies

I evaluated both Elo trajectories on the original P1 and P2 position sets with each set’s original frozen probe and probe-fitting recipe. The solid curves use the selected learning rate; dashed curves use the higher learning rate. Each comparison of models within a panel therefore shares its board positions, probe and definition of policy KL. Interpolated matched-KL values are reported only within the observed trajectory range.

At matched KL 0.01, the high-Elo copy has interpolated frozen-probe drift of **+0.113 points on the in-person domain and −0.056 on the bullet domain**. The grandmaster copies span +0.387 to +0.518 and +0.171 to +0.264, respectively. The Elo curves therefore do not collapse onto the earlier grandmaster curves. Policy KL alone is insufficient to predict probe drift across these experiments. The selected-learning-rate low-Elo trajectory never reaches KL 0.01 on either shared domain, so I do not extrapolate a value for it.

![All six specialists on the same two position domains](results/figs/matched_kl_all_six.png)

#### Does the board signal still affect moves?

On 400 positions, a medium-strength probe-direction edit toward an empty square reduced probability assigned to the selected piece’s legal moves by **7.82% in the base, 8.39% in the low-Elo copy and 8.16% in the high-Elo copy**. Equal-size random edits changed it by about 0.48–0.56%. The readable board signal continued to influence move predictions.

#### Do the differences compound through games?

All five models played from the same 600 starts used for the grandmaster comparison, for up to 30 plies against a greedy base-model opponent. I used the same context-length and game-ending rules, and compared only eligible sequences at each move count.

| Comparison | Games still identical after ten own moves |
|---|---:|
| Karpov vs Kasparov | 32% |
| Carlsen vs Tang | 29% |
| Low vs high Elo | {100*s['survival'][9]:.1f}% |
| Low Elo, two seeds | {100*seed_low['survival'][9]:.1f}% |
| High Elo, two seeds | {100*seed_high['survival'][9]:.1f}% |

For low vs high Elo, {s['n_end']} paired games reached 30 plies. Their final boards differed on an average of {s['squares_mean']:.1f} squares. The curves below preserve the same eligibility convention as the earlier report. Repeating the base rollout reproduced 594 of the 600 historical sequences exactly; six differed under the new inference batching, so exact rollout percentages have some numerical sensitivity.

![Elo and grandmaster rollout divergence from identical starts](results/figs/rollouts_comparison.png)

#### What this changes in the conclusion

The test across Elo bands supports frozen-probe transfer under a further form of specialization. It also revises my expectation that this rating gap would necessarily produce a larger policy difference than individual grandmaster styles. On the shared neutral boards, the Elo copies disagreed slightly less often.

The evidence supports different move preferences, preserved board readability, and continued influence of the board signal. The eight-habit style result and greedy move-quality results are more limited. A stronger test of the pretraining explanation would vary fine-tuning strength and data while holding evaluation positions fixed, and measure the resulting models’ playing strength directly.

#### Reproduction and saved results

The Elo configuration is in exp_elo/configs/elo.yaml. The preparation script reuses the exact neutral boards and cached engine scores, then samples the two Elo test sets with seed 0. The analysis scripts are under scripts/elo. Aggregated results, trajectory decompositions, confidence intervals and figures are saved under exp_elo/results. Raw policy arrays, game data, checkpoints and runtime logs are kept out of Git.

Run from the repository root with WC_ROOT=exp_elo and WC_CONFIG=elo. Use scripts/elo/prepare.py, the shared scripts/18b_style_policy.py and scripts/18c_style_engine.py, and scripts/19_rollouts.py. Then run scripts/elo/analyze.py, scripts/elo/quality.py, scripts/elo/rollout_summary.py and scripts/elo/matched_kl.py. The cross-domain launcher documents the exact checkpoint schedule. The original training and sweep launchers are included in this branch.
"""
examples=json.loads((R/"18/examples_verified.json").read_text())
example_blocks=["#### Selected move changes", "These positions were selected after the aggregate evaluation to illustrate both improvements and regressions. Both training seeds made the specialist choice in each example. Depth-18 checks compare the base and specialist candidates. Blue marks the specialist move and orange the base move; scores are from the side to move."]
for example in examples:
    def evaluation(move):
        score=example["depth18"][move]
        if score["mate"] is not None:
            return f"mate in {score['mate']}" if score["mate"]>0 else f"gets mated in {abs(score['mate'])}"
        return f"{score['centipawns']/100:+.2f} pawns"
    desc=example["descriptions"]
    example_blocks.append(f"**{example['band'].capitalize()} Elo, {example['side']} to move.** Base chose {desc['base_move']} ({evaluation(example['base_move'])}); the specialist chose {desc['copy_move']} ({evaluation(example['copy_move'])}).")
    example_blocks.append(f"![{example['band'].capitalize()} Elo selected {example['kind']} example](results/figs/{example['figure']})")
chapter=chapter.replace("#### Reproduction and saved results", "\n\n".join(example_blocks)+"\n\n#### Reproduction and saved results")

# Include compact matched-KL numerical comparisons on the shared domains.
lines=["| Position domain | Specialist | Frozen extra error at KL 0.01 | Refitted extra error at KL 0.01 |",
       "|---|---|---:|---:|"]
for key,value in matched.items():
    domain,player=key.split("|")
    point=value["dose"]["matched"].get("0.01")
    if point:
        lines.append(f"| {domain} | {player.capitalize()} | {point['frozen']:+.3f} pp | {point['refit']:+.3f} pp |")
chapter=chapter.replace("#### Does the board signal still affect moves?", "\n".join(lines)+"\n\n#### Does the board signal still affect moves?")
elo_path=target/"exp_elo/REPORT.md"
elo_path.parent.mkdir(parents=True,exist_ok=True)
elo_path.write_text(chapter.replace("### 3j. Third pair (P3) — fine-tuning across Elo bands",
                                  "# Fine-tuning chess models across Elo bands").replace("\n#### ", "\n## "))
main_path=target/"WRITEUP.md"
main=main_path.read_text()
main=main.replace('*Direction 3 ("What models represent"), adapted from the **Board games** starting point. P1 pilot, one training seed.*',
'*Direction 3 ("What models represent"), adapted from the **Board games** starting point. Grandmaster and Elo-band experiments with seed, probe, geometry and behavioural analyses.*')
marker="### 3j. Third pair (P3) — fine-tuning across Elo bands"
if marker in main:
    main=main[:main.index(marker)]+main[main.index("## 4. What would change",main.index(marker)):]
main=main.replace("## 4. What would change",chapter.replace("](results/","](exp_elo/results/")+"\n## 4. What would change",1)
main_path.write_text(main)

def inline(text):
    text=html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*",r"<strong>\1</strong>",text)

def render(markdown):
    blocks=[]
    for block in markdown.strip().split("\n\n"):
        if block.startswith("#"):
            level=len(block)-len(block.lstrip("#"))
            level=max(2,min(level,4))
            blocks.append(f"<h{level}>{inline(block.lstrip('# ').strip())}</h{level}>")
        elif block.startswith("!["):
            alt,path=re.match(r"!\[(.*?)\]\((.*?)\)",block).groups()
            file=ROOT/path
            data=base64.b64encode(file.read_bytes()).decode()
            mime="image/svg+xml" if file.suffix==".svg" else "image/png"
            width=' style="max-width:420px;margin:24px auto"' if file.suffix==".svg" else ""
            blocks.append(f'<figure{width}><img alt="{html.escape(alt)}" src="data:{mime};base64,{data}"/><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif block.startswith("|"):
            rows=block.splitlines()
            cells=lambda line: [inline(c.strip()) for c in line.strip("|").split("|")]
            head="<tr>"+"".join(f"<th>{v}</th>" for v in cells(rows[0]))+"</tr>"
            body="".join("<tr>"+"".join(f"<td>{v}</td>" for v in cells(line))+"</tr>" for line in rows[2:])
            blocks.append(f"<table><thead>{head}</thead><tbody>{body}</tbody></table>")
        else:
            blocks.append("<p>"+inline(block).replace("\n"," ")+"</p>")
    return "\n".join(blocks)

body=render(chapter)
chapter_heading="<h3>3j. Third pair (P3) — fine-tuning across Elo bands</h3>"
full_body=body.replace(chapter_heading,'<h2 id="elo-rating-bands">14. Third pair — fine-tuning across Elo bands</h2>',1)
standalone_body=body.replace(chapter_heading,"<h1>Fine-tuning chess models across Elo bands</h1>",1)
full_body=full_body.replace("<h4>","<h3>").replace("</h4>","</h3>")
standalone_body=standalone_body.replace("<h4>","<h2>").replace("</h4>","</h2>")
full_path=target/"report.html"
full=full_path.read_text()
start="<!-- ELO CHAPTER START -->"
end="<!-- ELO CHAPTER END -->"
if start in full:
    full=full[:full.index(start)]+full[full.index(end)+len(end):]
anchor='<h2 id="14-related-work-and-sources">'
assert anchor in full
full=full.replace(anchor,start+'\n<section id="elo-experiment">\n'+full_body+'\n</section>\n'+end+'\n'+anchor,1)
full=full.replace('>14. Related work and sources</h2>', '>15. Related work and sources</h2>')
full_path.write_text(full)
(target/"exp_elo/report.html").write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    '<title>Chess fine-tuning across Elo bands</title><style>'
    'body{font:17px/1.6 Georgia,serif;max-width:1000px;margin:auto;padding:30px;color:#20252b}'
    'h2,h3,h4{line-height:1.25;font-family:system-ui}img{width:100%;height:auto}figure{margin:24px 0}'
    'table{border-collapse:collapse;width:100%;font:14px/1.4 system-ui}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
    'figcaption{font:14px system-ui;color:#555} @media print{body{font-size:10pt}figure,table{break-inside:avoid}}'
    '</style><main>'+standalone_body+'</main></html>')
print("Built Elo Markdown and both HTML reports in",target)
