# Data audit: Elo-band Lichess dataset (exp_elo/data/games.jsonl)

Scope: read-only audit of 203,000 records (100,000 low, 100,000 high, 3,000 other). Full-file pass for metadata, splits, TC, colour, leakage, clock filter (all 200k games). Text integrity and SAN replay on 2,500 random games per band. No data or code was modified.

## Verdict

**Not ready for training as-is.** Blocking issues:

1. **Test set is not account-disjoint from policy (high band).** 99.1% of high-band test games have a target account that also appears in policy, and 44.2% of them come from accounts with more than 100 policy games. Test scores for the high band will mostly measure recognition of known players, not rating.
2. **The "other" probe pool contains online games.** 142 games are "Titled Tue" events and 13 are "chess.com" events, which conflicts with the over-the-board champion-games design.
3. **19 high-band games are labelled `test_leak`.** The builder code never produces this label. The high-band test split has 14,981 games instead of 15,000, and all 19 games have both players in policy. Their origin needs to be explained, and they should be dropped or relabelled.

Confounds to control or report (not blocking on their own): the two bands differ in termination mix, game length and berserk rate (section 8).

Everything else checks out: band membership, time-control histograms, colour balance, metadata consistency, text integrity, SAN legality, clock-skip reproduction, and gid/text duplication.

## 1. Band membership: PASS

- Low: 0 out-of-band Elo values, 0 missing or non-numeric. Min 1500, median 1746, max 1999, p5/p95 1537/1959.
- High: 0 out-of-band, 0 missing. Min 2300, median 2388, max 3060, p5/p95 2308/2590.
- All records have `event_class` starting with `lichess_`.

## 2. Time controls: PASS, with minor per-split noise

- Overall histograms are identical between bands. Top TCs: 180+0 65,407 and 60+0 31,869 in each band (65.4% and 31.9%). Other TCs are rare (300+0 698, 120+0 696, 600+0 354, …).
- No time control has base < 60 s (0 games in either band).
- Per-split histograms are not exactly identical, because splits are random within each band and colour. Largest differences are small: e.g. test 60+0 is 4,802 (low) vs 4,862 (high); policy 180+0 is 45,765 vs 45,794.

## 3. Target colour and metadata consistency: PASS

- Overall colour: low w 49,943 / b 50,057; high w 50,094 / b 49,906.
- Per split (w/b): low policy 34,960/35,040, probe_fit 4,994/5,006, probe_dev 2,497/2,503, test 7,492/7,508. High policy 35,066/34,934, probe_fit 5,009/4,991, probe_dev 2,505/2,495, test 7,506/7,475.
- `meta.colors == {band: color}` for all 200,000 low/high games (0 mismatches).
- `meta.target_user` equals the lowercased name on the target's side for all 200,000 games (0 mismatches).
- Split sizes: low 70,000 / 10,000 / 5,000 / 15,000 (exact). High 70,000 / 10,000 / 5,000 / 14,981 + 19 `test_leak`.

## 4. Text integrity and SAN replay: PASS (2,500 random games per band)

- All sampled texts start with `;1.`.
- All spans match: `text[s:e] == sans[i]` for every ply.
- All sampled games replay legally in python-chess (`parse_san` then `push`). 0 failures in either band.
- Sample mean plies: low 71.3, high 86.0.

## 5. Clock filter: PASS (rule reproduced exactly)

- Recomputed skip plies for all 200,000 games with the builder's start-clock rule and with a rule based on `meta.berserk`. Both reproduce `meta.skip_plies` exactly (0 mismatched games).
- Target-move skip fraction (all target plies, both bands):

| band | skipped / target plies | fraction | policy-only |
|---|---|---|---|
| low | 225,784 / 3,626,113 | 6.23% | 6.19% |
| high | 341,946 / 4,262,002 | 8.02% | 8.00% |

- By TC, low vs high: 60+0 12.2% vs 14.1%; 180+0 3.6% vs 5.2%; 180+1 4.0% vs 9.1%; 300+0 1.8% vs 2.9%; 600+0 0.4% vs 2.3%.
- The summary file (`results/01_data_summary.json`) gives skipped_frac 5.99% / 7.84% and target_moves 2,352,112 / 2,719,858. These do not match the policy-only numbers (2,379,968 / 2,747,376 kept target plies). The gap is about 1%, so the summary probably uses a slightly different count definition. Minor, but worth reconciling.

## 6. Leakage: mostly PASS, one unexplained label

- Duplicate gids in the whole file: 0. Gids are unique within each band, and no gid appears in both bands.
- Identical move text across any records: 0. Identical move sequences across splits or bands: 0.
- `test_leak` split: 19 games, all high band, all with TC 180+0 (18) or 60+0 (1). Both players appear in high-band policy for 19/19. Their origin is not produced by the builder code, and the high test split is therefore 14,981 rather than 15,000.
- Player-level overlap (see also section 7): 99.1% of high test games have an opponent who is a policy target account. Low: 83.7% (12,558 of 15,000).

## 7. Account concentration and overlap: FAIL for high band

| | low | high |
|---|---|---|
| distinct target accounts | 29,161 | 3,164 |
| top-10 share of games | 0.43% | 6.03% |
| top-100 share of games | 3.16% | 32.58% |
| max games for one account | 47 | 1,000 (cap hit) |
| accounts with ≥100 games | 0 | 252 (covering 54,382 games, 54.4%) |

- Accounts appearing in both bands: 94.
- TEST games whose target account also appears in policy: low 12,524 / 15,000 (83.5%); high 14,839 / 14,981 (99.05%).
- TEST games whose target account has >100 policy games: low 0; high 6,627 / 14,981 (44.2%).
- Test games whose target account appears in policy in the high band: as above (99.05%). Policy top accounts in high: flamenko 719, lalarttu 501, farmville 398.

The split is by game, not by account, so the high-band test score mostly reflects the same ~250 heavy accounts seen in training. Fixing this needs an account-grouped split (or a per-account cap on policy).

## 8. Length, termination, berserk, arena: confounds between bands

| | low | high |
|---|---|---|
| mean / median plies | 72.5 / 69 | 85.2 / 82 |
| Normal | 59,204 (59.2%) | 71,481 (71.5%) |
| Time forfeit | 40,796 (40.8%) | 28,519 (28.5%) |
| target berserked (games) | 2,083 (2.08%) | 593 (0.59%) |
| either side berserked | 3.77% | 1.02% |
| arena share | 0.0% | 0.0% |
| sample: final move is mate (2,500 each) | 26.6% | 16.4% |

Time forfeit and game length differ materially between bands. A model could pick up these cues instead of strength. Berserk is about 3.5 times more common in low band. Arena is absent in both bands.

## 9. 'Other' pool: FAIL on source (online games present)

- Count: 3,000, all `probe_fit_other`, no colour or meta. 0 games with `white` or `black` matching carlsen|tang|karpov|kasparov (case-insensitive). Gids unique.
- Source is not stored in the file, so "no Lichess games" cannot be checked directly. From event names:
  - 142 events are "Titled Tue …" (an online tournament series).
  - 13 events are "chess.com …".
  - Several others (e.g. "AI Cup Play-In 2023") may also be online. The count is not exhaustive.
- Other date issues: 1,825 games have partial dates (contain `?`), 14 are dated 2026, and 256 are dated 2023 or later. Dates range from 1939 to 2026.
- Plies: min 32, median 77.
- Dominant players: Kramnik, Anand, Smyslov, Tal, Spassky, Petrosian (among the top White and Black names).

## 10. Other observations

- No duplicate-name issues: every target account has one consistent raw spelling in the file. 0 accounts with multiple raw spellings on the target side (both bands).
- 0 games where white and black are the same name.
- Account name `Anonymous` appears as opponent in 26 low and 74 high games. It is never a target.
- Time forfeits with fewer than 30 plies: 476 low, 153 high. These are probably abandoned or very short games; they are not filtered by the builder (min_plies = 20 only).
- Games with 200 or more plies: 17 low, 159 high.
- Summary discrepancy: the summary's skipped_frac and target_moves are about 1% off from a reproducible count (section 5).
- Builder code note: for a berserk side, the builder takes the post-first-move clock as the starting clock instead of base/2. On this data it produces no mismatches, so it has no effect, but it is an edge case.

## Recommended fixes before training

1. Group the split by target account (or cap policy games per account and drop accounts from test), at least for the high band.
2. Remove or regenerate the `other` pool using only OTB sources (drop Titled Tuesday and chess.com events, and games with partial or implausible dates).
3. Resolve the 19 `test_leak` games (drop them, or confirm their purpose and relabel).
4. Control for termination, length and berserk in the rating analysis, or report stratified results.
5. Reconcile the summary's skipped_frac and target_moves with the recomputed numbers.
