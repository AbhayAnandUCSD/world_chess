# Data audit: Karpov / Kasparov games (2026-11-05)

Scripts: scratchpad `audit.py`, `cls.py` (event classifier), `run*.py`, `extra.py` (not part of the repo).

## 1. Current data (PGN Mentor)

| File | games | FEN/SetUp | parse err | <2 plies | dup (exact SAN) | kept |
|---|---|---|---|---|---|---|
| Karpov.pgn | 3529 | 0 | 0 | 1 | 0 | 3528 |
| Kasparov.pgn | 2128 | 0 | 0 | 1 | 192 (all are the h2h games already in Karpov.pgn) | 1935 |

There are no within-file exact duplicates. I found about 3 near-duplicates: the same game appearing truncated or re-entered, e.g. Istratescu–Karpov 2004 at 43 and 83 plies, and Karpov–Needleman in a 2005 simul.

**Identity bugs.** These games are attributed to the wrong player because `who()` matches on substring:
- `Kasparov,Georgy`: 8 games from Kislovodsk Open 2013, played by a junior with no Elo. They are labelled Kasparov.
- `Karpov, Anantoly2`: 1 game from UKR-ch 1997. This is a different Karpov.
- The substring rule also accepts `Karpova` / `Kasparova` and any other Karpov. That doesn't matter for PGN Mentor, but it would on any larger database. KingBase contains **Kasparov, Sergey (1219 games)**, Kasparova Tatiana (271), Karpov Vadim/Alexander/Arkadiy/Viktor (~280) and Karpova L./E. (41). The fix is to match on the given name, e.g. `^Karpov\s*,\s*(A|An|Ana|Anatol\w*)$`. For `Karpov, A`, also require Elo ≥ 2500.

**Event class** (Event and Site matched by regex; heuristic):

| class | Karpov | Kasparov | h2h |
|---|---|---|---|
| classical | 1906 (1905 after fix) | 993 (985 after fix) | 167 |
| match/exh (time control unclear, mostly classical training matches) | 114 | 57 | 0 |
| rapid | 702 | 233 | 8 |
| blitz | 283 | 187 | 13 |
| blindfold (Amber) | 103 | 0 | 0 |
| simul (incl. clock simuls) | 196 | 384 | 0 |
| online (ICC, KasparovChess, MSN) | 23 | 45 | 0 |
| vs computer / advanced | 1 / 8 | 24 / 10 | 4 / 0 |
| consultation | 0 | 2 | 0 |

**Decade** (Karpov / Kasparov / h2h): 1960s 62/0/0, 1970s 520/81/0, 1980s 613/501/133, 1990s 1095/809/38, 2000s 722/409/21, 2010s 287/117/0, 2020s 37/18/0.

**Length:** median 81 plies for Karpov and 74 for Kasparov. Under 40 plies: 287 Karpov, 116 Kasparov. 150 plies or more: 126 Karpov, 25 Kasparov.

**Elo** (player's / opponent's): both present for 3059/3336 Karpov games and 1607/1935 Kasparov games. Kasparov has 269 games with no opponent Elo, mostly simuls.

## 2. Recoverable drops

There is essentially nothing to recover. The only drops are the two 1-ply games. The files have no FEN games, no parse errors, and no comments or variations, so mainline handling is irrelevant here. On KingBase, 6 Karpov/Kasparov games hit `errors`/FEN.

## 3. External sources

| Source | License / ToS | Format, size | New K / Ka games | Bulk OK? |
|---|---|---|---|---|
| **KingBase 2019** (archive.org mirror, downloaded) | "free" database, discontinued 2019; license unstated | PGN, 459 MB zip, 2.2M games, 1990–2019, Elo>2000 | **31 / 36** (classical 17/26, rapid 4/3, blitz 3/1, simul 5/6, exh 2/0) | yes (public archive) |
| PGN Mentor other champions (data/raw_other) | free | already local | **0 / 0** (484 K/Ka games, all already present) | – |
| Lumbra's Gigabase | CC BY-NC 4.0; click-through license | PGN; "Elite >2400" 207 MB, complete 2.66 GB | not tested (would cover pre-1990) | yes, after the license dialog; **needs your acceptance** |
| Lichess broadcast DB | CC0 | monthly .pgn.zst, 2020– | expected <50 each (2020s legends events) | yes |
| TWIC | "free for personal use only, all rights reserved" | weekly zips, 1994– | ≈0 beyond KingBase, which already ingests TWIC | personal use only |
| chessgames.com | CloudFront blocks automated access (403 on robots/terms) | – | unknown | **no**; don't scrape |
| Caissabase / millionbase | free; Scid format / old mirrors | >1 GB | likely small (same upstream sources) | unclear |
| Lichess masters | API only (position explorer) | – | – | not bulk |
| ChessBase Mega | commercial | – | likely the largest (more rapid/simul) | no |

New games were matched on exact SAN sequence, plus near-duplicates (same surnames and the same first 16 plies). That removed 94 more transcription variants. KingBase output: `data/extra_candidates/kingbase2019/kingbase_KK_subset.pgn`.

**Takeaway:** PGN Mentor is close to complete for free OTB data. Free sources add only about 1–3%.

## 4. Recommendation

1. Fix the identity bug: drop the 9 misattributed games and tighten `who()`.
2. Kasparov is the bottleneck. Expected games per player (Karpov / Kasparov):
   - classical + match: **~2019 / ~1042**
   - adding rapid: **~2721 / ~1275**
   - adding blitz too: ~3004 / ~1462
   - KingBase adds about 20 / 30 classical games.
3. Default suggestion: classical + match + rapid, with the class as a condition token or metadata. Exclude simul, blindfold, online, computer, advanced and consultation games. Simul opponents are weak, which skews the game toward easy wins; blitz has lower move quality.
4. Use the 192 head-to-head games for both players, with loss only on that player's moves. That adds 192 high-quality classical games to each side.
5. Caveats:
   - The classifier is regex-based. About 114/57 "match" games and some TV/KO events are uncertain.
   - Karpov's sample is weighted toward his late (2000s+) rapid exhibitions.
   - Kasparov has no games before 1973.
   - Time-control tags are missing in the source, so I couldn't verify classical vs rapid directly.
