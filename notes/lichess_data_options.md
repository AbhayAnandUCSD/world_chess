# Lichess data options for per-policy fine-tuning (checked 2026-11-05)

## 1. Candidate accounts (`/api/user/{name}`, pulled today; counts are rated games per perf type)

| User | Who (source) | Title | Bullet | Blitz | Rapid | Classical | First rated game / last seen |
|---|---|---|---|---|---|---|---|
| Kingscrusher-YouTube | Tryfon Gavriel (profile realName) | CM | 223,535 | 44,130 | 2,265 | 7 | 2013-05 / active |
| Zhigalko_Sergei | Sergei Zhigalko (profile) | GM | 82,057 | 17,868 | 446 | 0 | **2018-09** / active |
| opperwezen | Not identified | IM | 53,796 | 4,730 | 275 | 89 | 2016-02 / active |
| penguingim1 | Andrew Tang (profile) | GM | 47,990 | 5,251 | 237 | 6 | 2015-04 / active |
| Lance5500 | "LM Lance", coach (profile) | LM | 28,062 | 18,836 | 637 | 425 | 2014-01 / active. Only 61k of 175k games are rated (simuls, casual) |
| EricRosen | Eric Rosen (profile) | IM | 19,597 | 9,374 | 1,954 | 309 | 2015-06 / active |
| nihalsarin2004 | Nihal Sarin (profile) | GM | 18,937 | 1,333 | 0 | 0 | **2017-09** / active |
| Night-King96 | Oleksandr Bortnyk (profile) | GM | 16,804 | 4,034 | 81 | 0 | **2018-02** / active |
| RebeccaHarris | Naroditsky per a fan blog only (unverified) | GM | 11,948 | 2,364 | 14 | 4 | 2016-09 / last seen 2025-10 |
| Arka50 | Not identified | GM | 11,169 | 1,299 | 58 | 7 | 2015-09 / active |
| Fins | John Bartholomew (profile) | IM | 7,313 | 3,470 | 594 | 31 | 2015-04 / active |

Rejected: DrNykterstein (Carlsen) has about 10.4k games, about 9.6k of them bullet. alireza2003 has about 10.8k. DrDrunkenstein has under 1k. GMHikaru, ChessNetwork, Ginger_GM and mishanick are closed. DanielNaroditsky has 14 games.

**Style evidence is thin.** Only Eric Rosen's style is well documented: offbeat, trap-heavy play and his revival of the Stafford Gambit ([Wikipedia](https://en.wikipedia.org/wiki/Stafford_Gambit)). I found no reliable source for the styles of the other high-volume accounts, so treat those as unverified. Every account with 20k+ games is mostly **bullet**. Measure style contrast from the downloaded games (opening repertoire, gambit rate, average game length, sacrifice rate) instead of relying on reputation.

**Suggested pairs** (accounts created before 2017-06 should be exported with `since=2017-06-01`):
1. **Zhigalko_Sergei vs Kingscrusher-YouTube, bullet.** This pair has the most data on both sides (82k vs 223k games) and a large strength gap (GM about 3130 vs CM about 2580 bullet). Zhigalko has no overlap with the pretraining months. The contrast is mostly in strength rather than documented style.
2. **Zhigalko_Sergei vs Lance5500, blitz** (17.9k vs 18.8k). This is the best matched-size blitz pair, and blitz is less noisy than bullet.
3. **EricRosen vs penguingim1, bullet** (19.6k vs 48k). Rosen's trappy, offbeat style is documented. Tang plays mainstream openings (unverified). Rosen's 1,954 rapid games give an extra test at a slower time control.

## 2. Downloading
- `GET https://lichess.org/api/games/user/{u}` ([spec](https://github.com/lichess-org/api/blob/master/doc/specs/tags/games/api-games-user-username.yaml)). Parameters:
  - `since` / `until` (ms epoch), `max`, `rated=true`, `perfType=bullet,blitz`, `color`, `sort=dateAsc|dateDesc`
  - `moves` (default true), `pgnInJson` (requires ndjson), `clocks` and `evals` (both default false), `tags`, `opening`, `literate`
- Streaming throttle: **20 games/s anonymous**, 30 games/s with an OAuth token, 60 games/s for your own games. At 20 games/s, 100k games take about 1.4 h and Kingscrusher's 270k take about 3.7 h.
- Rate limits ([docs](https://lichess.org/api)): send one request at a time. After an HTTP 429, wait at least a minute.
- **Verified gotcha:** curl's default User-Agent got `{"error":"Not found"}`. Requests with a custom `-A` User-Agent worked.
- Licensing: database exports are **CC0** ([database.lichess.org](https://database.lichess.org/)). The [ToS](https://lichess.org/terms-of-service) allow API use for research, subject to caps and with no API abuse. The ToS do not state a licence for data pulled through the API (unverified; the games are public).

## 3. Elo-bin design
- **Karvonen `lichess_200k_elo_bins.zip`** (998 MB; a `_blocks` version is 1.15 GB):
  - It is a CSV with columns `WhiteElo, BlackElo, Result, transcript, elo_bin`. Bins are 100 Elo wide, e.g. `"[1600, 1700)"`, capped at 200k games per bin ([README](https://huggingface.co/datasets/adamkarvonen/chess_games)). I checked this by inflating the first 200 KB.
  - In the rows I saw, the bin follows **WhiteElo**.
  - Row indices go up to about 16.1M, which suggests the rows were sampled from the 16.49M-game `lichess_6gb` pool, i.e. the pretraining data. This is an inference and is unverified. **Do not use it for fine-tuning.**
- **Maia** ([arXiv 2006.01855](https://arxiv.org/abs/2006.01855)):
  - Data: games from 2016-01 to 2019-11, with 2019-12 kept for test.
  - Bins: 100-point bins from 1100 to 1900, keeping only games where **both** players are in the bin.
  - Filters: bullet and hyperbullet are dropped. At evaluation, the first 10 ply and any move made with under 30 s left on the clock are discarded.
  - Size: 12M training games per bin.
- **Lichess open database:** monthly `.pgn.zst` files. 2026-10 is 31 GB and 95.4M games. For comparison, 2016-06 was 1.09 GB / 6.1M games and 2017-05 was 3.59 GB / 11.7M. Files expand about 7.1x when decompressed. Partial downloads can be decompressed, so you can stream with `zstdcat f.pgn.zst | python filter.py` and filter on `WhiteElo`/`BlackElo`, `TimeControl` and `[WhiteTitle "BOT"]`. Clocks appear as `%clk` comments from April 2017 on, and about 6% of games carry `%eval` comments. pgn-extract can also filter by tag (not tested).
- **Concrete design:**
  - Data: use a month with no pretraining overlap, e.g. 2024-xx. Keep blitz games (180+0 to 300+3) where both players are rated 1100–1300 for one copy and both are rated 2200+ for the other.
  - Masking and budget: mask the loss to all moves, and give both copies the same number of target characters (about 20–50M each). Both bins have plenty of games in a single month.
  - Probe: keep Karvonen's held-out `lichess_100mb` (2016-01) as the probe set.
  - Why it matters: low-rated play contains more blunders and hung pieces. If fitting the weaker policy lowers linear-probe board accuracy, that suggests the world model is shaped by the move distribution. If probe accuracy holds, board tracking is policy-independent.
  - Elo bins give 100x more data per policy than any single account.

## 4. Pitfalls
- **Pretraining overlap:** penguingim1, opperwezen, EricRosen, Kingscrusher, Lance5500, Fins, RebeccaHarris and Arka50 all have games in or before 2016-06 and 2017-05. Use `since=2017-06-01`.
- **Bullet noise:** many bullet moves are premoves or clock-driven. Apply Maia's under-30 s filter, or prefer blitz.
- **Time control:** keep both copies on the same time control.
- **Cheaters:** check `tosViolation` on accounts. Closed accounts (such as GMHikaru) cannot be exported. Opponents may also be cheaters; consider dropping games where the opponent's account is now closed (not tested).
- **Format:** Karvonen's format is `;1.e4 e5 2.Nf3 ...`, with no space after the move number. The database and API use `1. e4 { [%clk 0:00:30] } 1... e5`. Strip `{...}` comments, `?!` annotations and `N...` move-number continuations, and drop the tags.
- **Identity:** anonymous titled accounts (RebeccaHarris, opperwezen, Arka50) may be shared or misattributed. Streamer accounts include many simuls and casual games; set `rated=true`.
