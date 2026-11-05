#!/usr/bin/env bash
# Wait until a Lichess monthly dump has fully downloaded (size == expected bytes), verify it, then filter it
# into exp_elo/data/raw_db/{high,low}_<month>.pgn.  Usage: scripts/elo_wait_filter.sh MONTH EXPECTED_BYTES
set -uo pipefail; cd "$(dirname "$0")/.."
M=$1; N=$2; F=$HOME/abhay/lichess_db/lichess_db_standard_rated_$M.pgn.zst; OUT=exp_elo/data/raw_db
mkdir -p $OUT exp_elo/runs
until [ "$(stat -c %s "$F" 2>/dev/null || echo 0)" = "$N" ]; do sleep 30; done
echo "[$(date +%T)] $M downloaded"; zstd -tq "$F" && echo "[$(date +%T)] $M integrity ok" || { echo "$M CORRUPT"; exit 1; }
python3 scripts/01c_lichess_db_filter.py "$F" $OUT 2300 1500 2000 0.05
echo "[$(date +%T)] $M FILTER_DONE"
