#!/bin/bash
# Waits for the table pipeline, syncs to origin/main, then runs runbook
# step 6 (canaries), E1 (thread scaling), E2 (SPLIT_TARGET), E3 (k8 cache).
# Writes logs/NIGHT_DONE at the end, which triggers the auto-deallocate.
set -uo pipefail
cd ~/A087725
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

say "waiting for table pipeline (logs/EXIT)"
while [ ! -f logs/EXIT ]; do sleep 60; done
if [ "$(cat logs/EXIT)" != "0" ]; then say "PIPELINE FAILED (exit $(cat logs/EXIT)) — stopping"; exit 1; fi
say "pipeline green"

say "sync to origin/main"
git stash >/dev/null 2>&1
git pull --ff-only origin main || { say "git pull failed"; exit 1; }
git log --oneline -1
cargo build --release --features sha 2>&1 | tail -1 || exit 1

say "step 6 canaries"
{
  echo "--- gate 1: zpdb8 through 146 (expect 269,180,917 / 8,539,130,554)"
  target/release/solve24 --position "$R" --prove-at-least 147 --zpdb8 --parallel
  echo "--- gate 2: clm2 at 144 (expect 134,801,951)"
  target/release/solve24 --position "$R" --prove-at-least 145 --clm2 --parallel
  echo "--- gate 3: cascade through 146 (expect 4,363,759,350)"
  target/release/solve24 --position "$R" --prove-at-least 147 --clm2 --zpdb8 --parallel
} > logs/canaries.log 2>&1
say "canaries done -> logs/canaries.log"

for W in 64 48 32 16; do
  say "E1 W=$W (exhaust-148)"
  RAYON_NUM_THREADS=$W target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel > logs/e1_w$W.log 2>&1
  grep -E "threshold 148 exhausted" logs/e1_w$W.log || say "W=$W produced no 148 line"
done
say "E1 SWEEP COMPLETE"

say "E2: rebuild with parallel-profile for the SPLIT_TARGET knob"
cargo build --release --features "sha parallel-profile" 2>&1 | tail -1 || exit 1
for S in 4096 16384 32768 65536; do
  say "E2 SPLIT_TARGET=$S (W=64, exhaust-148)"
  FLAT_SPLIT_TARGET=$S RAYON_NUM_THREADS=64 target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel > logs/e2_s$S.log 2>&1
  grep -E "threshold 148 exhausted" logs/e2_s$S.log || say "S=$S produced no 148 line"
done
say "E2 SWEEP COMPLETE"

say "E3: rebuild with probe-cache-stats for k8 hit rates"
cargo build --release --features "sha probe-cache-stats" 2>&1 | tail -1 || exit 1
for W in 64 16; do
  say "E3 W=$W (exhaust-148, instrumented)"
  RAYON_NUM_THREADS=$W target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel > logs/e3_w$W.log 2>&1
  grep -E "k8 cache|lm cache|lm2 cache" logs/e3_w$W.log || say "W=$W produced no cache lines"
done
say "E3 COMPLETE"

say "restoring the plain production build"
cargo build --release --features sha 2>&1 | tail -1
say "ALL NIGHT WORK COMPLETE"
echo done > logs/NIGHT_DONE
