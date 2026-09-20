#!/bin/bash
# E8: name the ~16% parallel loss at W=64.
#
# Efficiency is 83.6% at W=64 (4.014 vs 4.803 Mn/s per core at W=16) and the
# E7 cache fix did NOT explain it. The repo's own instrument decomposes
#
#     speedup / W  =  B  x  (c_1 / c_p)
#
# B = busy fraction (split policy, scheduling, tail), c_p = worker ns/node
# (memory contention, clock). These have unrelated fixes, so the decomposition
# is what distinguishes them. Run at W=64 and W=16 so c_1 has a reference.
#
# parallel-profile also exposes FLAT_SPLIT_TARGET, so pin it to the production
# value; the measurement itself is two Instant::now() per work unit, i.e. free.
set -uo pipefail
cd "$HOME/A087725"
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(TZ=America/New_York date +'%-I:%M %p ET')] $*"; }

while pgrep -x solve24 >/dev/null; do sleep 60; done

say "build with parallel-profile"
cargo build --release --features "sha parallel-profile" 2>&1 | tail -1 || exit 1

for W in 64 16; do
  say "E8 parallel-profile W=$W (exhaust-148)"
  FLAT_SPLIT_TARGET=32768 RAYON_NUM_THREADS=$W target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel > logs/e8_prof_w$W.log 2>&1
  grep -E "threshold 148 exhausted" logs/e8_prof_w$W.log
  echo "--- decomposition W=$W ---"
  grep -viE "^\[|threshold|image slide|cWD|k8 ready|flat engine|mmapping|root-orbit|search:|parallel:|Nodes|Iterations|Search time|Throughput|Lower bound|Wall-clock" \
    logs/e8_prof_w$W.log | grep -v "^$" | tail -30
done

say "restore production build"
cargo build --release --features sha 2>&1 | tail -1
say "E8 COMPLETE"
echo done > logs/E8_DONE
