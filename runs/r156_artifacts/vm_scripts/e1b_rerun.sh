#!/bin/bash
# E1b: re-measure the thread-scaling curve at the CURRENT constants.
#
# The original E1 ran with SPLIT_TARGET=4096 and K8_SHARED_BITS=21. Both have
# since changed (32768 and 27). The k8 cache pressure was thread-count
# dependent — hit rate 72.905% at W=16 vs 66.271% at W=64 — so it was itself a
# component of the 19% loss E1 reported. The curve should therefore be flatter
# now, not merely lower.
#
# Original E1 for reference (seconds at exhaust-148):
#   W=16 1929.41 | W=32 1045.82 | W=48 743.62 | W=64 592.83
set -uo pipefail
cd "$HOME/A087725"
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(TZ=America/New_York date +'%-I:%M %p ET')] $*"; }

while pgrep -x solve24 >/dev/null; do sleep 60; done

say "sync to origin/main and rebuild"
git checkout src/puzzle24/search/flat.rs 2>/dev/null
git pull --ff-only origin main 2>&1 | tail -1
git log --oneline -1
grep -nE "^const (SPLIT_TARGET|K8_SHARED_BITS)" src/puzzle24/search/flat.rs
grep -nE "^const K8_SHARED_BITS" src/puzzle24/search/flat.rs
cargo build --release --features sha 2>&1 | tail -1 || exit 1

say "warm page cache"
cat data/cwd_mm.bin data/cwd_lm_mm.bin data/cwd_lm1l_mm.bin \
    data/pdb24_k8_a.zbin data/pdb24_k8_b.zbin data/pdb24_k8_c.zbin > /dev/null

for W in 64 48 32 16; do
  say "E1b W=$W (exhaust-148, SPLIT_TARGET=32768, K8_SHARED_BITS=27)"
  RAYON_NUM_THREADS=$W target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel > logs/e1b_w$W.log 2>&1
  grep -E "threshold 148 exhausted" logs/e1b_w$W.log || echo "  !! no 148 line at W=$W"
done

say "E1b COMPLETE"
echo done > logs/E1B_DONE
