#!/bin/bash
# E9: per-worker (thread_local) cache sizing at W=64.
#
# Each worker holds ~27 MB of private caches. Eight workers share one CCD's
# 32 MiB of L3, so they collectively want ~216 MB — roughly 7x oversubscribed.
# The 18-bit sizing was tuned at 8 threads, where that pressure did not exist,
# and its stated justification ("the L2 budget is not the binding constraint")
# may no longer hold. Scale all four constants together to find the direction
# first; refine afterwards.
#
#   tier    WORKER  LM  LM2  LM1L   ~MB/worker   x64
#   small     16    16   17    13        7        0.45 GB
#   down      17    17   18    14       14        0.9 GB
#   base      18    18   19    15       27        1.7 GB   <- current
#   up        19    19   20    16       54        3.5 GB
set -uo pipefail
cd "$HOME/A087725"
. "$HOME/.cargo/env"
F=src/puzzle24/search/flat.rs
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

# Wait for any in-flight sweep to finish so the box stays idle for timing.
while pgrep -x solve24 >/dev/null; do sleep 60; done
say "box idle; starting E9"

# Hold K8_SHARED_BITS at the adopted value so this sweep varies one thing only.
sed -i "s/^const K8_SHARED_BITS: u32 = .*/const K8_SHARED_BITS: u32 = 24;/" $F

run_cfg() {
  local name=$1 w=$2 lm=$3 lm2=$4 lm1l=$5
  say "E9 $name: WORKER=$w LM=$lm LM2=$lm2 LM1L=$lm1l"
  sed -i "s/^const WORKER_CACHE_BITS: u32 = .*/const WORKER_CACHE_BITS: u32 = $w;/" $F
  sed -i "s/^const LM_CACHE_BITS: u32 = .*/const LM_CACHE_BITS: u32 = $lm;/" $F
  sed -i "s/^const LM2_CACHE_BITS: u32 = .*/const LM2_CACHE_BITS: u32 = $lm2;/" $F
  sed -i "s/^const LM1L_CACHE_BITS: u32 = .*/const LM1L_CACHE_BITS: u32 = $lm1l;/" $F
  grep -nE "^const (WORKER_CACHE_BITS|LM_CACHE_BITS|LM2_CACHE_BITS|LM1L_CACHE_BITS)" $F | tr '\n' ' '
  echo
  cargo build --release --features sha 2>&1 | tail -1 || return 1
  target/release/solve24 --position "$R" --prove-at-least 149 \
    --clm2 --zpdb8 --parallel > logs/e9_$name.log 2>&1
  grep -E "threshold 148 exhausted" logs/e9_$name.log || echo "  !! no 148 line for $name"
}

run_cfg small 16 16 17 13
run_cfg down  17 17 18 14
run_cfg base  18 18 19 15
run_cfg up    19 19 20 16

say "revert to committed constants, rebuild, reference canary"
git checkout $F
cargo build --release --features sha 2>&1 | tail -1
target/release/solve24 --position "$R" --prove-at-least 147 \
  --clm2 --zpdb8 --parallel > logs/canary_e.log 2>&1
grep -E "threshold 146 exhausted" logs/canary_e.log
say "E9 COMPLETE"
echo done > logs/E9_DONE
