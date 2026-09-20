#!/bin/bash
# Extend the K8_SHARED_BITS curve: still rising at 26 (512 MB).
# Known: 21 568.44 | 22 540.55 | 24 508.46 | 25 492.94 | 26 462.60
# NOTE: canary runs AFTER the revert this time (last script measured a
# bits=26 build and mislabelled it a reference canary).
set -uo pipefail
cd ~/A087725
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(date -u +%H:%M:%S)] $*"; }
for B in 27 28; do
  say "K8_SHARED_BITS=$B ($((2**(B-17))) MB slots array), rebuild"
  sed -i "s/^const K8_SHARED_BITS: u32 = .*/const K8_SHARED_BITS: u32 = $B;/" src/puzzle24/search/flat.rs
  cargo build --release --features sha 2>&1 | tail -1 || exit 1
  /usr/bin/time -v target/release/solve24 --position "$R" --prove-at-least 149 \
    --clm2 --zpdb8 --parallel > logs/bits_$B.log 2>logs/bits_$B.time
  grep -E "threshold 148 exhausted" logs/bits_$B.time logs/bits_$B.log 2>/dev/null | head -1
  grep -E "Maximum resident" logs/bits_$B.time
done
say "revert to committed, then TRUE reference canary"
git checkout src/puzzle24/search/flat.rs
cargo build --release --features sha 2>&1 | tail -1
target/release/solve24 --position "$R" --prove-at-least 147 --clm2 --zpdb8 --parallel > logs/canary_d.log 2>&1
grep -E "threshold 146 exhausted" logs/canary_d.log
say "BITS SWEEP 2 COMPLETE"
echo done > logs/BITS2_DONE
