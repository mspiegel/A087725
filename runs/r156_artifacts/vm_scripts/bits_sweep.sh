#!/bin/bash
# Map the K8_SHARED_BITS curve at W=64. Reference points already measured:
#   21 (16 MB) 568.44 s | 24 (128 MB) 508.46 s
set -uo pipefail
cd ~/A087725
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(date -u +%H:%M:%S)] $*"; }
for B in 22 25 26; do
  say "K8_SHARED_BITS=$B ($((2**B/1024/1024*8)) MB), rebuild"
  sed -i "s/^const K8_SHARED_BITS: u32 = .*/const K8_SHARED_BITS: u32 = $B;/" src/puzzle24/search/flat.rs
  grep -n "^const K8_SHARED_BITS" src/puzzle24/search/flat.rs
  cargo build --release --features sha 2>&1 | tail -1 || exit 1
  target/release/solve24 --position "$R" --prove-at-least 149 --clm2 --zpdb8 --parallel > logs/bits_$B.log 2>&1
  grep -E "threshold 148 exhausted" logs/bits_$B.log || echo "  !! no 148 line at bits=$B"
done
say "canary C (bracket end)"
target/release/solve24 --position "$R" --prove-at-least 147 --clm2 --zpdb8 --parallel > logs/canary_c.log 2>&1
grep -E "threshold 146 exhausted" logs/canary_c.log
git checkout src/puzzle24/search/flat.rs
say "BITS SWEEP COMPLETE"
echo done > logs/BITS_DONE
