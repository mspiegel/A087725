#!/bin/bash
# E6 (--hugepages) and E7 (K8_SHARED_BITS) A/Bs at exhaust-148, W=64,
# bracketed by canaries. Baseline from last night at SPLIT_TARGET=32768: 580.07 s.
set -uo pipefail
cd ~/A087725
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

run148() { local n="$1"; shift
  target/release/solve24 --position "$R" --prove-at-least 149 --clm2 --zpdb8 --parallel "$@" > logs/$n.log 2>&1
  grep -E "threshold 148 exhausted" logs/$n.log || echo "  !! no 148 line in $n"
}
canary() {
  target/release/solve24 --position "$R" --prove-at-least 147 --clm2 --zpdb8 --parallel > logs/$1.log 2>&1
  grep -E "threshold 146 exhausted" logs/$1.log || echo "  !! canary $1 failed"
}

say "warming page cache (~49 GB off Standard SSD; cold after deallocate)"
cat data/cwd_mm.bin data/cwd_lm_mm.bin data/cwd_lm1l_mm.bin \
    data/pdb24_k8_a.zbin data/pdb24_k8_b.zbin data/pdb24_k8_c.zbin > /dev/null
say "warm: $(grep -E "^Cached:" /proc/meminfo)"

say "canary A (bracket start)"; canary canary_a
say "E6 baseline: no hugepages"; run148 e6_base
say "E6 treatment: --hugepages"
( sleep 150; grep -E "AnonHugePages|Hugepagesize" /proc/meminfo > logs/e6_huge_meminfo.txt ) &
run148 e6_huge --hugepages
cat logs/e6_huge_meminfo.txt 2>/dev/null

say "E7: K8_SHARED_BITS 21 -> 24, rebuild"
sed -i "s/^const K8_SHARED_BITS: u32 = 21;/const K8_SHARED_BITS: u32 = 24;/" src/puzzle24/search/flat.rs
grep -n "const K8_SHARED_BITS" src/puzzle24/search/flat.rs
cargo build --release --features sha 2>&1 | tail -1 || exit 1
run148 e7_bits24
say "E7+E6 combined: bits24 + hugepages"; run148 e7_bits24_huge --hugepages

say "reverting K8_SHARED_BITS to committed value"
git checkout src/puzzle24/search/flat.rs
cargo build --release --features sha 2>&1 | tail -1
say "canary B (bracket end)"; canary canary_b
say "AB BATCH COMPLETE"
echo done > logs/AB_DONE
