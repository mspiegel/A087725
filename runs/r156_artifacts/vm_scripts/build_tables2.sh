#!/bin/bash
set -euo pipefail
cd ~/A087725
. "$HOME/.cargo/env"

echo "=== [$(date -u +%H:%M:%S)] stage 2: build_cwd_table (parallel lines)"
target/release/build_cwd_table

echo "=== [$(date -u +%H:%M:%S)] stage 3: build_cwd_artifacts all"
target/release/build_cwd_artifacts all

for g in a b c; do
  case $g in
    a) TILES=1,2,3,4,6,7,8,9;;
    b) TILES=5,10,14,15,19,20,23,24;;
    c) TILES=11,12,13,16,17,18,21,22;;
  esac
  echo "=== [$(date -u +%H:%M:%S)] stage 4$g: build_pdb24 k8_$g"
  target/release/build_pdb24 --zero-aware --tiles $TILES \
    --out data/pdb24_k8_$g.zbin --verify-sha data/pdb24_k8_$g.sha256
done

echo "=== [$(date -u +%H:%M:%S)] stage 5: sha256 sweep"
fail=0
for a in wd24.bin cwd_single.bin cwd_mm.bin cwd_lm.bin cwd_lm2.bin cwd_lm_mm.bin cwd_lm1l_mm.bin; do
  want=$(tr -d " \n" < data/$a.sha256); got=$(sha256sum data/$a | cut -d" " -f1)
  if [ "$want" = "$got" ]; then echo "OK   $a"; else echo "FAIL $a"; fail=1; fi
done
for g in a b c; do
  want=$(tr -d " \n" < data/pdb24_k8_$g.sha256); got=$(sha256sum data/pdb24_k8_$g.zbin | cut -d" " -f1)
  if [ "$want" = "$got" ]; then echo "OK   pdb24_k8_$g.zbin"; else echo "FAIL pdb24_k8_$g.zbin"; fail=1; fi
done
[ $fail -eq 0 ] && echo "=== [$(date -u +%H:%M:%S)] PIPELINE COMPLETE, ALL SHA OK" || { echo "=== SHA FAILURES"; exit 1; }
