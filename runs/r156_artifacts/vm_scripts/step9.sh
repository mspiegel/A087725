#!/bin/bash
# Step 9 — the calibration gate, with E5 (checkpoint rehearsal) folded in.
#
# Exhaust-150 on the final configuration (SPLIT_TARGET=32768,
# K8_SHARED_BITS=27, W=64, no --hugepages). Started under --checkpoint, killed
# hard 30 minutes in, then resumed with the identical command: the resumed run
# both proves crash recovery at scale and IS the calibration.
#
# What it produces: nodes and wall at exhaust-150, hence the 148->150 growth
# ratio — the number that sets the 152 and 154 projections and therefore
# whether the proof is a ~2-month or ~8-month commitment.
#
# Reference: exhaust-148 = 114,245,221,757 nodes in 444.73 s at W=64.
# Expectation: ~3e12 nodes, ~3.5 h.
set -uo pipefail
cd "$HOME/A087725"
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
say() { echo "=== [$(TZ=America/New_York date +'%-I:%M %p ET')] $*"; }

while pgrep -x solve24 >/dev/null; do sleep 60; done
say "confirming production build and constants"
git log --oneline -1
grep -nE "^const (SPLIT_TARGET|K8_SHARED_BITS)" src/puzzle24/search/flat.rs
cargo build --release --features sha 2>&1 | tail -1 || exit 1

rm -rf runs/ckpt156; mkdir -p runs/ckpt156

say "part 1: exhaust-150 under --checkpoint (will be killed at 30 min)"
target/release/solve24 --position "$R" --prove-at-least 151 \
  --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156 > logs/step9_run1.log 2>&1 &
PID=$!
sleep 1800
say "E5: hard-killing pid $PID (SIGKILL — no cleanup, the harshest case)"
kill -9 $PID 2>/dev/null
wait $PID 2>/dev/null
sleep 2
say "run1 reached:"
grep -E "threshold 1(4[468]|50) exhausted" logs/step9_run1.log || echo "  (no threshold completed in run1)"
say "checkpoint dir after kill:"
ls -la runs/ckpt156/ | tail -n +2 | head -8
wc -l runs/ckpt156/*.ckpt 2>/dev/null | tail -3

say "part 2: resume with the identical command"
target/release/solve24 --position "$R" --prove-at-least 151 \
  --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156 > logs/step9_run2.log 2>&1
say "restore evidence:"
grep -E "checkpoint:" logs/step9_run2.log | head -6
say "result:"
grep -E "threshold 1(4[468]|50) exhausted|Lower bound|Nodes|Search time|Throughput" logs/step9_run2.log
say "STEP 9 COMPLETE"
echo done > logs/STEP9_DONE
