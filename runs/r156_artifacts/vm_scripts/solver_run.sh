#!/bin/bash
# The proof run, as launched on boot by r156-solver.service.
#
# Everything here is idempotent: the checkpoint restores completed thresholds
# and finished units in under a second, so being started repeatedly (after a
# spot eviction, a reboot, or a manual restart) simply resumes.
#
# Controls:
#   logs/PAUSE   present -> exit immediately without searching. Use this to
#                stop deliberately without the supervisor fighting you.
#   logs/evictions.log   one line per start, so the eviction rate is
#                measurable after a few days.
set -uo pipefail
cd "$HOME/A087725"
. "$HOME/.cargo/env" 2>/dev/null || true

R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
TARGET=156            # --prove-at-least: exhausts through 154
CKPT=runs/ckpt156
mkdir -p logs "$CKPT"

stamp() { TZ=America/New_York date +'%Y-%m-%d %-I:%M:%S %p ET'; }

if [ -f logs/PAUSE ]; then
  echo "[$(stamp)] PAUSE present — not starting" >> logs/evictions.log
  exit 0
fi

# Uptime at launch distinguishes a fresh boot (eviction/restart) from a
# same-boot relaunch.
UP=$(awk '{printf "%d", $1}' /proc/uptime)
echo "[$(stamp)] solver starting (uptime ${UP}s, target >=${TARGET})" >> logs/evictions.log

exec target/release/solve24 --position "$R" --prove-at-least "$TARGET" \
  --clm2 --zpdb8 --parallel --checkpoint "$CKPT" \
  >> logs/proof_run.log 2>&1
