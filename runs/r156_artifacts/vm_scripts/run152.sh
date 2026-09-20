#!/bin/bash
# Threshold 152, resuming from the existing checkpoint (144-150 already banked).
# Interruptible at any time: kill it and every completed unit survives.
set -uo pipefail
cd "$HOME/A087725"
. "$HOME/.cargo/env"
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
echo "=== [$(TZ=America/New_York date +"%-I:%M %p ET")] threshold 152 start (resume from ckpt156)"
target/release/solve24 --position "$R" --prove-at-least 154 \
  --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156
echo "=== [$(TZ=America/New_York date +"%-I:%M %p ET")] 152 RUN ENDED"
