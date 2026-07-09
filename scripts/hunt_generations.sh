#!/bin/bash
# hunt_generations.sh — drive N Phase-2 flywheel generations back-to-back.
#
# Each generation: candidates24 (reseed from the previous gen's deepest + fresh
# frame seeds) -> gen_corridors --mode score (rank) -> LB pass (ladder24, CPU 8
# threads) ∥ UB pass (gen_corridors ubfile, GPU) CONCURRENTLY -> catalog24
# --ingest (append-only, per-generation durability). Uses the prebuilt release
# binaries directly (no cargo build-lock contention between the concurrent LB/UB).
#
# Usage: scripts/hunt_generations.sh <first_gen> <last_gen>
#   reads data/reseed_g<first_gen>.txt as the initial seed file; writes
#   data/pool_g<G>.txt, runs/{score,lb,ub}_g<G>.tsv, data/reseed_g<G+1>.txt,
#   data/escalate_g<G>.txt, and appends to data/catalog24.tsv.
set -euo pipefail

REPO="/Users/michaelspiegel/Projects/atom-planet-embrace/A087725"
cd "$REPO"
mkdir -p runs

CAND=target/release/examples/candidates24
LADDER=target/release/examples/ladder24
CATALOG=target/release/examples/catalog24
GC=target/release/gen_corridors
FWD=data/ml24_frame2/value_latest.safetensors
PAIR=data/ml24_pair/value_latest.safetensors

FIRST="${1:?first gen}"
LAST="${2:?last gen}"
PREV_RESEED="data/reseed_g${FIRST}.txt"

for G in $(seq "$FIRST" "$LAST"); do
  echo "===== generation $G start ($(date +%H:%M:%S)) ====="
  POOL="data/pool_g${G}.txt"
  SCORE="runs/score_g${G}.tsv"
  LB="runs/lb_g${G}.tsv"
  UB="runs/ub_g${G}.tsv"
  NEXT=$((G + 1))
  RESEED="data/reseed_g${NEXT}.txt"
  ESC="data/escalate_g${G}.txt"

  # 1. generate (reseed from the previous generation's deepest + fresh frame diversity)
  "$CAND" --out "$POOL" --reseed "$PREV_RESEED" --frame-seeds 50 --perturb 8 \
    --iters 2000 --restarts 4 --per-seed 12 --min-dist 6 --pool-cap 100 --seed "$G"

  # 2. learned ranking (GPU, quick)
  "$GC" --mode score --boards "$POOL" --checkpoint "$FWD" --pair-checkpoint "$PAIR" \
    --out-tsv "$SCORE"

  # 3. LB pass (CPU) ∥ UB pass (GPU) — concurrent, different hardware
  RAYON_NUM_THREADS=8 "$LADDER" --pdb-dir data --heuristics select-k6 --mode bounded \
    --from "$POOL" --bound-budget-secs 60 --max-bound 200 --parallel --quiet --out-tsv "$LB" &
  LB_PID=$!
  "$GC" --mode ubfile --boards "$POOL" --checkpoint "$FWD" --bwas-budget 1000000 \
    --weight 2.5 --out-tsv "$UB" &
  UB_PID=$!
  wait "$LB_PID"
  wait "$UB_PID"

  # 4. join into the append-only catalog; emit next-gen seeds + escalation shortlist
  "$CATALOG" --catalog data/catalog24.tsv --ingest --date "g${G}" \
    --pool "$POOL" --lb-tsv "$LB" --ub-tsv "$UB" --score-tsv "$SCORE" \
    --rank 10 --reseed-out "$RESEED" --reseed-top 20 \
    --escalate-out "$ESC" --escalate-top 6 --gap-min 12

  PREV_RESEED="$RESEED"
  echo "===== generation $G complete ($(date +%H:%M:%S)) ====="
done
echo "ALL GENERATIONS COMPLETE"
