# The 24-puzzle deep-board hunt: a bracket catalog of hard instances

**Result (2026-07-08).** A construct → score → bound → re-seed loop produced the
first **population of certified-deep 24-puzzle boards** — **191 instances across
two flywheel generations**, each bracketed by a **proven lower bound** (bounded
IDA\* exhaust) and a **replay-verified learned upper bound** (BWAS with a general
value net). **164 of the 191 carry a proven optimal-depth LB ≥ 132; 38 reach
≥ 138; 19 reach ≥ 140**; the canonical hard board **R** was pushed to a proven
**≥ 148** (our own, this session) with a matching **UB 156**. No such catalog
exists in the literature — the published 24-puzzle record is `R ∈ [152, 156]`,
random boards averaging ~100, and the diameter interval `[152, 205]`.

The re-seed loop **enriches for depth**: the fraction of boards with proven
LB ≥ 140 rose from 5% (generation 1) to 10% (after generation 2), and the mean
pool WD floor rose 127.1 → 129.3 — the deepest finds seed deeper candidates, not
just more of them.

This is a *lower-bound-side* result: we **populate and certify** deep boards; we
do not (and on this hardware cannot) prove any board deeper than R, nor move the
published 152 diameter floor.

---

## 1. What the hunt produces

A ranked, reproducible registry (`data/catalog24.tsv`) of hard boards, each
certified into an interval `[proven LB, replay-verified UB]`:

- **Floor** — a `solve24`/`ladder24` bounded exhaust. Every heuristic is
  consistent, so an exhausted IDA\* threshold is a theorem: `optimal(board) ≥ K`.
- **Ceiling** — a solution length from a general learned solver (BWAS guided by
  the frame2 value net), re-applied from scratch to confirm it reaches GOAL.

The bracket is what makes an entry scientific rather than suggestive: a board at
`[138, 160]` is a *certified-deep* instance whose true optimum is pinned to a
22-wide window; "WD says 128" is not.

## 2. The method (a loop, not a pipeline)

Five stages, each a committed tool:

1. **Generate** (`examples/candidates24.rs`) — seeds = `R`, `reflect(R)`, and
   frame-conformant constructions (the 2B frame rule, `puzzle24::frame`), plus
   flywheel re-seeds. Hill-climb by **solvability-preserving** mutations (3-cycles
   and double-swaps are even permutations, so the even-inversion invariant and
   the blank are untouched), scored by Walking Distance — an admissible score, so
   it is itself a free proven LB. Dedup under the reflection symmetry; a Hamming
   floor keeps the pool from collapsing onto near-`R` clones.
2. **Rank** (`gen_corridors --mode score`) — the frame2 forward net and the
   target-conditioned pair net score every board. Above WD's saturation ceiling
   (140, attained only by `R`) the learned value is the only signal that
   separates candidates.
3. **Bound below** (`ladder24 --from --mode bounded --out-tsv`) — parallel
   bounded IDA\* with the `select-k6` router (max(LC,WD) on deep boards), proven
   LBs at an escalating per-board budget.
4. **Bound above** (`gen_corridors --mode ubfile --out-tsv`) — BWAS with the
   frame2 net, every solution replay-verified.
5. **Join + flywheel** (`examples/catalog24.rs`) — an append-only evidence log
   reduced to the best bracket per board (join key = the reflection-canonical
   board, since `dist(s) = dist(reflect(s))` and ladder labels are
   line-index-relative). Emits the deepest boards as re-seeds for stage 1 and the
   wide-gap, budget-limited boards as an escalation shortlist.

## 3. The catalog — two flywheel generations

**Pools:** generation 1 = 100 boards (WD 124–140, mean pairwise Hamming 21);
generation 2 = 100 boards reseeded from gen-1's 15 deepest (WD 126–140, mean
floor lifted to 129.3). Merged and deduped under the reflection symmetry: **191
distinct boards**.

**Proven lower bounds** (60 s/board budget, `select-k6` = max(LC,WD)), across the
merged 191-board catalog:

| proven LB | boards | (gen-1 alone) |
|---:|---:|---:|
| ≥ 132 | **164** | 80 |
| ≥ 134 | 107 | 43 |
| ≥ 136 | 61 | 17 |
| ≥ 138 | 38 | 11 |
| ≥ 140 | **19** | 5 |
| ≥ 142 | 5 | 1 |
| **148** (R) | 1 | 1 |

The population roughly doubled while the deep tail grew **more** than
proportionally (LB ≥ 140: 5 → 19; the ≥ 140 *fraction* 5% → 10%) — the flywheel
enriches for depth. These LBs are **budget-limited, not fundamental**: 60 s/board
buys only a few plies above each board's root h (the ~29×/+2 wall, §4); every
board hit the 60 s cap. The escalation shortlists (`data/escalate_g{1,2}.txt`)
mark the boards worth a multi-hour exhaust.

**Upper bounds** (BWAS 1M-node, frame2 net): **200/200 solved and
replay-verified** across both generations, zero budget failures. Gen-2's UBs run
deeper — 136–**162** (one 162-move solution; 20 boards at 156, 12 at 158) vs
gen-1's 138–160 — longer solutions than R's 156, though from a suboptimal solver,
so they bound depth above, not below.

**Top brackets** (rank by proven LB, tie-break UB; merged catalog):

| proven LB | learned UB | gap | v_fwd | seed lineage |
|---:|---:|---:|---:|---|
| **148** | 156 | 8 | 163.4 | `R` |
| 142 | 158 | 16 | 159.9 | **gen-2 reseed** (deepest gen-1 finds) |
| 140 | **162** | 22 | 158.9 | **gen-2 reseed** |
| 140 | 160 | 20 | 158.1 | `R` + perturb |
| 138 | 160 | 22 | 157.7 | **frame#11** (non-R-derived, gen-1) |
| 140 | 158 | 18 | 157.3 | **frame#34** (non-R-derived, gen-2) |

Two things matter here. The **reseed rows** (gen-2 boards climbed from gen-1's
deepest) now populate the top of the catalog — the flywheel is finding new deep
boards *around* the deep ones, not just re-finding them. And the **frame-seeded
rows** are certified deep (proven LB ≥ 138, learned UB up to 160) while **not**
derived from R at all — exactly the non-`R`-anchored deep population the diameter
question needs.

## 4. R, and the calibration that governs everything

The 2A run proved **`optimal(R) ≥ 148`** by a bounded exhaust of threshold 146:
**998.0 billion nodes**, 1.9 h wall, 146 Mnodes/s across 8 threads. This both
raises our own proven floor on R (146 → 148) and **validates the ~29×/+2
node-growth model** — it predicted ~1.1 T nodes for this threshold; actual was
0.998 T (within ~10%). The bracket on R is now `[148, 156]`: our floor is our
own, and our ceiling equals the published best-known (the general learned solver
of FINDINGS_R.md, which never saw R).

The same ~29×/+2 wall governs the whole hunt: each +2 to a board's proven LB
costs ~29× the nodes. A 60 s pilot pass therefore lands a few plies above root h;
only a handful of finalists justify multi-hour exhausts, and **none can be pushed
to a 152-class floor here** (~29 T nodes / ~2.6 days for ≥150 on R; ~months for
≥152).

## 5. The honest frontier

- We **push and populate the lower-bound side**: a novel catalog of certified-deep
  non-R boards, plus an improved own-floor on R (≥ 148) and a UB matching the
  world best (156).
- We **cannot prove any board deeper than R**, and we do not move the published
  **152** diameter lower bound — that needs an LB > 152-class exhaust, calibrated
  at ~months/board on this hardware.
- The exact deepest boards and the **205** upper bound remain out of reach here
  (they need optimal deep solving / different techniques).

## 6. The flywheel — demonstrated, and continuing

Generation 2 **ran** (this session): reseeding candidates24 from gen-1's 15
deepest boards (`data/reseed_g2.txt`) produced a pool with a higher WD floor and,
after bounding, a deeper LB tail (§3) — the re-seed loop measurably enriches for
depth rather than merely accumulating boards. The two passes ran **concurrently**
(LB on 8 CPU threads, UB on the GPU), ~1.75 h wall for the generation.

The loop is primed to continue: `catalog24 --reseed-out` emitted the current 15
deepest as `data/reseed_g3.txt` (generation-3 seeds), and `--escalate-out` the 6
widest-gap, budget-limited boards as `data/escalate_g{1,2}.txt` for the finalist
treatment (multi-hour LB exhaust + `solve_ff` FF-MITM UB). Because the catalog is
append-only and joins on canonical board, each generation's evidence accumulates
and re-ingestion is idempotent — gen-2 added 91 new distinct boards and merged 9
that recurred, with no double-counting.

## 7. Reproducibility

- **Tools:** `examples/{candidates24,catalog24,frame24}.rs`,
  `src/bin/{ladder24 (example),gen_corridors,solve24}.rs`, `src/puzzle24/frame.rs`.
- **Pilot commands:** see PUZZLE24.md §"Pilot cycle (generation 1)".
- **Artifacts:** `data/pool_g{1,2}.txt` (the pools), `data/catalog24.tsv`
  (append-only evidence + brackets, both generations), `data/reseed_g{2,3}.txt`
  (each generation's deepest → next-gen seeds), `data/escalate_g{1,2}.txt`,
  `data/phase2a_calibration.txt` (the R ≥ 148 run). Raw per-board run TSVs
  (`runs/`) are regenerable and gitignored; their evidence is inlined in the
  catalog.
- **Nets:** `data/ml24_frame2` (forward UB solver), `data/ml24_pair` (pair-distance
  ranker) — see FINDINGS_R.md.

## 8. Summary

Starting from the observation that no population of known-deep 24-puzzle boards
exists, a construct→score→bound→re-seed loop produced **191 certified-deep
instances across two flywheel generations** — 164 with proven optimal-depth ≥ 132
(19 ≥ 140), R pushed to a proven ≥ 148 (UB 156), and a set of frame-seeded boards
certified deep *without* reference to R. The re-seed loop measurably enriched for
depth (LB ≥ 140 fraction 5% → 10% gen-1 → gen-2). The reusable result is the
**bracket**: a proven LB from
consistent-heuristic exhaustion and a replay-verified UB from a general learned
solver together pin each hard board's true optimum to a narrow window, turning
"probably deep" into "certified in `[LB, UB]`" — a new, extensible catalog for the
open diameter question.
