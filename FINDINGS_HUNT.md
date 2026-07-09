# The 24-puzzle deep-board hunt: a bracket catalog of hard instances

**Result (2026-07-09).** A construct → score → bound → re-seed loop produced the
first **population of certified-deep 24-puzzle boards** — **542 instances across
six flywheel generations**, each bracketed by a **proven lower bound** (bounded
IDA\* exhaust) and a **replay-verified learned upper bound** (BWAS with a general
value net). **504 of the 542 carry a proven optimal-depth LB ≥ 132; 204 reach
≥ 138; 106 reach ≥ 140; 19 reach ≥ 142**; the canonical hard board **R** was
pushed to a proven **≥ 148** (our own, this session) with a matching **UB 156**.
No such catalog exists in the literature — the published 24-puzzle record is
`R ∈ [152, 156]`, random boards averaging ~100, and the diameter interval
`[152, 205]`.

**The re-seed loop enriches for depth, then converges to a frontier.** The
fraction of each generation's pool proven at LB ≥ 140 climbed 6.9% → 19.6% →
31.4% → 37.3%, then **plateaued at ~33–39%** (generations 4–6) — a 5.4× enrichment
that saturates, the signature of the loop reaching the deep boundary its
generator + 60 s LB budget can reach. Meanwhile new-distinct-board yield stayed
healthy (~85–91 per generation, no collapse), so the *population* keeps growing
even as the *depth fraction* levels off: the reachable deep region is large, but
its certifiable-at-this-budget floor is a real frontier. Across all 542 boards
and 2,713 evidence rows, **zero LB > UB bracket inversions** — the two
independent solvers never contradicted each other.

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

## 3. The catalog — six flywheel generations

Six generations of 100 boards each (generation 1 seeded from R/reflect(R) +
frame-conformant constructions; generations 2–6 each reseeded from the previous
generation's 15–20 deepest, plus 40–50 fresh frame seeds). Merged and deduped
under the reflection symmetry: **542 distinct boards**, 2,713 evidence rows.

**Proven lower bounds** (60 s/board budget, `select-k6` = max(LC,WD)):

| proven LB | boards |
|---:|---:|
| ≥ 132 | **504** |
| ≥ 134 | 385 |
| ≥ 136 | 271 |
| ≥ 138 | 204 |
| ≥ 140 | **106** |
| ≥ 142 | 19 |
| **148** (R) | 1 |

**The depth-enrichment trajectory** (per generation, from each generation's own
LB pass — the headline result):

| gen | mean LB | max LB | LB ≥ 138 | LB ≥ 140 | frac ≥ 140 |
|---:|---:|---:|---:|---:|---:|
| 1 | 133.4 | 144 | 13 | 7 | 6.9% |
| 2 | 135.7 | 144 | 37 | 20 | 19.6% |
| 3 | 136.4 | 144 | 46 | 32 | 31.4% |
| 4 | 137.1 | 144 | 61 | 38 | **37.3%** |
| 5 | 137.2 | 144 | 57 | 34 | 33.3% |
| 6 | 137.4 | 144 | 59 | 40 | **39.2%** |

The deep fraction rises 5.4× (6.9% → 37%) over generations 1–4 then **plateaus at
~33–39%** — the loop has reached the deepest region its generator and 60 s LB
budget can certify. **Max proven LB is pinned at 144 in every generation**: that
is the 60 s budget ceiling (the ~29×/+2 wall, §4), *not* a generator limit — the
plateau is about how many boards reach the ceiling, not the ceiling itself. These
LBs are therefore **budget-limited, not fundamental**; the escalation shortlists
(`data/escalate_g{1..6}.txt`) mark the boards worth a multi-hour exhaust.

**Upper bounds** (BWAS 1M-node, frame2 net): **600/600 solved and
replay-verified** across all six generations, zero budget failures. UBs run
136–**164** (deeper than R's 156, from a suboptimal solver, so they bound depth
above not below); the deep UB tail thickened over generations (13 boards at UB
162, one at 164).

**Top brackets** (merged catalog, rank by proven LB then UB):

| proven LB | learned UB | v_fwd | seed lineage |
|---:|---:|---:|---|
| **148** | 156 | 163.4 | `R` |
| 142 | **162** | 163.4 | gen-4 reseed |
| 142 | 160 | 160.7 | gen-4 reseed |
| 142 | 158 | 161.4 | gen-6 reseed |
| 142 | 158 | 159.5 | `reflect(R)` + perturb |

18 boards reach proven LB **142** (the deepest non-R floor at this budget), and
the top of the catalog is dominated by **reseed-lineage** boards — the flywheel is
manufacturing new deep boards *around* the deepest, not merely re-finding them.
Frame-seeded boards remain certified deep **without** reference to R, the
non-`R`-anchored deep population the diameter question needs.

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

## 6. The flywheel — six generations, and what it converged to

All six generations ran (`scripts/hunt_generations.sh`, ~1.75 h each with the LB
and UB passes concurrent on CPU/GPU). The trajectory (§3) is the substantive
finding, and it has two distinct signals:

- **Depth enriches, then saturates.** The LB ≥ 140 fraction climbs steeply for
  four generations (6.9 → 37.3%) then flattens (~33–39%). Reseeding from the
  deepest finds keeps steering the generator into deeper territory *until it hits
  the boundary of what its construction prior + 60 s LB budget can certify* — a
  genuine frontier, not a bug. Pushing past it needs a different lever (bigger LB
  budget per board, against the ~29×/+2 wall) or a generator that reaches boards
  with root h above the WD ≈ 132–134 that non-R deep boards top out at.
- **Population keeps growing.** New-distinct-board yield stayed ~85–91/generation
  (100 → 91 → 91 → 87 → 83 → 90) with no collapse toward re-finding the same
  boards — the reachable deep region is large. Six generations → 542 distinct
  boards; more generations would keep adding breadth at a flat depth fraction.

The loop remains primed (`data/reseed_g7.txt`, `data/escalate_g{1..6}.txt`).
Because the catalog is append-only and joins on canonical board, every
generation's evidence accumulated idempotently — recurring boards merged with no
double-counting, and no generation ever produced a `LB > UB` inversion.

## 7. Reproducibility

- **Tools:** `examples/{candidates24,catalog24,frame24}.rs`,
  `src/bin/{ladder24 (example),gen_corridors,solve24}.rs`, `src/puzzle24/frame.rs`.
- **Driver:** `scripts/hunt_generations.sh <first> <last>` runs N generations
  back-to-back (generate → rank → LB∥UB → ingest). Pilot commands: PUZZLE24.md
  §"Pilot cycle".
- **Artifacts:** `data/pool_g{1..6}.txt` (the pools), `data/catalog24.tsv`
  (append-only evidence + brackets, all six generations), `data/reseed_g{2..7}.txt`
  (each generation's deepest → next-gen seeds), `data/escalate_g{1..6}.txt`,
  `data/phase2a_calibration.txt` (the R ≥ 148 run). Raw per-board run TSVs
  (`runs/`) are regenerable and gitignored; their evidence is inlined in the
  catalog.
- **Nets:** `data/ml24_frame2` (forward UB solver), `data/ml24_pair` (pair-distance
  ranker) — see FINDINGS_R.md.

## 8. Summary

Starting from the observation that no population of known-deep 24-puzzle boards
exists, a construct→score→bound→re-seed loop produced **542 certified-deep
instances across six flywheel generations** — 504 with proven optimal-depth ≥ 132
(106 ≥ 140), R pushed to a proven ≥ 148 (UB 156), and many frame-seeded boards
certified deep *without* reference to R. The re-seed loop measurably enriched for
depth and then **converged to a frontier** — the LB ≥ 140 fraction rose 5.4×
(6.9% → 37%) and plateaued — while the population kept growing (~85–91 new boards
per generation). The reusable result is the **bracket**: a proven LB from
consistent-heuristic exhaustion and a replay-verified UB from a general learned
solver together pin each hard board's true optimum to a narrow window, turning
"probably deep" into "certified in `[LB, UB]`" — a new, extensible catalog for the
open diameter question.
