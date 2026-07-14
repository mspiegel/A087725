# Solving the R board with a learned general solver: 204 → 156

**Result (2026-07-08).** A learned value network — trained by adversarial
co-training and used to guide search, and which **never saw the R board or any
state on its solution path** — solves R in **156 single-tile moves**, matching
the published best-known solution. The solution is independently replay-verified
(`data/r156_ours_solution.txt`). The literature's 156 was *hand-constructed* from
R's rotational symmetry; ours was *discovered* by generic learned search.

`optimal(R) ∈ [152, 156]` (both bounds published; see below). Our upper bound now
**equals the world's best**, four above the proven floor.

---

## 1. The problem

The **R board** is the 24-puzzle (5×5 sliding puzzle) goal rotated 180°: blank at
cell 0, tile `25 − i` at cell `i` (row-major `0 24 23 … 2 1`). It is the canonical
hard instance — the unique board (up to symmetry) at the global Walking-Distance
maximum, `WD(R) = 140`, and the position behind both published diameter bounds.

**Published bounds** (Domain of the Cube Forum, node 238, linked from OEIS
A087725; Rokicki / Hannanov):
- **Lower bound 152** — Rokicki's optimal solver exhausted a ply-150 search on R
  with no solution (~10 days); parity forces the next candidate to 152.
- **Upper bound 156** — constructive: the 90°-rotated goal `W = ρ(GOAL)` solves
  optimally in 78 moves, so two consecutive quarter-turn solves give a 78 + 78 =
  156 solution to the 180° board. Rokicki found 12,225 length-156 solutions and
  none shorter, believing R is *exactly* 156.

So `optimal(R)` is unknown in `[152, 156]`. The exact value is an open problem;
optimally solving a WD-140 board is at/beyond the edge of feasibility (proving
≥ 152 on a single 32 GiB machine was calibrated at ~75 days — since cut to **~2
days** by cWD plus duplicate-pruning and root-symmetry levers, §8–8b, with **R ≥ 150
now proven**; proving ≥ 156 — exact optimality by exhaustive search — stays out of
reach: extending the measured ~×33-per-+2 node growth three thresholds past R ≥ 150
lands at **years-to-decades** even with these levers, and the ratio is expected to
climb at depth).

We attack the **upper bound**: find the shortest solution a *general* learned
solver can produce, and see how close a method with no hand-tuned knowledge of R
gets to the human/literature best.

## 2. The method

An adversarial co-training loop (DeepCubeA-style):
- **Solver** — a residual-MLP value network `V(s)` estimating cost-to-go,
  trained by DAVI (Bellman bootstrap toward `min_neighbor(1 + V)`), and consumed
  at solve time by batch-weighted-A\* / beam search.
- **Generator** — produces training boards; the key innovation was replacing a
  learned walk-policy (which plateaus at shallow depth) with a **WD-maximizing
  beam** (`generator-as-search`) that constructs genuinely deep boards.
- **Admissible base** — Walking Distance (`WD`), the strongest admissible
  heuristic for this puzzle; `WD(R) = 140` is exact and available at R even
  though R is far outside anything trainable.

All solutions below are **replay-verified** (the move sequence is re-applied to R
from scratch and checked to reach GOAL); the two headline results are additionally
verified by an *independent* from-scratch checker.

## 3. The ladder — and what each rung taught

| R (moves) | method | the lesson |
|---:|---|---|
| **204** | WD-beam baseline | classical search, no learning |
| **174** | direct net + anytime weighted-A\* | learning helps; loose walk-length labels give an *inflated but monotone* V that guides greedy search |
| **168** | meet-in-the-middle (front-to-end) | MITM beats unidirectional, but the meet is forward-dominated (150/18 split) — a wide forward beam with an optimal short endgame, not a true middle-meet |
| **164** | hybrid front-to-front MITM | aiming each beam at the *other frontier* balances the meet, but caps here due to **heuristic asymmetry** (strong forward V, weak backward WD-to-R) |
| **160** | frame-corridor-trained net + FF-MITM | training V on optimal *geodesic-corridor* states with tight labels, extended past the WD-search depth ceiling by **frame-rule** board generation |
| **156** | deeper-frame-corridor net + FF-MITM | pushing the training labels to depth ~170 lets V stay monotone-informative all the way to R's depth ~156 |

Two rungs are worth their own note because they were *counterintuitive*:

- **The residual experiment (a productive failure).** Reparameterizing the net to
  predict `optimal − WD` (so `V = WD + residual`) made V dramatically *more
  accurate* at R — `V(R) ≈ 146`, near the true value, versus the direct net's
  inflated ~195. Yet R got **worse** (174 → 190 → 206 as V grew more accurate).
  The mechanism: weighted-A\* with an *inflated* heuristic drives hard toward the
  goal and pops a solution; an *accurate/admissible* one behaves like true A\*,
  spreads over the f-contour, and returns a *longer* solution within budget. This
  produced the governing law of the whole project (§4).

- **Meet-in-the-middle needs symmetric guidance.** Front-to-front genuinely
  balances the meet (measured 96/96 split), but with only weak Manhattan guidance
  it finds *long* balanced solutions (192). The strong-base + FF-convergence hybrid
  is what recovers short *and* balanced.

## 4. The governing law

> **R-solve quality tracks whether `V` stays *monotone-informative* above depth
> ~126 — not whether `V` is accurate.**

Every net was probed against R's held-out 156 corridor (the replicated published
solution, used only as an out-of-distribution measurement, never for training):

| net | V(R) | R solved |
|---|---|---|
| direct (loose labels, inflated-but-monotone) | ~195 | **164** |
| residual (accurate, near-truth) | 146 | 190-206 |
| corridor (best-calibrated; but **saturates** at its 126 label ceiling) | 127.6 (< the WD floor!) | 174 |
| frame-corridor (labels to ~150) | 159.5 | **160** |
| deeper-frame-corridor (labels to ~170) | 163.4 | **156** |

The corridor net was the *best-calibrated* net we built (held-out error +40 → −13)
yet a *worse* R-solver, because it saturated: its deepest training label was ~126,
so it predicted **below** the provable WD floor on deeper states and flattened the
ranking exactly where R's first moves live. A `max(V, WD)` inference clamp repaired
the *value* but not the *ranking* — confirming ranking, not accuracy, is what the
search consumes.

## 5. The coverage fix (what actually got us from 164 to 156)

The corridor diagnosis (comparing V along R's true 156 geodesic vs. our own path)
localized the failure to **label coverage above depth 126**. Our board generator
provably plateaus at `WD ~126` — R lives at 140 — so V had *no training signal*
in the 126-152 band and could only extrapolate. The fix, in three parts:

1. **A non-R deep-board generator.** The 15-puzzle *frame rule* (deep boards have
   their corner/frame tiles in the reversed arrangement) generalizes to the
   24-puzzle: frame-conformant construction shifts the proven-lower-bound
   distribution +30 over random (mean WD 108 vs 78, near-disjoint), and bounded
   search certifies these boards at **proven depth ≥ 132** with true depth ~150 —
   squarely in the band the WD-search constructor cannot reach, and *not anchored
   on R*.

2. **Optimal geodesic-corridor labels.** Solving these deep boards yields their
   entire solution paths; suffix-optimality makes every path state's label exact
   (or, for the deep tail, a near-tight upper bound — audited at **mean slack
   0.04** in the exactly-solvable band). This replaced the old loose walk-length
   labels (+40 error) that had corrupted earlier training.

3. **Retrain, no ceiling.** Fine-tuning on exact + WD-search + frame corridors
   (labels to ~170) makes V monotone-informative to depth ~156. The out-of-
   distribution probe on R's corridor flattened to a uniform +5..+12 across all
   depth bands, and the search cashed the deep gradient into the 156-move solution.

A negative result that *closed* a line: the frame rule holds for board *positions*
but **not** for solution *corridors* — measured directly (frame-conformance is
100% at the antipode, 0% one move in) — so "frame-waypoint search" has no basis.
The frame rule is a *generator* prior, not a search prior.

## 6. What it would take to close the last four (156 → 152)

This is where the project meets the open problem. `optimal(R)` may not even *be*
152 — the true value is unknown in `[152, 156]`, and 156 may be optimal. Closing
the gap means either:
- **Finding a shorter solution** — but the learned upper-bound side is near its
  ceiling: frame boards cap at true depth ~162, and a bigger-capacity net or
  optimal bidirectional search would be a large investment with uncertain payoff;
- **Proving the lower bound higher** — the cWD heuristic plus duplicate-pruning and
  root-symmetry levers (§8–8b) have since made this far cheaper (R ≥ 150 is now
  *proven*, and reaching the published 152 is a ~2-day run rather than ~75 days), but
  a higher proven floor still would not lower *our* solution.

We have matched the published frontier with a general method; the remaining four
moves are a research problem, not an engineering one.

## 7. Reproducibility

- **Champion net:** `data/ml24_frame2/` (residual MLP, hidden 1024 / 6 blocks).
- **Solution:** `data/r156_ours_solution.txt` (156 moves, replay-verified).
- **Solve command:**
  ```
  solve_r --checkpoint data/ml24_frame2/value_latest.safetensors --hidden 1024 --blocks 6 \
          --bidir-ff --width 20000 --max-layers 300 --budget 40000000 \
          --ff-anchors 16 --ff-base-weight 2.0 --ff-weight 0.5
  ```
- **Training data:** `data/corridors_{exact,deep,frame,frame_deep}.txt`
  (`gen_corridors --mode {exact,frame}`); frame boards from the constructor in
  `src/puzzle24/ml/corridor.rs` / `examples/frame24.rs`.
- **Key tools:** `src/puzzle24/ml/{bidirectional,corridor,davi,wdsearch}.rs`,
  `src/bin/{solve_r,gen_corridors,train_ml24}.rs`, `examples/{frame24,corridor15}.rs`.
- **Provenance / bounds:** OEIS A087725; forum.cubeman.org node 238 (fetch with a
  browser User-Agent). See the R appendix in `PUZZLE24.md`.

## 7b. Follow-on: a general target-conditioned pair-distance net

The winning FF-MITM solver is lopsided (the backward beam contributes ~10%),
because the backward heuristic (WD-to-target) is weak next to the learned forward
V. We built the general fix: a **target-conditioned** value net `V(x | b) ≈
dist(x, I_b)`. The basis is that tile-relabeling is a graph automorphism, so
`dist(x, t) = dist(relabel_t(x), I_b)` for `b` = t's blank cell — backward search
toward *any* target becomes forward search in the target's relabeled frame. One
net conditioned on the blank-class does it; training labels are free from
corridor infixes (`dist(sᵢ, sⱼ) = j − i` on an optimal path). Verified end to
end (WD-invariance of the relabeling; zero-init conditioning reproduces the
forward net; 158k pair labels, all 25 blank-cells covered).

Trained (warm-started from the R=156 net, 40k steps): **held-out pair RMSE ~1-2
across all distance bands** — an accurate any-board-pair distance predictor.
Three findings on *using* it:

- **It reproduces the strong-backward result generally.** `frame2`-forward +
  pair-net-backward solves R in **156** (120/36 split) — matching the champion,
  with the *learned* backward contributing 3× more than weak WD-to-R (36 vs 12),
  and *without* any R-specific symmetry.
- **It ties-or-beats Manhattan on unseen targets.** On held-out frame boards:
  156=156 (tie), and 152 vs 154 (win). Equal-or-better, never worse, general —
  but a *modest* margin, because the forward-dominated meet limits how much the
  backward binds (the split is a thermometer of backward strength: 12 → 36 → 78
  as WD-to-R → pair-net → forward-strength symmetry, all at length 156).
- **It is not a better forward *solver*.** Used as the forward heuristic, its
  accuracy *hurts* greedy solution length (R = 198) — the same
  accurate-heuristic-hurts-greedy-search wall seen with the residual and corridor
  nets. Calibration ≠ search-guidance.

**Verdict:** the pair net's value is as **infrastructure** — an accurate any-pair
distance / eccentricity oracle and a general (non-symmetry) backward heuristic —
directly serving the *diameter* goal (arbitrary distance queries, eccentricity
estimation, non-R-anchored deep-board generation), not as a lever to push R below
156 (which Phase 0.5 had already shown a strong backward cannot do).
Reproducibility: `train_pairnet` / `solve_ff --fwd-checkpoint`; net `data/ml24_pair`;
pairs regenerable via `gen_corridors --mode frame --pairs-out`.

## 8. The other half: accelerating the *lower*-bound proof with cWD

Everything above pushes R's solution *down* toward the floor. The complementary
question is how high we can prove the floor *is* — an exhaustive IDA\* that certifies
no solution shorter than `N` exists. On this machine the proven bound had reached
**R ≥ 148** (2026-07-08; WD, 998 B nodes, 1.90 h), and reaching the published 152
was calibrated at **~75 days** — infeasible. The lever is a stronger *admissible*
heuristic: every extra unit of `h` prunes the exhaustive tree geometrically.

**cWD — escape-constrained Walking Distance.** WD is exact on the row/column
sorting relaxation but blind to one thing: tiles that belong in a row yet sit there
in reversed order cannot sort *in place* — some must physically leave the row and
re-enter. cWD adds exactly that as a *side constraint* on WD's own move budget
(never an addend, so admissibility is preserved): for each goal line `g`, any plan
must make at least `x_g = residents − LIS(their goal-cross order)` escape moves.
On R this gives `cWD(R) = 144` (WD 140 + 4). The forced-escape bound that makes it
sound is **machine-checked in Lean 4 / mathlib** (`proofs/puzzle15-wd`, no `sorry`;
Manhattan and WD admissibility too) — correctness insurance, since a mis-derived
heuristic would certify a *false* lower bound.

**From +4 at the root to a fast solver.** A root gain is worthless if it doesn't
propagate. Sampling R's actual search tree measured the per-node surcharge
`δ = cWD − WD`: **node-weighted `δ̄ ≈ 4.26`** over a fully-exhausted iteration — the
+4 is tree-wide, not an R-special case. That justified building a table:
- the full sharp multi-line table is memory-infeasible, but a **single-line-max**
  approximation retains **98%** of the node-weighted gain (measured) and *is*
  buildable — a per-contingency surcharge curve over all 65.65 M WD states
  (`data/cwd_single.bin`, SHA-pinned; validated three ways, including matching a
  reference constrained-A\* on thousands of entries and reproducing WD exactly on
  every contingency);
- an **incremental evaluator** (a move changes at most one line's demand) brings
  per-node cost to **within ~6% of WD**, verified against full recompute over
  30 000 make/unmake steps.

**Measured on R** (parallel, 8 threads; each row is the same exhaustive proof):

| proof | exhausts thr | WD | cWD | node reduction | wall speedup |
|---|---|---|---|---|---|
| R ≥ 146 | 144 | 36.9 B / 4.8 min | 1.78 B / 14.9 s | **20.7×** | **19.4×** |
| R ≥ 148 | 146 | 998 B / 1.90 h | 85.1 B / 10.8 min | **11.7×** | **10.5×** |

cWD re-proved the current record (R ≥ 148) in **11 minutes** instead of 2 hours.

**The honest caveat — the reduction decays with depth.** 20.7× at exhaust-144
fell to 11.7× at exhaust-146 (~1.8× per +2 threshold): deeper searches admit more
near-solved, low-surcharge nodes, diluting the gain. Extrapolating the *shallow*
number overstates the deep proofs. Tempered (2-point decay, may plateau):

- **R ≥ 150** (exhaust 148): ~6× wall → **~10 hours**
- **R ≥ 152** (exhaust 150): ~3–4× wall → **~2–3 weeks**

*(Both since measured with three further levers — see §8b. R ≥ 150 was proven in
**1.56 h of search**, not ~10 h; R ≥ 152 is revised down to **~2 days**.)*

So cWD does not *lower* R's solution (§6's upper-bound problem), but it turns the
lower-bound proof toward the published 152 from *infeasible* (~75 days) into a
**feasible multi-week run** — the first time reaching the published floor is on the
table for this hardware.

*Reproducibility:* `solve24 --heuristic cwd --prove-at-least N --parallel`; table
build `examples/build_cwd_table.rs` → `data/cwd_single.bin`; heuristic
`src/puzzle24/search/cwd.rs`; soundness `proofs/puzzle15-wd`; calibration
`data/phase2a_calibration.txt`.

## 8b. Three more levers — and R ≥ 150 *proven* (2026-07-11)

Since the §8 cWD baseline, three further node-cutters were stacked on the proof —
each **sound for a lower-bound proof** (optimality-preserving, so an exhausted
threshold stays a valid floor) and all on by default in `solve24`:

- **Move-pruning DFA** (Taylor–Korf duplicate-subtree elimination; 41,396 states /
  687 KiB, L2-resident, history window 11). Skips a move whose subtree is provably
  reached by an equal-or-shorter sequence. ~13–19% fewer nodes, **growing with
  depth**; the `build()` self-verifies it caught every duplicate (panics otherwise,
  never running an unsound pruner).
- **cWD neighbor-prune.** A child is bounded from its parent's cached neighbor-WD
  and pruned *before* its position is even computed/probed — ~1.9× fewer nodes.
- **Root-orbit-split** (OPTIMIZATION.md lever #1). R is a fixpoint of the
  goal-preserving diagonal reflection σ (`reflect(R) == R`), so the root's children
  fall into σ-orbits; R's corner blank gives a single orbit `{Down, Right}`, so the
  proof searches **one** representative → a clean, near-2× cut (measured 1.99×) that
  does not decay with depth. `solve24` auto-detects the symmetry and applies it only
  at the true root (deeper canonical pruning would be unsound). Soundness rides on σ
  being a length-preserving automorphism *plus* the DFA's own optimality-preservation
  — it needs no heuristic symmetry.

`solve24` also now brackets the search with `search: start`/`search: end` timestamp
logs and reports **search time separately from table-load setup** (~24 s to load
`data/wd24.bin` + `data/cwd_single.bin`). The numbers below are search-only.

**Measured on R** (parallel, 8 threads; full stack cWD + DFA + neighbor-prune +
root-orbit-split; search time excludes setup):

| proof | exhausts thr | nodes | **search time** | throughput |
|---|---|---:|---:|---:|
| R ≥ 146 | 144 | 422 M | 3.31 s | 128 M/s |
| R ≥ 148 | 146 | 18.19 B | 2.63 min | 115 M/s |
| **R ≥ 150** | 148 | **595.86 B** | **1.56 h** (5,633 s) | 106 M/s |

**R ≥ 150 is now proven** — a new record for this machine — in **1.56 h of search**,
where §8 had projected ~10 h. Against the §8 cWD-only baseline the node count fell
**4.7×** at R ≥ 148 (85.1 B → 18.19 B); root-orbit-split alone contributes ~1.9× of
that, essentially undiminished at depth (unlike the surcharge gain, which §8 showed
decaying). The three levers are multiplicative on nodes and near-free per node, so
throughput stays compute-bound at ~106–128 M nodes/s.

**Revised path to the published floor.** Node growth per +2 threshold is
*decelerating* — ×43 (144→146) then ×32.7 (146→148). Extrapolating ~×33 from the
measured 595.86 B: **R ≥ 152** (exhaust 150) ≈ **~20 T nodes ≈ ~2 days** of search
on this machine — down from §8's ~2–3 weeks and the original ~75-day calibration.
Reaching the published lower bound of 152 is now a weekend run, not a research
gamble. Two thresholds beyond that — **R ≥ 156** (exhaust 154), i.e. proving R
*optimal* by exhaustion — is a further ~×33² ≈ 1,000×, so **years-to-decades** even
on this stack (and the per-+2 ratio is expected to climb back toward the raw
branching factor at depth, making that a floor). Exact optimality via exhaustive
search stays out of reach: the [152, 156] gap remains a research problem (§6), not a
compute one.

**Extrapolated ladder** (from the measured exhaust-148, applying ~×33 per +2; throughput
held at ~90–100 M nodes/s — an order-of-magnitude projection, uncertainty compounding
per step):

| exhaust thr | proves | est. nodes | est. search |
|---|---|---:|---:|
| 150 | R ≥ 152 | ~19.7 T | ~2 days |
| 152 | R ≥ 154 | ~650 T | ~11 weeks |
| 154 | R ≥ 156 | ~2.1×10¹⁶ | **~7 years** (floor; ~decades if the ratio climbs) |

The exhaust-154 figure is optimistic — it assumes the ×33 ratio *stays* flat, whereas
the shallow-friendly gains (surcharge, duplicates) saturate at depth and push the
ratio back toward ~×45–50. And it is moot for *pinning* optimal(R): exhaust-154 only
proves R ≥ 156, which closes the problem **iff** 156 is truly optimal; if optimal(R) <
156, that search instead *finds* the shorter solution below threshold 154 and stops early.

*Reproducibility:* identical command as §8 with the defaults on (DFA, neighbor-prune,
and root-orbit-split auto-enable); e.g. `solve24 --position "0 24 23 … 2 1"
--parallel --heuristic cwd --max-bound {146,148}`. Levers:
`src/puzzle24/search/{move_dfa,cwd,symmetry}.rs`; the split-loop orbit filter and
search-timing logs in `src/puzzle24/search/idastar.rs` / `src/bin/solve24.rs`;
ranking rationale in `OPTIMIZATION.md`.

## 8c. Lazy combiner evaluation — the cWD+zPDB max becomes deep-board-viable (2026-07-13)

The §8b-era `cwd-zpdb` combiner (`MaxInc(Cwd, ZpdbInc)`, commit bfc5215) buys a
node reduction that *grows* with depth (1.39× @146 → 1.73× @148) but paid an
eager ~2.4–2.6× per-node cost — break-even was extrapolated at thr 152–154.
**Lazy evaluation moves the break-even to ≈ thr 150.**

Mechanism (`--heuristic cwd-zpdb-lazy`, `LazyMaxInc` in
`src/puzzle24/search/heuristic.rs`; `IncHeuristicMut::make_bounded` hook in
`idastar.rs`): the search passes each child's pruning budget `bound − (g+1)` to
the heuristic, and the combiner skips the zPDB advance whenever cWD alone
already prunes the child — the dominant per-node cost (137 ns zPDB probe) is
paid only on children cWD fails to prune. **Sound and search-tree-identical**:
the returned value is always admissible (cWD's own value), and a skipped child
was pruned at first touch either way (unit test proves node-identity end-to-end;
all A/B node counts below match bit-for-bit).

Measured on R (full default stack, same machine/session, search-only time;
log `data/cwdzpdb_lazy_ab.txt`):

| thr | engine | nodes | search | vs pure cWD |
|---|---|---:|---:|---:|
| 146 | pure cWD | 18.19 B | 158.6 s | 1× |
| 146 | cwd-zpdb (eager) | 13.09 B | 266.3 s | 1.68× slower |
| 146 | **cwd-zpdb-lazy** | 13.09 B | 237.0 s | **1.49× slower** |
| 148 | pure cWD (§8b) | 595.86 B | 5,633 s | 1× |
| 148 | cwd-zpdb (eager, §8b) | 343.84 B | 8,508.8 s | 1.51× slower |
| 148 | **cwd-zpdb-lazy** | 343.84 B | **6,755.7 s** | **1.20× slower** |

Wall gap vs pure cWD shrinks 1.49× → 1.20× per +2 threshold while the node
reduction keeps growing — extrapolation puts **break-even at exhaust-150** (the
R ≥ 152 run: coin-flip, pure cWD still fine) and a **clear combiner win at
exhaust-152+** (the ~11-week R ≥ 154 scale, where ~20 %+ faster is ~2 weeks
saved). The complementary heuristic is no longer a wall-clock loser at depth.

Negative result, so it isn't re-proposed: a stronger **Lipschitz-deferral**
(defer the zPDB across interior moves, catching up only when `v + k` could
crest the budget — sound since the zPDB is exactly 1-Lipschitz per move) was
built and measured **dead**: +7 % alone, ~0 % on top of the prune-skip. In
IDA* the budget shrinks 1/level while the pending slack grows 1/level, so the
deferral window closes at 2/level and the bottom-heavy tree catches up almost
everywhere; zPDB probes concentrate exactly where the nodes are. The skip-on-
prune case (kept) is the entire win.

## 8e. Off-R validation — the combiner is a *deep-board* heuristic, and its design space is closed (2026-07-13)

Every §8c measurement was on R — the one board where cWD is pathologically
strong (R is max-WD). Measured on three generic deep catalog boards (proven
LB 142 via cheap root, UB ≥ 158; `data/catalog24.tsv` g2/g3/g4), exhausting
threshold 144 with the full default stack (log `data/cwdzpdb_deepboard_ab.txt`):

| board | engine | nodes | search | node red. | wall vs cWD |
|---|---|---:|---:|---:|---:|
| B1 (g2) | pure cWD | 269.59 B | 2,927.7 s | — | 1× |
| B1 | cwd-zpdb-lazy | 145.27 B | 3,039.0 s | **1.86×** | 1.04× slower |
| B2 (g3) | pure cWD | 192.57 B | 1,901.2 s | — | 1× |
| B2 | cwd-zpdb-lazy | 108.53 B | 2,212.2 s | **1.77×** | 1.16× slower |
| B3 (g4) | pure cWD | 73.87 B | 727.3 s | — | 1× |
| B3 | cwd-zpdb-lazy | 47.54 B | 964.5 s | **1.55×** | 1.33× slower |

Versus R@146 (1.39× reduction at 1.49× slower): on generic deep boards the
zPDB complement removes ~45 % of the tree at near wall parity **already at
gap-2 thresholds**, so with the depth-growth of the node reduction (§8c) the
break-even sits at ≈ thr 144–146 here instead of R's ≈ 150. The complementary
heuristic's home turf is exactly the general deep-board regime Phase 1C's
structural argument predicted. Byproduct: all three boards' proven LBs rose
142 → **≥ 146**.

Three neighboring designs measured DEAD the same day (same log), closing the
combiner's local design space:

- **k7 tables (7-7-7-3) in the combiner, on R @146**: 15.48 B / 311.0 s vs
  k6's 13.09 B / 237.0 s — the weak 3-group partition loses at the root (120
  vs 126) *and* at depth, and its 508 MB tables probe slower. (Consistent
  with Phase 1C's "k=7 retired from tuning".)
- **Reflection-frame probe**: not a candidate — `ZpdbInc` already Korf-maxes
  the normal and diagonal-reflected views on every node; it has been inside
  every measurement all along.
- **Two-partition compound complement** (`cwd-zpdb2-lazy` =
  `LazyMaxInc(cWD, max(zpdb_canonical, zpdb_rowband))`, new row-band
  6-6-6-6 set `data/pdb24_rb_*.zbin` (SHA-pinned; ~30 s each to rebuild via
  `build_pdb24 --zero-aware --tiles …`), groups {1–6}{7–12}{13–18}{19–24};
  the CLI wiring was measured and then reverted, not kept):
  B3 @144 39.46 B / 1,276.2 s, B1 @144 129.51 B / 4,352.1 s. The marginal
  node reduction over the single complement (1.20× / 1.12×) is well under its
  marginal probe cost (~1.5–1.6× per unpruned node) on both boards — a net
  wall loss at any practical threshold. The codec-spec §5 collection lever
  does not survive contact with the lazy-gated cWD combiner: cWD already
  prunes most of what a second partition would.

Net: **single-partition k6 zPDB, lazily maxed under cWD, is the right-sized
complementary heuristic for deep boards** — validated off-R, with k7 /
reflection / compound / Lipschitz-deferral all measured and closed around it.

## 8f. Congestion location is free; multi-line coupling is real but marginal (2026-07-13)

Two candidate second-order cWD strengtheners tested end-to-end
(`examples/center_congestion.rs`, log `data/center_congestion_probe.txt`).

**Center-congestion hypothesis — NULL, twice.** Hypothesis: congestion near
the board center costs more than the same congestion at the edges.
(a) *Controlled*: identical within-line motifs (3-cycle, double swap) placed
in every row/column of an otherwise-solved board, solved exactly. Slack
`d* − cWD` is flat across placements (6 for every rot3 position, 14 for every
swap2, 16 only in the blank's own line) — cWD's escape constraint already
charges blank travel to the congested line, and location adds nothing.
(b) *Observational*: 320 random-walk boards (len 70/90) solved exactly;
slack correlates ≈0 (|r| ≤ 0.13, mixed sign) with every location feature —
center-boundary WD-flow share, misplaced-tile center mass, center-line escape
demand — including partials controlling for total congestion. **Closed.**

**Multi-line joint escape demand — real gap, marginal payoff.** The motif scan's
one deviation (motifs in rows {0,4}: slack 12 vs 8 for all other pairs) traced
to a genuine soundness-preserving gap: the production fast path charges
`max` over *single-line* surcharge curves, while the joint demand-vector A\*
charges more when demands span ≥2 lines. Measured on R:

- Root: fast = joint = 144 on R (and on the three §8e catalog boards) — no gap.
- Interior (thr-146 reservoir, 6k samples): joint−fast = **+0.21/node**,
  growing with depth (0.12 @ d24–31 → 0.27 @ d40–47); +2 on 10.3% of nodes.
  Pair-restricted demand captures it all (pair 0.2090 vs full 0.2093) —
  the coupling is entirely pairwise.
- **Real A/B** (exhaust R @ thr 144, identical serial driver both arms):
  726,797,221 nodes fast vs 708,142,073 with the pair bonus =
  **1.026× node reduction** — the δ̄-predicted 29^(0.21/2) ≈ 1.4× collapses
  to 2.6%. Second confirmation of the §8c lesson: sampled per-node δ̄ wildly
  overestimates IDA\* pruning (a +2 bonus only kills nodes already within 2
  of the threshold); only exhaustive A/B counts.
- Engineering note that survives: the joint/pair value depends only on
  `(contingency key, demand vector)`, and the thr-144 exhaust touched just
  **18,155 distinct pairs** (99.98% memo hit rate) — a lazily-memoized
  surcharge cache is essentially free, so if a future variant finds a
  bigger per-node gap, no offline table build is needed.

Verdict: pair-coupling surcharge **closed as a wall-clock lever** (≤ ~3%
nodes on R; demands on generic deep boards are too sparse for it to fire).
cWD's remaining 12–16 band slack is not location, not pairwise line
coupling, and not PDB-reachable — whatever closes it must couple the axes
or see actual tile identity at range.

## 8g. Slack anatomy — the missing 12–16 is transit-yield churn (2026-07-13)

Rather than test another hypothesis, we instrumented where the missing moves
physically go: `examples/slack_anatomy.rs` replays optimal paths and accounts
for every move against the WD/cWD charges, using the exact per-axis identity

    V = F_v + 2·RT_v      ⇒      slack_v = V − WD_row = 2·RT_v − (WD_row − F_v)

(every vertical move crosses exactly one row boundary; F = crossings forced by
start→goal line intervals; RT = excess round-trip pairs, i.e. churn). The
identities are asserted per board. Populations: the §8f random-walk boards
re-solved to optimality (`walks` mode, now multi-threaded — 8 workers gave
~5.4×, capped by the slowest board), plus R's 156-move path (`rpath` mode).

| population (n, d̄)   | slack d−cWD | churn 2·RT | corr(slack,2RT) | RT home-adj | RT far | cWD demand |
|----------------------|------------|-----------|-----------------|-------------|--------|------------|
| len-70  (200, ~47)   | 12.31      | 15.18     | 0.900           | 6.83        | 0.76   | 1.14       |
| len-90  (120, ~56)   | 14.30      | 17.47     | 0.888           | 7.83        | 0.90   | 1.41       |
| len-110 (50,  ~63)   | 15.48      | 18.56     | 0.868           | 8.24        | 1.04   | 1.42       |
| R 156-move path      | 12         | 44        | —               | 13          | 9      | 8          |

Findings:

- **Slack is churn, nearly 1:1, on band boards.** WD sits almost exactly at
  its naive flow floor there (over-flow charge ~1.2), so slack ≈ 2·RT; the
  per-board correlation is 0.87–0.90.
- **~90% of round trips are home-adjacent step-asides**: a tile hops across a
  boundary adjacent to its own goal line and returns. Far detours are rare
  (≤1 pair/board). R is atypical (60/40 home/far, WD ferry charge 28) —
  more evidence R does not represent the band.
- **cWD certifies almost none of it**: mean LIS escape demand is ~1.4 against
  ~8 observed round-trip pairs. The yielders are tiles whose within-line order
  is already LIS-consistent — they are forced aside by **through-traffic**,
  a mechanism invisible to every charge we have.
- Location stays flat across boundaries (third null confirmation, including
  the fresh len-110 leg: |r| ≤ 0.10 vs every centrality feature), and excess
  events are uniform along the path.

Verdict: the band slack finally has a physical identity — a **transit-yield
tax**, one step-aside each by many order-consistent tiles letting crossing
traffic pass. This is the constraint target: lower-bound forced exits of
line-g residents as a function of through-flow vs free capacity, chargeable
in the *same* product-graph escape-counter machinery as cWD (a yield is
exactly the abstract event cWD's counters already track — a goal-g tile
leaving line g). Per the §8f lesson, the next step is a reference-bound gap
measurement (how much any demand-vector rule can certify, and the ceiling of
the whole escape-counter family via observed-path escape counts) before any
fast implementation. Data: `data/sa_walks_{70,90,110}.tsv`,
`data/cc_walks_110.tsv` (completes the §8f regression set),
`data/slack_anatomy_probe.txt`.

## 8h. Transit-yield gap measurement — the family ceiling reaches the slack; soundness is the whole problem (2026-07-13)

A yield is exactly the abstract event cWD's counters already track (a goal-g
tile exiting line g), so candidate transit-yield charges are just bigger
demand vectors fed to the same vector-constrained A\*. `yieldgap` mode
evaluates, per board (370 re-solved walk boards, exact d\* oracle): three
static rules and a **family ceiling** — demands set to the observed per-line
exits of the board's own optimal path, sound for that path by construction
and an upper envelope for every per-line escape-demand rule.

| population | ceiling gain / slack | residual d\*−hCEIL | hCEIL = d\* exactly |
|------------|----------------------|--------------------|---------------------|
| len-70 (200)  | 10.40 / 11.87 (88%) | 1.47 | 90/200 |
| len-90 (120)  | 12.20 / 13.97 (87%) | 1.77 | 46/120 |
| len-110 (50)  | 12.96 / 15.04 (86%) | 2.08 | 17/50 |

- **The escape-demand family can express 86–88% of the band slack** — the
  first mechanism whose ceiling actually reaches it (every previous family
  topped out far below). The counter machinery needs nothing new; the entire
  open problem is a *provably sound* demand rule. Zero A\* budget fallbacks.
- Static root rules certify almost none of it: R1 (provable occupancy rule —
  full home line + forced through-traffic ⇒ 1 exit) fires ~never on
  scrambles; R2 (k≥4) ~0; R3 (k≥3 ∧ thru≥1) gains <1 and is **unsound**
  (witness: len-70 board 26, d\*=34, hR3=36). Static per-line rules stronger
  than full-line occupancy are structurally suspect: a passer can cross in
  any of the 5 columns, so only a physically full line forces an exit.
- R is untouched (CEIL 144 = cWD; its slack is ferry/far-churn). This lever
  belongs to generic deep boards, like the zPDB complement (§8e).
- Nuance for later: demands re-fire at every search NODE, and lines fill up
  mid-search even from scrambled roots — R1's per-node firing rate is a
  separate (A/B-only, per the §8c/§8f lesson) question from its root rate.

Data: `data/yg_{70,90,110}.tsv`, `data/yield_gap_probe.txt`.

## 9. Summary

Starting from a classical baseline of 204 moves, a sequence of measured
experiments — generator-as-search, the residual counterexample, front-to-front
MITM, the corridor diagnosis, and frame-rule deep-board generation — drove a
general learned solver down the ladder **204 → 174 → 168 → 164 → 160 → 156**,
matching the published best-known solution to the 24-puzzle's hardest board
*without* the hand-constructed rotational trick that produced the literature's
156. The reusable finding: for deep-board search, a value function must be
**monotone-informative to the target depth**, which requires **training labels
that actually reach that depth** — supplied here, non-R-derived, by frame-rule
construction and optimal geodesic-corridor labeling.
