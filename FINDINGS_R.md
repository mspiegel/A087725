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

Mechanism (`--heuristic cwd-zpdb6-lazy`, `LazyMaxInc` in
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
| 146 | **cwd-zpdb6-lazy** | 13.09 B | 237.0 s | **1.49× slower** |
| 148 | pure cWD (§8b) | 595.86 B | 5,633 s | 1× |
| 148 | cwd-zpdb (eager, §8b) | 343.84 B | 8,508.8 s | 1.51× slower |
| 148 | **cwd-zpdb6-lazy** | 343.84 B | **6,755.7 s** | **1.20× slower** |

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
| B1 | cwd-zpdb6-lazy | 145.27 B | 3,039.0 s | **1.86×** | 1.04× slower |
| B2 (g3) | pure cWD | 192.57 B | 1,901.2 s | — | 1× |
| B2 | cwd-zpdb6-lazy | 108.53 B | 2,212.2 s | **1.77×** | 1.16× slower |
| B3 (g4) | pure cWD | 73.87 B | 727.3 s | — | 1× |
| B3 | cwd-zpdb6-lazy | 47.54 B | 964.5 s | **1.55×** | 1.33× slower |

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

## 8i. Engine throughput — the deep-board heuristic is compute-bound, and only *work-reduction* pays (2026-07-19)

Everything above sharpens the *heuristic* (fewer nodes). §8i is orthogonal:
make each node **cheaper**. The production combiner `cwd-zpdb6-zpdb8-lazy`
(`max(cWD, k6-zpdb, k8-zpdb)`, lazily) was profiled and tuned on R at
exhaust-144 (proves ≥146; warm, 8 threads; node-identical A/Bs throughout —
every change here preserves the 268,071,922-node tree exactly).

**Where the time goes (line-level, `sample` + dSYM on the search phase).** The
hot path is *compute-bound*, not memory-bound: the k8 table read (`diff_lookup`)
is only **1.8%** — the ~2000×-reused working set stays cache-resident (the
30.5 GiB tables held resident on a 32 GiB box with *one* hard fault, RSS growth
all clean file-backed pages). The real cost is index arithmetic and projection
bookkeeping: `ZpdbLayout::rank` ~18–22%, `ProjectedState::apply_in_place`
~20–23%, `Cwd::make` ~16% (of which `pack` alone ~5.8%).

**Three wins — all genuine work-reduction, all node-identical:**

| commit | change | Mn/s |
|--------|--------|------|
| `abd66d8` | drop k6's *reflected* Korf-max view (stop maintaining it) | 34 → 38.7 |
| `936b6ce` | hoist the shared blank/step out of `ZpdbInc`'s per-group `apply_in_place` | 38.7 → 44.0 |
| `7e09bca` | maintain cWD's packed WD-table key incrementally (no per-node re-`pack`) | 44.0 → 44.9 |

Cumulative **≈ 34 → 45 Mn/s (+32%)** search throughput on R, at zero node cost.
Details: k6's reflected view binds 22.9% of *evals* but costs only **+0.39%
nodes** to drop (vs k8's +9.16%, which grows +10.7%→+17.3% at 144→146 and stays
valuable) — so k8 keeps reflection, k6 drops it by default (`--k6-reflect`
restores). The hoist removes 3–4 redundant `blank_pos`/`step` recomputes per
node (a view's projections share the board blank). The cWD key is rebuilt from
the 5×5 matrix every node, but a move touches one axis / two cells → O(1)
update (`key_row == pack(&m_row, br)` invariant, `debug_assert`-checked).

**The governing law (the reusable finding):** on the Apple-silicon
out-of-order core, **a sampling profiler's % on cheap, predictable, vectorized
instructions is not reclaimable time** — only removing real *multi-instruction
work* moves the needle. Every "shave instructions the core already hides"
attempt was neutral-to-negative, even when it looked like the biggest hot spot:

| dead end | result | why |
|----------|--------|-----|
| software prefetch / MLP (within-node + cross-child) | −0.7% / **−37%** | k8 working set is cache-resident; nothing to hide. probes/make = 2.000 (the 3 k8 groups partition all 24 tiles → one group/move) |
| incremental `rank` (maintain occ/sr/perm) | −0.7% | the from-scratch ascending walk (the 15.5% part) is a tight cache-hot loop; the incremental `slide` costs the same |
| `get_unchecked` in `rank`/`apply` (10+9 bounds checks) | −1.8% | bounds branches are predicted-not-taken (free); `rank` shrank 322→111 instrs for *no* cycle win |
| `h == 0` gate on the `s == GOAL` compare | +0.4% (noise) | the 25-byte compare is ~2 vectorized NEON ops; the 3.3% of *samples* is ~0 cycles |
| huge pages for the k8 mmap | infeasible | macOS can't superpage a file mmap without an anonymous copy that breaks the 30.5/32 GiB residency |

The contrast is the whole lesson: cWD `pack` (a 20-iteration *loop*) removed →
+3.6%; the goal compare (2 instructions) removed → 0. Same "eliminate the hot
line," opposite outcome, because one was work and one was already free.

Instrumentation kept: the gated `zpdb-locality` feature (`c4032e7`, off by
default) — exact `make`/probe counters. A/B logs: `data/r_k8lazy_*_ab.txt`.

## 8j. Drop the k6 tier entirely — `cwd-zpdb8-lazy`, a 2-tier `max(cWD, k8)` (2026-07-19)

§8i established that k6's *reflected* view is nearly free to drop (+0.39%
nodes). §8j asks the natural next question: what does the *entire* k6 tier buy
on top of `max(cWD, k8)`? A new combiner `cwd-zpdb8-lazy` —
`LazyMaxInc(cWD, k8)`, cWD every node and the three-group 8-tile ZPDB advanced
only where cWD fails to prune — was A/B'd against the production 3-tier
`cwd-zpdb6-zpdb8-lazy` (`max(cWD, k6-zpdb, k8-zpdb)`) on R, warm, 8 threads. Both
default (k6 reflection already off per §8i), so this measures the value of k6's
*primary* view on top of cWD+k8. Node counts are the exact §8i trees
(268,071,922 @144; 8,742,025,201 @146), so the 3-tier column below matches §8i
node-for-node.

| bound | | 2-tier `cwd-zpdb8-lazy` | 3-tier `cwd-zpdb6-zpdb8-lazy` | Δ (2-tier) |
|-------|--|-------------------------|------------------------------|------------|
| **144** (proves ≥146) | search time | **4.80 s** | 5.93 s | **−19% faster** |
| | throughput | **56.04 Mn/s** | 45.23 Mn/s | +24% |
| | nodes | 269,180,930 | 268,071,922 | +0.41% |
| **146** (proves ≥148) | search time | **180.78 s** | 221.04 s | **−18% faster** |
| | throughput | **48.73 Mn/s** | 39.55 Mn/s | +23% |
| | nodes | 8,808,311,484 | 8,742,025,201 | +0.76% |

**The finding:** the whole k6 tier is nearly redundant against `max(cWD, k8)` on
R — removing it costs only **+0.41% → +0.76%** nodes through bound 146, while
every non-cWD-pruned node gets **~23% cheaper** (no k6 projection/rank/probe),
so search finishes **~18–19% faster**. This is the §8i lesson taken to its
conclusion: k6's marginal pruning does not pay for its per-node compute in the
cWD+k8 regime. The node penalty does creep up with depth (0.41% → 0.76%), so a
much deeper bound could eventually reach a crossover where k6 earns its keep —
but there is no sign of one through 146. On R in this regime, the 2-tier is the
faster combiner.

Caveat: *search time* is the fair metric — wall-clock adds the identical ~25 s
k8 mmap + cWD load both tiers pay. Off-R this is untested; like the rest of the
combiner family (§8e) the balance is expected to shift on boards where cWD is
weaker and k6 carries more of the max. A/B log: `data/r_k8lazy_2tier_vs_3tier_ab.txt`.

## 8k. The pruning-tier trade is iso-time — you only win by moving *one* axis (2026-07-19)

§8i established the engine is *compute-bound* at its pruning ceiling, so on a
first order **time ≈ nodes × per-node-cost**. §8j and a follow-on forced-move
experiment pin down the consequence: **trading node-count against per-node cost
is a wash in either direction.** The two are coupled, and perturbing the tier
structure just slides along an iso-time curve.

The forced-move experiment (branch `single-survivor-promote`, `--promote-forced`,
not merged). The single-survivor histogram (§8j instrumentation follow-on) found
**36% of expanded nodes have exactly one child surviving the cWD neighbor
prune** — "forced" moves with no branching decision. The idea: on those nodes
skip the lazy combiner's upper tier (advance cWD, defer/skip k8) and collapse the
forced move in-frame instead of recursing. Sound (admissible ⇒ same proven bound)
but not node-identical. Measured on R, 8 threads:

| combiner / bound | Δ nodes | Δ throughput | Δ search time |
|------------------|---------|--------------|---------------|
| `cwd-zpdb6-lazy` / 144 | +1.9% | +2.8% | **−0.7%** |
| `cwd-zpdb8-lazy` / 146 | +5.6% | +6.2% | **−0.6%** |

Skipping k8 lifts throughput (fewer catch-ups/probes + no recursion frame) by
almost exactly what the extra nodes (weaker pruning) cost. **≈ break-even both
regimes** — and the node penalty scales with how much the skipped tier prunes
(k8 > k6), while the throughput gain scales with its per-node cost, so the two
track each other by construction.

**The governing law, generalized (the reusable finding).** On a compute-bound
heuristic at its pruning ceiling, an effort pays **only if it moves one axis
while holding the other fixed**:

- **cheaper at fixed nodes** — §8i (drop k6's reflected view, hoist, incremental
  key): **+32% throughput, zero node change**. Pure win.
- **fewer nodes at ~zero per-node cost** — the move-DFA (Taylor–Korf duplicate
  FSA, L2-resident): cuts nodes for a table lookup, not a heuristic probe.
- **a tier that isn't paying its way** — §8j (drop k6 entirely): +0.76% nodes but
  ~18% faster, because k6 added almost no pruning yet cost real compute. That is
  *work-reduction disguised as a tier drop*, not a genuine pruning trade.

Anything that moves **both** axes at once — add a tier (−nodes, +compute), or
drop/skip a paying tier (+nodes, −compute) — is iso-time. The forced-move result
also settles a structural question: skipping k8 *costs* +5.6% nodes, so **k8 is
paying its way** (unlike k6). That makes **cWD + k8, k6 dropped, at §8i's tuned
throughput the efficient frontier** for this heuristic family on R — every local
perturbation of its tier structure is a wash.

To beat the frontier you must move it, not slide along it: (1) a **cheaper cWD**
— paid on *every* node (§8i profiled `Cwd::make` at ~16% of the 3-tier's samples,
and its share is larger in the k6-dropped 2-tier where less ZPDB work runs), so
any node-identical speedup is pure win; (2) more **near-free structural node
reduction** (move-DFA-class, finite state, not another heuristic tier); or (3) a
heuristic with a better **pruning-per-cycle** ratio — not simply more or less
pruning.

## 8l. Four node-identical throughput wins on the 2-tier frontier (2026-07-19)

§8k's prescription — *move one axis, hold the other fixed* — executed. Re-profiled
the settled frontier `cwd-zpdb8-lazy` (the §8i profile was on the retired 3-tier;
`sample` + dSYM on the search phase, 8 threads, exhaust-146). The 2-tier split:
**search loop 29%, ZPDB/k8 ~40% (`rank` 21% + `apply`/`make` 13% + `unmake` 5%),
cWD 23%, LazyMax glue 8%; the 30.5 GiB k8 probe itself is 0.05%** (§8i's
"compute-bound, not the table read" holds). Four changes followed, **every one
node-identical** (bit-for-bit `8,808,311,484` nodes @146) — the only kind §8k
says pays — each landed by a back-to-back A/B on R:

| change | attacks | mechanism | A/B (R, thr146, 8t) |
|--------|---------|-----------|---------------------|
| demand-LIS LUT (`c21078e`) | top cWD line | per-node `n − LIS(...)` → a 32 KiB table keyed by the line's residency pattern | −2.5…4.7% |
| undo-record trim (`b3f427a`) | cWD make/unmake | drop the *dead* `curves` field; recompute the packed `key` on unmake (`CwdUndo` 40→20 B) | −2.0% |
| `search_inc_mut` arg-bundle (`5b75923`) | search-loop spills | bundle the 6 loop-invariant args behind one `&SearchInv` (15→10 args) | −2.0% |
| zPDB `rank` co-location (`5496995`) | `rank` cache misses | merge `counts`/`cohort_base`/`labels[sr]` into one ~40 B `ShapeInfo` (3 scattered lookups → 1 cache line) | −2.9% |

Cumulative throughput rose ~**48 → 53 Mn/s** across the round (≈ +10%; exact
per-change deltas blur under cross-session thermal drift, so the table cites each
change's own same-session A/B). Two of the four were **disassembly-/layout-driven,
not profile-percentage-driven** — the profiler said *where*, but the *why* and the
fix came from reading the generated code:

- **The fat argument list was a real spill site.** Disassembling the hot
  `search_inc_mut` showed a 384 B frame, all 10 callee-saved registers used, and
  **85 stack spill/reload instructions** — 7 of its 15 args stack-passed and
  re-loaded per node. Bundling the invariants cut it to a 288 B frame / 53 spills
  (−31% code). A stack-spill survey of the other hot functions found **none
  close** — `rank`/`ZpdbInc::make` are compute-bound, not spill-bound.
- **`rank` was *partly memory-stall-bound*, correcting §8i.** §8i read the ZPDB
  working set as cache-resident, but that was the diff table; the *ranking* tables
  (`labels` ≈27 MB, `cohort_base` ≈8.6 MB for k8, indexed by an effectively-random
  `sr`) miss. Co-locating the three per-shape reads into one cache line — cutting
  the *number* of misses 3→1 — won 2.9%. The lesson: reducing miss *count* pays
  where the hardware can't help you.

**Reconfirmed dead ends (kept honest by the A/Bs).** A software prefetch of
`shape_info[sr]` (overlapping the `perm_rank` window) was **neutral** — the
out-of-order core already issues that load speculatively, so §8i's "prefetch is
neutral on this core" holds for a second reason (the hardware beats us to it), and
it was reverted. Also measured-and-dropped this round: the forced-move `promote`
combiner (§8j follow-on, ≈break-even) and the ZPDB lazy-blank apply (sound but
shelved — k8's reflected view needs `TAU`/`SIGMA` owning-group handling that
doubles the intricacy for the same ~4% apply-side prize).

**Still open (untried levers, ranked):** shrink `ShapeInfo` (nibble-pack the
region `labels`, ~40→27 B, more resident shapes); the lazy-blank apply (the
largest single work-reduction left, ~4%); and — only if attacking the reflected
view — note the reflected *rank integer* is **not** derivable from the normal one
(the transpose scrambles both the shape and permutation ranks), though the
reflected *positions* are, which would save the reflected *apply*, not the *rank*.

## 8m. Min-cut / cut-packing congestion bounds are provably capped at Manhattan (2026-07-19)

§8f/§8h ruled out the per-line escape-demand family (ceiling proven) and said
"whatever closes R's slack must couple the axes or see tile identity at range."
The one genuinely-unbuilt *sound-by-construction* candidate that survived was a
**min-cut / operator-counting congestion bound** — the LP/flow generalization of
WD's per-boundary linear flow. `examples/mincut_ceiling.rs` measured its ceiling
on R the §8h way (path-informed, fail-fast) — and the answer is a **theorem, not
a number: the whole cut-packing family cannot exceed Manhattan(R) = 112.**

**Construction.** A *cut* is a subset `S` of the 25 cells; a sound lower bound on
the solution moves that cross it is `lb(S) = #{tiles whose start,goal straddle S}`
(each crossing move carries exactly one tile across; the directional balance pins
the minimum to `a+b`). A move slides one tile along one grid edge, so a move
crosses `S` iff that edge is a cut-edge of `S`. Hence for any **edge-disjoint**
family of cuts, no move crosses two of them and `d* ≥ Σ lb(S)`. Maximizing that
sum is the min-cut / operator-counting ceiling for the family.

**The Menger cap.** `Σ_{S∈F} lb(S) = Σ_tiles #{S∈F separating that tile}`. For a
fixed tile the separating cuts are pairwise edge-disjoint, and by **Menger's
theorem** the max number of edge-disjoint edge-cuts between two vertices equals
their graph distance = that tile's **Manhattan** distance. Summing:
`Σ lb(S) ≤ Σ_tiles Manhattan(t) = Manhattan(R)`, with equality achieved by the 8
axis half-plane cuts. The tool confirms it numerically: axis-packing = best of
200 000 randomized packings = **112 = Manhattan(R)**; greedy (lb-biased) even
*under*performs at 108. Every selected cut's observed crossings on R's 156-move
path dominate its `lb` (soundness witness holds).

**Consequence.** The min-cut congestion direction is **dead for R by proof**, not
by budget: it is capped at Manhattan = 112, *below* WD (140) — because WD already
beats Manhattan via the one thing a single-commodity cut is blind to, **intra-line
ordering** (LIS). So neither non-axis rectangle cuts, center-isolating cuts, nor
any edge-disjoint packing can add anything WD/cWD don't already have. This closes
the last sound-by-construction *single-commodity* lever. What Menger does **not**
cap — and what therefore remains the only live frontier for R's residual slack —
is exactly what §8f named: intra-line **ordering** (already exploited) and true
**axis-coupling** (a joint row×col relaxation, i.e. a *multi-commodity* LP whose
tiles compete for shared edge capacity — the operator-counting bound proper, which
Menger's single-tile argument does not bound). `mincut_ceiling.rs` is retained as
the constructive proof witness (the "measured-dead, harness retained" pattern).

## 8n. The multi-commodity (fractional) cut LP is *also* exactly Manhattan — congestion coupling is dead (2026-07-19)

§8m capped the *integral* edge-disjoint cut packing at Manhattan(R)=112 by Menger,
and named the one thing Menger does **not** cap: the *fractional* relaxation, where
cuts share edge capacity — the multi-commodity / operator-counting coupling.
`examples/axis_lp_ceiling.rs` built and solved it exactly (a one-phase primal
simplex on the packing LP, over rectangles + multi-order prefix sweeps). **It is
Manhattan too — 112.0000, with the 8 axis cuts as its entire support.**

**The LP.** Let `k_e ≥ 0` be the crossings of grid edge `e`; then `d* = Σ k_e` and
every cut obeys `Σ_{e∈∂S} k_e ≥ lb(S)`. So `min Σ k_e` s.t. those constraints (the
**covering LP**) lower-bounds `d*`; its dual is the fractional cut packing
`max Σ lb(S)y_S` s.t. `Σ_{S∋e} y_S ≤ 1`. The hope: sharing edges fractionally
escapes the Menger integrality cap.

**Theorem (it doesn't).** Route each tile on a shortest path; the induced edge
counts `k_e` are feasible for **every** cut (a straddling tile crosses each
separating cut an odd ≥1 times ⇒ `Σ_{e∈∂S} k_e ≥ lb(S)`) and `Σ k_e =
Manhattan`. So the full covering LP `≤ Manhattan ≤` (axis lower bound) covering
LP `⇒ = Manhattan` exactly — over *all* cuts, fractional included. The solver
exhibits it: optimum 112, support = the 8 axis cuts (each `y=1`), observed
crossings on R's path dominate every `lb` (soundness witness holds).

**Why coupling fails here.** A cut/flow bound charges edge *traversals*, but
sliding-puzzle edges have **unbounded per-plan capacity** — a tile may recross an
edge for free in the relaxation — so congestion never bites. The constraint that
actually makes R hard is intra-line **ordering** (two tiles can't pass through
each other): a *conflict*, not a *cut*. That is exactly what WD/cWD/linear-conflict
already charge, and no single-commodity-*or*-multi-commodity cut/flow LP can see it.

**Consequence — the whole cut/flow congestion family is closed for R.** Integral
(§8m) and fractional (§8n), single- and multi-commodity, all collapse to Manhattan
= 112, below WD (140) < cWD (144). Coupling the axes *through shared edge capacity*
buys nothing. The only structurally-distinct lever left is **cost-partitioning an
ordering bound with a cross-axis tile-identity abstraction** — i.e. optimally
combining cWD (ordering) with a PDB (cross-axis identity) via an LP *over
abstractions, not cuts*. That is a genuinely different object; but the prior is
poor: the strongest such abstraction we have (the 30.5 GiB k8 ZPDB) already
**collapses to ≤126 on R**, and cWD saturates the full unit move-cost reaching 144,
so a cost-partition of cWD⊕PDB has little slack to redistribute. `axis_lp_ceiling.rs`
is retained as the constructive proof witness. Cf. [[mincut-manhattan-cap]].

## 8o. The axis-coupled ("blank-ferry") WD is vacuous — column-abstracted coupling = plain WD (2026-07-19)

After §8m/§8n closed the cut/flow congestion family, the remaining structurally
distinct lever was a *conflict*-style coupling: a single WD automaton over the
**product** state `(M_r, M_c, blank_cell)` — one blank serving both axes — to
charge the ferry cost cWD's two independent per-axis solves miss (§8g: R's slack
is ferry/far-churn; Step A re-confirms it — blank round-trip pairs = 74, `rt_far`
= 9 on R's path). Two cheap root measurements, no table.

- **Step A (available slack).** `slack_anatomy rpath`: R's gap over cWD is 12
  moves (row 8 + col 8 over WD; cWD surcharge 4). Plenty of physical ferrying to
  *try* to charge.
- **Step B (can a coupled WD charge it?).** `examples/coupled_wd_root.rs` computes
  the coupled bound at R's root by IDA* over the product state (admissible: every
  real move → one abstract move, goal-class in the blank's column chosen freely;
  `h = WD_row+WD_col` is consistent, so exact). Result: **coupled-WD(R) = 140 =
  WD(R) exactly** (found in one downhill plunge at threshold 140).

**Theorem (vacuous).** The blank's ROW trajectory is driven solely by the vertical
moves and its COLUMN trajectory solely by the horizontal ones; a vertical move's
availability depends only on `M_r` (any class present in the adjacent row, chosen
freely) and never on the blank's column, symmetrically for horizontal. So any
interleaving of an optimal row-WD sequence with an optimal col-WD sequence is a
valid product path ⇒ coupled-WD ≤ WD_row+WD_col, and ≥ h = the same ⇒ **= WD
exactly**. The shared blank couples nothing, because column-abstraction has
forgotten *which* tile sits in the blank's column — so the relaxation lets the
blank always find a useful move.

**Consequence.** A coupling that actually bites must retain **tile-column
identity** — i.e. a PDB — and PDBs collapse to ≤126 on R. So the tractable
axis-coupling (#2 of the "beating cWD" menu) reduces to the PDB direction, already
known weak on R; it is closed in its column-abstracted form. Net after §8m/§8n/§8o:
every *sound-by-construction* admissible-bound family that does not see tile
identity at range is measured-out for R — cut/flow (= Manhattan) and product-WD
(= WD). What is left is not construction but **combination** (cost-partition cWD ⊕
PDB) or **a sound strengthening of cWD's own escape rules** (§8h's transit-yield
ceiling reaches R's slack; only the soundness proof is missing). `coupled_wd_root.rs`
retained as the witness. Cf. [[mincut-manhattan-cap]].

## 8p. Transit-yield soundness diagnostic (Phase 1) — the unsound cause is free-slot escape (2026-07-19)

Of the "beating cWD" menu, transit-yield (#3) is the only family whose *ceiling*
already exceeds cWD (§8h/R4 reaches R's slack) — so the entire problem is
soundness. Phase 1 localizes exactly where the aggressive rules R2/R3 lose it.
`slack_anatomy yielddiag` (3000 random len-55 boards, exact-solved) reports the
board-level unsound rate (cwd_axis with the rule's demands > d*) and buckets every
fired transit demand by `(k = goal-g residents in line g, thru = transiters
spanning g)`, measuring the "over-fire" (optimal path made fewer exits than
demanded).

**Results.** R2 is unsound on **3/3000 (0.1%)**, R3 on **19/3000 (0.6%)**, mean
overshoot ~2.7–2.8 when unsound. Over-fire by bucket:

| bucket | over-fire rate | reading |
|--------|---------------:|---------|
| **k=5**, any thru | **0%** | fully saturated line ⇒ transiter *must* force a yield — the provable R1 core |
| k=4, thru=1 (R2/R3) | 29% | one gap in the line |
| k=4, thru=2+ demanding 2 (R3) | 29% | two transiters, one gap |
| k=3, thru=1/2+ (R3) | 27% / 13% | two gaps |

**The cause is free-slot escape.** A line with `k` goal-g residents has `5−k`
non-resident cells = **gaps**. A transiter can pass *through a gap* without
displacing any resident, so demanding a yield is unsound whenever a gap is
reachable. This is why **k=5 is the unique safe condition** (0 gaps ⇒ 0% over-fire
= exactly R1) and every `k≤4` demand (all of R2/R3's extensions) is unsound. The
`k=4, thru≥2 → demand 2` case adds a second failure — **shared yield**: two
transiters share the single gap, so only one exit is forced, not two.

**Implication for Phase 2.** Gating on the *count* `k` alone cannot beat R1
soundly — the count can't tell whether the line's gaps are *reachable by the
transiter*. A sound strengthening must be **position-aware**: demand a yield only
when the transiter's path is blocked by residents on *both* sides (no reachable
gap). That needs within-line **column order** — the exact information WD's
count-only contingency abstracts away (the same crux as §8o). So #3 is not dead,
but Phase 1 reframes it precisely: the target is a *gap-blocked* refinement
carrying minimal within-line position, not a bigger count-demand. Encouragingly,
R2 is already unsound on only 0.1% of boards, so the guard that must be added is
narrow. Diagnostic saved to `data/yielddiag_len55.txt`; `yielddiag` mode retained.

## 8q. Gap-blocked rule refuted; the sound transit-yield demand is redundant with LIS on R (Phase 2 first cut) (2026-07-19)

§8p's Phase 1 said a sound rule beyond R1 must be *position-aware* (fire only
when the transiter has no reachable gap). `slack_anatomy gbtest` built and tested
the first cut — a **gap-blocked** rule (fire when every gap in the line is
INTERIOR, cols 1–3, flanked by residents, hypothesised unreachable) — against the
same exact-solved bulk set (3000 boards) plus R's 156-move path.

**Both halves came back negative.**
- **The gap-blocked hypothesis is refuted.** GB is **unsound — 0.07% (2/3000)**.
  An interior gap does *not* block a transiter: the transiter reaches that column
  by ferrying **in its own row**, not line g, so line-g gap geometry is irrelevant
  to reachability. §8p's free-slot escape cannot be fixed by any position feature
  read from line g alone.
- **The only sound demand (R1, k=5) is gainless on R.** R1 is sound (0/3000) but
  never fires on shallow boards (0 gain), and where it *does* fire on R's corridor
  (6/157 path states) it **binds on 0 of them** (max gain +0): cWD's LIS ordering
  demand already covers those exits. R1 ⊆ LIS on the states that matter for R.

**Consequence — the per-line escape-demand family is closed for R.** Feeding a
scalar per-line demand to the vector-constrained WD (`cwd_axis`) cannot beat cWD
soundly: the sound subset (R1) is subsumed by cWD's existing LIS demands, and
every gainful extension (R2/R3, and now GB) is unsound via free-slot escape. This
operationalizes §8h's verdict ("the family ceiling reaches the slack; soundness is
the whole problem") into a mechanism: the gain lives exactly in the exits that
free-slot escape makes unprovable. A sound gain would require a relaxation *richer
than a scalar per-line demand* — carrying within-line column order (→ a PDB, which
collapses to ≤126 on R) or cost-partitioning cWD with a PDB (candidate #1, still
unmeasured). `gbtest`/`yielddiag` retained; results in `data/gbtest_len55.txt`.

Net after §8m–§8q: of the "beating cWD" menu, cross-axis congestion (cut/flow =
Manhattan), blank-ferry coupling (= WD), and per-line transit-yield (sound ⊆ LIS)
are all measured-out for R. The last unmeasured lever is **cost-partition cWD ⊕
PDB** (#1) — an LP over abstractions, cheap to try, with a poor prior (k8 ≤126).

## 8r. Yield-or-detour ceiling — R's transit churn is detour-dominated, not yield-dominated (2026-07-19)

The one cross-axis coupling §8m–§8q left untested: a transiter dodging a row-yield
by slipping through a gap must *reach that gap's column*, costing column moves — so
each transit crossing is a **yield** (through a saturated row), a **detour** (at an
off-band column, extra column travel), or **free** (an on-band gap slip).
`slack_anatomy ydceiling` classifies every transit crossing on R's 156-move path:

| | transit | YIELD | DETOUR | FREE |
|---|---:|---:|---:|---:|
| row (vertical) | 53 | 1 | 22 | 30 |
| col (horizontal) | 52 | 1 | 19 | 32 |

**Findings.**
- **Yields are negligible (2/105).** Saturation-forced yields barely occur on R —
  independent confirmation of §8q (R1 binds 0). Row-saturation charging is dead here.
- **The coupling that shows up is DETOUR (41/105).** Transiters routinely cross at
  off-band columns (off-band excess 53) — R's ferry/far-churn made concrete, and it
  lives on the *column* axis, not row saturation.
- **FREE is the plurality (62/105)** — on-band gap slips no sound bound can reach.

**But it is not a win, and the tool says so.** These are **gross** path crossings,
not net-of-cWD: cWD_col already prices most column churn (R's slack over cWD is 12,
while the detour-excess alone is 53), so the *net* recoverable is bounded by the 12
slack and is subject to the R4 trap — a detour on *this* optimal path can be
rerouted on another, i.e. it may be optional. So §8r **redirects** the search
(from row-yield, dead, to column-detour, the live mechanism) without yet
establishing sound net gain. The decisive next step is a **soundness-checked detour
rule** (gbtest-style: a state-computable "transiter with no on-band gap must detour
≥ k columns" demand, tested for 0% unsound + gain over cWD on the bulk set). Prior
is guarded — path detours are reroutable — but detour is now the specific,
localized target, which yields never were. Saved to `data/ydceiling_r.txt`;
`ydceiling` mode retained.

## 8s. The detour rule is unsound — yield-or-detour closes; R's churn is provably-optional to a static bound (2026-07-19)

§8r localized R's transit churn to **detour** (off-band column travel), not yield.
The Phase-2 second cut, `slack_anatomy dettest`, built the natural state-computable
detour rule and validated it the §8q way. **DET(s):** a tile that must transit a
row is *blocked* if in some transit row every cell within its natural column band
is a resident (no on-band gap) → it must yield-or-detour, +1 (symmetric for
columns). Test: is `h0(LIS cWD) + DET ≤ d*`?

**Unsound, decisively.** Bulk: unsound on **0.57% (17/3000)**, worst overshoot +5.
On R's path it is far worse — DET fires on 84/157 states but `h0+DET` exceeds the
suffix length (a *valid* solution bound) on **77 of 157 states**, DET reaching 38.
The block is **illusory**: a transiter facing no on-band gap *now* reroutes, or a
gap opens as other tiles move, before it must cross — so the detour is **not
forced**. The free-slot/reroute wall that killed the yield branch (§8p/§8q) kills
the detour branch identically.

**This closes yield-or-detour, and with it the last cross-axis coupling.** The deep
reason, now demonstrated on both branches: R's residual churn is **optional** — it
appears on R's optimal path but can be rerouted across the space of optimal
solutions, so no *static, state-computable* sound bound can charge it (only a
path-specific one can, which is §8h's R4 — sound for one path, not a heuristic).
That is precisely why cWD's sound demands are ⊆ LIS and every strengthening is
unsound. `dettest`/`ydceiling` retained; results in `data/dettest_len55.txt`.

**Net over §8m–§8s — the admissible-bound search over cWD is complete for R.** Every
sound-by-construction family has been built and measured out: cut/flow congestion
= Manhattan (§8m/§8n), blank-ferry/product coupling = WD (§8o), per-line yield
(§8p/§8q) and cross-axis detour (§8r/§8s) both defeated by reroutable/optional
churn, and cost-partition ⊕ PDB = max by cWD saturation. **cWD(R) = 144 is the
ceiling for tractable admissible heuristics on R**; the residual 12 to d*≤156 is
optional churn no static sound bound reaches. The practical route to R ≥ 152 is the
~2-day exhaustive compute (§8b), not a tighter heuristic — which is exactly what R
being the global WD-maximum instance predicts.

## 8t. Deep-board complement profile — k8's complement vanishes toward R (2026-07-19)

Correctly targeting the *traversal* (not h(R)): what matters for proving R ≥ 152 is
the heuristic across the R→goal tree. `examples/deepboard_profile.rs` evaluates cWD
and the k8 zPDB standalone on a large sample of the states the proof visits (4000
random walks 1..60 from R, mean cWD 126) plus R's 156-corridor, and profiles where
k8 *complements* cWD (k8 > cWD).

**k8 complements cWD almost never — and it collapses toward R.**

| cWD bucket | states | k8 > cWD | mean gain |
|---|---:|---:|---:|
| 100–109 | 8 | 62% | 2.4 |
| 110–119 | 718 | 4% | 2.2 |
| 120–129 | 2040 | 0.1% | 2.0 |
| 130–139 | 962 | **0%** | — |
| 140–149 | 272 | **0%** | — |

Overall k8 > cWD on **1.0%** of near-R states; on R's corridor **0/157**, with
`max(cWD,k8) = cWD` exactly (k8 is strictly looser — mean looseness 5.55 vs cWD's
3.06 against the suffix-length d* proxy). And cWD's looseness is **concentrated at
R**: 12 at the root, ~3 on the corridor.

**Reconciles the coarse A/B and closes the traversal-complement question.** The
earlier deep-board A/B (k8-lazy cut ~40% of nodes, iso-time) drew its win from the
cWD≈100–119 boards where k8 fires ~4%; but the complement **vanishes at cWD ≥ 130**
— the near-R region where the R≥152 proof is bottlenecked. The structural reason is
the same one that makes R hard: near the antipode, aggregate per-axis flow (cWD)
dominates any partial-tile-identity bound (PDB), so k8 — and, by the collapse of all
8-tile PDBs on R, any PDB partition — has nothing to add on the deepest states. So
the traversal-complement lever, correctly targeted this time, is **dry for the hard
region**: k8 helps only on shallower deep boards and only iso-time there.

The remaining honest unknown is which cWD-region actually dominates the R-proof
tree's node count (near-R cWD≥130, where no complement exists, vs the cWD~110–125
shell, where k8 fires but pays for it). That needs instrumenting the real search's
per-node cWD distribution — a node-count profile, not a heuristic probe.
`deepboard_profile` retained; results in `data/deepboard_complement_profile.txt`.

## 8u. The R-proof workload lives at cWD 100–125 — where the complement DOES fire (corrects §8t) (2026-07-20)

§8t's complement profile used random walks from R and concluded "k8's complement
vanishes toward R (cWD≥130)." That measured the wrong distribution. The real
question is the cWD value of the nodes the *proof search actually expands*, so I
instrumented the search directly: feature `cwd-node-hist` histograms `h_val` for
every node passing the `f ≤ bound` check — the nodes cWD does NOT prune, which are
exactly the nodes where a complementary heuristic is probed. A bounded R search
(`--heuristic cwd --max-bound 144`, proves ≥146; 319M expanded nodes) gives:

| cWD region | share of expanded nodes | k8 fires here (§8t) |
|---|---:|---:|
| ≤ 109 | **44.4%** | high (62% at 100–109) |
| 110–125 | **55.3%** (peak ~112) | ~4% at 110–119 |
| ≥ 126 | **0.33%** | ~0% |

**§8t was misleading; the complement lever is reopened.** The proof workload is
dominated by cWD **100–125**; the near-R cWD≥126 region where k8 is useless is a
negligible **0.33%**. §8t's random-walk sample (mean cWD 126) over-weighted the
high-cWD near-R states that barely appear in the real tree — the threshold
constraint `g + cWD ≤ bound` puts the node mass at intermediate depth / mid cWD,
not at the root. So k8 *does* fire on the dominant workload (consistent with the
~40% node cut in the coarse A/B), and the barrier is **iso-time** (per-node cost),
not absence of a complement.

**A cWD-value gate does NOT convert this to a win — the incremental cost blocks it.**
An earlier draft proposed probing k8 only when cWD ≤ ~112 (skipping the ~half of
expanded nodes at cWD 113–125 where k8 fires ~4%). But k8 is an *incremental*
heuristic: `ZpdbInc::make` runs `apply_in_place_at` unconditionally per group per
move (position maintenance — unavoidable), and the value is a *diff* from the prior
`n_h` (`diff_lookup(n_idx, ctx.n_h[i])`). So a gated node can't skip-and-resume
cheaply: the apply is always paid, and recovering a deferred value needs a cold
rank (`cold_lookup_proj`) that re-pays the rank *plus* a cold 10 GiB access. This is
the same defer-and-replay economics that made the `--zpdb-depth` gate a measured
wash. The gate is retracted.

**What stands, and where the wall-clock lever actually is.** The solid, load-bearing
result is the *correction of §8t*: the proof workload lives at cWD 100–125 where k8
DOES fire, so the ~40% node cut is genuine and on-workload — the complement is not
absent. The barrier is the intrinsic per-node incremental cost (apply + diff-rank,
~40% of runtime, §8l), which no value-gate can dodge. So converting the node-cut to
wall-clock is a *throughput* problem (cheapen k8's per-node cost — the §8i–§8l
frontier) or a *structurally cheaper complement* problem (a partition with lower
per-node apply+rank cost that still fires on the cWD 100–125 workload), not a gating
problem. The `cwd-node-hist` instrumentation lives on branch `cwd-node-hist` (kept off
main); results in `data/cwd_node_hist_r144.txt`.

## 8v. cWD is pinned to depth on the proof contour — the cWD-gate IS the depth-gate (2026-07-20)

Extending `cwd-node-hist` to record depth `g` alongside cWD (branch `cwd-node-hist`)
and re-running the same bounded R search answers "how do the cWD bands correlate with
depth?" — **exactly, and trivially**: every expanded node has `cWD = 144 − g`.

The per-depth table shows mean-cWD = `144 − g` at every depth with **zero spread**
(each depth is 100% within one cWD band), i.e. `g + cWD = 144` for *every* expanded
node — the last IDA\* iteration explores only the `f = 144` contour, as expected for a
consistent heuristic rooted at `f = threshold`. So the cWD bands are *depth* bands:

| band | cWD | depth g | share |
|---|---|---|---:|
| k8 dead | ≥ 126 | g ≤ 18 | 0.33% |
| k8 weak | 110–125 | g 19–34 | 55.3% |
| k8 fires | ≤ 109 | g ≥ 35 | 44.4% |

(node count peaks at g≈32, cWD 112.)

**Consequence — this rigorously confirms §8u's retraction.** Because `cWD = 144 − g` on
the dominant contour, a cWD-value gate (probe k8 only when cWD ≤ 112) is *identical* to
a depth gate (probe k8 only when g ≥ 32) — which is exactly the `--zpdb-depth`
experiment, already measured a wash. The cWD value carries no information beyond depth
on the contour, so there is nothing a cWD-keyed decision can do that a depth-keyed one
cannot — and the depth-keyed one lost. The wall-clock lever is throughput or a
structurally cheaper complement, not any gate. Results in `data/cwd_depth_hist_r144.txt`;
tool on branch `cwd-node-hist`.

## 8w. Mid-tree transpositions are already gone — the move-DFA leaves only ~7% (2026-07-20)

§8v left one lever that *isn't* blocked by the admissible-bound wall: don't tighten
the bound on the dominant depth-19–34 nodes, **eliminate them as duplicates**. The
move-DFA (Taylor–Korf) prunes redundant move *sequences* statelessly but not full
transpositions (one board reached by two paths); a transposition table would. The
probe (`midtree-dedup` feature, branch `midtree-dedup`, sequential R search at
bound 144) collects the actual boards at target depths and counts same-depth
repeats:

| depth g | visits | distinct | dup-rate | redundancy |
|---:|---:|---:|---:|---:|
| 13 | 17 841 | 17 841 | 0.00% | 1.000× |
| 22 | 3 699 141 | 3 541 534 | 4.26% | 1.045× |
| 28 | 14 902 743 | 13 969 823 | 6.26% | 1.067× |
| 34 | 18 443 783 | 17 208 548 | 6.70% | 1.072× |
| 40 | 11 506 524 | 10 704 512 | 6.97% | 1.075× |

**Same-depth = total transposition redundancy here** (a clean consequence of §8v):
on the `f = 144` contour cWD is a function of the board, so `board → cWD → g` is
deterministic — every board appears at exactly one depth, and cross-depth
transpositions are *impossible*. So the probe captures all of it, and it's only
**~7%** in the mid-tree. The move-DFA already eliminates the rest. A transposition
table's ceiling is therefore a ~7% node cut, bought with a table storing billions
of boards — not worth it. The duplicate-pruning lever is dry.

**Synthesis (the traversal thread, §8t–§8w).** The dominant depth-19–34 mid-tree
nodes are **largely irreducible** for the current method: (1) a tighter *admissible*
heuristic there means beating cWD on cWD-110–125 states, closed generally by
§8m–§8s; (2) the *complement* (k8) fires but is iso-time, and gating it is the
already-failed depth-gate (§8u/§8v); (3) they are *not* transposition-redundant
(§8w, ~7%). So R ≥ 152 stays a **compute** problem — the ~2-day exhaustive run
(§8b) — and the wall-clock frontier is throughput (§8i–§8l) or a structurally
cheaper complement, not a tighter bound, a gate, or a transposition table. The
`midtree-dedup` tool is on branch `midtree-dedup`; data in
`data/midtree_dedup_r144.txt`.

## 8x. The cWD ladder ratio is ~43×/+2, not 29×, and throughput *falls* with depth (2026-07-29)

*(Numbering: the unmerged branch `blank-tour-disjunction` also uses §8x…§8x-8,
for unrelated lower-bound work that is not being merged — it carries no
performance changes. If that branch is ever revived, its sections need
renumbering, not this one.)*

Two corrections to figures this file and `PUZZLE24.md` have been carrying, both
measured on the rewritten flat engine (`src/puzzle24/search/flat.rs`, sequential,
cWD + move-DFA + neighbour-WD pre-prune + root σ-orbit split).

**1. The ~29×/+2 growth rate is a *WD* number and does not apply to cWD.**
`PUZZLE24.md:378-381` derives it from `wd --prove-at-least 144` (exhausts 142,
1.23 B) → `wd --prove-at-least 146` (exhausts 144, 36.9 B) ≈ 30×. It has since
been quoted as if it governed the cWD proof ladder (`PUZZLE24.md:533`, `:590`).
It does not. Measured on R with cWD:

| exhausts | nodes (cumulative) | ratio vs previous |
|---|---:|---:|
| 144 | 422,379,806 | — |
| 146 | 18,189,473,636 | **43.1×** |
| 148 | 595.86 B (§8b) | **32.8×** |

Iteration-only, the 146 pass is 17,767,093,830 nodes = **42.1×** the 144 pass.
The ratio *falls* with depth (43 → 33), so a single constant is the wrong model
in both directions: it understates the next step and overstates the one after.

**2. Throughput degrades with depth — same binary, same board.**

*(Corrected 2026-07-29, later the same day: this section originally claimed the
degradation is progressive **within** a threshold, inferred from a budgeted run
showing −10.6% against a full pass showing −17.7%. A cleaner measurement refutes
that. Two 35 s counter windows taken from the **same process**, at 3% and 71%
through a full exhaust-146 pass, show IPC 3.156 → 3.173 — flat to 0.5%, and no
top-down category moving more than 0.6pp. The effect is a **step between
thresholds**, not a drift within one: same run, 144 at 31.74 Mn/s vs 146 at
27.75 Mn/s = −12.6%. Whatever produced the budgeted-vs-full gap happens in the
first few percent of the pass or is cross-run noise; the two cannot be separated
from the data. The practical consequence is the opposite of what the original
text implied: a budgeted `--max-nodes` window IS representative of its
threshold, so profiling need not chase the deep tail.)*

| exhausts | search time | throughput |
|---|---:|---:|
| 144 | 13.41 s | **31.50 Mn/s** |
| 146 | 679.94 s | **26.75 Mn/s** |

−15%. A deeper threshold walks a larger and more diffuse slice of the merged cWD
table, so probe locality falls; this is the same effect §8i's hit-rate table
shows from the other side. **Every wall-clock extrapolation in this repo assumes
a fixed rate and is therefore optimistic**, including the ones made on the day
the engine landed.

**Revised projection.** Proving R ≥ 152 requires exhausting threshold 150.

| exhausts | nodes | 1 thread @ ~25 Mn/s |
|---|---:|---:|
| 148 | 595.86 B (measured, §8b) | ~6.6 h |
| 150 | ~17 T (extrapolated, ratio ~28×) | **~8 days** |

The 150 row is an extrapolation twice over — the ratio is still falling so 28× is
a guess, and the rate at that depth will be below 25 Mn/s. Treat it as an order
of magnitude. It is nonetheless a materially different picture from the 75-day
figure at `PUZZLE24.md:590`, which was WD-based and predates cWD, the σ-orbit
split, and this engine.

**Engine state.** The flat engine reached this via seven node-identical
work-removal wins (C, D1, A, F, G3, H1, widemul-fold hash): 21.83 s → 13.41 s at
exhaust-144, **−39% search time / +63% throughput**, unchanged nodes. Five
loop-*restructuring* attempts all lost (prefetch −37%, index table −3.5%, probe
cache **at 1 K** +5.8%, H5 +16%, H5b +12.7%). Full log:
`data/r_flat_engine_ab.txt`.

*(Corrected 2026-07-29: this list previously read "probe cache −5.8%", which
inverted the sign and dropped the size qualifier. −5.8% is the **winning** 64 K →
256 K sizing delta (`data/r_flat_engine_ab.txt:911`); the reject was the 1 K
variant at +5.8%, i.e. slower (`:990`). The engine ships a 256 K probe cache and
§8y makes it the single largest lever in the parallel regime, so listing "probe
cache" among the failures was actively misleading.)*

**Scope, and it matters.** Every one of those A/Bs ran at **exhaust-144**. The
counters above show the regime changes with depth — back-end share rises, IPC
falls 15% — so the *memory-shaped* rejects do not generalise. The prefetch
entry's own recorded reason for failing, "working set already resident; nothing
to hide", is a statement about the 144 regime and is exactly what breaks at 146.
The seven wins are unaffected: they remove work from the node's dependency chain,
and that work is removed at every depth.

**Method note.** The exhaust-146 run doubles as the strongest correctness check
the engine has: 18,189,473,636 nodes matching the pure-cWD figure recorded at
§8c bit-for-bit, ~500× the coverage of the 180-case frozen oracle
(`src/puzzle24/search/flat_oracle.rs`) and at a depth the oracle's 200–700-step
walk boards never reach.

## 8y. The parallel flat engine — 7.4× on 8 threads; the loss is per-node, never the schedule (2026-07-29)

*(Numbering: the unmerged `blank-tour-disjunction` branch also uses §8y…§8y-2 for
the quotient budget-ladder work, per the note at §8x. Neither carries performance
changes; if that branch is revived its sections need renumbering, not these.)*

§8x left the R proof single-threaded: 675 s to exhaust 146. This section
parallelises it and, more usefully, **measures where the missing speedup goes** —
because the first answer (6.33×) was wrong about the cause, and acting on the
obvious suspicion would have wasted the runs.

**The driver.** Tree-splitting plus work-stealing: a cheap sequential expansion
grows a frontier of ~4096 subtree roots, then rayon runs the *unmodified*
sequential loop on each and the results reduce (SUM the stats, MIN the next
threshold, first `Solved` wins). A worker searches its subtree at `bound − g`,
since `g + d + h ≤ bound ⟺ d + h ≤ bound − g`, so it needs no notion of the depth
it sits at. Exhausting a threshold in parallel still proves `dist ≥ next`: MIN is
order-independent and every node with `f ≤ bound` is visited exactly once. Node
counts are identical to the sequential engine at every worker count and cache
size tested — 422,379,806 at 144 and 17,767,093,830 at 146.

**The instrument** (`--features parallel-profile`). Parallel efficiency factors
exactly:

```
    speedup / W  =  B  ×  (c₁ / c_p)
```

`B` = busy fraction = Σ(unit wall time) / (W × span); `c_p` = worker ns/node. The
two have unrelated fixes — `B` is the split policy and scheduler, `c_p` is
contention, cache sizing and clock — so **a lone speedup number cannot say which
to attack.** Validated on the first W=8 run: `0.977 × 0.794 = 0.776` against a
measured 0.777, and again at 146 post-fix: `0.989 × 0.883 = 0.873` against 0.8727.

It also prints three makespan bounds — `work/W` (perfectly divisible), **LPT**
(best possible given that a unit is *indivisible*: a worker drawing a huge subtree
runs it to completion), and actual — so `LPT − work/W` is the split's granularity
cost and `actual − LPT` is scheduling loss. Plus per-thread ns/node, because the
busy fraction is **blind to core heterogeneity**: a thread parked on an E-core
still reads as busy while its nodes cost 2–3× more.

**What was ruled out, and this is the finding.** At *both* thresholds (W=8, final
18-bit configuration):

| | exhaust-144 | exhaust-146 |
|---|---:|---:|
| busy fraction `B` | 98.6% | 99.4% |
| granularity (LPT − work/W) | **0.00 s** | **0.00 s** |
| scheduling (actual − LPT) | 1.4% | 0.6% |
| serial split phase | 0.00 s | 0.00 s |
| largest unit, as % of span | 4.1% | 3.9% |
| per-thread ns/node spread | 35.0–36.0 | 40.3–41.5 |

Granularity is **exactly zero** — LPT equals `work/W` — on a tree 43× larger with
the same 4096 units. `SPLIT_TARGET` is therefore **closed as a lever**; the
going-in suspicion (one giant indivisible subtree holding the tail) is refuted,
and no runs were spent sweeping it. No thread lands on an E-core at W=8. The
schedule was never the problem at any depth.

**The defect it found instead: the probe cache was rebuilt per *work unit*.**
`Arena::new()` and `ProbeCache::with_bits()` sat inside the `map` closure, so both
were discarded and re-zeroed 4096 times per threshold. A unit averages ~100 K
nodes; the cache is direct-mapped and warms over millions of probes. It never
warmed.

| exhaust-144, W=1 | hit rate | misses | ns/node | setup |
|---|---:|---:|---:|---:|
| sequential, 256 K | 99.665% | 1.42 M | 34.04 | — |
| per-unit, 64 K | 95.147% | 20.50 M | 37.97 | 1.3% |
| per-unit, 256 K | 95.495% | 19.03 M | 40.59 | 4.6% |
| per-unit, 1 M | 95.613% | 18.53 M | 50.03 | 15.2% |

14.5× the misses — and enlarging the cache 16× moved the hit rate 0.47 pp while
re-zeroing pushed setup from 1.3% to 15.2%. That shape is the signature of
**compulsory** misses (first touch per unit), which no capacity can fix, and it is
why §8i's sequential 64 K→256 K result did not transfer: that measured capacity
misses, and these are not those. Fixed with one `thread_local!` per rayon worker,
reused across units and thresholds — not `map_init`, which re-runs its initialiser
once per producer-split leaf rather than once per thread. **+10.2%**, and W=1 came
within 0.6% of the sequential engine, i.e. closed rather than reduced.

**Then cache sizing, which refuted its own design note.** `WORKER_CACHE_BITS` was
16, reasoned from the L2 budget: 4 P-cores share 16 MB, so 8 threads at 18 bits
want 67 MB of a 32 MB total. Measured at exhaust-146, W=8:

| bits | per thread | hit rate | misses | ns/node | span |
|---|---|---:|---:|---:|---:|
| 16 | 2.1 MB | 96.026% | 706.1 M | 43.07 | 96.77 s |
| 17 | 4.2 MB | 97.617% | 423.4 M | 42.14 | 94.73 s |
| **18** | 8.4 MB | 98.592% | 250.1 M | 40.96 | **91.52 s** |
| 19 | 16.8 MB | 99.265% | 130.6 M | 40.20 | 91.96 s |

18 bits wins by **5.7%** while overflowing L2 several times over. What the cache
buys is not L2 residency but *not probing the 4.4 GB merged table* — a miss is a
SwissTable lookup that TLB-misses into multi-GB territory, costing far more than
an L2 miss on the cache itself. So the quantity to minimise is misses and the L2
budget is not the binding constraint. At 18 bits the eight per-worker caches
together take 250.1 M misses against one sequential 256 K cache's 249.7 M — parity
— which drops per-node contention from **+13.3% to +7.7%**. 19 bits lowers ns/node
further but ties on wall clock for 2× the memory, so 18 is the knee.

**Contention roughly doubles with depth**, which is why none of this was decidable
at 144: `c_p/c₁ − 1` is +5.8% at 144 against +13.3% at 146 with 16-bit caches, and
+3.8% against +7.7% with 18-bit ones. The mechanism is in the sequential cache
counters — misses per node
run 0.0034 at 144 against **0.0141 at 146**, 4.2× — the same collapse in probe
locality §8x measured from the TLB side. It saturates by W=4 at both depths, so it
is a shared-memory-hierarchy effect, not a per-core-count one.

**The residual splits 50/50 between clock and memory — and only half is
addressable.** `powermetrics` under root, sampling inside the 146 threshold with
both runs back-to-back in one script (necessary: sequential exhaust-146 has ranged
over 640–680 s across sessions, so only a within-script pairing is sound):

| | P0-Cluster | P1-Cluster | E-Cluster | ns/node |
|---|---:|---:|---:|---:|
| sequential | 3399.3 MHz | 3441.2 MHz | 972 MHz @ 54% | 37.44 |
| W=8 | **3264.0** MHz | **3264.0** MHz | 1075 MHz @ 55% | 41.12 |

Since `cycles/node = ns/node × GHz` is frequency-independent, this separates the
two: clock 3420 → 3264 MHz inflates ns by **4.8%**, and cycles/node rises 128.1 →
134.2, another **4.8%**. `1.0482 × 1.0478 = 1.0983` reproduces the measured ns ratio
exactly; taking the clusters as bounds rather than averaging gives clock 4.2–5.4%
and memory 4.2–5.5%, so the even split is robust.

Two things follow. First, **linear speedup is unattainable here by construction**:
eight busy P-cores run 4.6% below one *migrating* thread — the sequential baseline
gets the higher one-core-per-cluster boost, and its per-cluster residency visibly
swings 1.8%→100% as the OS moves it — so the ceiling is `8 × 0.9544 × 0.994 ≈
7.6×`. The measured 7.26× is ~96% of that, and only the 4.8% memory term is
software-addressable at all.

Second, **the memory term is purely a depth effect.** Applying the same measured
clocks to exhaust-144 (`c₁` 34.22, `c_p` 35.59) gives 117.0 → 116.2 cycles/node —
zero, arguably slightly negative:

| exhausts | ns/node penalty | of which clock | of which memory |
|---|---:|---:|---:|
| 144 | +4.0% | +4.8% | ~0 |
| 146 | +9.8% | +4.8% | +4.8% |

The clock term is depth-independent, being a function of how many cores are busy;
the memory term tracks footprint, with the sequential miss counters supplying the
mechanism. So parallel efficiency will keep sliding as the proof goes deeper, which
makes the projection below optimistic beyond §8x's existing fixed-rate warning.

The E-cluster reads the same ~54–55% residency in *both* runs — background OS work
plus `powermetrics` itself — so no rayon thread migrated to an E-core at W=8,
independently confirming the per-thread ns/node table above.

**Final state** (this session, AC; note it ran ~8% slow against the §8x session,
so compare only within the table):

| exhausts | sequential | W=8, 18 bits | speedup | efficiency |
|---|---:|---:|---:|---:|
| 144 | 14.45 s / 29.22 Mn/s | **1.90 s / 221.8 Mn/s** | 7.61× | 95.1% |
| 146 | 675.43 s / 26.30 Mn/s | **91.52 s / 194.1 Mn/s** | 7.38× | 92.2% |

The residual at 146 is 0.6% scheduling, 0.0% granularity, 4.8% all-core clock and
4.8% memory contention — see the split above. Against the **recursive** engine's 8-thread figures at §8b — 3.31 s /
128 Mn/s at 144 and 157.8 s / 115 Mn/s at 146 — this is **1.74× and 1.72×**. It
also exceeds the recursive engine's best throughput ever recorded in any
configuration (146 Mn/s, `FINDINGS_HUNT.md:135`, on the far cheaper WD heuristic
over a 55× larger tree).

**Declined: the E-cores.** W=12 exhausts 146 in **79.93 s / 222.3 Mn/s**, a further
**+21.1%** over W=8, with a 99.4% busy fraction — the four E-cores pay for
themselves even though `c_p` rises 24.5% (43.07 → 53.62). Not adopted: it saturates
the machine and leaves nothing for interactive use. W=10 is the worst of both
(93.3% busy, 6.7% scheduling) — 8 P + 2 E is the asymmetric case where stealing
cannot hide two slow threads, while at 12 all four participate and it rebalances.
Note rayon's default pool here is **8**, not 12, so W=12 needs
`RAYON_NUM_THREADS=12` and is available on demand for a run nobody needs to share.

**Revised projection.** Exhaust-148 was measured at 595.86 B nodes (§8b).
Extrapolating this engine's depth trend (−12.5% throughput per +2, so ~170 Mn/s at
148 and ~149 at 150) against §8x's ~28× ladder ratio:

| exhausts | proves | nodes | W=8 estimate |
|---|---|---:|---:|
| 148 | R ≥ 150 | 595.86 B (measured) | **~1 h** |
| 150 | R ≥ 152 | ~17 T (extrapolated) | **~1.3 days** |

R ≥ 150 becomes an hour rather than §8b's 1.56 h recursive, and **R ≥ 152 — the
published floor — becomes an overnight-to-weekend run**, against §8x's ~8 days
single-threaded and `PUZZLE24.md`'s original 75-day WD-era figure. The 150 row is
an extrapolation twice over (the ratio is still falling and the rate at that depth
is a guess); treat it as an order of magnitude.

**The reusable finding.** Every candidate cause of sub-linear speedup except one
measured to be *zero or near-zero* — granularity exactly 0.00 s, scheduling ~1%,
E-core placement absent, redundant work nil, and cache *misses* at parity with
sequential once sized right. The loss was per-node cost throughout: first a cache
lifetime bug, then a sizing constant justified by the wrong constraint, and finally
an irreducible floor that is half all-core clock and half depth-driven memory
contention. The decomposition `B × c₁/c_p` is what made that visible; the aggregate
speedup number pointed at the scheduler, which was innocent at every depth and
every worker count. Full log: `data/r_flat_parallel_efficiency.txt`.

## 8z. The lazy k8 tier on the flat engine — iso-time at 148 again; adopted for the R ≥ 152 run (2026-07-30)

§8k measured the k8 trade iso-time on the recursive engine; §8y's flat engine
then made cWD 1.73× faster, which should have buried k8. Measured, it did not —
the crossing moved with it.

**The build.** `solve24 --zpdb8`: `max(cWD, k8)` with the three 8-tile ZPDBs
(32.8 GB mmap'd), consulted only at children cWD fails to prune. The recursive
`LazyMaxInc` deferral glue (8% of §8l's samples) is replaced by an arena
invariant — a child's k8 slot is written at the consult, every entered node has
therefore a current slot, and no catch-up can ever be needed. Two further flat-era
cuts: only the two cost-1 group-views are slid per consult (the projected-edge
law makes the four cost-0 slides no-ops; the stale blank is healed just before
that group's next rank), and the reflected view needs no `transpose_move` — its
cells and cost-1 group come from the σ maps directly. Consult = one slot copy +
2 slides + 2 ranks + 2 O(1) `diff_lookup`s. A from-scratch design is impossible,
not just slow: the zbins are differential, and `cold_lookup` reconstructs an
absolute value in O(h) graph-descent steps (§8u's retraction, sharpened).

**Gates.** Threshold 146: **8,539,130,554 — bit-for-bit** the recorded recursive
2-tier tree (§8j), sequential and parallel. Threshold 144: 269,180,917 vs the
recorded 269,180,930 — 13 nodes *smaller*, confined to the first iteration
(attributed to the deleted lazy driver's Lipschitz-deferred values; not chased,
since the 8.5 B-decision gate rules out any systematic h error). And the race
below reproduced the cWD-only 148 tree exactly (577,673,506,215; §8b's
595,862,979,851 total), extending flat-engine node identity to 600 B scale.

**The race (W=8, back-to-back, AC):**

| exhausts | 2-tier | cWD-only | verdict |
|---|---:|---:|---|
| 146 | 133.7 s | 91.5 s | loses 1.46× |
| **148** | 193.31 B / **3254.78 s** | 577.67 B / **3256.50 s** | **dead heat, 0.05%** |

The recursive engines tied at 148 too (5630.91 vs 5633 s). Both engines got
~1.73× faster, and the iso-time crossing stayed planted at exhaust-148 — the
cut ratio's growth with depth (1.57× → 2.03× → **2.99×** measured) exactly
paces the layer's cost through one full engine generation.

**Adopted for R ≥ 152.** Break-even at exhaust-150 requires the cut ratio to
stop growing *entirely*; every measured rung grew it ×1.29–1.47. Projected:
~3.8 T nodes / ~19.3 h with `--zpdb8` against ~16.8 T / ~28.6 h without —
**~9 h saved**, with the unmeasured 150 cut ratio the only soft spot.
Constraints that stand: the tables oversubscribe RAM (RSS plateaus at 19.9 GB
of 37.8 GB mapped; parallel contention +13.6% vs cWD's +7.7%), and the unpulled
levers — a per-worker (idx → h) memo cache, k8 slot sharing, W=12 — would each
convert the 148 tie into a win. Full program log: `data/r_flat_k8_lazy.txt`.

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
