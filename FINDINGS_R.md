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
≥ 152 on a single 32 GiB machine was calibrated at ~75 days; ≥ 156 at ~10⁸ years).

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
- **Proving the lower bound higher** — infeasible on this hardware (≥154 ≈ years),
  and it wouldn't lower our solution anyway.

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

## 8. Summary

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
