# A087725

Sliding-tile puzzle research, in Rust. Three things live here: an exact
24-puzzle lower-bound prover, an exact 15-puzzle solver with a complete
enumeration of its deepest boards, and a learned system that constructs and
solves hard 24-puzzle instances.

All depths are **single-tile moves (STM)**. The 15-puzzle diameter is 80
(Korf & Schultze 2005); the 24-puzzle diameter is known only to lie in
[152, 205].

```sh
cargo build --release --features sha
cargo test
```

---

## 1. A 24-puzzle solver specialized for deep boards

`solve24` is not a general solver — it cannot return a solution path for an
arbitrary board in reasonable time. It does one thing: **prove a lower bound**
on a specific hard board by exhausting IDA\* thresholds. Every heuristic in it
is consistent, so an exhausted threshold `b` is a theorem: `optimal(board) > b`.

The target is `R`, the canonical hard instance (blank at cell 0, tile 25−i at
cell i), for which the literature records `optimal(R) ∈ [152, 156]`.

**Status.** Thresholds 144, 146, 148 and 150 are exhausted — **2,405,729,385,972
nodes**, proving `optimal(R) ≥ 152`. The per-threshold record is committed in
`runs/ckpt156/`. Exhausting 154 would prove `≥ 156` and, with the published
upper bound of 156, close the problem: `optimal(R) = 156`.

| threshold | nodes |
|---|---:|
| 144 | 115,436,814 |
| 146 | 4,363,759,350 |
| 148 | 114,245,221,757 |
| 150 | 2,287,004,968,051 |

### How it works

The engine (`src/puzzle24/search/engine.rs`) is a flat, iterative IDA\* — no
recursion, no undo. Search state lives in a depth-indexed arena allocated once,
which removes the call frame the recursive engine spent 85 stack spill/reload
instructions per node maintaining. The per-axis cWD state is indexed by
`row_at[d]`/`col_at[d]` rather than by depth, so a move copies one axis and
shares the other with an ancestor at a cost of one byte.

Above the base cWD heuristic sit optional tiers, consulted only at nodes the
cheaper ones fail to prune:

| flag | tier | artifact |
|---|---|---|
| `--clm2` | constrained last-two-moves, jointly priced | 13.5 GB |
| `--zpdb8` | three additive 8-tile zero-aware PDBs, both σ-views | 30.5 GB |

The production stack is the cascade `--clm2 --zpdb8` (consult order
cWD → cLM2 → k8). `--parallel` splits each threshold into subtree work units
across rayon workers and is **node-identical** to the sequential driver;
`--checkpoint` makes a multi-day run resumable, each worker appending only to
its own file.

```sh
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
target/release/solve24 --config large --position "$R" \
  --prove-at-least 145 --clm2 --zpdb8 --parallel    # → 115,436,814 nodes
```

Tables are built locally (~49 GB) and SHA-256 pinned; they are not in the repo.
`RUNBOOK_R156.md` has the build procedure, measured timings, the pinned hashes
and the machine requirements. `CLAUDE.md` has the gates any change must pass —
chief among them node identity, since nothing may alter the search tree.

---

## 2. A 15-puzzle solver, and every board at depths 76–80

`solve15` solves any 15-puzzle position **optimally** via IDA\* with additive
pattern databases. The default heuristic is `korf-plus`:
`max(Korf 7-8 PDB, its reflection, linear conflict, walking distance)`. Even a
depth-80 antipode solves in seconds.

```sh
target/release/solve15 --pdb-dir data/ --position "<16 tokens>"
```

### The enumeration

Built on top of that: the **complete** set of boards at each depth 76–80 —
every board whose optimal solution length is exactly `D`, not a sample.

| depth | boards | = N(d)? |
|---|---:|---|
| 80 | 17 | ✓ |
| 79 | 70 | ✓ |
| 78 | 3,406 | ✓ |
| 77 | 26,638 | ✓ |
| 76 | 272,198 | ✓ |
| **total** | **302,329** | |

Each layer is checked against `N(d)` from the Korf–Schultze distribution
(`data/pdb15_depth_histogram.txt`), which serves as both the termination oracle
and a per-layer correctness gate. All five match exactly. Output is
`data/enum15/depthNN.ranks`, one 6-byte little-endian rank per board, full
symmetry orbits; `enum_expand15` renders a file to readable rows and re-verifies.

The method avoids searching the 10.46-trillion-state space by never leaving the
top layers, and is mostly **solve-free**. Two structural facts do the work:
neighbouring boards differ in optimal depth by exactly ±1 (parity), and
`depth(s) = 1 + min over neighbours`. So expanding a certified layer `d+1`
identifies its depth-`d` neighbours with no search at all, and boards missed
that way — strict local maxima — are recovered by a Bellman membership test over
2–4 neighbours. IDA\* is a fallback for the residue only, never the descent.

`ENUMERATION.md` has the algorithm in full. Note its status section predates the
current data, which reaches depth 76.

---

## 3. Learned search for deep 24-puzzle boards

The 24-puzzle has no ground truth past depth ~30, so a learned system can't be
graded against optima. This part is built around that: a solver and a generator
that improve each other, with every claim bracketed by independent evidence.

**Solver.** A cost-to-go value network `V(s)` — raw one-hot board in, no
admissible heuristic anywhere in its input or training. Trained by DAVI
(approximate value iteration, DeepCubeA-style) and deployed through Batch
Weighted A\* Search, `f(x) = g(x) + weight·V(x)`. Non-admissible, so its answers
are upper bounds and every one is replay-verified from scratch.

**Generator.** A policy network that builds a board by choosing moves from
`GOAL`, trained by REINFORCE — its reward depends on running the solver's actual
search, which is not differentiable. The reward is GANCO-style regret: the
learned solver's cost minus a fixed admissible baseline's, which targets boards
where the learned solver underperforms rather than boards that are merely hard.

Implementation is `src/puzzle24/ml/` on `candle` (Metal backend, CPU fallback);
`TRAINING.md` documents the design and the 15-puzzle proof of concept that
validated it, where exact ground truth exists.

### Results

**It solved `R` in 156 moves** — matching the best published solution — having
never seen `R` or any state on its solution path. The literature's 156 was
hand-constructed from R's rotational symmetry; this one was discovered by
generic learned search. Replay-verified in `data/r156_ours_solution.txt`.
See `FINDINGS_R.md`.

**A catalog of certified-deep boards.** A construct → score → bound → re-seed
loop produced **542 instances**, each bracketed by a proven lower bound
(bounded IDA\* exhaust) and a replay-verified learned upper bound: 504 with
LB ≥ 132, 204 at ≥ 138, 106 at ≥ 140, 19 at ≥ 142. Across 2,713 evidence rows,
**zero LB > UB inversions** — the two independent solvers never contradicted
each other. The registry is `data/catalog24.tsv`; see `FINDINGS_HUNT.md`.

The bracket is what makes an entry scientific rather than suggestive: a board at
`[138, 160]` is a certified-deep instance whose optimum is pinned to a 22-wide
window. "WD says 128" is not.

This is a lower-bound-*side* result. It populates and certifies deep boards; it
does not prove any board deeper than `R`, nor move the published diameter floor
of 152.

---

## Layout

```
src/puzzle24/search/engine.rs     the lower-bound prover (§1)
src/puzzle24/search/recursive.rs  generic IDA*, optimal solving + deadlines
src/puzzle15/enumerate/           the depth 76-80 enumeration (§2)
src/puzzle24/ml/                  value net, policy net, DAVI, BWAS (§3)
src/puzzle8/                      the 8-puzzle warmup: full ground truth
```

`DESIGN.md` explains the 8-puzzle-first approach and the compression question
the project started from. `WD.md` documents the walking-distance family the
prover's heuristic is built on. `records/` holds the measurement ledgers — grep
`records/r_flat_k8_lazy.txt` before calling any optimization idea untried.
