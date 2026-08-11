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

### What makes it fast

The tree is fixed — none of this changes a pruning decision. Each item either
makes the same nodes cheaper or avoids generating nodes that provably cannot
matter.

- **Iterative, not recursive.** Search state lives in a depth-indexed arena
  allocated once; there is no call frame to spill across.
- **No allocation on the per-node path.** The arena and every front cache are
  built once per worker and reused across work units and thresholds.
- **One axis copied per move, the other shared** with an ancestor for the cost
  of a one-byte index.
- **Move-pruning DFA.** Taylor–Korf duplicate elimination compiled to a
  41,396-state automaton (687 KiB), folded into the candidate mask.
- **Child pre-prune from the parent's neighbour-WD** — over-bound children are
  skipped before being built, with no table probe.
- **σ-orbit split at the root**, halving the tree on a σ-symmetric board.
- **cWD**: Walking Distance sharpened by escape demands.
- **Last-move refinements** `--lm` / `--lm2` / `--clm2`, pricing the forced
  endgame crossings on top of cWD.
- **Three additive 8-tile zero-aware PDBs**, each queried in both σ-views.
- **1 bit per PDB entry**, not 8 — distances reconstructed differentially.
- **Lazy cascade**: each tier is consulted only at nodes the cheaper ones failed
  to prune.
- **Compile-time tier selection**, so an unused tier is absent from the emitted
  loop rather than a well-predicted branch.
- **Incremental keys** — cWD, tracked lines and PDB keys are updated, never
  recomputed; cost-0 slides are skipped entirely.
- **A front cache in front of every table**, direct-mapped and per-worker.
- **Tables mapped, not parsed**, with `--hugepages` to cut the TLB footprint.
- **Node-identical parallel tree-splitting** across rayon workers.

Expanding on the ones that matter most:

**The arena is the reason the engine exists.** The recursive engine it replaced
carried a 384-byte frame doing 85 stack spill/reload instructions per node; a
flat loop removes the call boundary those spills exist to cross. The arena also
exploits a structural fact — a move changes only the row or the column
abstraction, never both — by indexing per-axis state with `row_at[d]`/`col_at[d]`
instead of depth. The untouched axis is shared with an ancestor and costs one
byte, against the 20-byte undo record the recursive engine wrote per node.

**The DFA prunes more than loops.** A continuation is *dominated* if it reaches
a board also reachable by a shorter — or equal-length, lexicographically smaller
— sequence. Skipping it removes duplicate work without ever removing a board, so
an exhausted threshold stays a sound lower bound. That predicate is regular in
the move string, so it compiles (Aho–Corasick collapse, then Myhill–Nerode
minimization) to a table that fits in L2. Per node the check is one
array-indexed transition — no hashing, no main-memory traffic, unlike a
transposition table.

**The heuristics are all admissible, and stack lazily.** Walking Distance
abstracts tiles into per-row and per-column bags, so each half is exactly
solvable and the two add. cWD sharpens that with escape demands — the moves a
blank must spend leaving a line it is obliged to cross. The last-move tiers go
further: `--lm` tracks one type-3 tile's forced 3→4 crossing, `--lm2` prices all
four last-two-move endgame branches, and `--clm2` lifts those branches with
single-demanded-line escape constraints, priced jointly. `WD.md` is a full
intuition-level account with worked examples.

**The k8 tables are 1 bit per entry.** The 24 tiles partition into three
disjoint 8-tile patterns whose distances sum admissibly (Korf–Felner).
*Zero-aware* means the blank belongs to the pattern; the 1-bit encoding
(Clausecker–Reinefeld) stores only whether a neighbour's distance rises or
falls, reconstructed differentially from a known start. That is what fits
30.5 GB where a byte per entry would need eight times as much. Because the
encoding is differential, the search must carry a running distance — which the
incremental machinery does anyway, by one XOR per moved tile, skipping the four
group-views a move leaves untouched.

**Caching is what makes the cascade affordable.** Every table sits behind a
direct-mapped front cache: the cWD cache runs at 99.665% hits sequentially, and
the k8 cache — keyed on packed tile positions, so a hit costs neither the rank
walk nor the mmap — measured 3.3× throughput (20.2 → 67.3 Mn/s at exhaust-144).
No cache is load-bearing for correctness; a hit must equal what the table would
have returned.

### What it proves

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

### Running it

The engine is `src/puzzle24/search/engine.rs`. The base stack is always
cWD + move-DFA + child pre-prune + root σ-orbit split; the tiers above it are
opt-in, and consulted only at nodes the cheaper ones fail to prune:

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
