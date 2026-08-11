# A087725

Sliding-tile puzzle research, in Rust. Three things live here: an exact
24-puzzle lower-bound prover, an exact 15-puzzle solver with a complete
enumeration of its deepest boards, and a learned system that constructs and
solves hard 24-puzzle instances.

All depths are **single-tile moves (STM)**. The 15-puzzle diameter is 80
(Korf & Schultze, 2005); the 24-puzzle diameter is known only to lie in
[152, 205] (Hannanov & Rokicki, 2011; Whitmore, 2018).

```sh
cargo build --release --features sha
cargo test
```

---

## 1. A 24-puzzle solver specialized for deep boards

### Optimizations

This project contributes three variations on the Walking Distance heuristic
(Takahashi, 2001), all admissible, all used by the prover:

- **cWD** — escape-constrained Walking Distance. WD sharpened by the moves a
  blank must spend leaving a line it is obliged to cross.
- **Last-moves Walking Distance** (`--lm`, `--lm2`) — WD refined by the forced
  final move or two of any solution. The last-moves idea is not itself novel
  (Korf & Taylor, 1996); pricing it against a WD abstraction is.
- **cLM2** (`--clm2`) — the two combined, with the last-two-move branches lifted
  by single-demanded-line escape constraints and priced *jointly*, which is
  stronger than taking the maximum of the two separately.

[`WD.md`](WD.md) derives all three from scratch with worked examples.

This project also adopts the following optimizations. If a citation below is
missing, please open a GitHub issue and it will be added.

- **Iterative, not recursive.** Search state lives in a depth-indexed arena
  allocated once; there is no call frame to spill across.
- **No allocation on the per-node path.** The arena and every front cache are
  built once per worker and reused across work units and thresholds.
- **One axis copied per move, the other shared** with an ancestor for the cost
  of a one-byte index.
- **Move-pruning DFA.** Taylor–Korf duplicate elimination
  (Taylor & Korf, 1993), compiled to a 41,396-state automaton (687 KiB) and
  folded into the candidate mask.
- **Child pre-prune from the parent's neighbour-WD** — over-bound children are
  skipped before being built, with no table probe.
- **σ-orbit split at the root** (Culberson & Schaeffer, 1994), halving the tree
  on a σ-symmetric board.
- **Three additive (Korf & Felner, 2002) 8-tile zero-aware PDBs**
  (Clausecker & Reinefeld, 2019), each queried in both σ-views.
- **1 bit per PDB entry** (Clausecker & Reinefeld, 2019), not 8 — distances
  reconstructed differentially.
- **Lazy cascade**: each tier is consulted only at nodes the cheaper ones failed
  to prune.

### What it proves

The target is `R`, the canonical hard instance — the goal rotated 180°, which
the literature calls the "turned 180-degree" configuration. It records
`optimal(R) ∈ [152, 156]` (Hannanov & Rokicki, 2011).

```text
      goal                    R

    1  2  3  4  5         · 24 23 22 21
    6  7  8  9 10        20 19 18 17 16
   11 12 13 14 15        15 14 13 12 11
   16 17 18 19 20        10  9  8  7  6
   21 22 23 24  ·         5  4  3  2  1
```

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
[`RUNBOOK_R156.md`](RUNBOOK_R156.md) has the build procedure, measured timings,
the pinned hashes and the machine requirements. [`CLAUDE.md`](CLAUDE.md) has the
gates any change must pass — chief among them node identity, since nothing may
alter the search tree.

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

[`ENUMERATION.md`](ENUMERATION.md) has the algorithm in full. Note its status
section predates the current data, which reaches depth 76.

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
[`TRAINING.md`](TRAINING.md) documents the design and the 15-puzzle proof of
concept that validated it, where exact ground truth exists.

### Results

**It solved `R` in 156 moves** — matching the best published solution — having
never seen `R` or any state on its solution path. The literature's 156 was
hand-constructed from R's rotational symmetry; this one was discovered by
generic learned search. Replay-verified in `data/r156_ours_solution.txt`.
See [`FINDINGS_R.md`](FINDINGS_R.md).

**A catalog of certified-deep boards.** A construct → score → bound → re-seed
loop produced **542 instances**, each bracketed by a proven lower bound
(bounded IDA\* exhaust) and a replay-verified learned upper bound: 504 with
LB ≥ 132, 204 at ≥ 138, 106 at ≥ 140, 19 at ≥ 142. Across 2,713 evidence rows,
**zero LB > UB inversions** — the two independent solvers never contradicted
each other. The registry is `data/catalog24.tsv`; see [`FINDINGS_HUNT.md`](FINDINGS_HUNT.md).

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

[`DESIGN.md`](DESIGN.md) explains the 8-puzzle-first approach and the
compression question the project started from. [`WD.md`](WD.md) documents the
walking-distance family the prover's heuristic is built on. `records/` holds the
measurement ledgers — grep `records/r_flat_k8_lazy.txt` before calling any
optimization idea untried.

---

## References

Clausecker, R., and Reinefeld, A. 2019. *Zero-Aware Pattern Databases with
1-Bit Compression for Sliding Tile Puzzles.* SOCS 2019, pp. 35–43. Improves on
the 1.6-bit mod-3 encoding of Breyer & Korf 2010. Construction details follow
Clausecker, *Notes on the Construction of Pattern Databases*, ZIB Report 17-59,
2017; see `docs/zpdb-codec-spec.md`.

Culberson, J. C., and Schaeffer, J. 1994. *Efficiently Searching the
15-Puzzle.* Technical Report TR 94-08, Department of Computing Science,
University of Alberta. §2.1, "Mirror Positions", gives the argument in the form
used here: reflecting a path across the main diagonal is the move replacement
l↔u, r↔d, and Lemma 2.1 says a bound on a position applies to its mirror. A
closing footnote proposes normalising the board so mirror positions never enter
the search at all. Published as *Searching with Pattern Databases*, CSCSI 1996,
LNAI 1081, pp. 402–416, and *Pattern Databases*, Computational Intelligence
14(3), 1998, pp. 318–334, where it becomes Lemma 2 with the automorphism proof
spelled out and the claim that "the effective search space for sliding-tile
puzzles is half the size previously thought"; Korf & Schultze (2005) cite the
1998 version for the factor of two. All three state the node-level result, of
which the root split used here is the special case — none phrases it as
root-specific. Taking the maximum of a PDB and its reflection is a separate
technique from the same papers (§4.3 in the 1998 version).

Hannanov, B. ("stannic"), and Rokicki, T. 2011. *Twenty-Four puzzle, some
observations.* Domain of the Cube Forum, node 238,
`forum.cubeman.org/?q=node/view/238`, linked from OEIS A087725. The thread that
produced both published bounds on `R`: Hannanov opens it by proving ≥ 140 STM
"using good heuristic developed by Ken'ichiro Takahashi (takaken)", and Rokicki
then reports 12,225 distinct length-156 solutions (2011-08-09) and a completed
ply-150 search with no solution, giving ≥ 152 by parity (2011-08-18). H.
Kociemba also contributes. The forum 403s ordinary fetchers; use a browser
User-Agent.

Hannanov, B. ("stannic") 2017. *Pattern databases for the 5x5 sliding puzzle.*
Domain of the Cube Forum, node 555, `forum.cubeman.org/?q=node/view/555`. Dates
Takahashi's heuristics to 2001/2002 and raises the Prieditis X-Y connection;
lists `R` as "rotate_180" and "a particularly bad case for disjoint pattern
databases". Its "Nodecounts" comment (2017-04-24) is the source of the 17
depth-80 antipodes in `data/pdb15_antipodes.txt`, which §2 seeds from.

Korf, R. E., and Taylor, L. A. 1996. *Finding Optimal Solutions to the
Twenty-Four Puzzle.* AAAI 1996, pp. 1202–1207. Introduces the last-moves
heuristic ("the last two are introduced here for the first time"); the
linear-conflict heuristic it also uses is Hansson, Mayer & Yung 1992.

Korf, R. E., and Felner, A. 2002. *Disjoint Pattern Database Heuristics.*
Artificial Intelligence 134(1–2).

Korf, R. E., and Schultze, P. 2005. *Large-Scale Parallel Breadth-First
Search.* AAAI 2005. The complete 15-puzzle depth distribution, which
`data/pdb15_depth_histogram.txt` reproduces and §2 gates each layer against.

Takahashi, K. ("takaken") 2001. *１５パズル自動解答プログラムの作り方*
[How to build an automatic 15-puzzle solver], describing the Walking Distance
and Invert Distance heuristics.
`ic-net.or.jp/home/takaken/nt/slide/solve15.html`, now offline; earliest
Internet Archive capture 2001-06-25. His *15puzzle Optimal solver* reached
v1.2 in May 2002. Walking Distance has no formal publication — the 2001 date is
the earliest archived capture of the page, corroborated by Hannanov (2017),
which also notes that WD may be a rediscovery of the X-Y heuristic (Prieditis,
1993). For a peer-reviewed work that formally cites the page, see Hasan, D. O.;
Aladdin, A. M.; Talabani, H. S.; Rashid, T. A.; and Mirjalili, S. 2023. *The
Fifteen Puzzle — A New Approach through Hybridizing Three Heuristics Methods.*
Computers 12(1):11.

Taylor, L. A., and Korf, R. E. 1993. *Pruning Duplicate Nodes in Depth-First
Search.* AAAI 1993, pp. 756–761. Introduces the finite-state machine that
enforces the pruning rules.

Whitmore, B. 2018. *5x5 sliding puzzle can be solved in 205 moves.* Domain of
the Cube Forum, node 559, `forum.cubeman.org/?q=node/view/559`. The 24-puzzle
diameter upper bound.
