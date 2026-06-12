# DESIGN.md

## Goal

Build an optimal solver for the **15-puzzle**, and eventually scale the same techniques to the **24-puzzle**.

By "optimal" we mean: given any solvable position, return a move sequence of minimum length. The 15-puzzle's diameter is 80 single-tile moves (Korf & Schultze, 2005); the 24-puzzle's diameter is known only to lie in [152, 182].

A trivially-correct optimal solver for the 15-puzzle exists: precompute the optimal distance to the goal for every reachable position and store it in a 10.5 TB lookup table. We are not interested in that. We are interested in solvers that are **as small as possible while still optimal**. The research question is:

> *What is the smallest correct representation of the optimal-move function?*

State-of-the-art is IDA* + additive pattern databases (~600 MB), roughly 20,000× smaller than the full table, still optimal, still fast. The floor below that is unknown.

## Approach

Single-threaded Rust. Start on the 8-puzzle (181,440 states, full table ~181 KB) as a tractable warmup where every technique can be implemented end-to-end, verified against ground truth, and benchmarked. Once each technique is correct on the 8-puzzle, lift it to the 15-puzzle. Multi-threading, disk-resident structures, and the 24-puzzle come later.

The 8-puzzle isn't the goal — it's the place to make all the bugs.

## Seven directions for compression

In rough order of increasing cleverness:

1. **Symmetry quotient.** Quotient the state space by the puzzle's goal-preserving symmetries. Lossless ~2× compression.
2. **Distance-only table.** Store only `dist(p)`; recompute optimal moves by querying neighbors. Trades lookups for storage.
3. **Pattern databases.** Decompose tiles into disjoint subsets; precompute admissible heuristics over each subset; drive IDA*. State of the art today.
4. **Macro mining.** Identify frequent subwalks in the optimal-move DAG; encode solutions as `(macro, residual)` pairs. Used by Rubik's-cube solvers; underexplored for sliding puzzles.
5. **Learned policy.** Train a neural net on `(position → optimal move)` pairs. Lossy; wrap with admissible search to keep correctness.
6. **Decision-tree / rule mining.** Induce interpretable rules from the optimal-move function. Produces a compressed *theory* of optimal play, not just a black box.
7. **Kolmogorov-complexity floor.** Run general-purpose compressors (zstd, xz, arithmetic coding) against the table to get an empirical upper bound on the true complexity floor.

DESIGN.md lists all seven. Which to implement, and in what order, will be decided after the 8-puzzle foundation is in place and the ground-truth table exists.

## Milestones

1. **8-puzzle foundation.** State, moves, lex-rank, backward BFS from goal, full distance table, IDA* with Manhattan + table-exact heuristics, diagonal-reflection symmetry. Verified against published results (diameter 31, 2 antipodes, full distance histogram).
2. **8-puzzle compression study.** Pick a subset of the seven directions, implement, benchmark each on a uniform `Report { bytes_stored, mean_solve_time, max_error }`. Establish which techniques are worth scaling.
3. **15-puzzle port.** Lift the proven techniques. New constraints: full table no longer fits in RAM; IDA* + PDBs becomes the practical baseline. Diameter 80 verified against Korf–Schultze; 17 antipodes reproduced.
4. **15-puzzle compression study.** Repeat the comparison at full scale. The interesting number is how close we can get to (or below) the ~600 MB state of the art.
5. **24-puzzle exploration.** Not "solve completely" — that's open research. Instead: extend the best 15-puzzle techniques, characterize where they break, and contribute to the open problem (current best upper bound 182, lower bound 152).

## Engineering principles

- **Single-threaded throughout the 8-puzzle work.** Parallelism comes in at the 15-puzzle stage where it actually matters.
- **Tests at every step.** Unit tests for invariants (parity, bijection, symmetry involution); integration tests asserting each compression direction produces optimal solutions on the full state space (or, for lossy directions, that the wrapped solver does).
- **Deterministic outputs.** The table is byte-stable across runs; SHA-256 pinned in the repo.
- **Minimal dependencies.** Core library stays dep-free. CLI tools and ML examples may pull dev-deps under feature flags.
- **No premature generalization.** Implement everything for the 8-puzzle concretely. Only lift to a generic `Puzzle` trait after the patterns are clear, when porting to the 15-puzzle.

## Verification

The 8-puzzle phase is complete when:
- `cargo test` passes (invariants, round-trip, compression contracts).
- The generated distance table has the expected SHA-256.
- The stats binary reports diameter 31, antipode count 2, total 181,440.
- Every implemented compression direction reports `max_error_plies == 0` on the full 181,440-state space.

The 15-puzzle phase is complete when the same checks pass at 16!/2 ≈ 10.46 trillion scale, with the Korf–Schultze depth histogram reproduced and the 17 antipodes recovered.

The 24-puzzle phase has no completion criterion — it's research, not engineering.
