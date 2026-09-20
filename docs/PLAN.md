# Session synthesis: use the frame rule to push the 24-puzzle lower bound

## Context

The project's open goal is bounding the 24-puzzle diameter (currently `[152, 205]` STM; see memory `bounds-24-puzzle-corrected.md`). In an earlier session we analyzed the existing antipode data files (`records/antipodes8.txt`, `data/pdb15_antipodes.txt`) and found empirical structure that constrains where antipodes live. The most actionable finding — the **frame rule** — is concrete enough to use as a search-space restriction at m=5.

This plan is focused on that single use: build 24-puzzle search infrastructure, then attack the frame-conformant subspace to either raise the lower bound past 152 or refute the frame rule at m=5.

## Status (updated 2026-06-15)

The **infrastructure half** is built and verified; the **research half** (frame rule → lower-bound push) is the remaining work.

- **Phase 1 (port to `puzzle24/`) — DONE + verified.** `src/puzzle24/{state,rank,symmetry}.rs`; 38 tests + `tests/puzzle24_state.rs`.
- **Phase 3a (generic IDA\* solver) — DONE + verified.** `solve24` + `build_pdb24` binaries; Manhattan + incremental Korf-max; optimal solves confirmed end-to-end.
- **Phase 2 (PDBs) — DONE + verified.** Pivoted from standard additive Korf 6-6-6-6 to **zero-aware PDBs** (Clausecker & Reinefeld, SOCS 2019). The 1-bit codec, `Z24D` file format, and the incremental `ZpdbInc` heuristic (with reflected view) are built and tested. **Four `data/pdb24_*.zbin.sha256` pins are committed; the `.zbin` artifacts themselves (22.6 MB × 4) are `.gitignore`d and regenerated locally via `build_pdb24 --zero-aware`** — matching the existing convention for `pdb15_*.bin` and friends. `solve24 --heuristic zpdb` solves end-to-end. Full technical spec: `docs/zpdb-codec-spec.md`; pivot rationale: memory `zpdb-pivot-phase2`.
- **Phase 3b (frame-restricted search) — NOT STARTED.**
- **Phase 4 (lower-bound push) — NOT STARTED.**

## What we established (carried over from session)

### The frame rule (15-puzzle, verified on all 17 known antipodes)

A state `s` is **frame-conformant** iff:
- (a) the four corner tiles `{_, 1, 4, 13}` sit at their antipodal corner positions `{0, 15, 12, 3}`, and
- (b) each of the 8 goal-neighbors of those corner tiles sits within Chebyshev ≤ 1 of its assigned anti-corner, with at most one exception of Chebyshev = 2.

All 17 depth-80 boards in `data/pdb15_antipodes.txt` satisfy (a) and (b). 15 of 17 satisfy (b) with zero exceptions; 2 of 17 (A8, A15) use the one-exception allowance, with the leaker landing at the same interior cell (1,1).

### Cross-puzzle evidence

| puzzle | corners forced | corner-neighbor rule | regime |
|---|---|---|---|
| 3-puzzle (m=2) | 4/4 | vacuous (no non-corner cells) | trivial |
| 8-puzzle (m=3) | 3/4 (blank leaks) | 62.5% — barely above 44% random baseline | rule has no room to bite |
| 15-puzzle (m=4) | 4/4 in 17/17 | 98.5% — far above 25% baseline | rule strongly active |
| 24-puzzle (m=5) | conjectured 4/4 | conjectured strong | rule should be *more* active than at m=4 |

Geometry: the union of 2×2 corner blocks covers 8/9 cells at m=3, 16/16 (with overlap) at m=4, and only 16/25 at m=5 — so 9 cells at m=5 are outside any corner block. The rule has more room to constrain, not less.

### Conjectured m=5 frame rule

The same predicate, translated:
- (a) corner tiles `{_, 1, 5, 21}` at antipodal corner positions `{0, 24, 20, 4}`.
- (b) each of the 8 goal-neighbors of those corner tiles (tiles `{2, 6, 4, 10, 16, 22, 20, 24}` — eight distinct tiles indexing the corner-adjacent goal cells) within Chebyshev ≤ 1 of its assigned anti-corner, at most one exception of Chebyshev = 2.

This is a *conjecture* at m=5, not a proven property. The plan tests it.

### What did not survive

- "Full 180° reversal of goal is an antipode" — false at m=3 and m=4.
- "The (n−1)-puzzle antipode shape recurs at the center of n-puzzle antipodes" — supported by exactly one data point (A5), and only because at m=2 the 3-puzzle antipode and the 180°-rotation coincide. Not propagable.
- A single closed-form candidate board for the 24-puzzle antipode — the right object is a *search space*, not a board.

## The plan: build 24-puzzle infrastructure and push past depth 152 inside the frame-restricted subspace

This is Milestone 5 in `DESIGN.md`. Rather than tackle the full ~7.76 × 10²⁴ reachable state space, attack the frame-conformant subspace and either (a) raise the lower bound past 152, or (b) refute the frame rule at m=5 and learn something specific.

### Phase 1 — port `puzzle15/` to `puzzle24/`  ✅ DONE

Mirrored the 15-puzzle module structure for a 5×5 board with tiles 1..24 and a blank.

- `src/puzzle24/{state,rank,symmetry}.rs`. (Moves are folded into `state.rs`, mirroring puzzle15 — there is no separate `moves.rs`.)
- State representation: `[u8; 25]` (chosen for clarity, matching puzzle15).
- Ranking: a u128 Lehmer rank/unrank bijection over `25!/2` (the generic permutation-ranking primitive; no full distance table is built).
- Symmetry: anti-diagonal reflection with compile-time `SIGMA`/`TAU`.
- Solvability: 24-puzzle is **odd-width** (like the 8-puzzle), so a state is solvable iff its inversion count is even — independent of the blank's position.

Verification: `tests/puzzle24_state.rs` (goal state, odd-width parity, hand-traced 5-move sequence, reflection involution) + module unit tests. Run with `cargo test --test puzzle24_state` (the crate is named `puzzle8`; there is no `-p puzzle24`).

### Phase 2 — zero-aware PDBs  ✅ DONE

**Pivot (see `docs/zpdb-codec-spec.md`).** The original plan called for standard additive Korf 6-6-6-6 PDBs. We instead target **zero-aware additive PDBs (ZPDBs)** (Clausecker & Reinefeld, SOCS 2019): the index tracks not just the pattern-tile positions but the blank's **zero-tile region**. This gives an ~8.6× stronger heuristic, and — crucially — the region-aware construction also *removes the build bottleneck*.

**Partition (canonical Korf 6-6-6-6):**
- `{1, 2, 3, 6, 7, 8}`, `{4, 5, 9, 10, 14, 15}`, `{11, 12, 16, 17, 21, 22}`, `{13, 18, 19, 20, 23, 24}`.

**Corrected sizes** (the original plan's "1.62 × 10⁹ entries, ~1.6 GB/PDB, 6.4 GB total" was wrong on all three counts):
- Standard additive (no-blank) PDB: `P(25, 6) = 127,512,000` entries ≈ **127 MB/PDB**, ~510 MB for four. (`2,422,728,000 = P(25, 7)` is the *transient BFS* count, not the stored PDB.)
- **Zero-aware PDB: 181,008,000 entries** (`6! · 251,400`, matching the paper's Table 1) — region-collapsed, ≈ 172 MB uncompressed, ≈ **21.6 MB at 1 bit/entry**.

**The build optimization = the zero-aware construction.** The standard 0/1 BFS over `(pattern_config, blank_cell)` (≈2.42 B states) is dominated at m=5 by a *serial* 0-cost closure — measured ~8 min, ~1.5 of 12 cores, ~15.5 GB peak for one 6-tile PDB. BFS over `(pattern_config, blank_region)` instead collapses the blank's free wandering into its zero-tile region: the 0-cost closure vanishes (every edge is a unit-cost pattern move → bipartite, `Δh = ±1`), the state space shrinks ~13×, and the BFS is fully parallel. **Result: full 6-tile ZPDB in 42.5 s (11.4× faster), 0 unvisited of 181,008,000, ~185 MB working set.**

Done + verified:
- `src/puzzle24/pdb/{pattern,build,db,heuristic}.rs` — standard additive engine (rayon 0/1 BFS, `P24D` save/load/mmap, Korf-max incremental evaluator). The **admissibility oracle**: ZPDB ≥ additive ≤ true.
- `src/bin/build_pdb24.rs` — `--part a|b|c|d`, `--tiles`, `--threads`, `--verify-sha`/`--write-sha`, **`--zero-aware`**.
- `src/puzzle24/pdb/zpdb.rs` — the `(m,p,r)` perfect-hash index (shape-rank × perm-rank × region-rank), proven a bijection; cached per-shape region tables; shape-parity (bipartite invariant) table for codec parity recovery.
- `src/puzzle24/pdb/zbuild.rs` — region-aware BFS (sequential + rayon). Verified **byte-identical** to `build::build` via min-over-regions (k=2,3,4); `ZPDB ≥ additive` pointwise.
- `src/puzzle24/pdb/zcodec.rs` — 1-bit codec: `pack_bits`, `lookup_bit`, `diff_lookup`, `parity_of_h`. The bipartite-parity invariant is **parity of the sum of pattern-tile cell indices** (not perm parity), recoverable from the index alone via `ZpdbLayout::shape_parity`.
- `src/puzzle24/pdb/zdb.rs` — `Z24D` file format (magic, version, pattern bitmask, reserved, u64 total, packed bits), `ZPatternDb::{from_dist, build, save, load, load_mmap}`, and `cold_lookup` (O(h) descent for the IDA\* root).
- `src/puzzle24/pdb/heuristic.rs::ZpdbInc<N>` — incremental Korf-max over `N` disjoint ZPDBs, normal + reflected views. Cold-seeded at root; per-move it skips the diff when an anon swap leaves `(m,p,r)` unchanged, else applies `diff_lookup`. Tests: root admissibility-oracle dominance over additive Korf-max, advance-matches-fresh-root over 1500-step random walks, IDA\* finds verifiable optima on shallow scrambles.
- `src/bin/solve24.rs` — `--heuristic zpdb` loads four `pdb24_*.zbin` and runs `ZpdbInc<4>` through `idastar_inc_with_stats`.
- `data/pdb24_{a,b,c,d}.zbin.sha256` — pinned SHA-256s for the four ZPDBs (22,626,024 bytes each). The `.zbin` artifacts themselves are `.gitignore`d (matching the existing `.bin` convention) and regenerated via `cargo run --release --bin build_pdb24 --features parallel,mmap,sha -- --zero-aware --part X --out data/pdb24_X.zbin --verify-sha data/pdb24_X.zbin.sha256` (~45 s/PDB, ~3 min for all four). End-to-end smoke-tested: `solve24 --heuristic zpdb` on the goal-rotated-by-blank position recovers a 68-move optimal solution.

### Phase 3a — generic IDA\* solver  ✅ DONE

`src/puzzle24/search/idastar.rs` + `src/bin/solve24.rs`, modeled on the puzzle15 templates.

- `h(s) = max(additive(s), additive(reflect(s)), Manhattan(s))` — incremental Korf-max (`KorfPdbInc`). The zero-aware 1-bit PDB will slot into the *same* `IncHeuristic` interface once the codec lands.
- `solve24 --state '<board>' --heuristic korf|manhattan` returns optimal depth + path.

Verification: `tests/puzzle24_solver.rs` — Manhattan and Korf-max solvers recover exact BFS distance to depth 8 and reach GOAL; the two heuristics agree on optimal length for depth-14 scrambles. CLI smoke-tested.

### Phase 3b — frame-restricted variant  ⬜ NOT STARTED

Add a frame predicate as an optional pruning hook on the Phase 3a search (e.g., a `--frame` flag on `solve24`, or a separate entry point that wraps the generic expander).

- **Frame pruning**: any state at the search frontier that fails frame conformance is discarded immediately. This is a heuristic restriction, not a sound pruning over all paths — it relies on the conjecture that antipodes are frame-conformant. A path to a frame-violating state is discarded; we will never *reach* a frame-violating state through this search.
- **Bidirectional search variant** (optional): also search from candidate frame-conformant boards (the analog of the 17 m=4 antipodes — for m=5, the natural seed is the full 180° rotation, despite knowing it's likely not an antipode itself, plus other corner-conformant high-Manhattan states).

Verification: port the frame predicate to a `solve15 --frame` mode and confirm that (a) unrestricted `solve15` recovers all 17 known depth-80 antipodes, and (b) `solve15 --frame` also recovers all 17 — i.e., the predicate doesn't lose any known antipode. This validates both the generic solver and the predicate before deploying on m=5.

### Phase 4 — the lower-bound push  ⬜ NOT STARTED

Three concrete experiments, in order:

1. **Seed-and-deepen** (generic solver). Start from the full 180° rotation of the 24-puzzle goal (corner-conformant by construction), run unrestricted IDA\* with increasing thresholds. The Manhattan-lower-bound of the rotation is 160 ≥ 153, so *if* it's reachable in 152 moves the existing bound suffices; *if* the optimal solution exceeds 152, we have a new lower bound. The rotation's actual depth is the first datum we want, and it does not depend on the frame conjecture.
2. **Enumerate frame-conformant high-Manhattan states** (frame-restricted solver). With corners forced, the remaining 21 tiles have ~21!/something placements; restrict to those with Manhattan ≥ 155 (or whatever bound makes the count feasible). Run frame-restricted IDA\* on each; track maximum depth found.
3. **Refutation check** (generic solver). Sample frame-violating states with high Manhattan (e.g., corner tiles misplaced but interior near-rotated), solve with the unrestricted Phase 3a solver, and look for any with optimal depth ≥ some m=5-equivalent of "deep." Any frame-violating state with optimal depth ≥ 153 *both* raises the lower bound *and* refutes the conjecture. A null result here strengthens (does not prove) the conjecture.

Compute budget: each IDA* run at depth ~150 on the 24-puzzle is non-trivial — single-state solves can take hours with PDBs. Budget should be capped (e.g., 100 candidate states × 2 hours each on a single core) before declaring inconclusive.

## Verification

End-to-end signals that the plan worked:

- `cargo test --test puzzle24_state` and `cargo test --test puzzle24_solver` pass (state, moves, symmetry, optimal round-trips). ✅
- `cargo run --release --bin build_pdb24 -- --zero-aware --part a --out data/pdb24_a.zbin --verify-sha data/pdb24_a.zbin.sha256` reproduces the committed PDB byte-for-byte, and similarly for b/c/d. ✅
- `cargo run --release --bin solve24 -- --state '<board>'` returns an optimal-depth-and-path for a small hand-constructed test state. ✅
- `solve15` (unrestricted) and `solve15 --frame` both recover all 17 known depth-80 antipodes — sanity check before deploying on m=5 (Phase 3b). ⬜
- Milestone writeup `docs/24-puzzle-lower-bound-push.md` reports one of: (a) a 24-puzzle state at optimal depth ≥ 153, raising the lower bound; (b) a frame-violating state with depth that contradicts the conjecture; (c) a null result with stated compute budget and per-state maximum depth found. ⬜

## Critical file references

- `data/pdb15_antipodes.txt` — empirical justification for the frame rule at m=4
- `records/antipodes8.txt` — m=3 comparison
- `docs/zpdb-codec-spec.md` — the zero-aware PDB + 1-bit codec spec (Phase 2)
- `src/puzzle24/pdb/{zpdb,zbuild}.rs` — the `(m,p,r)` index and the region-aware BFS
- `src/puzzle24/{state,rank,symmetry}.rs`, `src/puzzle24/search/`, `src/puzzle24/pdb/` — the built 24-puzzle stack
- `src/bin/{build_pdb24,solve24}.rs` — the 24-puzzle CLI
- `src/puzzle15/` — the port templates (state/rank/symmetry, IDA\*, Korf-max, PDB loader)
- `src/bin/verify15.rs` — template for any phase-3 sanity check
- `DESIGN.md` — Milestone 5
- Memory: `goal-24-puzzle-diameter.md`, `bounds-24-puzzle-corrected.md`, `zpdb-pivot-phase2.md`
