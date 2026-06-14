# Session synthesis: use the frame rule to push the 24-puzzle lower bound

## Context

The project's open goal is bounding the 24-puzzle diameter (currently `[152, 205]` STM; see memory `bounds-24-puzzle-corrected.md`). In this session we analyzed the existing antipode data files (`data/antipodes8.txt`, `data/pdb15_antipodes.txt`) and found empirical structure that constrains where antipodes live. The most actionable finding — the **frame rule** — is concrete enough to use as a search-space restriction at m=5.

This plan is now focused on that single use. Earlier drafts proposed a formal-proof exercise on the 15-puzzle, but the result that exercise would have produced (every depth-80 state is frame-conformant) is already in hand by direct inspection of `data/pdb15_antipodes.txt`. There is nothing to verify there that we haven't verified.

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

This is Milestone 5 in `DESIGN.md`. The session's contribution is to focus it: rather than tackle the full ~7.76 × 10²⁴ reachable state space, attack the frame-conformant subspace and either (a) raise the lower bound past 152, or (b) refute the frame rule at m=5 and learn something specific.

### Phase 1 — port `puzzle15/` to `puzzle24/`

Mirror the existing 15-puzzle module structure for a 5×5 board with tiles 1..24 and a blank.

- New module: `src/puzzle24/` with `state.rs`, `rank.rs`, `moves.rs`, `symmetry.rs`. Template: `src/puzzle15/state.rs`, `src/puzzle15/rank.rs`, `src/puzzle15/symmetry.rs`.
- State representation: 25 cells × 5 bits each = 128-bit packed, or `[u8; 25]` for clarity. Match whatever puzzle15 does.
- Ranking: the 24-puzzle state space (~7.76 × 10²⁴) is too large for full ranking; instead rank only within the frame-conformant subspace (see Phase 3) or rank tile-subsets for PDBs (Phase 2). Avoid building any full distance table.
- Symmetry: the 5×5 puzzle has the same anti-diagonal reflection as the 4×4, applied around the blank's goal corner. Mirror `src/puzzle15/symmetry.rs`.

Verification: write `tests/puzzle24_state.rs` checking goal state, solvability parity (24-puzzle is odd-width like 8-puzzle; inversions must be even), a hand-traced 5-move sequence, and reflection involution.

### Phase 2 — build Korf 6-6-6-6 PDBs

Mirror `src/bin/build_pdb15.rs` as `src/bin/build_pdb24.rs`. The 24-puzzle PDB build will be **multithreaded**, inheriting the rayon-fanned 0/1 BFS already implemented for the 15-puzzle in `src/puzzle15/pdb/build.rs::build_parallel` (atomic `Vec<AtomicU8>` distance arrays with `fetch_min(Relaxed)`, thread-local frontier emission, byte-deterministic output). Port the same structure to `src/puzzle24/pdb/build.rs`; expose `--threads N` on `build_pdb24` matching the existing 15-puzzle CLI.

Korf's well-known 6-6-6-6 partition of the 24 non-blank tiles:

- `{1, 2, 3, 6, 7, 8}` — top-left region
- `{4, 5, 9, 10, 14, 15}` — top-right region
- `{11, 12, 16, 17, 21, 22}` — bottom-left region
- `{13, 18, 19, 20, 23, 24}` — bottom-right region

Each PDB: 25 × 24 × 23 × 22 × 21 × 20 × 19 = `1.62 × 10⁹` entries. With 1 byte per entry (u8 sufficient up to depth ≤ 255), each PDB is ~1.6 GB; four total ≈ **6.4 GB on disk**. This is larger than the 15-puzzle PDBs but tractable on commodity hardware.

Alternative (smaller, slower): Korf's 7-8-9 partition is sometimes used but the disk cost grows further. Stick with 6-6-6-6 unless we hit a heuristic-quality wall.

Output files: `data/pdb24_a.bin`, `data/pdb24_b.bin`, `data/pdb24_c.bin`, `data/pdb24_d.bin`, plus SHA-256s.

Verification: run on a small handful of known states with hand-checked Manhattan + tile-displacement bounds, confirm each PDB is admissible.

### Phase 3a — generic IDA\* solver

Add `src/puzzle24/search/idastar.rs` modeled directly on `src/puzzle15/search/idastar.rs`. No frame logic — this is the unrestricted baseline solver and the thing `solve24` exposes.

- `h(s) = max(pdb_a(s) + pdb_b(s) + pdb_c(s) + pdb_d(s), Manhattan(s), reflected_pdb(s))` — standard Korf-max construction.
- Expose a `solve24` binary mirroring `src/bin/solve15.rs`: takes `--state '<board>'`, returns optimal depth and path.

Verification: hand-constructed test states with known short optimal depths solve correctly; `cargo test -p puzzle24` covers small-depth round-trips.

### Phase 3b — frame-restricted variant

Add a frame predicate as an optional pruning hook on the Phase 3a search (e.g., a `--frame` flag on `solve24`, or a separate entry point that wraps the generic expander).

- **Frame pruning**: any state at the search frontier that fails frame conformance is discarded immediately. This is a heuristic restriction, not a sound pruning over all paths — it relies on the conjecture that antipodes are frame-conformant. A path to a frame-violating state is discarded; we will never *reach* a frame-violating state through this search.
- **Bidirectional search variant** (optional): also search from candidate frame-conformant boards (the analog of the 17 m=4 antipodes — for m=5, the natural seed is the full 180° rotation, despite knowing it's likely not an antipode itself, plus other corner-conformant high-Manhattan states).

Verification: port the frame predicate to a `solve15 --frame` mode and confirm that (a) unrestricted `solve15` recovers all 17 known depth-80 antipodes, and (b) `solve15 --frame` also recovers all 17 — i.e., the predicate doesn't lose any known antipode. This validates both the generic solver and the predicate before deploying on m=5.

### Phase 4 — the lower-bound push

Three concrete experiments, in order:

1. **Seed-and-deepen** (generic solver). Start from the full 180° rotation of the 24-puzzle goal (corner-conformant by construction), run unrestricted IDA\* with increasing thresholds. The Manhattan-lower-bound of the rotation is 160 ≥ 153, so *if* it's reachable in 152 moves the existing bound suffices; *if* the optimal solution exceeds 152, we have a new lower bound. The rotation's actual depth is the first datum we want, and it does not depend on the frame conjecture.
2. **Enumerate frame-conformant high-Manhattan states** (frame-restricted solver). With corners forced, the remaining 21 tiles have ~21!/something placements; restrict to those with Manhattan ≥ 155 (or whatever bound makes the count feasible). Run frame-restricted IDA\* on each; track maximum depth found.
3. **Refutation check** (generic solver). Sample frame-violating states with high Manhattan (e.g., corner tiles misplaced but interior near-rotated), solve with the unrestricted Phase 3a solver, and look for any with optimal depth ≥ some m=5-equivalent of "deep." Any frame-violating state with optimal depth ≥ 153 *both* raises the lower bound *and* refutes the conjecture. A null result here strengthens (does not prove) the conjecture.

Compute budget: each IDA* run at depth ~150 on the 24-puzzle is non-trivial — single-state solves can take hours with Korf PDBs. Budget should be capped (e.g., 100 candidate states × 2 hours each on a single core) before declaring inconclusive.

## Verification

End-to-end signals that the plan worked:

- `cargo test -p puzzle24` passes (state, moves, symmetry round-trips).
- `cargo run --release --bin build_pdb24 -- --part a` produces `data/pdb24_a.bin` matching a committed SHA-256, and similarly for b/c/d.
- `cargo run --release --bin solve24 -- --state '<board>'` returns an optimal-depth-and-path for a small hand-constructed test state (Phase 3a).
- `solve15` (unrestricted) and `solve15 --frame` both recover all 17 known depth-80 antipodes — sanity check that the generic solver and the frame predicate are correct before deploying on m=5 (Phase 3b).
- Milestone writeup `docs/24-puzzle-lower-bound-push.md` reports one of: (a) a 24-puzzle state at optimal depth ≥ 153, raising the lower bound; (b) a frame-violating state with depth that contradicts the conjecture; (c) a null result with stated compute budget and per-state maximum depth found.

## Critical file references

- `data/pdb15_antipodes.txt` — empirical justification for the frame rule at m=4
- `data/antipodes8.txt` — m=3 comparison
- `src/puzzle15/{state,rank,symmetry}.rs` — port templates for `src/puzzle24/`
- `src/puzzle15/search/idastar.rs`, `src/puzzle15/search/heuristic.rs` — IDA* + Korf-max templates
- `src/puzzle15/pdb/mod.rs` — PDB loader/builder template
- `src/bin/build_pdb15.rs` — template for `build_pdb24`
- `src/bin/solve15.rs` — template for `solve24`
- `src/bin/verify15.rs` — template for any phase-3 sanity check
- `DESIGN.md` — Milestone 5
- Memory: `goal-24-puzzle-diameter.md`, `bounds-24-puzzle-corrected.md`
