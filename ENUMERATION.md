# ENUMERATION — complete enumeration of deep 15-puzzle boards (depth ≥ T)

Produce, for each depth `D ≥ T`, the **complete** set of 15-puzzle boards whose optimal
solution length is exactly `D`, walking `T` down from 80. No full breadth-first search of
the 10.46-trillion-state space: we only ever touch the boards in layers `[T..80]`.

The exact distribution `N(d)` (Korf & Schultze 2005, A087725) lives in
`data/pdb15_depth_histogram.txt` and is both the **termination oracle** and a **per-layer
correctness gate**. The 17 depth-80 antipodes are pinned in `data/pdb15_antipodes.txt`.

## Structural facts

- **Parity.** A board and any neighbor have opposite-parity optimal depths and are one move
  apart, so their depths differ by **exactly ±1** — every neighbor is strictly *down*
  (depth−1, toward GOAL) or strictly *up* (depth+1). No two neighbors share a depth.
- **Bellman.** `depth(s) = 1 + min over neighbors of depth(neighbor)`.
- The boards a top-down sweep cannot reach are **strict local maxima** (every neighbor
  shallower). They are common at the very top — the 17 antipodes are all blank-in-corner
  (2 moves each), touching ≤ 34 of the 70 depth-79 boards, so ≥ 36 depth-79 boards are
  local maxima — and become a vanishing fraction deeper down.

## Algorithm (`src/puzzle15/enumerate/`)

Process `d` from 79 down to `T`, keeping a `rank → exact-depth` map seeded with the
antipodes at depth 80.

1. **Descent (solve-free, unambiguous).** Expand every map board at depth `d+1`; each
   hash-miss neighbor is necessarily depth `d` — parity gives depth `d` or `d+2`, and all
   depth-`d+2` boards are already in the map (layer `d+2` was certified complete). No
   heuristic, no search.

2. **Local-maxima recovery (solve-free).** Build the depth-`d−1` *shell* (hash-miss
   neighbors of the depth-`d` boards — again necessarily depth `d−1`). A maximum at depth
   `d` is an up-neighbor of the shell, not yet in the map, **all of whose neighbors are in
   the map at depth `d−1`** (Bellman ⇒ depth exactly `d`). This is a membership test over
   2–4 neighbors — no solve. It recovers every "shell-anchored" maximum.

3. **Count gate.** Require `|Layer[d]| == N(d)`. Equal ⇒ the layer is provably complete;
   write it and descend. Short ⇒ the residue is local maxima inside disconnected pockets
   (down-neighbors not reachable from above). Fallback: `idastar` (`korf-plus`) on
   candidates near the deep set — feasible because even a depth-80 antipode solves in ~9 s.
   A persistent shortfall is the natural "uncomfortable" stopping point: we never write an
   incomplete layer or descend past one.

The heuristic (`korf-plus = max(Korf 7-8, reflected, linear-conflict, walking-distance)`,
`src/puzzle15/pdb/heuristic.rs`) is used **only** by the fallback, never by the descent or
the Bellman recovery. A zero-aware 15-PDB would further tighten it if pockets prove costly.

## Reuse

`State`/moves (`state.rs`), `rank`/`unrank` (`rank.rs`), `idastar` (`search/idastar.rs`),
the PDB/LC/WD heuristics (`pdb/heuristic.rs`, `search/*.rs`), the histogram and antipode
data files.

## Output

`data/enum/depthNN.ranks`, one file per depth, each board as a 6-byte little-endian rank
(`rank()`), all boards (full symmetry orbits). `enum_expand15` renders a file to 16-token
rows and re-verifies. Sizes: depth77 ≈ 160 KB, depth75 ≈ 9 MB, depth70 ≈ 5.2 GB.

## Walk-down / status

### Measured status (2026-06-15) — discomfort boundary is the very top

- **Layer 80**: 17/17, complete, written, = N(80). ✓
- **Layer 79, descent** (solve-free): **34/70** — exactly the antipode-neighbors. The other
  36 are strict local maxima, and ~34 of those are not adjacent to the antipode-reachable
  depth-78 shell, so the shell/Bellman recovery never generates them as candidates.
- **Layer 79, band BFS `[78,80]`**: converges (108 s) to a **disconnected 36-board
  component**. The remaining 34 depth-79 boards sit in other `[78,80]` components, bridged
  only through depth-77.

**Conclusion:** storage/time gets uncomfortable immediately at **T=79**. Completing it needs
floor ≤ 77 ≈ 10⁵ deep solves at ~7 s each (hours, possibly still disconnected → lower); each
lower layer multiplies the cost, and T=75 would need ~10⁷ solves (infeasible). The band
engine is correct but only practical at the very top.

**The scalable fix** is a **zero-aware 15-PDB** tight enough to classify a hash-miss neighbor
as up (depth d+1, a local maximum) vs down (depth d−1) *without solving* — that makes the
solve-free descent + Bellman recovery complete each layer with no idastar, the only route
that reaches T=75. Memory (not compute) would then be the constraint: the map holds the
cumulative `N(≥T)` ranks (≈ 1.8 M / 40 MB at T=75; ≈ 1.3 B / 8 GB at T=70).
