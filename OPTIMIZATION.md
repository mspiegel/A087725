# OPTIMIZATION.md

Optimization opportunities for the 8- and 15-puzzle solvers, prioritized by
impact on the hot path (the 15-puzzle IDA\* inner loop). The 8-puzzle is fast
enough via its exact distance table that it matters mainly as a shared-code
testbed; most items below are in code shared by both, or in the 15-puzzle PDB
path.

## Status (measured on the Korf-100 bench, warm page cache)

Implemented so far, in order:

- **#2 + #3 — table-driven move-gen + threaded blank, `debug_assert!`** (commit
  `6598dd9`). Behavior unchanged (identical node counts). No measurable change on
  the korf PDB workload (PDB memory latency dominates there); a controlled
  Manhattan A/B showed ~1.2× (34.3 → 41.4 Mnodes/s) — the per-node CPU win,
  masked under the PDB heuristic.
- **#1 — incremental Korf evaluator** (`KorfPdbInc` + `IncHeuristic` +
  `idastar_inc_with_stats`). Maintains each PDB's projected board for the normal
  view (advanced by the move) and the reflected view (advanced by the transposed
  move), so no per-node re-projection or `reflect`. Value is byte-identical to
  the scratch `max(additive, reflected)` composition (proven by unit tests +
  identical Korf-100 node counts). **Result: korf 1.04s → 0.42s, a ~2.5×
  speedup (4.2 → 10.4 Mnodes/s).** A/B with `--scratch` in the bench.

Tried and **rejected on this workload** (the bench earned its keep):

- **#4 — move ordering** (expand lowest-`h` child first). Implemented and
  measured: **0.07%** fewer nodes, and wall-clock *rose* ~6% (419 → ~444 ms)
  from the per-node sort. IDA\* with a tight heuristic visits nearly all
  `f ≤ bound` nodes regardless of order — ordering only skips siblings *after*
  the goal is found, a negligible fraction. Reverted.
- **#5 / #6 — duplicate elimination (FSM / transposition table).** `don't-undo`
  already removes every girth-2 cycle; the next duplicates are girth-12 (the
  blank circling a 2×2 block ×3). `examples/dup_probe.rs` measures the ceiling:
  of 4,378,068 nodes only **10.5% are redundant re-expansions** — and the FSM
  catches only a subset of those, while a TT's per-node hash probe would erase
  the saving exactly as #4's sort did. Not worth it at m=4. **Deferred to m=5**,
  where depth ~150 (vs ~53) makes any branching-factor cut compound far more,
  and where Korf used the FSM; `dup_probe` is the tool to re-confirm there.

So the per-node re-projection/reflect cost — not just memory latency — was the
dominant lever for the 7-8 partition (#1, 2.5×). With the korf heuristic this
directed, node-count micro-optimizations (#4/#5/#6) do **not** pay at m=4. The
remaining real levers are at m=5 scale (24-puzzle) and in representation (#6
packed-state mainly as a copy/compare win, not for its TT).

### Profile (samply, 10 warm iterations, arm64)

Self-time of the default `korf` solve splits cleanly (99.8% in our code, CPU-bound):

| self% | function | cost |
|---:|---|---|
| ~50% | `KorfPdbInc::value` | the 4 PDB lookups/node: `rank()` (`count_ones` loop) + the random table load |
| ~29% | `search_inc` | recursion, `apply_at`, `s == GOAL` compare, move-gen |
| ~21% | `ProjectedState::apply` | the incremental projected-board advance (×4/node) |

Two profile-suggested tweaks were **tried and rejected (measured)**:

- **Port the `NEIGHBOR` table into `ProjectedState::apply`** (the 21% line):
  **~4% slower** (423 → 440 ms), reverted. Reason: `W = 4` is a power of two, so
  the apparent `div`/`mod` already compiles to a shift + mask (free); the real
  21% is the two 16-byte array copies (`cells` + `pos_of`) per call, inherent to
  the immutable `Copy` context. Only a **packed `ProjectedState`** (#6-style)
  would shrink that — a bigger change, deferred.
- **#7 `target-cpu=native`**: **neutral** on arm64 (no measurable change).
  `count_ones()` already lowers to baseline ARMv8 `CNT`/`ADDV`, so there's no
  POPCNT-style win like on x86. Not committed (a forced `.cargo/config.toml`
  hurts build reproducibility for zero gain here); enable via `RUSTFLAGS` on an
  x86 build host if ever relevant.

Takeaway: at m=4 the korf solver is at a local optimum — `value` (rank + PDB
load) and the projected-board copies dominate.

### Binary inspection (llvm-objdump) — two real wins the source hid

Disassembling `value` (the 50% function) showed `count_ones` already lowers to
`cnt.8b`/`addv.8b` (optimal — confirms `target-cpu=native` can't help), but also
two pure-overhead patterns, each **4× per node**:

1. **Redundant mmap re-slice.** Every PDB lookup recomputed `Storage::dist()`
   (`&mmap[HEADER..]`) — a length-checked re-slice + pointer adjust — even though
   it's invariant. Fixed: `KorfPdbInc` caches each PDB's `dist: &[u8]` (and
   `Pattern`) at construction.
2. **Bounds check on `dist[rank]`.** Replaced with `get_unchecked` — sound
   because `rank() ∈ [0, num_projected_states) == dist.len()` by the PDB
   construction invariant (debug-asserted).

Measured (Korf-100, warm): **~424 → ~409 ms (Copy path), ~3–4%**; identical node
counts, 100/100 optimal. Stacks with `--mut`: combined **~396 ms, 11.0 Mnodes/s**.

`ProjectedState::apply` (21% bucket) has the same pattern — 3 bounds checks
(`cells[n]`, swap index, `pos_of[swapped]`), all on indices `< 16` — but the
vectorized copy dominates that bucket (the checks are ~1%), and `apply` is on the
SHA-pinned PDB-build path, so `unsafe` there isn't worth ~1%. Left as-is.

Net: the genuinely *next* gains remain structural (packed representation) or at
m=5 scale, but binary inspection still found a clean ~4%.

## Headline finding: the incremental machinery exists but the search doesn't use it

`ProjectedState` carries a `pos_of: [u8; 16]` table and an incremental `apply`
(`src/puzzle15/pdb/pattern.rs:204`) whose docstring explicitly justifies itself:
*"At k=8 and billions of IDA\* node expansions per antipode, the saving is
material."* But the actual solver never calls it. `idastar::search` operates on
full `State`, and `PatternDb::h` (`src/puzzle15/pdb/db.rs:104`) calls
`ProjectedState::from_state(s, pattern)` — a fresh **O(16) scan rebuilding
`cells` and `pos_of`** — on *every node*. The incremental path is dead code
outside PDB construction.

For the default `korf` heuristic the per-node cost is roughly:

- `MaxHeuristic(additive, reflected(additive))`
- → `additive.h(s)`: 2× `from_state` (O(16)) + 2× `rank` (O(k))
- → `reflected.h(s)`: 1× `reflect(s)` (O(16)) + 2× `from_state` + 2× `rank`

So **~5 full 16-cell passes + 4 ranks + a reflect, recomputed from scratch at
every node**, when a single move changed exactly one tile.

---

## Tier 1 — per-node cost in the inner loop

### 1. Incremental heuristic evaluation (biggest CPU win)

Thread the heuristic state through the DFS instead of recomputing.

- **Manhattan / Linear Conflict:** a move shifts one tile by one cell, so
  Δ(Manhattan) = ±1 (the test `manhattan_changes_by_at_most_1_per_move` proves
  it). Pass the running `h` down the recursion and update in O(1); for LC only
  the moved tile's row + column lines need recompute.
- **PDB:** maintain one `ProjectedState` per db (and per reflected view) in the
  search context; on each move call the existing incremental `apply` and
  re-`rank` (O(k)). This deletes all the per-node `from_state` scans.
- **Reflected view without a stored reflected PDB:** the move on `reflect(s)` is
  just the transposed move (Up↔Left, Down↔Right). Maintain a reflected
  `ProjectedState` and apply the transposed move incrementally — keeps the
  storage savings the design deliberately chose, but pays zero per-node
  `reflect()`.

Needs a small trait change (e.g. an `IncrementalHeuristic` with
`init(state) -> (h, ctx)` and `step(ctx, move, blank, new_blank) -> (h, ctx)`),
or a concrete fused search specialized to the Korf heuristic.

**Caveat:** for PDB-driven search the *dominant* cost is often the
cache-missing random reads into the ~519 MB tables (4 lookups/node for korf),
which this doesn't remove — expect a solid constant factor on CPU work and a
large win for the Manhattan/LC/WD heuristics, less so against PDB memory
latency. The node-count reductions in Tier 2 stack multiplicatively on top.

### 2. Precomputed neighbor/move tables + thread the blank position

`State::blank_pos()` (`src/puzzle15/state.rs:113`) is a linear scan, called by
`legal_moves()` *and* again inside every `apply()` (`src/puzzle15/state.rs:136`).
Per node that's `1 + children` scans (up to 5) plus repeated div/mod. Replace
with const lookup tables:

```rust
const LEGAL: [MoveSet; 16];        // LEGAL[blank]
const NEIGHBOR: [[u8; 4]; 16];     // NEIGHBOR[blank][move]
```

`legal_moves` becomes one array read; `apply(m, blank)` becomes
`next.swap(blank, NEIGHBOR[blank][m as usize])`. Since the child's blank is
`NEIGHBOR[blank][m]`, thread it through the search and no scan ever runs.
Eliminates all blank scans + all coordinate arithmetic.

### 3. `assert!` → `debug_assert!` in `State::apply`

`src/puzzle15/state.rs:141-156` (and the 8-puzzle equivalent) use `assert!`,
which runs in *release*. The search only ever applies moves drawn from
`legal_moves()`, so they're 4 dead branches per node. `ProjectedState::apply`
already correctly uses `debug_assert!`; make `State::apply` match. (The
neighbor-table rewrite in #2 subsumes this.)

---

## Tier 2 — node-count reduction (multiplies with Tier 1)

### 4. Move ordering

In the final IDA\* iteration, expand the child with the lowest `h` first. The
first goal hit is still optimal (admissible h), but you reach it after exploring
fewer siblings.

> **Verdict (measured, m=4): rejected.** 0.07% fewer nodes, ~6% slower
> wall-clock from the per-node sort. See the Status section.

### 5. Generalized duplicate-move pruning (Taylor–Korf FSM)

The solver already prunes immediate undos (`src/puzzle15/search/idastar.rs:65`).
The published finite-state-machine extension prunes longer redundant sequences
(the smallest are girth-12: the blank circling a 2×2 block ×3), cutting the
effective branching factor below the ~2.13 you get from undo-pruning alone. Pure
node savings, optimality-preserving.

> **Verdict (measured, m=4): deferred to m=5.** `examples/dup_probe.rs` shows
> only 10.5% of nodes are redundant re-expansions (the ceiling), and the FSM
> catches a subset of those while a TT pays #4-style per-node overhead. Worth
> revisiting at 24-puzzle depth (~150), where Korf used it. See the Status
> section.

---

## Tier 3 — representation & build flags

### 6. Packed u64 (nibble) state

Each tile is 0–15 → the whole 4×4 board fits in 64 bits. Makes `State`
copy/compare a single register op (vs 16-byte memcmp against `GOAL` every node)
and makes a hash-based **transposition table** cheap to add. More invasive
(swap = nibble extract/insert), so treat as opt-in, but it's the standard
fast-solver representation and the enabler for memoizing across the IDA\*
re-search.

### 7. `target-cpu=native` for local research builds

`rank()` leans on `count_ones()`; without POPCNT/BMI it's a fallback loop. Add
`.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

Determinism / SHA pinning is safe — POPCNT returns identical values, and the PDB
bytes are arithmetic-deterministic regardless of target. Consider
`panic = "abort"` in the release profile too (smaller, no unwind tables).
`lto = true, codegen-units = 1` are already set — good.

> **Verdict (measured, arm64): neutral, not committed.** On Apple Silicon
> `count_ones()` already lowers to baseline `CNT`/`ADDV`, so there was no
> measurable change. The POPCNT argument is x86-specific. See the Status
> section.

---

## Tier 4 — PDB build time/memory

### 8. `bfs_dist` should be a bitset, not `Vec<u8>`

`src/puzzle15/pdb/build.rs:65`: its stored depth is *never read back* — only
compared `== UNVISITED`. A 1-bit visited set suffices, cutting the transient
from `num_bfs_states` bytes to bits: **P8 drops 4.15 GB → ~519 MB**. The
parallel path (`src/puzzle15/pdb/build.rs:295`) does the same — swap
`fetch_min` for an atomic `fetch_or` on the bit and keep the "I flipped it"
frontier-dedup logic.

### 9. Avoid `frontier_d.clone()` in the 0-cost closure

`src/puzzle15/pdb/build.rs:78`: at deep layers this clones millions of states
each iteration. Process the closure via an index into `frontier_d`, appending
in place, instead of cloning into a separate `work` stack.

### 10. Port `pos_of` back to the 8-puzzle

`puzzle8::ProjectedState::rank` (`src/puzzle8/pdb/pattern.rs:200`) still does an
O(k·9) linear scan per tile that the 15-puzzle version already eliminated. Minor
for solve time, but the 8-puzzle is the verification/benchmark bed, so keeping
the two implementations algorithmically aligned matters.

---

## Process note

The **Korf-100 benchmark** is the measurement harness for this work:
`examples/korf100_bench.rs`, driven by `data/korf100.txt` (Korf 1985 Table 1
instances + published optimal lengths). Run before/after each change:

```text
cargo run --release --features mmap --example korf100_bench -- --pdb-dir data
```

It reports node count and wall-clock *separately* per the split that matters:
incremental-h and `target-cpu` move wall-clock at fixed nodes; move-ordering and
the duplicate-pruning FSM move nodes. It also asserts every solution is optimal
against Korf's table (exit non-zero on any mismatch), so it doubles as an
optimality regression test. Node counts come from `idastar_with_stats`
(`SearchStats { nodes, iterations }`), which is heuristic-implementation-agnostic
— so the node metric stays stable across the incremental-h refactor (#1), letting
you attribute its gains purely to wall-clock.

Baseline (korf, warm cache, before #1): 100/100 optimal, total optimal 5305,
mean 43,780 nodes/instance, ~4.2 Mnodes/s (1.04s). After #1 (incremental
evaluator, default `korf` path): same nodes, ~10.4 Mnodes/s (0.42s). Use
`--scratch` to reproduce the pre-#1 path for A/B.

## Suggested order of implementation

1. **#2 + #3** — mechanical, low-risk, immediately measurable.
2. **#1** — the structural win, behind a new incremental-heuristic trait so the
   existing `Heuristic` API and tests stay intact.
3. **#8** — build memory.

Do them as separate commits on a branch with a small Korf-100 bench added first
so each step is measured.
