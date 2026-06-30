# Zero-Aware PDB (ZPDB) + 1-bit codec — implementation spec

Pinned-down spec for the Phase 2 zero-aware pattern databases, from the primary
sources. Port target for `src/puzzle24/pdb/`.

**Sources**
- Clausecker & Reinefeld, *Zero-Aware Pattern Databases with 1-Bit Compression
  for Sliding Tile Puzzles*, SOCS 2019, pp. 35–43. (the spec below)
- Clausecker, *Notes on the Construction of Pattern Databases*, ZIB Report
  17-59, 2017. (the perfect-hash index / `(m,p,r)` ranking)
- Reference C implementation: `github.com/clausecker/24puzzle` (BSD-2). Files to
  port: `bitpdb.{c,h}`, `index.{c,h}`, `pdbgen.c`, `heuristic.c`,
  `tileset.{c,h}`. Distributed build follow-up: Nagahashi & Takahashi, ICCSA
  2025, DOI 10.1007/978-3-031-97000-9_17.
- Prior art the 1-bit scheme improves on: Breyer & Korf 2010a (1.6-bit / mod-3).

---

## 1. What a ZPDB is

A standard **additive PDB (APDB)** (Korf & Felner 2002) stores, for each
arrangement of the *k* pattern tiles, the number of moves to bring those tiles
home, **disregarding the blank**. Blank-*compression* (Felner et al. 2004)
stored the min over blank positions → smaller but **inconsistent**.

A **zero-aware PDB (ZPDB)** instead keeps track of **which *zero-tile region*
the blank lies in** — the connected set of cells the blank can reach without
moving a pattern tile. One entry per `(permutation, zero-tile region)`. This is
admissible **and consistent**, and gives a strictly higher *h*. On the 24-puzzle
it improves pruning **1.61×** over the plain 6-tile APDB; full enhancement stack
(ZPDB + optimal partitioning + collection) = **8.59×**.

Index is a tuple `(m, p, r)`:
- `m` — which grid cells are occupied (the pattern's *shape*),
- `p` — permutation of the pattern tiles within those cells,
- `r` — which zero-tile region the blank is in.

`m, p` are computed by bit-fiddling; `r` and the region count for a given `m`
come from an auxiliary table shared across all PDBs of the same tile count. A
6-tile ZPDB splits into C(24,6)=134,596 cohorts, each a rectangular array over
`(p, r)`. (Perfect-hash details: ZIB 17-59 / `index.c`.)

## 2. Sizes (24-puzzle, Table 1 of the paper) — CORRECTED

ZPDB stores per zero-tile *region*, **not** per exact blank position. The avg
region count per config is small, so ZPDB ≈ APDB × ~1.4, **not** `P(25,7)`.

| k | APDB entries | **ZPDB entries** | avg regions | uncompressed | **1-bit** |
|---|---|---|---|---|---|
| 6 | 127,512,000 | **181,008,000** | 1.42 | 172.62 MiB | **21.58 MiB** |
| 7 | 2,422,728,000 | 4,066,655,040 | 1.68 | 3.79 GiB | **484.78 MiB** |
| 8 | 43,609,104,000 | 87,358,400,640 | 2.00 | 81.36 GiB | **10.17 GiB** |

**6-tile partition we build (4 PDBs): ~181M entries each → ~21.6 MiB each at 1
bit → ~86 MiB for all four.** (Earlier "P(25,7)=2.42e9, ~303 MB/PDB" was wrong —
that was the exact-blank count, not the region-collapsed ZPDB count.)

Uncompressed build array is ~172.62 MiB/PDB (1 byte/entry) — the transient, not
multi-GB. `zstd` on top compresses the 1-bit form a further ~3.8× on disk
(21.58 → 5.66 MiB avg) — optional, decompressed at load.

## 3. The 1-bit encoding (the codec)

**Why 1 bit suffices.** The puzzle (and the ZPDB quotient graph) is **bipartite**:
every move changes *h* by exactly ±1. So along any search path *h*'s parity
(its LSB) flips every step and is fixed by the configuration's parity — it need
not be stored. Breyer-Korf stored `h mod 3` (log₂3 ≈ 1.585 bit). Clausecker-
Reinefeld store `h mod 4` (2 bits) **and discard the LSB** (parity-derived),
leaving **exactly 1 stored bit per entry: bit 1 of *h* (the 2's place).**

**Store** (`bitpdb_from_pdb`): for each full-PDB byte `h`, keep `(h >> 1) & 1`,
packed 8 per byte. (Requires `search_space_size % 8 == 0`.)

**Bit lookup** (`bitpdb_lookup_bit`): returns the stored bit shifted to value 0/2:
`entry = (data[off/8] >> (off%8) & 1) << 1`.

**Differential step** (`bitpdb_diff_lookup_idx`) — given current absolute `old_h`
and a neighbor's `entry`, return the neighbor's absolute *h* = `old_h ± 1`:

```c
next_h = old_h + 1 - ((entry ^ old_h ^ (old_h << 1)) & 2);
```

Derivation: `old_h+1` and `old_h-1` differ in bit 1, so the neighbor's stored
bit 1 selects between them; `old_h<<1` injects bit 0 (parity) to handle the
carry/borrow. `& 2` extracts the decision → subtracts 0 or 2 from `old_h+1`.
**This is the O(1) per-move update the IDA\* search uses.**

**Absolute lookup from cold** (`bitpdb_lookup_puzzle`): seed a dummy-high anchor
`DUMMY_HVAL(256) | partial_parity(p) | lookup_bit`, then repeatedly take a
*happy move* (one that lowers *h* via the differential step) — descending the
quotient graph until no move lowers *h* (goal region reached). Return
`initial_h - cur_h` = number of descent steps = true *h*. O(*h*) per cold
lookup; never used on the hot path.

**Admissibility/consistency:** the reconstruction is *lossless* (exact pattern
distance), so the ZPDB stays admissible and consistent — the 1 bit is pure
representation, no precision loss. Valid for any bipartite state space.

## 4. Use in search — must be incremental

Because a stored bit yields a *delta*, the heuristic is maintained
incrementally: keep each PDB's running `h` (and its index) in the search
context and apply `diff_lookup` on each move. This is exactly the repo's
existing 15-puzzle incremental evaluator (**OPTIMIZATION.md #1**,
`KorfPdbInc`/`IncHeuristic`) — port that structure; do **not** add a scratch
absolute-lookup path on the hot loop.

**Reflected view (ZPDB-specific):** a reflection τ must keep the zero tile fixed,
so the transformed lookup is `π_τ = (0, τ(0)) ∘ τ ∘ π ∘ τ⁻¹` (Eq. 2) — an extra
zero-swap vs the APDB case (Eq. 1). Account for this in the reflected
incremental view. Transposition can be invalid if the region containing the
zero tile changes; guard per the paper.

## 5. Partitioning

- **Canonical 6-6-6-6** (Korf & Felner 2002; Fig 6a): h̄ = 81.82. The standard,
  what PLAN.md targets — **start here.**
- **Optimal single 6-6-6-6** (Fig 6b): h̄ = 82.06 (marginal; comparable nodes).
- **PDB collection** (Fig 8, §5): max over several partitionings → **5.04×**
  fewer node expansions at ~3.5× per-node cost. A Phase-4 upgrade, not Phase 2.

Building *all* 29,285 distinct 6-tile ZPDB classes (for partition search) cost
161.79 GiB — **out of scope**; we build only the 4 PDBs of the chosen partition.

## Implementation status (2026-06-14)

- **Done + verified:** the standard *additive* PDB engine (`src/puzzle24/pdb/`),
  the Manhattan + incremental Korf-max heuristics, the IDA\* solver, and the
  `build_pdb24` / `solve24` binaries — a working optimal solver and the
  admissibility oracle (ZPDB ≥ additive). Plus the **zero-tile-region
  decomposition** (`src/puzzle24/pdb/zpdb.rs`), validated against Table 1
  (Σregions over all C(25,6) shapes = 251,400 ⇒ 181,008,000 ZPDB entries; max 5).
- **Remaining (the zero-aware port):** the full `(m,p,r)` perfect-hash *rank*
  (shape-rank × permutation-rank × region-rank, a bijection into
  `[0, 181,008,000)`), the region-aware retrograde BFS over `(m,p,r)`, and the
  §3 1-bit codec on the resulting bipartite graph. The region decomposition
  above is the geometric foundation these build on.

## 6. Construction

Retrograde BFS from the goal over `(m,p,r)` states; every edge is a tile move
(unit cost in the blank-tracked graph). Build the uncompressed byte PDB
(~172.62 MiB/PDB), then compress to the 1-bit form (§3). Port the rayon
0/1-BFS *structure* from `src/puzzle15/pdb/build.rs::build_parallel`, but over
the ZPDB index, not the no-blank `rank`. The transient visited set is the
OPTIMIZATION.md #8 bitset target (now ~181M bits ≈ 21.6 MiB, not GBs). Output
must stay byte-deterministic (SHA-pinned); gate with parallel == sequential
tests at k≤6.
