# Plan: Find the hardest 24-puzzle instances — Phase 1 (solver parity + calibration)

## Context

**Goal.** Find the deepest (largest optimal STM length) 24-puzzle boards we can, to push
understanding of the 24-puzzle diameter (known only to lie in **[152, 205]**; antipodes
unknown). No A087725-style histogram or known antipodes exist for the 24-puzzle, so we hunt
for *deep individual boards*, not a complete layer.

**The open question that gates everything: is our IDA\* solver fast enough on 24-puzzle
boards?** It can't be answered from the docs — it must be measured. Two findings reshape how:

**1. The 24-puzzle solver is out of date.** `puzzle15`'s zero-aware `ZpdbInc` was *originally
ported from `puzzle24`* (`src/puzzle15/pdb/zheuristic.rs:1`), but `puzzle15` then gained the
full incremental zero-aware "plus" stack and `puzzle24` was left behind. Parity gap (from the
module exports):

| current `puzzle15` | `puzzle24` today | action |
|---|---|---|
| `ZpdbPlusInc` = incremental `max(zpdb, LC, WD)` (zheuristic.rs:217) | only `ZpdbInc` (PDB term alone) | A5 |
| `LinearConflictInc`/`LcCtx`; `WalkingDistanceInc`/`WdCtx` | neither module exists | A3, A4 |
| `IncHeuristic` threads `&mut SearchStats`; `verifier-stats` counters | no stats in trait | A1 |
| `IncHeuristicMut` + `idastar_inc_mut_with_stats` (make/unmake) | absent | A1 (optional) |
| non-inc `ZpdbHeuristic`/`AdditiveZpdbHeuristic` (for ranking) | absent | A5 |

So Phase 1A is mostly **porting proven `puzzle15` code back to `puzzle24`** — each port is "copy
the module, change `W` 4→5 and the sizes, keep the `#[cfg(test)]` suite." This supersedes the
earlier draft's bespoke `IncMax`/`AsInc` combinators — `ZpdbPlusInc` already *is* that, tested.

**2. zpdb must stay 1-bit / incremental.** The zero-aware PDB's 1-bit codec stores a *delta*
bit, not an absolute distance; the only absolute lookup (`cold_lookup`) is O(h) (~100×/node).
So every term must be advanced incrementally — exactly what `ZpdbPlusInc` does (zpdb via the
O(1) differential step; LC/WD via localized per-move updates).

**Why deep boards need maximum heuristic strength.** IDA\* cost grows ~exponentially in
(true depth − root *h*). The canonical hard board — the **180° rotation** `R` (Rokicki LB
**152**; a 156-move solution known; true optimum **unknown**, an open problem) — has plain
Manhattan only **112**, a ≥40 gap. Random instances are shallow (Korf & Taylor 1996: avg
102.6, max 114); the "171/177" speedsolving figures are **suboptimal** outputs, not real
depths. Deep boards must be **structured**, not random.

**A bigger PDB is the single highest-leverage upgrade (confirmed: also build k=7).** Each +1 to
root *h* roughly halves the search. Corrected ZPDB `.zbin` sizes (1 bit/entry; from Table 1
entry counts, validated against the k=6 files on disk = 181,008,000 ÷ 8 = 22 MB each ✓):

| k | ZPDB entries | 1-bit / PDB | partition | total disk | verdict (36 GiB free, 32 GiB RAM) |
|---|---|---|---|---|---|
| 6 (current) | 181,008,000 | 21.6 MiB | 6-6-6-6 | 86 MiB | on disk |
| **7** | 4,066,655,040 | **485 MiB** | 7-7-7-3 | **~1.45 GiB** | ✅ ~1 hr build, ~3.8 GiB RAM/PDB transient |
| 8 | 87,358,400,640 | **10.17 GiB** | 8-8-8 | ~30.5 GiB | ❌ disk too tight, ~30 GiB runtime RAM, 81 GiB build transient |

> `docs/zpdb-codec-spec.md:52` mis-states k=8 as 1-bit "~1.27 GiB" (it is **10.17 GiB**) and
> "uncompressed 10.17 GiB" (it is **81.4 GiB**) — both ~8× too small. **Fix that line.** k=7's
> 4.07e9 just fits a `u32` index (max 4.29e9).

**Decision (confirmed with user): calibrate first, then pause.** Bring `puzzle24` to parity,
add the lower-bound search mode and the k=7 PDB, build a calibration harness, run it, report
the feasibility frontier. Only then design the hunt — its shape depends on these numbers.

**Encoding (matches speedsolving / `solve24`):** 25 ints, row-major, blank `0`/`_`, tiles
`1..24`, goal `1 2 … 24 0`. `parse_position` (solve24.rs:87) already accepts it.

---

## Phase 1A — Bring puzzle24 to parity with puzzle15, then extend

**Sequencing (baseline-first):** A1 (trait parity, no builds) + A2 (bounded mode) + the B
harness → **baseline calibration on the existing k=6 `ZpdbInc`** for the first frontier numbers,
*before* committing to the expensive WD/k=7 work. Then A3 (LC) + A4 (WD) + A5 (`ZpdbPlusInc`) +
A7 (k=7 build) → full calibration matrix → **pause**.

### A1. Search-infra parity (foundational, no builds)
- Update the `puzzle24` `IncHeuristic` trait (idastar.rs:110) so `root`/`advance` take
  `&mut SearchStats`, matching puzzle15 (idastar.rs:174-192); update the existing `IncManhattan`
  (heuristic.rs:67) and `ZpdbInc` (heuristic.rs:222) impls and the `idastar_inc_*` drivers.
- Extend `puzzle24` `SearchStats` with the `verifier-stats` per-component counters
  (`lc_advances`, `wd_advances`, `zpdb_advances`, `zpdb_rank_calls`, `proj_applies`) + `add()`,
  mirroring puzzle15 (idastar.rs:33-70). Off by default; ~5-10% when enabled.
- **Optional (perf follow-on):** port `IncHeuristicMut` + `idastar_inc_mut_with_stats` +
  `search_inc_mut` (puzzle15 idastar.rs:277-364). The make/unmake driver removes the per-node
  context copy (profiled at ~21% on puzzle15); payoff is *larger* on puzzle24 (context = 4 PDBs
  × normal+reflected `ProjectedState` + `LcCtx` + `WdCtx`). Gate on whether baseline shows the
  copy dominating.

### A2. Bounded / lower-bound ("failed search") mode — *new to both puzzles*
All our heuristics are **consistent**, so the live IDA\* threshold is always a proven lower
bound; a capped search that exhausts threshold `b` proves `depth ≥ next` (the `Step::Bound`
value `search_inc` already returns). This is how Rokicki proved ≥152. In idastar.rs, reusing
`search_inc`/`Step`/`SearchStats`:

```rust
pub enum BoundedOutcome { Solved(Vec<Move>), ProvedAtLeast(u8), Unsolvable }
pub fn idastar_inc_bounded_with_stats<E: IncHeuristic>(
    start: &State, e: &E, max_bound: u8) -> (BoundedOutcome, SearchStats)
```

Loop mirrors `idastar_inc_with_stats`; before each iteration `if bound > max_bound { return
ProvedAtLeast(bound) }`. Re-export from search/mod.rs. Optional `…_telemetry` variant firing a
closure per iteration with `(threshold, cumulative stats, iter_elapsed)` — the calibration
signal, zero hot-path cost.

### A3. Port `linear_conflict.rs` to puzzle24
Copy `src/puzzle15/search/linear_conflict.rs` → `src/puzzle24/search/linear_conflict.rs`
(`LinearConflictHeuristic`, `LinearConflictInc`, `LcCtx`); add to search/mod.rs. W=5 changes;
**drop the 16-bit LUT** (5×5 tiles need 5 bits → 2^25/line) — recompute each dirty line
directly (5-cell scan + LIS≤5; a vertical slide dirties 2 rows + 1 col, horizontal 2 cols + 1
row). Port the module's `#[cfg(test)]` suite verbatim.

### A4. Port `walking_distance.rs` to puzzle24
Copy puzzle15 → `src/puzzle24/search/walking_distance.rs` (`WalkingDistanceHeuristic`,
`WalkingDistanceInc`, `WdCtx`, `warm_up`); add to search/mod.rs. **First sub-step (gate the
rest): verify table size** — the 5×5 row/col contingency-table BFS (margins ≈ (5,5,5,5,4));
expected low-millions of states into `HashMap<u128,u8>`. **`pack` → `u128`** (25 cells × 3 bits
+ blank ≈ 78 bits overflows u64). If startup build time is material, promote to a SHA-pinned
`data/wd24*.bin`. Port the test suite.

### A5. Port `ZpdbPlusInc` to puzzle24 — the up-to-date heuristic
New `src/puzzle24/pdb/zheuristic.rs` mirroring puzzle15: `ZpdbPlusInc<N>` (`{zpdb: ZpdbInc<N>,
lc: LinearConflictInc, wd: WalkingDistanceInc}`; `Ctx = ZpdbPlusCtx{zpdb, lc, wd}`; `root`/
`advance` return `h_zpdb.max(h_lc).max(h_wd)`, all incremental) — the 1-bit-incremental
`max(zpdb, MD+LC, WD)` we need. Plus the non-incremental `ZpdbHeuristic` (single `cold_lookup`)
and `AdditiveZpdbHeuristic` (sum) for candidate ranking in Phase 2. Re-export from pdb/mod.rs.
(Existing `ZpdbInc`/`ZpdbCtx` stay in pdb/heuristic.rs, or move into zheuristic.rs to match
puzzle15 — impl-time judgment.) Port puzzle15's `zpdb_plus_inc_dominates_korf_plus_pointwise`
and admissibility tests.

### A6. solve24 CLI wiring
Add `--heuristic zpdb-plus` (`ZpdbPlusInc<4>`, via `idastar_inc_with_stats` / the mut driver if
ported); `--max-bound T` / `--prove-at-least T` (bounded driver → `Lower bound: depth >= K`);
k=6/k=7 PDB selection via `--pdb-dir` + file names (`ZpdbInc<4>` unchanged — 7-7-7-3 is still
four PDBs). Retain `manhattan` and plain `zpdb` for attribution.

We deliberately **do not build the full plain additive 6-6-6-6 `.bin`** (`korf`): by the
15-puzzle precedent, `enumerate15` and the whole deep-board/residue toolkit run on
`ZpdbPlusInc` (zero-aware), while `solve15`/`korf100_bench` use plain `korf` only as a
solve/benchmark baseline. The additive PDB stays only as the small-pattern `ZPDB ≥ additive`
test oracle (already exercised in tests).

### A7. Build the k=7 (7-7-7-3) ZPDB partition
~485 MiB/PDB × 3 + a tiny k=3 → ~1.45 GiB disk; ~3.8 GiB RAM transient/PDB; ~16 min/PDB
(~1 hr total) via the existing region-aware BFS.
- **Generalize the ZPDB index/build from its k=6 specialization** (`src/puzzle24/pdb/{zpdb,
  zbuild,zdb}.rs`): generate the k=7 region-count table (Σregions = 806,876) and **relax
  `assert_eq!(max_regions, 5, "…k=6…")` (zpdb.rs:618)** to be k-parameterized. Confirm the
  `(m,p,r)` rank holds 4.07e9 (fits `u32`; widen if any intermediate overflows). The
  retrograde-BFS structure (`zbuild.rs`) is k-generic.
- **Mixed-k partition:** 7-7-7-3 needs two region tables (k=7, k=3); ensure `ZPatternDb`/
  `ZpdbInc` carry per-PDB `k`. Define the partition beside the 6-6-6-6 (build_pdb24.rs:30); a
  reasonable geometric 7-7-7-3 grouping (exact tiles tune h̄).
- **`build_pdb24`**: add `--partition k7`; emit `data/pdb24_k7_{a,b,c,d}.zbin`. `.gitignore`
  covers `*.zbin`; commit only the new `.sha256` pins. Keep output byte-deterministic.
- **Determinism gate:** rebuild the **k=6** partition through the *same generalized path* and
  confirm it reproduces the SHA-pinned `pdb24_*.zbin` byte-for-byte — proves the
  k-generalization didn't regress the k=6 case before we trust the k=7 output.

---

## Phase 1B — Calibration harness + run (the gate)

**New `examples/ladder24.rs`** (`[[example]] required-features=["mmap"]`), modeled on
`examples/korf100_bench.rs` (reuse its `Row`/`run`/`print_summary` shapes; copy `load_zpdbs`
from solve24.rs:129).

**Board sources:** (1) **random walks from GOAL** at `--walk-lengths 20,40,…,120` × `--reps` ×
`--seed` (reuse the no-immediate-undo xorshift walk at idastar.rs:243) — the *optimal-solve*
regime (saturates ~100-114); (2) **structured deep boards** `R` = 180° rotation
(`R.0[0]=0; R.0[i]=25-i`, solvable: 276 inversions), `reflect(R)`, D4 images (filter
`is_solvable`) — the *lower-bound* regime; (3) **`--from FILE`** 25-int boards (Phase 2 feeds in).

**Two measurement modes (run both):**
- **Optimal-solve cost vs depth** — `idastar_inc_with_stats` on the random walks. "To what
  depth can we optimally solve in 1 min / 10 min / 1 hr?"
- **Bounded-LB cost vs threshold** — `idastar_inc_bounded_with_stats` at rising `--max-bound`
  on the structured boards. "How high a threshold can we exhaust within budget?" = the
  **realistic frontier** for the deepest boards.

**Record per (board × heuristic):** `root_h`, nodes, iterations, wall-clock, nodes/sec, outcome
(`Solved(len)`/`ProvedAtLeast(K)`). Output a table + summary (deepest solved; highest threshold
on `R`).

**Two-pass run (both wrapped in `caffeinate -dimsu`):**
1. **Baseline** (after A1+A2+B): `manhattan`, k=6 `zpdb`. First frontier numbers + `root_h(R)`.
2. **Full matrix** (after A3-A7): `{k6, k7} × {zpdb, zpdb-plus}` + standalone `manhattan`, LC,
   WD for attribution — does k=7 alone close most of the gap; do LC/WD still add on top of k=7?

Then **PAUSE** and report the feasibility frontier — the decision gate for the hunt.

---

## Verification

Mirror the codebase's conventions: **inline `#[cfg(test)]` per module** + **`tests/`
integration files** + **parallel==sequential / BFS-oracle gates** for PDB builds. The ports
inherit puzzle15's existing suites — adjust `W`/sizes and keep every test.

**Unit tests (inline):**
- *Bounded mode* (idastar.rs): on BFS depth `d ≤ 10`, `…bounded(s,&IncManhattan,d)` →
  `Solved(len==d)`; `max_bound=d-1` → `ProvedAtLeast(d)`; `h0>max_bound` → `ProvedAtLeast(h0)`;
  never exceeds the unbounded optimum.
- *Linear conflict* (port from puzzle15): `LC(GOAL)=0`; a row swap and a column swap each add
  2; `LC ≥ MD`, `LC ≤ truth` on shallow BFS; `LinearConflictInc.root == scratch`, `advance ==
  fresh root` over a ~1500-step walk; `lc_removals` cases.
- *Walking distance* (port): `WD(GOAL)=0`; `pack`/`unpack` round-trip; deterministic table;
  `WD ≥ MD`, `WD ≤ truth`; inc `root == scratch`, `advance == fresh root` over a long walk.
- *ZpdbPlusInc / zheuristic* (port puzzle15's suite): admissible vs truth; `zpdb ≥ additive`
  pointwise; `advance == fresh root`; **cost-predicts-index-change** (the projected-edge
  invariant); IDA\* returns optimal lengths; **`zpdb-plus ≥ korf-plus` pointwise** and ≤ truth.
- *Search-infra parity* (idastar.rs): stats counters increment as expected; if `IncHeuristicMut`
  is ported, the mut driver returns identical optimal lengths to the copy driver.
- *k=7 index* (zpdb.rs): k=7 region table totals Σregions = 806,876; `(m,p,r)` rank is a
  bijection into `[0, 4,066,655,040)` on a sample (round-trip).
- *ladder24 helpers*: board parser round-trips the 25-int encoding; the walk generator emits no
  immediate-undo; `R` is solvable.

**Integration tests (`tests/`):** `tests/puzzle24_{bounded,linear_conflict,walking_distance,
zpdb_plus}.rs` end-to-end against `bfs_distances` (reuse the helper, search/mod.rs:13); every
returned solution replays to `GOAL` (korf100_bench.rs:263).

**PDB-build gates (A7):** parallel == sequential at small k; ZPDB ≥ additive oracle (small
patterns); **rebuilding k=6 via the generalized path reproduces the pinned `pdb24_*.zbin`
byte-for-byte**; built k=7 PDBs admissible/consistent (solver matches k=6 optimal lengths on
shallow scrambles); `k7-zpdb ≥ k6-zpdb` on sampled states; byte-deterministic, SHA-pinned.

**Headline milestone:** bounded search on `R` reproduces **depth ≥ 152** (Rokicki) — validates
the full stack against a published result and is a real calibration data point.

**Side task:** fix the k=8 size row in `docs/zpdb-codec-spec.md:52`.

`cargo test --features puzzle24,mmap` + `cargo build --release --features mmap`.

---

## Phase 1C — ZPDB partition decision (after the 1B pause)

**Verdict from Phase 1B calibration + reference study** (Clausecker's `24puzzle`; Clausecker &
Reinefeld SoCS 2019; Clausecker & Schintke SoCS 2021 "eta"; ICCSA 2025 distributed build):

- **Single-partition tuning is a weak lever.** The optimal single 6-6-6-6 raises h̄ only
  81.82 → 82.06 (`docs/zpdb-codec-spec.md:116`); our hand-built k=7 (7-7-7-3) *underperformed* k=6
  on `R` (root h 120 vs 126; `data/ladder24_calibration.txt`). **k=7 is retired from active
  tuning** — the built artifacts and SHA pins are kept, but we don't invest in tuning the grouping.
- **k=8 / k=9 are deferred.** Clausecker builds them only on distributed clusters (2304 nodes,
  MPI+OpenMP). On this 32 GiB machine our region-based k=8 ZPDB is 87.4e9 entries (10.17 GiB at
  1-bit, ~81 GiB uncompressed build transient); the rank-frontier build asserts `total ≤ u32::MAX`
  (`src/puzzle24/pdb/zbuild.rs`). Needs an external-memory / ≤2-bit frontier-free builder first.
- **The real lever is a *collection* — the MAX over several disjoint 6-6-6-6 ZPDBs** (Clausecker's
  `small-compound`; `docs/zpdb-codec-spec.md` §5 documents ~5× fewer node expansions at ~3.5×
  per-node cost). The infra already exists (`MaxInc` over two `ZpdbInc`; combining partitions needs
  no new combinator).
- **But a structural argument says the collection helps the *general* regime, not R-type
  antipodes.** An additive PDB counts only its k pattern tiles and treats the other tiles as free
  to displace — a relaxation that collapses on antipodal boards where *every* tile is far from home,
  whereas Walking Distance's row/col abstraction counts all tiles' migration. That is why WD = 140 >
  zpdb-k6 = 126 on `R`. A collection takes the **max** (not the sum) over partitions, so
  collection-on-R ≈ best-single-partition-on-R — still under WD.

**The bootstrap.** We cannot certify a heuristic's *average* quality on the true d≈150 layer without
a population of such boards — which is the project's goal. But per-board measurements (root `h`,
bounded-LB) are **self-certifying**: they need only the board, not its true depth, and a proven
lower bound is a valid result regardless of the heuristic. So the antipode question is settled
cheaply and directly on `R`, without first solving the bootstrap.

### Chosen path — Option 1: cheap R-ceiling, then decide

Compute the best zpdb Korf-max root `h` achievable by a curated set of k=6 partitions on a
*constructed* deep-board set {`R`, `reflect(R)`, D4 orbit, perfect tile-reverse, a few
frame-conformant}, and compare the **max** to WD (140) and the bounded-LB-on-`R` frontier (144).
Then:

- **best k6 R-score < WD** (expected): no k6 collection beats WD on antipodes → a collection's value
  is the **general regime** (ranking/solving the hunt's in-reach candidates). Build a small
  general-regime collection (`MaxArrayInc` over ≈3 partitions), validate on the *bootstrap-free*
  solvable population (`ladder24` optimal mode) vs `zpdb-k6`/`select-k6`, and ship it as a hunt tool;
  the selector keeps WD on antipodes.
- **best k6 R-score ≈/> WD** (surprising): a collection may help `R` → build it and measure
  bounded-LB on `R` vs WD (does the higher root overcome the higher per-node cost within budget?).

**Tool:** `examples/rceiling24.rs` — builds each candidate partition's 4 ZPDBs in RAM
(`build_zpdb_parallel` → `ZPatternDb::from_dist`, no `.zbin` saved), evaluates `ZpdbInc::root` on the
constructed deep boards, reports per-partition and the max vs WD/LC. Candidates: existing k6 Korf set
(`PART_*`), Clausecker's reinefeld-et-al / best-single (transformed `v ↦ 25 − v`, gated by a
disjoint+cover check), 2–3 corner/strata-aligned (RESIDUE-informed) partitions, a few seeded-random.
Optional exactness upgrade if the max lands within a few of 140: a projected single-board additive
solver (0-1 BFS over the region-collapsed projected graph).

**Result (measured, `examples/rceiling24.rs` → `data/rceiling24.txt`, 2026-06-30).** On `R`, the
best k6 zpdb Korf-max over 8 diverse partitions (korf, value-strata, spatial, 5 random) is **126**
(korf); every partition landed 118–126, all far below WD's **140**. So **no k6 collection beats WD
on antipodes** — the max over partitions ≈ the best single, because the additive/anon-free
relaxation collapses on `R` regardless of grouping (confirmed on `reflect(R)` = 126 too). On deep
*general* boards the order flips — korf zpdb beats WD (walk300: 72 vs 66; walk400: 74 vs 70;
walk200: 58 vs 50) — confirming PDBs are the general-regime tool and WD owns antipodes.
**Secondary finding:** korf *pointwise-dominated* every other candidate on all five boards, so a
naive collection of them ≈ korf alone (no diversity gain); a real collection win needs
*complementary* (η-optimized) partitions, which our quick geometric/random set does not provide.
**Decision:** keep single k6 zpdb via `select-k6` (WD on `R`, zpdb on general) as the working
heuristic; a general-regime collection is only worth building behind a proper
complementary-partition (η) sweep — **deferred to Phase 2** unless hunt throughput demands it.

### Option 2 — frame-rule deep bridge (deferred to Phase 2)

The fuller fix to the bootstrap: build a **frame-conformant constructed-board generator** (PLAN.md:
corners `{_,1,5,21}` at `{0,24,20,4}`, the 8 corner-neighbor tiles Chebyshev ≤ 1) producing a
diverse deep-board population *without search*, used as **both** the heuristic test population and
the hunt's seeds. This merges the two goals — every high proven LB on a constructed board is
simultaneously a heuristic validation and a deep-board discovery, so the population is search-free
and the bootstrap dissolves. Deferred because it is Phase-2 scope and rests on a conjecture.

> **Revisit: the frame rule is a conjecture.** "Deep ⇒ frame-conformant" (and the weaker converse)
> is unproven for the 24-puzzle — it generalizes 15-puzzle evidence (all 17 depth-80 antipodes
> frame-conformant; corner-neighbor rule 98.5% vs 25% baseline; `PLAN.md`). **Before relying on it
> for candidate generation or partition selection, test it on the 24-puzzle:** construct a sample of
> frame-conformant boards, bounded-LB them, and check whether high LB correlates with frame
> conformance (and conversely, whether known-deep boards are frame-conformant). Treat the frame rule
> as a heuristic prior until measured.

---

## Phase 2 — the hunt (the plan)

**Goal.** Tighten the diameter *lower* bound: produce a ranked catalog of the hardest boards we can
find, each with the highest *proven* optimal-depth lower bound. The realistic deliverable is bounds
+ a catalog, not exact antipodes (the optimal-solve frontier is ~d ≤ 90–100; the 205 *upper* bound
needs a different technique). The core is a **loop** — construct → score → bound → re-seed — not a
linear pipeline. Heuristic policy is settled (Phase 1C): WD on antipodes, k6 zpdb on general,
routed per board by `select-k6`.

### 2A — Reproduce R ≥ 152 (headline milestone + stack validation)
Push the proven LB on `R` from 144 (120 s) toward Rokicki's 152, validating the full stack against a
published result and calibrating the *budget → threshold* scaling that sizes 2D's budgets. Mostly a
run: `ladder24 --mode bounded --max-bound 152` (or `solve24 --prove-at-least 152`) on `R`, WD-led;
parity means exhausting threshold 150 proves depth ≥ 152. Extrapolating the 1B calibration (~3–5×
nodes per +2 threshold) this is ~hours–overnight, caffeinated/backgrounded. Independent of the rest —
run first / in parallel.

### 2B — Test the frame-rule conjecture (gates the generator)
Before betting the generator on it, measure whether frame-conformance predicts depth (the deferred
1C revisit). New: a frame-conformant constructor (corners `{_,1,5,21}@{0,24,20,4}` + Chebyshev-≤1
neighbors, `is_solvable` filter). Generate a frame-conformant sample **and** a random-solvable
control; bounded-LB both at a modest budget; compare LB distributions. **Decision:** frame-conformant
bounds ≫ random → seed 2C with it; else fall back to R-orbit + perturbation seeding. (We can only
weakly test "deep ⇒ frame" — one known-deep board, `R`; we *can* test "frame ⇒ tends-deep," which is
what the generator needs.)

### 2C — Candidate generator (`examples/candidates24.rs`)
Produce a diverse pool of hard candidates. Seeds = `R`, `reflect(R)`, D4 images, frame-conformant
constructions (if 2B passed), + perturbations of `R`. **Hill-climb** each seed by tile transpositions
that raise an admissible score (WD or `max(LC,WD,zpdb)`), keeping `is_solvable`; **dedup** via
`symmetry::canonical`; emit 25-int rows + scores → `ladder24 --from`. The score *is* a proven LB, so
this already yields cheap lower bounds. **Expect:** WD saturates near `R` (which nearly maximizes
row+col reversal), so root-h candidates won't exceed ~140 by much — generation's job is *diversity*
(boards possibly deeper than `R`); LBs above ~140 come from search budget in 2D.

### 2D — Bounded-LB hunt (the loop)
Feed top candidates into `ladder24 --from --mode bounded` with **escalating budgets** (cheap pass
over many; big budget on the few best); reuse `idastar_inc_ladder`; collect proven LBs. **Flywheel:**
the deepest confirmed boards become new seeds for 2C → generate around them → bound → repeat. This is
how the bootstrap resolves — construction seeds it, search certifies, deep finds re-seed.

### 2E — Catalog + bounds report
The deliverable: ranked catalog of hardest boards + best proven LBs; updated diameter lower-bound
claim; honest frontier statement (we push the *lower* bound; exact antipodes and the 205 upper bound
remain out of reach here).

### Conditional
- **2F — general-regime ZPDB collection (η sweep):** only if 2D throughput on *general* candidates is
  heuristic-bound (deferred from 1C; needs complementary η-optimized partitions, since 1C showed
  naive ones add nothing over single korf).
- **Optimal-solve conversion:** for candidates within the ~d ≤ 90–100 frontier, convert LB → exact
  depth via `select-k6`.

### Order & dependencies
```text
2A (R≥152) ───────────────┐  (parallel, long-running: validates stack + sizes budgets)
2B (frame test) ─→ decide ─→ 2C (generate) ─→ 2D (bound) ─→ 2E (report)
                               ▲__________________│   (flywheel: deepest finds re-seed)
```
`2A` runs early/parallel; `2B` before `2C` (don't build the generator on an untested conjecture);
`2C → 2D → 2E` with the `2D → 2C` flywheel. **Most concrete next action: 2A** (near-pure run,
headline result, budget calibration); **cheapest new code: 2B**.

---

## Risks & expectations

- **The frontier may sit well below 152.** Optimally solving the deepest boards is an open
  problem; the realistic deliverable is **a ranked catalog of hard candidates + the best
  proven lower bounds**, not optimal antipodes. The harness must report this honestly.
- **Parity port is real work, but low-risk:** it's copy-from-puzzle15 + change `W`/sizes + run
  the same tests. The one genuine algorithm change is dropping the LC LUT (A3) and the WD
  `u128`/table-size question (A4).
- **WD at 5×5**: table-size/build-time and `u128` packing — gate the rest of A4 on the size
  check.
- **k=7 build (A7)** generalizes the k=6-specialized ZPDB index/region tables; mitigate with
  the validated region-decomposition and the parallel==sequential / ZPDB≥additive gates.
  **k=8 is ruled out** on this machine (disk ~30.5/36 GiB, ~30 GiB runtime RAM, 81 GiB build
  transient) — revisit only with more disk/RAM and a ≤2-bit builder.
- **u8 cost ceiling** is safe (diameter ≤205<255); keep `g+h` in `u8` with `saturating_add`.
