# OPTIMIZATION.md

Optimization opportunities for the IDA\* hot path, focused on `enumerate15`
throughput. The earlier round of work (incremental Korf evaluator, table-driven
moves, threaded blank, incremental WD/LC) flattened the per-node CPU cost.
What's left are structural changes that cut **calls to IDA\*** or **work per
IDA\* call**, not per-node cost.

The three items below are independent — each can be implemented and measured
on its own. They stack multiplicatively in principle, though the gains usually
compound less than the product because they target overlapping inefficiencies.

## Headline

For deep boards (depth ≥ 70), the dominant costs in `enumerate15`'s steady
state are:

1. **Redundant IDA\* on reflection partners.** Reflection preserves depth
   (`src/puzzle15/symmetry.rs:11`). Yet `verify` is called independently on
   `B` and `reflect(B)` whenever both surface as BFS neighbors. **~50% of
   calls to IDA\* are redundant in fully symmetry-closed regions.**
2. **Excessive IDA\* iterations.** Textbook IDA\* sets the next f-bound to the
   minimum exceeding value, which for the 15-puzzle increments by 2 (parity).
   Boards at depth 78 with `h_start = 60` do ~10 deepening iterations, each
   re-exploring nearly everything the previous iteration did.
3. **Within-iteration duplicate exploration.** The undo-prune at
   `src/puzzle15/search/idastar.rs:65` catches girth-2 cycles, but longer
   redundant sequences (girth ≥ 12 — blank circling a 2×2 ×3) are still
   re-expanded. Per `examples/dup_probe.rs`'s prior measurement, ~10.5% of
   nodes are redundant re-expansions inside a single IDA\* call.

Each of the three items below targets one of these.

---

## 1. Symmetry-aware solving (call-count reduction)

The cheapest meaningful win. Reflection preserves depth, so if `B` is solved,
`reflect(B)` doesn't need to be re-solved.

### Mechanism

For every board `B` the BFS asks to verify:

1. Compute `r = rank(B)` and `r_refl = rank(reflect(B))`.
2. Cache lookup on **both** ranks before invoking IDA\*. If either hits, use
   that depth for both.
3. After an IDA\* solve on `B`: write `(r, d)` AND `(r_refl, d)` to the cache.

### Where it lives

`src/bin/enumerate15.rs`, the `verify` closure (≈line 122), and the
`run_band` cache-hit/miss triage in `src/puzzle15/enumerate/frontier.rs:239`.

### Expected gain

- **Steady-state ~2× reduction in IDA\* calls** when the BFS is exploring a
  symmetry-closed region (which is most of the work — most of the cache is
  symmetry-closed).
- Self-symmetric boards (`reflect(B) == B`) get no benefit but are rare
  (≤ 1% of any layer at depth 78).
- No effect on per-call IDA\* cost, only call count.
- Cache size grows linearly with new entries; reflection partners are
  half-extra storage but cheap (`rank::rank` + `reflect` are O(16)).

### Risks

- `reflect` and `rank` are deterministic and side-effect-free; correctness is
  guaranteed if and only if depth is reflection-invariant (which is proven by
  `tests::reflect_preserves_solvability` and the docstring at
  `src/puzzle15/symmetry.rs:11`).
- The auto-seed-from-cache loop would also auto-seed both partners, since
  both are now cached.

### Measurement

A/B `enumerate15` on a fresh run (no warm cache) with `--floor 73` for a
fixed budget. Compare IDA\* solve counts (visible in the stderr summary) and
wall-clock. A 2× reduction in `solves` and a similar wall-clock improvement
should be unambiguous.

---

## 2. IDA\* with controlled re-expansion (per-call iteration reduction)

The standard "increment f-bound to the minimum exceeding f" is conservative:
each new iteration is just barely larger than the last, so total work scales
poorly when the heuristic is much smaller than the true cost. **IDA\*\_CR**
(Sarkar et al. 1991) sets the next bound higher than the minimum, accepting
some over-exploration in exchange for fewer iterations.

### Mechanism

Replace the bound update in `idastar.rs:109, 220, 311`:

```rust
// current (textbook IDA*)
bound = next;
// IDA*_CR
bound = bound_cr(next, &iteration_history);
```

where `bound_cr` picks a higher value — commonly such that the next iteration
expands ≈ N× the previous iteration's nodes for some target N (typical N = 2).
There are several variants:

- **Fixed-step**: `bound += STEP` (e.g. `STEP = 4`). Easiest. Loses
  optimality without a refinement step.
- **History-based**: track node counts per iteration; choose next bound to
  hit a target growth ratio.
- **Distance-doubling**: each iteration `bound = min(next, prev_bound + (prev_bound - h_start))`. Asymptotically O(log d) iterations.

### Optimality preservation

IDA\*_CR can return a non-optimal solution because it may search paths whose
cost exceeds the true optimum. **Refinement step**: after finding any
solution at bound `T*`, run one more textbook iteration with
`bound = h_start` and `cutoff = T* - 1` (depth-first, looking only for
strictly better solutions). If none found, the original is optimal.

For our use case, **all calls require optimal depth**, so the refinement is
mandatory.

### Where it lives

`src/puzzle15/search/idastar.rs` — three iteration loops (`search`,
`search_inc`, `search_inc_mut` variants). Add a private `bound_cr` helper.
The refinement loop is a fourth call shape; could be a method that takes
`max_bound: u8`.

### Expected gain

- **2–3× speedup per IDA\* call** on hard 15-puzzle instances (well-published
  result; Korf's own measurements on m=4).
- **Smaller gain on instances with tight h** (early iterations short, refinement
  becomes a larger fraction of total work).
- Stacks with #1 (symmetry-aware): per-board work goes down 2–3× *and* calls go
  down 2× → ~4–6× total in favorable conditions.

### Risks

- Refinement step is easy to get wrong. The optimality-preservation property
  is a code-review hotspot; add a unit test that runs IDA\*\_CR and standard
  IDA\* on a sample of `data/korf100.txt` boards and asserts equal lengths.
- The `iteration_history` adds a small per-call allocation; keep it stack-only.

### Measurement

`examples/korf100_bench.rs` is already set up for this. A/B textbook IDA\* vs
IDA\*\_CR on the full 100 instances. Look at:

- `SearchStats::iterations` per board (textbook ≈ 5–10, IDA\*\_CR ≈ 2–4).
- `SearchStats::nodes` (IDA\*\_CR should be ≤ 2× textbook with refinement;
  much less without).
- Wall-clock (the real headline number).

---

## 3. Transposition table inside IDA\* (per-iteration duplicate elimination)

Within a single IDA\* iteration, the same state can be reached via multiple
paths. The undo-prune catches girth-2 cycles but not the girth-12+ ones (blank
circling a 2×2 block three times). Empirically (from `dup_probe.rs`), ~10.5%
of nodes are redundant re-expansions.

A bounded transposition table catches these:

```rust
struct TTEntry {
    state_hash: u64,
    g: u8,
    bound: u8,
}
```

### Mechanism

At each node:

1. Compute `state_hash` (cheap if `State` is packed — see below).
2. Probe TT. On hit, if `entry.g <= current_g` and `entry.bound >= current_bound`,
   the previous visit explored everything reachable at this f-bound — skip.
3. On miss or stale hit, expand the node. On return, store `(state_hash,
   current_g, current_bound)`.

### Sizing

A power-of-two-sized direct-addressed array (no chaining), replace-always policy.
Sizes:

- 2^20 entries × 10 bytes = ~10 MB per IDA\* call.
- 2^24 entries = 160 MB per call.

For `enumerate15`, IDA\* calls are short and many; size 2^16–2^20 to keep
construction cost low (one `vec![0u8; ...]` per call, or thread-local reused).

### Dependency on #6 (packed u64 state)

A `[u8; 16]` `State` hash is a 16-byte hash — slow per node. The packed
`u64` representation deferred from the old OPTIMIZATION.md (`State` as nibbles
in a `u64`) makes TT cheap: hash = the state itself (or its splitmix).

Implementing TT *before* packing `State` works but pays a real per-node cost
(probably eats 30–50% of the dup-elim savings). **Recommended: do #6 first,
then TT.**

### Where it lives

`src/puzzle15/search/idastar.rs`, the recursive `search_inc` body. The TT
itself sits in the `SearchContext` (new field, allocated per call or reused
via a thread-local pool).

### Expected gain

- **Per `dup_probe.rs`: ceiling is ~10.5% node reduction.** Real-world: probably
  6–8% with imperfect dedup.
- This is much smaller than #1 or #2 — the optimization document for m=4
  explicitly deferred this to m=5 where depth ~150 (vs ~53) compounds the
  branching factor cut. **But for `enumerate15` at depth 78** the boards are
  deeper than the Korf-100 average (which targets depths in the 40s–50s), so
  the dedup benefit should be 2–3× higher than the dup_probe estimate.
- Stacks with #1 and #2 multiplicatively.

### Risks

- TT correctness: the stored `(g, bound)` must dominate the current visit for a
  skip to be sound. Easy to get the inequality backward.
- TT collisions: with replace-always, false hits are impossible (we still check
  `state_hash`), but **false misses** (collisions causing eviction) just reduce
  the dedup rate, not correctness.
- Memory pressure if pool-sized too large for many parallel IDA\* calls.

### Measurement

Same `korf100_bench.rs` rig. Also re-run `examples/dup_probe.rs` to confirm
the dedup rate matches expectation. Watch for regressions on shallow boards
(where TT overhead might exceed savings).

---

## Implementation order

In rough order of bang-per-buck:

1. **Symmetry-aware solving (#1)**. ~50 lines. Easy to verify. ~2× call
   reduction in steady state. No representation changes needed. Strongly
   recommended as a first step.
2. **IDA\*\_CR (#2)**. ~200 lines including refinement. ~2–3× per-call.
   Test against textbook IDA\* on Korf-100. Stacks with #1.
3. **Packed `State` (prerequisite for #3)**. Inheriting the deferred #6 from
   the old doc — a couple of hundred lines, mostly mechanical, biggest pain
   is `State::swap` becoming nibble extract+insert.
4. **TT inside IDA\* (#3)**. ~100 lines on top of packed `State`. ~6–8% per
   call, more on deep boards.

The full stack on a depth-78 verify should be roughly **4–8× faster** vs
today (point estimates: 2× from symmetry, 2× from IDA\*\_CR, 1.1× from TT,
multiplied for non-overlapping effects, discounted for overlap).

## Process

Each item should be a separate PR with:

- Before/after `korf100_bench` numbers (use `--scratch` for the pre-#1
  baseline path).
- For #1: also report `enumerate15` solve count and wall-clock on
  `--floor 73 --down-to 77 --budget 1000000` against a fresh cache.
- Unit tests: optimality vs textbook IDA\* on a Korf-100 sample (#2 especially).
- For #3: re-run `examples/dup_probe.rs` to validate the dedup rate.

Don't merge speculative wins. The previous round of optimization saved one
~6% regression (move ordering) by relying on measured Korf-100 numbers as the
gate.
