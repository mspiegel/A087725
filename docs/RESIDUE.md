# Closing the 15-puzzle residue at the antipodal end

## 1. The goal

Enumerate *every* 15-puzzle state at the deepest layers — `d ∈ {76, 77, 78}` —
that the band-BFS enumerator cannot reach from the antipode shell within
practical wall-time. We call these boards the **residue**: their true optimal
distance to the goal is `d`, but every path from them to the antipode shell
that stays within the band `[floor, 80]` requires a detour below `floor`.

Layer status (expected counts = A087725):

| Depth | Expected | State |
|------:|---------:|-------|
| 80 | 17 | complete (antipodes) |
| 79 | 70 | complete |
| 78 | 3,406 | complete |
| 77 | 26,638 | **complete** |
| 76 | 272,198 | **COMPLETE** (closed 2026-06-29; see §6) |

**Why the top is hard.** The enumerator does a BFS outward over the band
`[floor, 80]`, seeded from every cached board in the band, solving each
cache-miss neighbor with admissible IDA\*. It only ever reaches the connected
component of its seed mass. But the cached set is **closed under one
puzzle-move at the top** (every neighbor of a cached `d≥75` board is already
cached), so the missing boards are **strict local maxima** — every neighbor is
at `d−1` and also missing — sitting in pockets of the puzzle-move graph that
are disconnected from the antipode-reachable mass at any practical `floor`.
Lowering the floor reconnects them eventually but is very slow: deep-board
IDA\* runs only ~10 solves/s/thread, and round 1 of a low-floor run must verify
millions of candidates (≈16 h at `floor=73`, worse below).

The residue therefore has to be attacked by **structure**, not by brute band
enumeration.

## 2. What worked

**Reflection-closure first.** The cache is kept symmetry-closed (every find is
mirrored), and the symmetry is the main-diagonal transpose-with-relabel
(`2↔5, 3↔9, 4↔13, 7↔10, 8↔14, 12↔15`; `1,6,11` fixed). So the missing boards
of a layer form a closed set under reflection — almost always a **reflection
pair**. Consequence: you only ever need to find **one** representative; its
partner is constructed for free. Diffing the layer against its reflection also
recovers a board outright if closure was ever broken by a lost insert.

**Maxima profiling.** Split a layer into *maxima* (no neighbor one level
deeper) and *non-maxima*. The non-maxima are fully derivable by expanding the
already-complete layer above, so **the residue is always maxima**. Profiling
the known maxima by blank cell and heuristic `h` bounds the search: at `d=77`
there are 19,440 maxima, only at blank cells `{1,3,4,6,9,11,12,14}` (symmetric
under reflection), with `h` ranging 61–71.

**The frame structure + frame-completion — the closer for d=77.** The decisive
lever. In a deep maximum the **high tiles `{9..15}` plus the blank occupy the
top 8 cells and the low tiles `{1..8}` fill the bottom 8** (this is the
"rows-reversed" antipodal arrangement; the transpose of such a board puts the
high tiles in the *left* columns, which is its reflection partner). The maxima
cluster tightly into **frames** = top-8 arrangements: the 19,440 `d=77` maxima
span only ~4,980 frames (≈3.9 maxima/frame, one frame holding up to 76), and
members of a frame differ *only by permutations of the bottom 8 cells*. Those
are Hamming distances up to 8 that are **bottom-only** — exactly the moves a
small-radius neighborhood search cannot make.

So: for each known maximum, **fix its top-8 frame and enumerate all `8!`
permutations of the bottom tiles**, verifying the solvable, cache-miss, high-`h`
completions with IDA\*. Sweeping this over *all* maxima frames closed `d=77` in
minutes after seed-based methods had failed for days. The pair it found differ
from an already-cached maximum purely by a full bottom reshuffle.

**Hamming-neighborhood enumeration around a good seed (closed d=78).** When the
residue *is* geometrically isolated (the `d=78` case), enumerate every state
within Hamming-K of a well-chosen reference and IDA\*-verify the high-`h`
cache-misses. K=4 around a structural outlier found the first `d=78` pair; K=8
around a *single* known-good seed reaches the next pocket (K=8 around the whole
layer does not — the balls overlap and collectively reach no further than one
of them).

**Placement-rarity seeding (helped d=77).** Score each board by its rarest
`(tile, cell)` placement across the layer; boards containing a globally-rare
placement are good references. `d=77` residue was *placement*-isolated even
though it was not Hamming-isolated.

**Cascade from every find.** One puzzle-move expansion from a freshly found
residue board yields its neighbors at `d±1`; cache-miss neighbors are residue
too. This is the cheap multiplier that turns one find into a pocket.

**Band enumeration at `floor = T−4` or `T−5` (the workhorse for the bulk).**
Practical on a single machine; auto-seeding from *all* cached band boards (not
just the antipode-connected ones) lets it cascade through the upper layers and
recover the large majority of residue in hours. It will not reach the final
disconnected maxima, but it clears everything else cheaply.

**Symmetry-aware everything.** Mirror every find on insert; probe both a board
and its reflection before solving.

## 3. What did not work

**Band-BFS at a fixed floor for the last maxima.** At `floor ≥ 74` the final
pockets are disconnected from the antipode mass, so no number of 1-step BFS
rounds reaches them. Dropping to `floor` 73/72 reconnects them only after the
slow, millions-of-solves round-1 grind.

**Hamming sweeps for the final maxima.** Radius-≤6 sweeps around every seed
class tried — best rarity/residue seeds, freshly found lower-depth residue, and
*all* high-`h` maxima — returned **zero** `d=77`. The final pair were Hamming
> 6 from every known board. Hamming distance was simply the wrong metric for
this residue.

**Seeded puzzle-move pocket walks.** A miss-only band-BFS from cache-miss
residue maps a disconnected pocket completely and cheaply — but only the pocket
*containing a seed*. The final pair's pocket contained none of the thousands of
seeds tried. At a low floor the same walk floods the enormous connected
cache-miss "web" without converging.

**Hamming/geometric singleton clustering for d=77.** It closed `d=78` (which
was Hamming-isolated) but found nothing at `d=77`, whose residue is
frame/placement-isolated, not geometrically isolated.

**Guessing a structural signature from prior finds.** Fixing a hand-picked
skeleton (a shared bottom row, a shared tile placement) and enumerating its
completions gave zero. The productive frame must be *derived* from the maxima
distribution, not guessed.

**Counting/symmetry to localize the gap.** Because the layer is
reflection-symmetric within every `(blank, h)` bucket, adding the missing pair
keeps every bucket balanced — there is no asymmetry or odd count to point at.

**"Hardest residue ⇒ highest h."** A tempting heuristic (the `d=78` residue
sat at the `h=70` ceiling) that proved **false** at `d=77`: the final pair were
`h=67`, a common mid bucket (8,184 maxima), not the rare `h=69/71` tail.
Targeting only the high-`h` frames missed them; sweeping all maxima frames
caught them.

**PDB-shell enumeration (filter `h ≥ T`, verify).** The admissible heuristic
ceiling is ~71 and the mean `h` near the top is only ~66–67, so to catch every
maximum admissibly `T` must be low enough that the candidate set is ~10⁸–10⁹ —
infeasible to verify without a tighter heuristic.

## 4. Recommendations for closing remaining 15-puzzle layers

To close a deep layer `T` (next: `d=76`), in order:

1. **Reflection-closure check.** Confirm the gap is reflection pairs (halving
   the work) and recover any board whose mirror is already present.

2. **Maxima profiling.** The residue is the maxima; the non-maxima fall out of
   the complete layer above for free. Profile maxima by blank cell and `h` to
   scope the hunt and to choose the `h` cutoff in step 3.

3. **Frame-completion over *all* maxima frames — do this first.** Fix each
   maximum's top-8 frame, enumerate every bottom-8 permutation, verify the
   cache-miss completions. This is the fast, seed-independent closer; prefer it
   over any sweep or band run. Two rules learned the hard way: sweep **all**
   maxima frames, not just the high-`h` ones, and set the `h` cutoff a few
   below the layer's maximum maxima-`h` (the residue is not necessarily
   high-`h`). Limitation: it only reaches boards whose frame is shared with a
   cached maximum — if that yields nothing, the residue's frame is novel; go to
   step 5.

4. **Band enumeration at `floor = T−4/T−5` for the bulk.** The workhorse for
   the thousands of ordinary residue boards; cascade one puzzle-move from each
   find, mirror every insert. `d=76` is far larger than `d=77`, but frame-
   completion plus this should close it much faster than a low-floor run.

5. **If the residue's frame is genuinely novel** (shared with no cached
   maximum), escalate, cheapest first:
   - **Generative per-tile-cell enumeration.** Restrict each tile to the cells
     it actually occupies across known maxima and enumerate bijections; the
     high-top/low-bottom invariant factors this into two independent 8-cell
     sub-problems, keeping it tractable. Unlike frame-completion this can
     synthesize frames the cache has never seen.
   - **Low-floor global band run** as the guaranteed-but-slow fallback (keep
     the machine awake — it pauses on sleep and silently stops making
     progress).
   - **Long-horizon: a tighter admissible heuristic** (multi-partition or
     larger ZPDB) to shrink the antipodal slack, which finally makes PDB-shell
     enumeration viable — the only fully seed-independent closer, worth the
     build cost if this residue pattern recurs (e.g. for the 24-puzzle).

**Operational invariants.** Never run point inserts while the enumerator is
running — its periodic autosave overwrites them. Back up the cache before any
mutation. Mirror every find under reflection.

**The cross-layer lesson.** Each layer's residue hides behind a *different*
metric: `d=78` was Hamming-isolated, `d=77` was placement- and
frame-isolated. Hamming distance and puzzle-move connectivity both failed for
`d=77`; the **frame / bottom-permutation** structure was the right lens. When a
metric stops finding residue, the residue has not run out — the metric has.

---

## 5. Session addendum (2026-06-28): the d=76 attack

`d=76` stands at 272,176 / 272,198 — **22 boards (11 reflection pairs) still
missing**, all necessarily maxima (d=77 is complete, so every non-maximum d=76
board falls out of expanding d=77). 211,766 of the layer are maxima, 60,410
non-maxima.

### The quadrant decomposition (a better frame than top-8/bottom-8)

The d=76 maxima structure **radiates from the four corners**, not in a top/bottom
band (the band split holds for only ~24% of the layer). Per-cell entropy is
monotonic in Chebyshev distance from the nearest corner: the four corners are the
tightest cells (mean H≈1.25; cells 3,12 are 83% a single tile, 15 is 74%, 0 is
39%), every other cell is looser (mean H≈2.90), and the dead-center 2×2
{5,6,9,10} is loosest. So the productive partition is **the four 2×2 quadrants**
(TL{0,1,4,5} TR{2,3,6,7} BL{8,9,12,13} BR{10,11,14,15})), each anchored by a
locked corner and holding a characteristic **3-tile core + 1 floating tile**:
TL≈{11,12,15}, TR≈{9,13,14}, BL≈{3,4,8}, BR≈{1,2,5} (TR/BL are exact reflection
twins; top corners hold high tiles, bottom corners low).

Dedupe the maxima by their 4-quadrant **tile-set partition** (which 4 tiles live
in each quadrant): only **9,598 distinct partitions**, avg 22 maxima each, and
**98.5% of maxima share a partition with another** — beating the top-8 band
(31,581 frames, 94.3% reach) and the entropy-tightest-8 single frame (26,818,
95.3%). **Quadrant-factored generative completion** (`d76_quadrant_complete.rs`):
for each partition, enumerate every within-quadrant arrangement (4! per corner,
24⁴ total), keep solvable + cache-miss. Dry run: 3.18B enumerated → 1.59B
solvable → **1.54B cache-miss (96.7% novel)** — a strong, reach-98.5% generator,
the four-way refinement of step-5's two-way top/bottom factorization.

### What did NOT crack the d=76 residue (the heuristic-slack wall)

The ZPDB-plus heuristic has ~11 moves of slack near the top (real d=76 boards
have **even** h ∈ [60,70], mean 65, mode 66; maxima bottom out at h=60). Every
*cheap heuristic-landscape* filter therefore fails:

- **h-cutoff.** A safe cutoff (≤62, to not exclude low-h residue) leaves 180M+
  candidates; a tight cutoff (≥66) drops ~46% of the maxima. Dead either way.
- **Pit-filter** (board where every neighbor has higher h; the heuristic is
  consistent + parity-flipping, so min(neighbor h) = h∓1). Pit-ness rises as h
  drops (57% of d=76 at h=60, 0.3% at h=68) and *enriches* d=76 by 5–29×, but
  retention is unsafe (drops 43–80% of low-h d=76) and the low-h pit sets are
  still 7.5M–29M. **Cheap shot run: the 130,594 h=64 pit candidates all verified
  to depth 66–74 — zero d=76 finds.** Confirms the residue is h<64 or non-pit.
- **Neighbor-depth lookup** is out: the cache is complete only top-down to ~d=72
  (d=71 count < d=72 → incomplete frontier below), and the residue is a connected
  pocket whose d=75 neighbors are themselves missing.
- **ML classifier *with* h as a feature** just relearns "high h → deep": 100%
  d=76 recall needs keeping 57.8% of candidates (no filtering); low-h recall ~0.

### What WORKED: the structure-without-h discriminator

**Withhold h (and all distance features) and train a GBM on board structure
alone — 16 tile-at-cell values + blank cell.** This separates a true d=76 from
quadrant-generated d=70–74 boards **at fixed h**, where every heuristic method is
blind. Validated against the fair negatives (quadrant-generated, same
distribution as inference; `export_features_ranks.rs` + sklearn
`HistGradientBoostingClassifier`):

- fixed h=64, d=76 vs quad-cands(d66–74): **AUC 0.999, recall 0.99 at a 100× cut**
- **pit-controlled** (both classes 100% pit): AUC 0.999 — not a pit artifact
- depth-stratified, even vs d=74 (hardest lookalike): AUC 0.993
- within-h-bucket AUC ≥0.99 at h=60/62/64

The model reads true depth off the *within-quadrant arrangement* given the same
corner frame, at fixed h, with no heuristic at all — exactly the
residue-vs-known-maximum distinction. h was a crutch drowning out this signal.
This is the first discriminator that beats the slack wall and works in the low-h
region where the residue lives.

**Status: validated on held-out KNOWN d=76; the full scoring run is NOT yet
done.** Caveats: (1) generalization to the *exact* 22 missing boards is unproven
— they may be the hardest-to-classify d=76, so run at a generous keep-fraction
(top ~5%, not 1%); (2) the generator only reaches partitions shared with a known
maximum (98.5%) — a genuinely novel-partition residue board is invisible to it.

**Next step:** deploy the structure model at scale — Rust generator emits
candidate features in shards → batch-score in Python → keep top ~5% by structure
score → `verify_depths` (IDA*) → `cache_insert --from-pairs` (mirrors +
backs up). Estimated hours, not weeks.

### Session bookkeeping

Cache grew by the 130,594 verified pit candidates (d=66–74, all previously
cache-miss): 86,725,302 → **86,855,896** entries (backup at
`solve_cache.bin.bak`). Tools added this session: `d76_maxima_dump`,
`d76_quadrant_complete` (generator + pit stats), `verify_depths`,
`export_depth_features` / `export_features_ranks` (classifier features),
`d76_h_hist`, `d76_min_neighbor_h`, `cache_depth_hist`, `joint_pairwise`;
`cache_insert` gained a `--from-pairs FILE` mode.

---

## 6. CLOSING d=76 (2026-06-29): the layer-augmentation method

**d=76 is COMPLETE — 272,198 / 272,198.** The §5 plan worked, but the last ~10
boards needed one idea §5 didn't have: **borrow partition vocabulary from the
adjacent shallower layer.** This is the decisive, reusable result.

### The structural key: nested partition hierarchy

A board's **quadrant tile-set partition** (which 4 tiles occupy each 2×2 corner
quadrant — see §5) is **depth-agnostic**: the same partition has solvable
realizations at many depths and both parities. Measured over the full layers:

> **d76 partitions (9,598) ⊆ d75 partitions (35,155) ⊆ d74 partitions (102,612).**

Each shallower layer's partition set strictly *contains* the deeper one's and
adds more (d75 adds 25,557 novel; d74 adds 67,457 novel beyond d75). The
quadrant generator can only emit boards whose partition is in its seed set, so a
residue board in a **novel partition** (held by no *cached* board of its own
layer) is unreachable — *unless you seed the generator from a layer whose
partition set is richer.* The residue sits in progressively rarer partitions;
each augmentation reaches one shell deeper:

| seed partitions | closed |
|---|---|
| d76 maxima (9,598) | the typical residue |
| **+ d75 (35,155)** | 3 d=76 pairs in partitions novel to d76 |
| **+ d74 (102,612)** | the final 3 d=76 pairs, novel to d75 |

Every closed pair was verified to be in a partition novel to the layer below it
— direct confirmation that the residue is *partition*-isolated and that
borrowing from below is the right lever.

### The closure pipeline (reusable)

1. **Structure-without-h model** (§5) scores candidates; it is *blind to novel
   partitions* (trained on the layer itself), but **per-partition top-K** rescues
   it — fix K (=500) arrangements per partition regardless of absolute score, so
   each novel partition's best candidates are always verified.
2. **Interleaved parity hunt** (`d75d76_hunt`): enumerate each partition's 24⁴
   arrangements ONCE; keep solvable + cache-miss + h≥59; route by parity —
   **even-h → d76 candidates, odd-h → d75 candidates** — score, keep per-partition
   top-K of each, and write a score-merged list. One enumeration hunts **two
   layers at once** (a board's depth parity = its h parity, so even-h boards can
   only be d=even, odd-h only d=odd).
3. **Auto-extending sweep** (`sweep.sh`): verify the merged list in 500K chunks
   top-down by score, `cache_insert` after each, stop when a chunk's yield drops
   below THRESH. A saturating fit `cum = T·n/(K+n)` to the per-chunk yield
   predicts the total findable and the asymptote (it landed on the known gap,
   confirming near-total reach); a power-law fit is the wrong model (exponent→1
   diverges).
4. **Bellman closure** (`bellman_closure`): any cache-miss board whose EVERY
   neighbor is cached has exact depth `1 + min(neighbor depths)` — free, no IDA*.
   Iterates to a fixpoint, mirrors inserts. **Limited to the boundary**: it fills
   holes adjacent to the known mass but cannot crack a *disconnected,
   mutually-missing pocket* (the deep residue), so it yielded only a handful of
   d=75 here — useful cleanup, not a closer.
5. Standard throughout: IDA*-verify every find, mirror under reflection on
   insert, `caffeinate -dimsu` all jobs, checkpoint pairs incrementally
   (crash-safe), back up the cache before each mutation.

### What did NOT help (this session)

- **Retraining the model with on-distribution hard negatives backfired** — the
  d72–74 quadrant boards are structurally near-identical to the residue, so
  adding 600K as negatives taught the model to suppress that whole region; the
  known residue's score collapsed (~0.99 → ~0.04). More data hurts when the
  negatives crowd the positives.
- **h-cutoff / pit-filter / ML-with-h** all failed for the low-h residue (§5);
  the *structure-without-h* model + per-partition top-K is what works.

### The cross-layer lesson (updated)

The residue's hiding metric again differed: d=78 Hamming-isolated, d=77
frame-isolated, **d=76 quadrant-partition-isolated** — and specifically isolated
in partitions *novel to its own layer but present one layer down*. For deeper or
harder layers (incl. the 24-puzzle), the **layer-augmentation hierarchy** is the
general tool: to close layer L, seed the partition generator from L ∪ (L−1) ∪
(L−2)… until the residue's partition is covered; interleave even/odd to close two
layers per pass.

### Bookkeeping (final)

d=76: 272,176 → **272,198 (complete)**. d=75: 1,501,962 → 1,507,336 (closed
~5,374; ~1,260 still short of 1,508,596, reachable by continuing the sweeps).
Cache: ~86.7M → **95.8M** entries. New tools: `d75d76_hunt` (parity hunt),
`bellman_closure`, `dump_grids`, `gen_from_domains`, `d76_quadrant_score`,
`model_score_check` + `common/tree_model.rs` (Rust port of the sklearn
HistGBM, bit-exact); driver `sweep.sh`; the structure model is `model.txt`
(dumped from `dump_model.py`, scored in Rust via `tree_model`).
