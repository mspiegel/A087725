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
| 76 | 272,198 | a few thousand still missing |

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
