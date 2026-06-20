# Closing the 15-puzzle residue at the antipodal end

## 1. The goal

Enumerate every 15-puzzle state at the deepest layers — specifically `d ∈
{76, 77, 78}` — that the band-BFS strategy in `enumerate15` cannot reach
from the antipode shell within practical wall-time. We call this the
**residue**: boards whose true optimal distance to the goal is `d`, but
which are *not* reachable from the antipode shell through paths that stay
within the band `[floor, 80]` in fewer than a small number of below-floor
detours.

Starting state at the beginning of this work:

| Depth | Cache count | Expected `N(d)` | Missing residue |
|------:|------------:|----------------:|----------------:|
| 78    | 3,402       | 3,406           | **4**           |
| 77    | 26,514      | 26,638          | **124**         |
| 76    | 241,295     | 272,198         | **30,903**      |

Layers 79 and 80 were already complete (70/70 and 17/17) — those came
from prior runs and from the canonical antipode list. The residue at `d=78`
in particular has resisted multi-day enumeration runs.

### How `enumerate15` works (band-BFS)

The enumerator's pipeline (see `src/bin/enumerate15.rs` and
`src/puzzle15/enumerate/frontier.rs`):

1. **Seed** a `Store` with the 17 known antipodes at `d=80` plus, optionally,
   every cached `(rank, depth)` with `depth ≥ floor` (auto-seed-from-cache).
2. **BFS** outward over the band `[floor..=80]`. Each round:
   - Take the current frontier (boards inserted in the prior round, or the
     full seed set on round 1).
   - For each, generate one-blank-move neighbors (`fresh_neighbors`).
   - For every cache-miss candidate, run admissible IDA\* (`ZpdbPlusInc`
     heuristic = max of zero-aware 7+8 ZPDB sum, linear conflict, walking
     distance) to determine the exact optimal depth.
   - Symmetry-aware: probe both `r` and `reflect(r)` in the persistent
     solve cache before solving; mirror every solve into both halves of
     the reflection pair.
   - Insert candidates at `depth ≥ floor` into the Store; queue them for
     the next round.
3. **Repeat** until the frontier is empty or a budget cap is hit.

The auto-seed step makes the BFS heavily reliant on the persistent
`solve_cache.bin`: once it has seen a board, future runs skip the IDA\*
work. The K-step expansion variant (added experimentally as
`ENUM_K_STEPS`, then later reverted) lets each round expand `≤K` blank
moves to bridge thin pockets of below-floor states, at the cost of larger
candidate fan-out.

### Why this misses residue at the top

The 4 missing `d=78` boards have the property that *every* band path from
them to the antipode shell dips below the chosen `floor`. At `floor=72`,
they are unreachable by 1-step BFS no matter how many rounds run.

This was confirmed empirically:

- A 24-hour `floor=74` enumerate15 run finished with `d=78 = 3401/3406`
  (5 boards short), having performed 4.85M IDA\* solves.
- A `floor=72`, `ENUM_K_STEPS=3` long-run made **zero** `d=77` or `d=78`
  inserts over many hours at the same time-per-solve rate (we observed
  ~102 solves/s, with d77 frozen at the seeded count across every status
  heartbeat).
- A `floor=72`, K=8 explicit Hamming expansion from all 3,402 known
  `d=78` boards (570K candidates after `h ≥ 61` filter) found exactly
  zero new `d=78` boards.

The residue is structurally separated from the antipode-reachable mass
in the band-72 connectivity graph. No amount of band-BFS at this floor
will close it.

## 2. What has worked

### Tile-position clustering reveals geometric outliers

Treating each `d=78` board as a 16-dim tile vector and running
connected-component clustering on Hamming distance at `eps=3` reveals
structure that depth/blank-cell univariate analysis misses:

- The bulk of `d=78` boards form one dense connected mass (3,367 of 3,402
  at `eps=3`).
- **27 boards are geometric singletons** — Hamming distance ≥ 4 from any
  other `d=78` board. These are the "extreme outlier" wings.
- A few size-2/3 clusters at blank cells with structural quirks.

The singletons include the "perfect-reverse" arrangement (rank
`653837183999`) — every tile in strict descending order, the structural
anti-pattern to the goal, at `h=70` (maximum heuristic value observed
across the entire cache).

### K=4 Hamming-neighborhood enumeration around the perfect-reverse singleton

Enumerate every solvable state within Hamming distance 4 of the
perfect-reverse pattern, IDA\*-verify each cache-miss candidate with
`h ≥ 60` filter. From ~17K raw states and ~4.5K candidates verified in
~3 minutes:

- **2 NEW d=78 boards** (a reflection pair): ranks `638008168979` and
  `616213568008`, both blank@0
- 306 new d=76 boards as a side cascade

This was the breakthrough. K=6 around the same reference adds 8 d=77
residue and 1,444 d=76 residue, but no new d=78 — the *additional* d=78
maxima live outside this neighborhood.

### Cascade from found residue

Once any residue board at depth `d` is known, expanding 1 puzzle move
(K=1) finds its neighbors. Cache-miss neighbors at adjacent depths
(`d-1` and `d+1`) are residue too if not already cached. K=2/K=3 from
residue seeds reaches further:

- K=2 from 1,685 d=76 residue boards → 17 NEW d=77 residue + 324 NEW
  d=76 (~10 min, 9.4K verifies)
- K=3 from 27 d=77 residue boards → 5 more d=77 + 36 more d=76
- K=4 from ~2,000 d=76 residue boards → 8 more d=77 + 529 more d=76
  (47 min, 70K verifies)

This is the workhorse for filling d=77 and the upper d=76 layer.

### Sub-clustering by blank cell

Re-running the tile-position clustering after filtering to a *specific
blank cell* exposes additional singletons that were buried in the global
mass. The most fragmented blank cells:

- **blank@10**: 34 d=78 boards → 14 clusters (8 singletons + 6 small)
- **blank@15**: 68 d=78 boards → 13 clusters (6 singletons + 6 pairs +
  size-50)

These look like the right hunting territory if the remaining d=78
residue is at a non-blank@0 cell.

### Symmetry-aware cache enrichment

Every find is inserted with its reflection partner. This keeps the
solve cache symmetry-closed and means the BFS probe path
`cache.get(r) || cache.get(reflect(r))` lands a hit on either half of
each pair.

### Compound effect

The combined approach is dramatically more efficient than band-BFS
alone for the top of the distribution:

| Approach | d=78 found | d=77 found | d=76 found | Wall-time |
|---|---|---|---|---|
| enumerate15 K=1 @ floor=72 (24h) | 0 | 0 | ~50K | 24 h |
| enumerate15 K=3 (1+h) | 0 | 0 | ~3K | 1 h |
| Residue strategy | **4** | **~120** | **~6,200** | **~6 h** |

### Closing d=78: structural-signature sweep, not singleton clustering

After the initial 2 d=78 finds from K=4 around the perfect-reverse
singleton, an extensive geometric singleton/sub-singleton/pair-member
exploration *failed* to find the other 2 d=78 boards. The winning move
was simpler: **sweep K=4 around every cached d=78 board sharing the
heuristic signature of the original successful reference**
(`h=70, blank@0` — 57 boards in the cache). Reference #44 in that
sweep (rank `641121866279`) produced the final 2 d=78 boards as K=4
Hamming neighbors:

- rank `510134722843` blank@0: `__ 12 10 13 / 15 11 14 1 / 7 8 6 5 / 4 3 2 9`
- rank `510134887077` blank@0: `__ 12 10 13 / 15 11 14 9 / 7 8 6 5 / 4 1 2 3`

The two reflection pairs of d=78 residue all lived in the same
structural neighborhood — `blank@0`, `h=70` — but at different specific
tile arrangements. The targeted hunt could have closed in ~1 hour by
sweeping the 57 `h=70 blank@0` boards as the second experiment instead
of the much larger singleton-clustering exploration that consumed most
of the session.

**The lesson:** when a single targeted enumeration produces residue
finds, the next move should be to sweep *every* board sharing the
successful reference's structural signature (blank cell + heuristic
bucket), not to fan out into the geometric-clustering machinery.

## 3. What has not worked

### Band-BFS at `floor=72`

Both 1-step and K=3 K-step variants. At `floor=72`, the band-graph is
disconnected such that no d=78 residue is reachable via 1-step paths
from any known d=78 board, and the K=3 reach is too short to close the
gap (a single K=3 probe found 1 d=77 hit in 17 min).

### K-step Hamming expansion from arbitrary d=78 seeds

K=2, K=4, K=6, K=8 from the *entire* known d=78 set, and K=4 around
each of 16 d=78 singletons individually: **all yielded 0 new d=78
finds.** The missing 2 d=78 boards are at Hamming distance > 8 from
every known d=78 board.

### K=4 sweep of d=77 singletons (112 boards)

Found a steady ~5 new d=77 residue boards and ~50 d=76 boards over the
full sweep, but **zero d=78 finds**. The d=77 singleton neighborhoods
don't intersect the missing d=78 sub-pocket.

### K=2/3/4 cascade from d=77 residue → d=78

K=1, K=2, K=3 expansions from the 32 known d=77 residue boards found
many new d=76 and a few more d=77 — but **zero d=78**. The known d=77
residue boards have all their d=78 neighbors already in the cache. The
missing d=78 boards' d=77 neighbors are among the still-unknown d=77
residue.

### K=2/K=4 cascade from d=76 residue → d=78

K=2 from ~1,685 d=76 residue and K=4 from ~2,000 d=76 residue: thousands
of new d=74/75/76/77 finds, **zero d=78**. The K=4 reach (4 puzzle moves)
from any d=76 residue board does not include a missing d=78 board.

### PDB-shell enumeration via ZpdbPlus

Filtering all states with `h(s) ≥ T` and verifying with IDA\*. Our
heuristic ceiling is ~71 across 15M cached entries; the slack at `d=78`
is ~11 (mean h = 66.76). To capture all d=78 boards admissibly, the
threshold must be `T ≤ 62`, which in the wild catches a vastly larger
candidate set than is feasible to verify (~10⁸–10⁹ candidates per
estimates). Tighter heuristics (10-tile ZPDB or multi-partition) would
close this slack — but at the cost of weeks of engineering and storage.

### Cluster-1 signature space enumeration

`{blank=0, tile 12@1, tile 13@3, tile 4@12, tile 1@15}` — the dominant
tile signature of the 6 known `h=62` d=78 maxima. Yields 11!/2 = ~20M
solvable states, ~2.6M cache-miss candidates after `h ≥ 61` filter.
Aborted partway through verification once the perfect-reverse Hamming
approach yielded results faster. In retrospect, the 6 h=62 boards live
in the global main mass — the cluster-1 signature space probably does
*not* contain the residue.

### `enumerate15` with auto-seed enrichment

Restarting the BFS with the now-richer cache (every residue board
inserted) reproduces the same trickle pattern. The residue boards are
in the cache, but they don't help the BFS reach the still-missing
d=78 boards — their d=77 neighbors are still unknown.

## 4. Recommended strategy

The data converges on a single approach: **the residue strategy *is* the
path forward.** No band-BFS configuration at this floor closes the top
layers in practical wall-time. The recommendation is to formalize the
techniques that have worked, sequence them deliberately, and avoid the
detours we've already eliminated.

### Tactical sequence to close d=78 (revised after this session closed it)

When any K=4 Hamming-neighborhood enumeration produces a d=78 find,
**immediately sweep K=4 around every cached d=78 board sharing the
successful reference's structural signature** — same blank cell and
same heuristic bucket. This catches reflection-pair siblings sitting
in the same neighborhood without needing further clustering. For the
4 missing d=78 boards in this session, all 4 lived at `h=70 blank@0`
and were captured by just 2 K=4 enumerations (perfect-reverse + one
ordinary h=70 blank@0 board). Total compute: ~1 hour if done from
the start.

If the same-signature sweep finds nothing, the missing residue at
the layer is at a different heuristic bucket or different blank cell.
At that point, the right escalation is:

1. **Sweep K=4 around the same blank cell at the next-lower heuristic
   bucket** (e.g. `h=68` boards at `blank@0`). For d=78 the layer
   distribution is ~50 at `h=70`, ~1,300 at `h=68`, ~1,700 at `h=66`,
   etc. — the high-h bucket is small and fast to sweep.

2. **Pivot to a different blank cell** based on which non-blank@0 cells
   appear in the residue (the `above_degree == 0` Bellman-max boards
   we know about already tell us which blank cells are populated).
   At the rare non-blank@0 cells (`blank@7`, `blank@10`, `blank@13`,
   `blank@15`), the populations are 34–88 each — fully sweepable at K=4
   in well under an hour per cell.

3. **K=6 around a single high-yield reference** only when K=4 sweeps
   across all candidate blank cells have come back empty. K=6 covers
   the next Hamming shell but with 100× more candidates per reference.

The lesson from this session: the per-blank-cell sub-singleton
clustering work was a detour. It produced useful d=76 cascade but
none of it converged on the d=78 residue. The d=78 residue lived in
the **dense high-h bucket within the blank cell where the first finds
occurred** — not in the geometric-outlier sub-pockets at sparser blank
cells.

### Tactical sequence to close d=77

The d=77 residue is down from 124 missing to 70 missing, primarily via
cascade from found d=78 + Hamming-K from d=76 residue. To close it:

1. **Generalize the sub-singleton-by-blank-cell technique to d=77.**
   Re-run d77_clusters with the blank-cell filter at each odd-parity
   blank (1, 3, 4, 6, 9, 11, 12, 14) and identify sub-singletons within
   each. Sweep K=4 around each. Expected yield: most or all of the 70
   remaining d=77 residue.

2. **Once any new d=78 is found, its d=77 neighbors are also residue.**
   This couples the d=77 closure to d=78 closure: closing one helps the
   other.

3. **The remaining d=77 residue might cascade for free when all 4 d=78
   are known.** Once the BFS has every d=78 board in store, its
   1-step expansion finds every d=77 board adjacent to any d=78 —
   which by Bellman optimality is every d=77 with at least one d=78
   neighbor. The residual d=77 (with no d=78 neighbor — pure d=77
   Bellman maxima) is a much smaller set and is addressed by the
   sub-singleton sweep above.

### Tactical sequence to close d=76

d=76 still has ~28K missing — too many for hand-targeted sweeps. The
cascade we've already done has been incidental; for a focused close:

1. **Cluster d=76 boards at eps=3** (similar to d=77 / d=78). The
   structure will likely be a single dense mass with thousands of
   singletons — too many to sweep individually.

2. **K=2 from all known d=77 residue + new** (after d=77 closes). This
   sweeps every 1-puzzle-move neighborhood of the d=77 layer for d=76
   cache misses. Cheap and should close most of d=76.

3. **Accept partial closure** if needed. d=76 is much less load-bearing
   than the top — a 90%+ completion may be sufficient for the broader
   diameter-bounding work.

### Methodology improvements to land

The session has revealed a working pattern that should be codified:

- **Cluster the deepest layer by tile-arrangement Hamming distance**,
  globally and per-blank-cell.
- **Identify the geometric singletons and sub-singletons** as the
  hunting territory.
- **Hamming-K=4 enumeration around each** as the primary find
  operation.
- **Cascade via 1-step puzzle-move expansion** from every new residue
  insert to find adjacent residue at neighboring depths.
- **Symmetry-aware everything** — every find is mirrored on insert.
- **A general `enumerate_signature_neighborhood` tool** that accepts
  an arbitrary reference state + Hamming-K + heuristic floor would
  reduce future per-target setup to a one-liner.

### Long-horizon investment (out of scope for closing d=78)

If a similar problem at d=78-equivalent depth shows up for the 24-puzzle
or a different domain, the cleanest investment is a **tighter
admissible heuristic** that reduces the slack at the antipodal end.
This unlocks PDB-shell enumeration as a viable closure technique,
making the residue-hunt pipeline fast even when no structural
clustering is identified. For the 15-puzzle, this means a multi-
partition or larger-pattern ZPDB (storage cost a few tens of GB; build
time days; engineering cost weeks). Whether it's worth it depends on
how often the residue-hunt pattern recurs.

The current state of the cache (and the techniques in this repository)
is sufficient to close d=78 within a few more wall-clock hours of
targeted enumeration following the sequence above.
