//! Zero-aware PDB primitives — the geometric core of the `(m, p, r)` index.
//!
//! A **zero-aware** PDB (Clausecker & Reinefeld, SOCS 2019) indexes not just the
//! pattern-tile positions but also *which zero-tile region the blank lies in* —
//! the connected component (4-adjacency) of the cells **not** occupied by
//! pattern tiles. This module implements that region decomposition, the piece
//! that distinguishes a ZPDB from the standard additive PDB.
//!
//! Why it matters for the 1-bit codec (`docs/zpdb-codec-spec.md`): collapsing
//! all blank positions within one region into a single index contracts the
//! 0-cost ("anon") moves, leaving a quotient graph where **every edge costs 1**
//! (a pattern-tile move) — bipartite with `Δh = ±1`, the property the 1-bit
//! compression requires. The standard additive PDB, which keeps those 0-cost
//! edges, is *not* amenable to it.
//!
//! This module is verified against Table 1 of the paper: summed over all
//! `C(25,6) = 177,100` six-tile shapes, the region counts total `251,400`
//! (so the full 6-tile ZPDB has `6! · 251,400 = 181,008,000` entries — exactly
//! the paper's figure), averaging `1.42` regions per shape with a maximum of 5.

use super::pattern::{Pattern, ProjectedState, ANON};
use crate::puzzle24::state::{N_CELLS, W};

/// Marker in a region labeling for a cell occupied by a pattern tile.
pub const OCCUPIED: u8 = 0xFF;

/// 4-neighbours of cell `c` on the 5×5 board, written into `out`; returns the
/// count. Order is Up, Down, Left, Right (deterministic).
#[inline]
fn neighbours(c: usize, out: &mut [usize; 4]) -> usize {
    let (r, col) = (c / W, c % W);
    let mut n = 0;
    if r > 0 {
        out[n] = c - W;
        n += 1;
    }
    if r < W - 1 {
        out[n] = c + W;
        n += 1;
    }
    if col > 0 {
        out[n] = c - 1;
        n += 1;
    }
    if col < W - 1 {
        out[n] = c + 1;
        n += 1;
    }
    n
}

/// Decompose the board into zero-tile regions given the set of cells occupied by
/// pattern tiles (`occupied` is a 25-bit mask, bit `c` set ⇔ cell `c` holds a
/// pattern tile). Returns `(region_count, label)` where `label[c]` is the
/// region index in `0..region_count` for a free cell, or [`OCCUPIED`] for an
/// occupied cell.
///
/// Regions are numbered by ascending smallest-cell-in-region, so the labeling is
/// deterministic and order-stable — the foundation for a perfect-hash `r`.
pub fn regions(occupied: u32) -> (u8, [u8; N_CELLS]) {
    let mut label = [OCCUPIED; N_CELLS];
    for c in 0..N_CELLS {
        if occupied & (1u32 << c) != 0 {
            label[c] = OCCUPIED;
        } else {
            label[c] = u8::MAX - 1; // "free, unlabelled" sentinel (distinct from OCCUPIED)
        }
    }
    const UNLABELLED: u8 = u8::MAX - 1;

    let mut next_region: u8 = 0;
    let mut stack: Vec<usize> = Vec::with_capacity(N_CELLS);
    let mut nb = [0usize; 4];
    // Visit cells in ascending order so region 0 contains the smallest free cell.
    for start in 0..N_CELLS {
        if label[start] != UNLABELLED {
            continue;
        }
        let region = next_region;
        next_region += 1;
        label[start] = region;
        stack.push(start);
        while let Some(c) = stack.pop() {
            let k = neighbours(c, &mut nb);
            for &m in &nb[..k] {
                if label[m] == UNLABELLED {
                    label[m] = region;
                    stack.push(m);
                }
            }
        }
    }
    (next_region, label)
}

/// Number of zero-tile regions for a given occupancy mask.
#[inline]
pub fn region_count(occupied: u32) -> u8 {
    regions(occupied).0
}

// ---------------------------------------------------------------------------
// (m, p, r) perfect-hash index
// ---------------------------------------------------------------------------

/// Pascal's-triangle table for `C(n, k)` with `n ≤ 25`, `k ≤ 8` — the only
/// arguments `shape_rank`/`rank` ever use (and big enough for `C(25, 8)`).
const MAXN: usize = N_CELLS + 1;
const MAXK: usize = 9;
const BINOM: [[u64; MAXK]; MAXN] = build_binom();
/// `k!` for `k ≤ 8`, covering every `factorial`/`perm_rank` argument here.
const FACT: [u64; MAXK] = build_fact();

const fn build_binom() -> [[u64; MAXK]; MAXN] {
    let mut t = [[0u64; MAXK]; MAXN];
    let mut n = 0;
    while n < MAXN {
        t[n][0] = 1;
        let mut k = 1;
        while k < MAXK {
            let above = if n > 0 { t[n - 1][k] } else { 0 };
            let left = if n > 0 { t[n - 1][k - 1] } else { 0 };
            t[n][k] = above + left;
            k += 1;
        }
        n += 1;
    }
    t
}

const fn build_fact() -> [u64; MAXK] {
    let mut t = [1u64; MAXK];
    let mut i = 1;
    while i < MAXK {
        t[i] = t[i - 1] * i as u64;
        i += 1;
    }
    t
}

/// Binomial coefficient `C(n, k)`. Exact for the small `n ≤ 25` used here; an
/// O(1) table read (symmetry folds `k > n/2` into the tabulated `k ≤ 8` range).
#[inline]
fn binom(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    BINOM[n][k.min(n - k)]
}

/// `k!`. O(1) table read for the `k ≤ 8` arguments used here.
#[inline]
fn factorial(k: usize) -> u64 {
    FACT[k]
}

/// Combinatorial-number-system rank of a sorted cell set `c_0 < c_1 < … < c_{k-1}`:
/// `Σ_i C(c_i, i+1)`, a bijection onto `[0, C(25, k))`.
fn shape_rank(sorted_cells: &[u8]) -> u64 {
    let mut r = 0u64;
    for (i, &c) in sorted_cells.iter().enumerate() {
        r += binom(c as usize, i + 1);
    }
    r
}

/// Lehmer (factorial-number-system) rank of a permutation of `0..k`, onto
/// `[0, k!)`. O(k): the Lehmer digit at position `i` is `perm[i]` minus the
/// count of already-placed (left) values smaller than it, found with a popcount
/// over a used-value bitmask — no O(k²) right-scan.
fn perm_rank(perm: &[u8]) -> u64 {
    let k = perm.len();
    let mut used: u32 = 0;
    let mut rank = 0u64;
    for i in 0..k {
        let v = perm[i] as u32;
        let smaller_left = (used & ((1u32 << v) - 1)).count_ones() as u64;
        rank += (v as u64 - smaller_left) * factorial(k - 1 - i);
        used |= 1u32 << v;
    }
    rank
}

/// Enumerate every `k`-subset of the 25 cells, calling `f(sorted_cells, mask)`.
/// Cells are yielded ascending; iteration order is lexicographic (the consumer
/// re-indexes by [`shape_rank`], so order is irrelevant).
fn for_each_ksubset(k: usize, mut f: impl FnMut(&[u8], u32)) {
    let mut idx = [0u8; 8];
    for (i, slot) in idx.iter_mut().enumerate().take(k) {
        *slot = i as u8;
    }
    loop {
        let mut mask = 0u32;
        for &c in &idx[..k] {
            mask |= 1u32 << c;
        }
        f(&idx[..k], mask);
        let mut i = k as isize - 1;
        while i >= 0 && idx[i as usize] as usize == N_CELLS - k + i as usize {
            i -= 1;
        }
        if i < 0 {
            break;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Perfect-hash layout for a zero-aware PDB over a `k`-tile pattern.
///
/// An entry is keyed by `(m, p, r)`: the pattern-tile **shape** `m` (which `k`
/// cells are occupied), the **permutation** `p` (which tile sits in each, in
/// ascending order), and the blank's **zero-tile region** `r`. The index is
///
/// ```text
/// rank = cohort_base[shape_rank(m)] + perm_rank(p) * regions(m) + r
/// ```
///
/// where `cohort_base[s] = Σ_{s' < s} k! · regions(shape s')` groups all
/// `k!·regions(m)` entries of one shape into a contiguous cohort. The total is
/// `k! · Σ_m regions(m)` — e.g. `720 · 251,400 = 181,008,000` for `k = 6`, the
/// figure from Clausecker & Reinefeld Table 1.
pub struct ZpdbLayout {
    k: usize,
    kfact: u64,
    /// `slot_of[v]` = the permutation slot of tile value `v` (its index in the
    /// pattern's iteration order), and `tiles[j]` = the tile value with slot
    /// `j`. Precomputed so `rank` finds the occupied cells via `pos_of` in O(k)
    /// — no full-board scan and no per-call tile-value search.
    slot_of: [u8; N_CELLS],
    tiles: [u8; 8],
    /// Length `C(25, k)`, indexed by [`shape_rank`].
    cohort_base: Vec<u64>,
    /// Region count per shape, indexed by [`shape_rank`].
    counts: Vec<u8>,
    /// Region labels per shape, indexed by [`shape_rank`]. Caching these (≈4.4 MB
    /// for `k = 6`) turns each `rank`/move-gen lookup into an array index instead
    /// of a `regions()` flood-fill — essential for the 181M-state build.
    labels: Vec<[u8; N_CELLS]>,
    /// Bipartite-parity of `h` for any entry in shape `s`: parity of the sum
    /// of the `k` cell indices the pattern tiles occupy. The 1-bit codec
    /// recovers `h`'s bit 0 (the unstored parity) as `shape_parity[s]`.
    ///
    /// Why this is the right invariant: every ZPDB edge moves one pattern
    /// tile by ±1 (horizontal) or ±5 (vertical) — both odd offsets — so the
    /// sum-of-cells parity flips on every edge. Goal entries have
    /// `shape_parity = parity(sum of goal cells of pattern)` and `h = 0`
    /// (even); the rest follow by induction.
    shape_parity: Vec<u8>,
    total: u64,
}

/// Sorted occupied cells from a mask, written to `out`; returns the count.
#[inline]
fn sorted_cells(mask: u32, out: &mut [u8; 8]) -> usize {
    let mut n = 0;
    let mut m = mask;
    while m != 0 {
        out[n] = m.trailing_zeros() as u8;
        n += 1;
        m &= m - 1;
    }
    n
}

impl ZpdbLayout {
    /// Build the layout for `pattern`, precomputing the cohort prefix sums and
    /// the per-shape region tables.
    pub fn new(pattern: Pattern) -> Self {
        let k = pattern.size() as usize;
        let kfact = factorial(k);
        let nshapes = binom(N_CELLS, k) as usize;

        // Permutation slot of each tile value (pattern iteration order) and its
        // inverse, for the O(k) occupied-cell lookup in `rank`.
        let mut slot_of = [0u8; N_CELLS];
        let mut tiles = [0u8; 8];
        let mut slot = 0u8;
        for t in pattern.iter() {
            slot_of[t as usize] = slot;
            tiles[slot as usize] = t;
            slot += 1;
        }

        // Bipartite parity reference: parity of the sum of pattern-tile cell
        // indices at the goal. Tile `t` lives at goal cell `t - 1`.
        let mut goal_sum: u32 = 0;
        for t in pattern.iter() {
            goal_sum += (t - 1) as u32;
        }
        let goal_parity = (goal_sum & 1) as u8;

        let mut counts = vec![0u8; nshapes];
        let mut labels = vec![[OCCUPIED; N_CELLS]; nshapes];
        let mut shape_parity = vec![0u8; nshapes];
        for_each_ksubset(k, |cells, mask| {
            let sr = shape_rank(cells) as usize;
            let (cnt, lab) = regions(mask);
            counts[sr] = cnt;
            labels[sr] = lab;
            let mut sum: u32 = 0;
            for &c in cells {
                sum += c as u32;
            }
            shape_parity[sr] = ((sum & 1) as u8) ^ goal_parity;
        });

        let mut cohort_base = vec![0u64; nshapes];
        let mut acc = 0u64;
        for s in 0..nshapes {
            cohort_base[s] = acc;
            acc += kfact * counts[s] as u64;
        }
        Self { k, kfact, slot_of, tiles, cohort_base, counts, labels, shape_parity, total: acc }
    }

    /// Cached `(region_count, region_labels)` for an occupancy mask — an array
    /// index, not a flood-fill.
    #[inline]
    pub fn regions_for(&self, occupied: u32) -> (u8, &[u8; N_CELLS]) {
        let mut cells = [0u8; 8];
        let n = sorted_cells(occupied, &mut cells);
        let sr = shape_rank(&cells[..n]) as usize;
        (self.counts[sr], &self.labels[sr])
    }

    /// Number of pattern tiles.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Total number of `(m, p, r)` entries — the ZPDB size.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// `k!` (permutations per shape).
    pub fn perms(&self) -> u64 {
        self.kfact
    }

    /// Cohort prefix-sum table: `cohort_base[s]` is the global index of the
    /// first entry of shape `s` (`shape_rank(cells)`). Length `C(25, k)`.
    /// Exposed for the 1-bit codec's `idx → (shape, perm_rank, region)`
    /// decomposition.
    pub fn cohort_base(&self) -> &[u64] {
        &self.cohort_base
    }

    /// Region count per shape, indexed by `shape_rank`. Length `C(25, k)`.
    pub fn region_counts(&self) -> &[u8] {
        &self.counts
    }

    /// `h(s) & 1` for any reachable entry whose shape is `s`. The bipartite
    /// invariant: every ZPDB edge moves a pattern tile by ±1 (horizontal) or
    /// ±5 (vertical) cells — both odd — so the parity of the sum of pattern
    /// cell indices flips on every edge. Stored XOR'd with the goal-shape's
    /// parity, so the goal entries (h = 0) come out at parity 0.
    pub fn shape_parity(&self) -> &[u8] {
        &self.shape_parity
    }

    /// Map a projected state to its `(m, p, r)` index in `[0, total())`. The
    /// `pattern` is implicit in the precomputed `slot_of`/`tiles` maps built
    /// from it.
    ///
    /// O(k), not O(board): the occupied cells come straight from `pos_of`
    /// (no full-board scan), `shape_rank` is accumulated inline over the
    /// ascending bit-iteration of the occupancy mask, and the permutation slots
    /// feed the O(k) [`perm_rank`].
    pub fn rank(&self, proj: &ProjectedState, _pattern: Pattern) -> u64 {
        // Occupancy bitmask from the k pattern tiles' positions — no board scan.
        let mut occ: u32 = 0;
        for j in 0..self.k {
            occ |= 1u32 << proj.pos_of(self.tiles[j]);
        }
        let blank = proj.blank_pos() as usize;

        // Walk occupied cells ascending: accumulate shape_rank and read each
        // cell's tile slot (cells live in one cache line, hot in L1).
        let mut sr = 0u64;
        let mut perm = [0u8; 8];
        let mut j = 0usize;
        let mut m = occ;
        while m != 0 {
            let c = m.trailing_zeros() as usize;
            sr += BINOM[c][j + 1];
            perm[j] = self.slot_of[proj.cells[c] as usize];
            m &= m - 1;
            j += 1;
        }
        debug_assert_eq!(j, self.k);

        let sr = sr as usize;
        let pr = perm_rank(&perm[..self.k]);
        let count = self.counts[sr] as u64;
        let r = self.labels[sr][blank] as u64;
        debug_assert!(r < count);
        self.cohort_base[sr] + pr * count + r
    }

    /// Reconstruct a **representative** projected state for entry `rank` — the
    /// inverse of [`rank`](Self::rank) up to the blank's exact cell within its
    /// region.
    ///
    /// The `(m, p, r)` index pins the pattern-tile placement (shape `m` +
    /// permutation `p`) and the blank's *region* `r`, but not which cell of that
    /// region the blank occupies. We place the blank at the region's smallest
    /// cell. That is sufficient for the region BFS ([`super::zbuild`]): a region
    /// is connected, so every cell in it yields the same successor set under
    /// `gen_moves`. The round-trip invariant holds exactly:
    /// `rank(unrank_representative(x)) == x` (guarded by tests).
    ///
    /// This lets the build store its frontier as 4-byte ranks instead of 50-byte
    /// `ProjectedState`s — the dominant build-memory term — at the cost of an
    /// O(log C(25,k) + k²) reconstruction per expanded node.
    pub fn unrank_representative(&self, rank: u64) -> ProjectedState {
        debug_assert!(rank < self.total, "rank {} >= total {}", rank, self.total);

        // 1. Decompose rank = cohort_base[sr] + pr * count[sr] + region.
        let sr = match self.cohort_base.binary_search(&rank) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let rem = rank - self.cohort_base[sr];
        let count = self.counts[sr] as u64;
        let pr = rem / count;
        let region = (rem % count) as u8;
        self.unrank_in_cohort(sr, pr, region)
    }

    /// Reconstruct the representative for a state already decomposed into its
    /// shape-rank `sr`, permutation-rank `pr`, and blank `region` — steps 2–5 of
    /// [`unrank_representative`](Self::unrank_representative), skipping the cohort
    /// binary-search. Used by the frontier-free 2-bit build sweep, which already
    /// knows `sr` from cohort iteration and would otherwise pay the search per
    /// (re-)expanded node.
    pub fn unrank_in_cohort(&self, sr: usize, pr: u64, region: u8) -> ProjectedState {
        debug_assert!(sr < self.counts.len(), "shape rank {sr} out of range");

        // 2. Combinadic unrank of the shape `sr` → k ascending occupied cells.
        let mut occ_cells = [0u8; 8];
        let mut rem_sr = sr as u64;
        for i in (0..self.k).rev() {
            let mut c = i;
            while binom(c + 1, i + 1) <= rem_sr {
                c += 1;
            }
            debug_assert!(c < N_CELLS, "combinadic cell {c} out of range");
            occ_cells[i] = c as u8;
            rem_sr -= binom(c, i + 1);
        }

        // 3. Lehmer (factorial-number-system) unrank of the permutation `pr` →
        //    the pattern-tile slot at each ascending occupied cell.
        let mut avail = [0u8; 8];
        for (j, a) in avail.iter_mut().enumerate().take(self.k) {
            *a = j as u8;
        }
        let mut navail = self.k;
        let mut rem_pr = pr;
        let mut slot_at = [0u8; 8];
        for slot in slot_at.iter_mut().take(self.k) {
            let f = factorial(navail - 1);
            let d = (rem_pr / f) as usize;
            rem_pr %= f;
            *slot = avail[d];
            for x in d..navail - 1 {
                avail[x] = avail[x + 1];
            }
            navail -= 1;
        }

        // 4. Place tile `tiles[slot_at[j]]` at cell `occ_cells[j]`; rest ANON.
        let mut cells = [ANON; N_CELLS];
        for j in 0..self.k {
            cells[occ_cells[j] as usize] = self.tiles[slot_at[j] as usize];
        }

        // 5. Blank at the region's smallest cell (regions are numbered by
        //    ascending smallest cell, so the first match is that cell).
        let labels = &self.labels[sr];
        let blank = labels
            .iter()
            .position(|&l| l == region)
            .expect("region label must exist in its shape");
        cells[blank] = 0;

        ProjectedState::from_projection(cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::State;

    /// All permutations of `items` (small `k` only).
    fn permutations(items: &[u8]) -> Vec<Vec<u8>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut out = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let x = rest.remove(i);
            for mut p in permutations(&rest) {
                p.insert(0, x);
                out.push(p);
            }
        }
        out
    }

    /// Build a ProjectedState placing each `(cell, tile)` and the blank, filling
    /// the rest with arbitrary non-pattern tiles.
    fn make_proj(pattern: Pattern, placement: &[(u8, u8)], blank: u8) -> ProjectedState {
        let mut cells = [255u8; N_CELLS];
        for &(c, t) in placement {
            cells[c as usize] = t;
        }
        cells[blank as usize] = 0;
        let mut fill = (1..=24u8).filter(|t| !pattern.contains(*t));
        for v in cells.iter_mut() {
            if *v == 255 {
                *v = fill.next().unwrap();
            }
        }
        ProjectedState::from_state(&State(cells), pattern)
    }

    /// Enumerate every `(shape, permutation, region)` of `pattern` and assert
    /// `ZpdbLayout::rank` is a bijection onto `[0, total())`.
    fn assert_bijection(pattern: Pattern) {
        use std::collections::HashSet;
        let layout = ZpdbLayout::new(pattern);
        let k = pattern.size() as usize;
        let tiles: Vec<u8> = pattern.iter().collect();
        let mut seen: HashSet<u64> = HashSet::new();
        for_each_ksubset(k, |cells, mask| {
            let (count, labels) = regions(mask);
            // one representative blank cell per region
            let mut rep = vec![u8::MAX; count as usize];
            for (c, &l) in labels.iter().enumerate() {
                if l != OCCUPIED && rep[l as usize] == u8::MAX {
                    rep[l as usize] = c as u8;
                }
            }
            for perm in permutations(&tiles) {
                let placement: Vec<(u8, u8)> =
                    cells.iter().zip(&perm).map(|(&c, &t)| (c, t)).collect();
                for &b in rep.iter() {
                    let proj = make_proj(pattern, &placement, b);
                    let r = layout.rank(&proj, pattern);
                    assert!(r < layout.total(), "rank {} >= total {}", r, layout.total());
                    assert!(seen.insert(r), "duplicate rank {r}");
                }
            }
        });
        assert_eq!(seen.len() as u64, layout.total(), "rank not surjective onto [0,total)");
    }

    #[test]
    fn binom_and_factorial_known_values() {
        assert_eq!(binom(25, 6), 177_100);
        assert_eq!(binom(25, 7), 480_700);
        assert_eq!(factorial(6), 720);
        assert_eq!(factorial(0), 1);
    }

    #[test]
    fn shape_rank_is_bijection_small() {
        use std::collections::HashSet;
        for k in [1usize, 2, 3] {
            let mut seen: HashSet<u64> = HashSet::new();
            for_each_ksubset(k, |cells, _| {
                let r = shape_rank(cells);
                assert!(r < binom(N_CELLS, k));
                assert!(seen.insert(r));
            });
            assert_eq!(seen.len() as u64, binom(N_CELLS, k));
        }
    }

    #[test]
    fn perm_rank_is_bijection() {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::new();
        for p in permutations(&[0, 1, 2, 3]) {
            let r = perm_rank(&p);
            assert!(r < 24);
            assert!(seen.insert(r));
        }
        assert_eq!(seen.len(), 24);
    }

    /// The O(k) popcount `perm_rank` must equal the plain count-inversions
    /// Lehmer rank *value-for-value* (not just bijectively) — the on-disk ZPDBs
    /// are indexed with it, so any drift would silently corrupt lookups.
    #[test]
    fn perm_rank_matches_inversion_reference() {
        fn reference(perm: &[u8]) -> u64 {
            let k = perm.len();
            let mut rank = 0u64;
            for i in 0..k {
                let smaller = perm[i + 1..].iter().filter(|&&p| p < perm[i]).count() as u64;
                rank += smaller * (1..=(k - 1 - i) as u64).product::<u64>().max(1);
            }
            rank
        }
        for k in 1..=7u8 {
            let items: Vec<u8> = (0..k).collect();
            for p in permutations(&items) {
                assert_eq!(perm_rank(&p), reference(&p), "perm_rank diverged on {p:?}");
            }
        }
    }

    #[test]
    fn zpdb_rank_bijection_k2() {
        assert_bijection(Pattern::new(&[1, 2]));
    }

    #[test]
    fn zpdb_rank_bijection_k3() {
        assert_bijection(Pattern::new(&[1, 7, 13]));
    }

    #[test]
    fn zpdb_layout_total_matches_paper_k6() {
        let layout = ZpdbLayout::new(Pattern::new(&[1, 2, 3, 6, 7, 8]));
        assert_eq!(layout.total(), 181_008_000);
        assert_eq!(layout.perms(), 720);
    }

    #[test]
    fn empty_occupancy_is_one_region() {
        let (n, label) = regions(0);
        assert_eq!(n, 1);
        assert!(label.iter().all(|&l| l == 0));
    }

    #[test]
    fn full_occupancy_is_zero_regions() {
        let occ = (1u32 << N_CELLS) - 1; // all 25 cells
        let (n, label) = regions(occ);
        assert_eq!(n, 0);
        assert!(label.iter().all(|&l| l == OCCUPIED));
    }

    #[test]
    fn labels_are_contiguous_and_min_ordered() {
        // A vertical wall down column 2 (cells 2,7,12,17,22) splits the board
        // into a left region (contains cell 0) and a right region (contains 3).
        let mut occ = 0u32;
        for c in [2, 7, 12, 17, 22] {
            occ |= 1u32 << c;
        }
        let (n, label) = regions(occ);
        assert_eq!(n, 2);
        assert_eq!(label[0], 0, "smallest free cell must be in region 0");
        assert_eq!(label[3], 1, "the far side must be region 1");
        // Every free cell labelled in 0..n; every wall cell OCCUPIED.
        for c in 0..N_CELLS {
            if occ & (1u32 << c) != 0 {
                assert_eq!(label[c], OCCUPIED);
            } else {
                assert!(label[c] < n);
            }
        }
    }

    #[test]
    fn diagonal_isolation_creates_singleton_region() {
        // Wall off the top-left corner (cell 0) with cells 1 and 5: cell 0 is a
        // singleton region.
        let occ = (1u32 << 1) | (1u32 << 5);
        let (n, label) = regions(occ);
        assert_eq!(n, 2);
        assert_eq!(label[0], 0); // singleton corner is region 0 (smallest cell)
        // Region 0 must be exactly {0}.
        assert_eq!(label.iter().filter(|&&l| l == 0).count(), 1);
    }

    #[test]
    fn matches_paper_table1_for_k6() {
        // SOCS 2019 Table 1: the 6-tile ZPDB has 181,008,000 entries =
        // 6! · sum_over_shapes(regions). So the region counts summed over all
        // C(25,6) shapes must be 181,008,000 / 720 = 251,400, averaging 1.42,
        // with a maximum of 5 regions per shape.
        let mut total_regions: u64 = 0;
        let mut shapes: u64 = 0;
        let mut max_regions: u8 = 0;
        for_each_ksubset(6, |_, mask| {
            let n = region_count(mask);
            total_regions += n as u64;
            shapes += 1;
            if n > max_regions {
                max_regions = n;
            }
        });
        assert_eq!(shapes, 177_100, "C(25,6)");
        assert_eq!(total_regions, 251_400, "sum of regions over all 6-tile shapes");
        assert_eq!(max_regions, 5, "max zero-tile regions for k=6 (paper Table 1)");
        // 6! · 251_400 == 181_008_000 (the paper's ZPDB entry count).
        assert_eq!(720 * total_regions, 181_008_000);
        // Average 1.42 (paper).
        let avg = total_regions as f64 / shapes as f64;
        assert!((avg - 1.42).abs() < 0.005, "avg regions {avg} not ≈ 1.42");
    }

    #[test]
    fn matches_paper_table1_for_k7() {
        // The k=7 index gate (A7): the 7-tile ZPDB has 4,066,655,040 entries =
        // 7! · sum_over_shapes(regions). So region counts summed over all
        // C(25,7)=480,700 shapes must be 4,066,655,040 / 5040 = 806,876,
        // averaging ≈ 1.68 (paper). The max-regions bound rises with k (more
        // pattern tiles can carve the free cells into more components), so unlike
        // the k=6 case it is not pinned to a paper figure — we only assert it is
        // sane (≤ a generous board-geometry ceiling).
        let mut total_regions: u64 = 0;
        let mut shapes: u64 = 0;
        let mut max_regions: u8 = 0;
        for_each_ksubset(7, |_, mask| {
            let n = region_count(mask);
            total_regions += n as u64;
            shapes += 1;
            if n > max_regions {
                max_regions = n;
            }
        });
        assert_eq!(shapes, 480_700, "C(25,7)");
        assert_eq!(total_regions, 806_876, "sum of regions over all 7-tile shapes");
        // 7! · 806_876 == 4,066,655,040 — fits u32 (< 4,294,967,296) and matches
        // the corrected docs/zpdb-codec-spec.md Table 1 entry.
        assert_eq!(5040 * total_regions, 4_066_655_040);
        assert!(max_regions <= 8, "max zero-tile regions {max_regions} implausible for k=7");
        let avg = total_regions as f64 / shapes as f64;
        assert!((avg - 1.68).abs() < 0.01, "avg regions {avg} not ≈ 1.68");
    }

    /// The k=7 `(m,p,r)` rank lands in `[0, total())` and is injective — checked
    /// on a *sample* (the full space is 4.07e9 entries, far too large to
    /// enumerate). We walk a 7-tile projected state and confirm every rank is
    /// in range, deterministic, and collision-free across distinct projections.
    #[test]
    fn k7_rank_in_range_and_injective_on_sample() {
        use crate::puzzle24::state::GOAL;
        use std::collections::HashMap;
        let pattern = Pattern::new(&[1, 2, 3, 4, 5, 6, 7]);
        let layout = ZpdbLayout::new(pattern);
        assert_eq!(layout.total(), 4_066_655_040);

        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // Map rank -> the (m,p,r) signature: pattern-tile placement plus the
        // blank's REGION (rank collapses the blank's exact cell within a region,
        // so two states with the blank in the same region must share a rank —
        // that is correct, not a collision). A genuine collision is two distinct
        // (placement, region) sharing a rank.
        let mut seen: HashMap<u64, ([u8; N_CELLS], u8)> = HashMap::new();
        let mut s = GOAL;
        for _ in 0..20_000 {
            let proj = ProjectedState::from_state(&s, pattern);
            let r = layout.rank(&proj, pattern);
            assert!(r < layout.total(), "rank {} >= total {}", r, layout.total());

            let mut occ: u32 = 0;
            for (c, &v) in proj.cells.iter().enumerate() {
                if v != 0 && v != crate::puzzle24::pdb::pattern::ANON {
                    occ |= 1u32 << c;
                }
            }
            let (_, labels) = layout.regions_for(occ);
            let region = labels[proj.blank_pos() as usize];
            let mut placement = proj.cells;
            placement[proj.blank_pos() as usize] = crate::puzzle24::pdb::pattern::ANON; // drop exact blank cell
            let sig = (placement, region);

            if let Some(prev) = seen.get(&r) {
                assert_eq!(*prev, sig, "rank {r} collides on distinct (m,p,r)");
            } else {
                seen.insert(r, sig);
            }
            let opts: Vec<_> = s.legal_moves().iter().collect();
            s = s.apply(opts[(next() as usize) % opts.len()]);
        }
    }

    /// `rank(unrank_representative(x)) == x` for every entry — the inverse used
    /// by the rank-frontier build. Exhaustive at small k, sampled at k=7 (4.07e9
    /// entries is too many to enumerate).
    #[test]
    fn unrank_representative_round_trips_rank() {
        // Exhaustive small patterns (varied tile values / regions).
        for pattern in [
            Pattern::new(&[1, 2, 3]),
            Pattern::new(&[1, 7, 13, 19]),
            Pattern::new(&[3, 4, 5, 9, 10]),
        ] {
            let layout = ZpdbLayout::new(pattern);
            for x in 0..layout.total() {
                let proj = layout.unrank_representative(x);
                assert_eq!(layout.rank(&proj, pattern), x, "round-trip failed at {} (k={})", x, layout.k());
            }
        }
        // k=7 (production size): sample ~200k entries spread across the range,
        // plus the first/last few.
        let pattern = Pattern::new(&[1, 2, 3, 6, 7, 8, 11]);
        let layout = ZpdbLayout::new(pattern);
        let total = layout.total();
        let step = total / 200_000;
        let mut x = 0u64;
        while x < total {
            let proj = layout.unrank_representative(x);
            assert_eq!(layout.rank(&proj, pattern), x, "k=7 round-trip failed at {x}");
            x += step;
        }
        for x in [total - 1, total - 2, total - 3] {
            assert_eq!(layout.rank(&layout.unrank_representative(x), pattern), x);
        }
    }
}
