//! Zero-aware PDB primitives for the 15-puzzle — the geometric core of the
//! `(m, p, r)` index. Ported from `puzzle24::pdb::zpdb` (Clausecker &
//! Reinefeld, SOCS 2019) with the 4×4 board constants (`N_CELLS = 16`,
//! `W = 4`, tile range `1..=15`, `C(16, k)`).
//!
//! A **zero-aware** PDB indexes not just the pattern-tile positions but also
//! *which zero-tile region the blank lies in* — the connected component
//! (4-adjacency) of the cells **not** occupied by pattern tiles. Collapsing
//! all blank positions within one region into a single index contracts the
//! 0-cost ("anon") moves, leaving a quotient graph where **every edge costs 1**
//! (a pattern-tile move). The resulting value dominates the blank-agnostic
//! additive PDB while remaining admissible.
//!
//! # 4×4 vs 5×5 note
//!
//! The 24-puzzle codec recovers `h`'s parity bit from the parity of the sum of
//! pattern-tile cell indices, which flips on every edge there because both
//! horizontal (±1) and vertical (±5) moves are odd offsets. On the **4×4**
//! board the vertical offset is ±4 (even), so that cell-sum parity argument
//! does **not** hold and the 1-bit codec would be unsound. We therefore store
//! the raw distance bytes (see `zdb.rs`) and do not port `shape_parity` / the
//! 1-bit codec. The geometry below (regions, ranking) is metric-agnostic and
//! ports verbatim.

use super::pattern::{Pattern, ProjectedState};
use crate::puzzle15::state::{N_CELLS, W};

/// Marker in a region labeling for a cell occupied by a pattern tile.
pub const OCCUPIED: u8 = 0xFF;

/// 4-neighbours of cell `c` on the 4×4 board, written into `out`; returns the
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
/// pattern tiles (`occupied` is a 16-bit mask, bit `c` set ⇔ cell `c` holds a
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

/// Pascal's-triangle table for `C(n, k)` with `n ≤ 16`, `k ≤ 8` — the only
/// arguments `shape_rank`/`rank` ever use (and big enough for `C(16, 8)`).
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

/// Binomial coefficient `C(n, k)`. Exact for the small `n ≤ 16` used here; an
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
/// `Σ_i C(c_i, i+1)`, a bijection onto `[0, C(16, k))`.
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

/// Enumerate every `k`-subset of the 16 cells, calling `f(sorted_cells, mask)`.
/// Cells are yielded ascending; iteration order is lexicographic.
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
/// where `cohort_base[s] = Σ_{s' < s} k! · regions(shape s')`. The total is
/// `k! · Σ_m regions(m)`.
pub struct ZpdbLayout {
    k: usize,
    kfact: u64,
    /// `slot_of[v]` = the permutation slot of tile value `v` (its index in the
    /// pattern's iteration order). Precomputed so `rank` skips the per-call
    /// `asc` rebuild and the O(k²) tile-value search.
    slot_of: [u8; N_CELLS],
    /// The pattern's tile values in slot order (`tiles[j]` has slot `j`). Lets
    /// `rank` find the occupied cells via `pos_of` in O(k) — no full-board scan.
    tiles: [u8; 8],
    /// Length `C(16, k)`, indexed by [`shape_rank`].
    cohort_base: Vec<u64>,
    /// Region count per shape, indexed by [`shape_rank`].
    counts: Vec<u8>,
    /// Region labels per shape, indexed by [`shape_rank`].
    labels: Vec<[u8; N_CELLS]>,
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

        // Permutation slot of each tile value, in pattern iteration order — the
        // same indexing the old `while asc[idx] != v` search produced — plus the
        // inverse map (slot -> tile value) for the O(k) occupied-cell lookup.
        let mut slot_of = [0u8; N_CELLS];
        let mut tiles = [0u8; 8];
        let mut slot = 0u8;
        for t in pattern.iter() {
            slot_of[t as usize] = slot;
            tiles[slot as usize] = t;
            slot += 1;
        }

        let mut counts = vec![0u8; nshapes];
        let mut labels = vec![[OCCUPIED; N_CELLS]; nshapes];
        for_each_ksubset(k, |cells, mask| {
            let sr = shape_rank(cells) as usize;
            let (cnt, lab) = regions(mask);
            counts[sr] = cnt;
            labels[sr] = lab;
        });

        let mut cohort_base = vec![0u64; nshapes];
        let mut acc = 0u64;
        for s in 0..nshapes {
            cohort_base[s] = acc;
            acc += kfact * counts[s] as u64;
        }
        Self { k, kfact, slot_of, tiles, cohort_base, counts, labels, total: acc }
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
    /// first entry of shape `s`. Length `C(16, k)`.
    pub fn cohort_base(&self) -> &[u64] {
        &self.cohort_base
    }

    /// Region count per shape, indexed by `shape_rank`. Length `C(16, k)`.
    pub fn region_counts(&self) -> &[u8] {
        &self.counts
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
        // cell's tile slot (cells live in one 16-byte line, all in L1).
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::state::State;

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

    fn make_proj(pattern: Pattern, placement: &[(u8, u8)], blank: u8) -> ProjectedState {
        let mut cells = [255u8; N_CELLS];
        for &(c, t) in placement {
            cells[c as usize] = t;
        }
        cells[blank as usize] = 0;
        let mut fill = (1..=15u8).filter(|t| !pattern.contains(*t));
        for v in cells.iter_mut() {
            if *v == 255 {
                *v = fill.next().unwrap();
            }
        }
        ProjectedState::from_state(&State(cells), pattern)
    }

    fn assert_bijection(pattern: Pattern) {
        use std::collections::HashSet;
        let layout = ZpdbLayout::new(pattern);
        let k = pattern.size() as usize;
        let tiles: Vec<u8> = pattern.iter().collect();
        let mut seen: HashSet<u64> = HashSet::new();
        for_each_ksubset(k, |cells, mask| {
            let (count, labels) = regions(mask);
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
        assert_eq!(binom(16, 7), 11_440);
        assert_eq!(binom(16, 8), 12_870);
        assert_eq!(factorial(6), 720);
        assert_eq!(factorial(0), 1);
    }

    #[test]
    fn shape_rank_is_bijection_small() {
        use std::collections::HashSet;
        for k in [1usize, 2, 3, 4, 5, 6, 7, 8] {
            let mut seen: HashSet<u64> = HashSet::new();
            for_each_ksubset(k, |cells, _| {
                let r = shape_rank(cells);
                assert!(r < binom(N_CELLS, k), "shape_rank {r} >= C(16,{k}) for {cells:?}");
                assert!(seen.insert(r), "shape_rank COLLISION at {r} for cells {cells:?} (k={k})");
            });
            assert_eq!(seen.len() as u64, binom(N_CELLS, k), "shape_rank not surjective at k={k}");
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
    fn empty_occupancy_is_one_region() {
        let (n, label) = regions(0);
        assert_eq!(n, 1);
        assert!(label.iter().all(|&l| l == 0));
    }

    #[test]
    fn full_occupancy_is_zero_regions() {
        let occ = (1u32 << N_CELLS) - 1; // all 16 cells
        let (n, label) = regions(occ);
        assert_eq!(n, 0);
        assert!(label.iter().all(|&l| l == OCCUPIED));
    }

    #[test]
    fn labels_are_contiguous_and_min_ordered() {
        // A vertical wall down column 1 (cells 1,5,9,13) splits the board into a
        // left region (contains cell 0) and a right region (contains cell 2).
        let mut occ = 0u32;
        for c in [1, 5, 9, 13] {
            occ |= 1u32 << c;
        }
        let (n, label) = regions(occ);
        assert_eq!(n, 2);
        assert_eq!(label[0], 0, "smallest free cell must be in region 0");
        assert_eq!(label[2], 1, "the far side must be region 1");
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
        // Wall off the top-left corner (cell 0) with cells 1 and 4.
        let occ = (1u32 << 1) | (1u32 << 4);
        let (n, label) = regions(occ);
        assert_eq!(n, 2);
        assert_eq!(label[0], 0);
        assert_eq!(label.iter().filter(|&&l| l == 0).count(), 1);
    }
}
