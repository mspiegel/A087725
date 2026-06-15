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

/// Binomial coefficient `C(n, k)`. Exact for the small `n ≤ 25` used here.
fn binom(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u64 = 1;
    for i in 0..k {
        // r == C(n, i); r * (n-i) / (i+1) == C(n, i+1) exactly.
        r = r * (n - i) as u64 / (i + 1) as u64;
    }
    r
}

/// `k!`.
fn factorial(k: usize) -> u64 {
    (1..=k as u64).product::<u64>().max(1)
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
/// `[0, k!)`.
fn perm_rank(perm: &[u8]) -> u64 {
    let k = perm.len();
    let mut rank = 0u64;
    for i in 0..k {
        let mut smaller = 0u64;
        for &p in &perm[(i + 1)..] {
            if p < perm[i] {
                smaller += 1;
            }
        }
        rank += smaller * factorial(k - 1 - i);
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
    /// Length `C(25, k)`, indexed by [`shape_rank`].
    cohort_base: Vec<u64>,
    /// Region count per shape, indexed by [`shape_rank`].
    counts: Vec<u8>,
    /// Region labels per shape, indexed by [`shape_rank`]. Caching these (≈4.4 MB
    /// for `k = 6`) turns each `rank`/move-gen lookup into an array index instead
    /// of a `regions()` flood-fill — essential for the 181M-state build.
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
        Self { k, kfact, cohort_base, counts, labels, total: acc }
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

    /// Map a projected state to its `(m, p, r)` index in `[0, total())`.
    ///
    /// Reads the pattern-tile cells and the blank from `proj.cells`; `pattern`
    /// supplies the ascending tile order for the permutation component.
    pub fn rank(&self, proj: &ProjectedState, pattern: Pattern) -> u64 {
        // Ascending tile-value → index, for the permutation component.
        let mut asc = [0u8; 8];
        let mut ki = 0usize;
        for t in pattern.iter() {
            asc[ki] = t;
            ki += 1;
        }
        debug_assert_eq!(ki, self.k);

        // Scan cells (ascending) → occupied cells, their tiles, the blank.
        let mut occ_cells = [0u8; 8];
        let mut perm = [0u8; 8];
        let mut n = 0usize;
        let mut blank = 0u8;
        for (c, &v) in proj.cells.iter().enumerate() {
            if v == 0 {
                blank = c as u8;
            } else if v != ANON {
                occ_cells[n] = c as u8;
                // index of tile v in ascending pattern order
                let mut idx = 0usize;
                while asc[idx] != v {
                    idx += 1;
                }
                perm[n] = idx as u8;
                n += 1;
            }
        }
        debug_assert_eq!(n, self.k);

        let sr = shape_rank(&occ_cells[..n]) as usize;
        let pr = perm_rank(&perm[..n]);
        let count = self.counts[sr] as u64;
        let r = self.labels[sr][blank as usize] as u64;
        debug_assert!(r < count);
        self.cohort_base[sr] + pr * count + r
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
                    assert!(seen.insert(r), "duplicate rank {}", r);
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
        let occ = ((1u32 << N_CELLS) - 1) & !0; // all 25 cells
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
        assert!((avg - 1.42).abs() < 0.005, "avg regions {} not ≈ 1.42", avg);
    }
}
