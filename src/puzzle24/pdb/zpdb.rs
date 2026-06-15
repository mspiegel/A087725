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

#[cfg(test)]
mod tests {
    use super::*;

    /// Iterate all `C(25, k)` k-cell occupancy masks, calling `f(mask)`.
    fn for_each_ksubset(k: usize, mut f: impl FnMut(u32)) {
        let mut idx = [0usize; 8];
        for (i, slot) in idx.iter_mut().enumerate().take(k) {
            *slot = i;
        }
        loop {
            let mut mask = 0u32;
            for &c in &idx[..k] {
                mask |= 1u32 << c;
            }
            f(mask);
            // Advance the combination (lexicographic).
            let mut i = k as isize - 1;
            while i >= 0 && idx[i as usize] == N_CELLS - k + i as usize {
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
        for_each_ksubset(6, |mask| {
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
