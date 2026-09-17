//! Tile placements counted and sampled by Manhattan distance.
//!
//! η of a heuristic close to Manhattan distance is carried by the rare states
//! with small Manhattan distance, which uniform draws almost never reach. Here
//! `count(m)` is the exact number of placements of tiles 1..=24 on the 25 cells
//! (any blank cell, both permutation parities) with Manhattan distance `m`, and
//! `sample(m)` draws one of them uniformly, so a stratum `MD = m` can be sampled
//! directly and weighted by its exact size.
//!
//! Solvable states are not exactly half of each level, so callers keep
//! unsolvable draws as zero-valued samples instead of rescaling by one half.
//!
//! Table: `t[S][j]` is the number of placements of tiles 1..=|S| on exactly the
//! cells of `S` with Manhattan distance `j ≤ cap`. Tile |S| sits on some cell
//! `c ∈ S`, so `t[S][j] = Σ_c t[S∖c][j − md(|S|, c)]`. Rows are laid out by
//! (|S|, colex rank of S), so each layer is contiguous and reads only the
//! previous one. Counts are `f64`: exact up to 2^53 and within ~1e-16 relative
//! beyond, far below any sampling error they are used with.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::puzzle24::eta::rng::Rng;
use crate::puzzle24::state::{State, N_CELLS, W};

/// Placement counts by Manhattan distance for a square board of any width.
struct Levels {
    width: usize,
    cap: usize,
    binom: Vec<Vec<usize>>,
    /// First row of the layer with `p` placed tiles.
    offsets: Vec<usize>,
    table: Vec<f64>,
}

fn manhattan(width: usize, tile: usize, cell: usize) -> usize {
    let home = tile - 1;
    (home / width).abs_diff(cell / width) + (home % width).abs_diff(cell % width)
}

impl Levels {
    fn build(width: usize, cap: usize) -> Levels {
        let cells = width * width;
        assert!(cells <= 25, "cell masks are u32 and table rows 2^cells");
        let mut binom = vec![vec![0usize; cells + 1]; cells + 1];
        for n in 0..=cells {
            binom[n][0] = 1;
            for k in 1..=n {
                binom[n][k] = binom[n - 1][k - 1] + if k < n { binom[n - 1][k] } else { 0 };
            }
        }
        let offsets: Vec<usize> = (0..cells)
            .scan(0usize, |acc, p| {
                let start = *acc;
                *acc += binom[cells][p];
                Some(start)
            })
            .collect();
        let rows = (1usize << cells) - 1;
        let c = cap + 1;
        let mut levels = Levels {
            width,
            cap,
            binom,
            offsets,
            table: vec![0.0; rows * c],
        };
        levels.table[0] = 1.0;
        let mut table = std::mem::take(&mut levels.table);
        for p in 1..cells {
            let (done, rest) = table.split_at_mut(levels.offsets[p] * c);
            let prev = &done[levels.offsets[p - 1] * c..];
            let cur = &mut rest[..levels.binom[cells][p] * c];
            #[cfg(feature = "parallel")]
            cur.par_chunks_mut(c)
                .enumerate()
                .for_each(|(r, row)| levels.fill_row(p, r, row, prev));
            #[cfg(not(feature = "parallel"))]
            cur.chunks_mut(c)
                .enumerate()
                .for_each(|(r, row)| levels.fill_row(p, r, row, prev));
        }
        levels.table = table;
        levels
    }

    fn cells(&self) -> usize {
        self.width * self.width
    }

    /// Colex rank of a `p`-subset: Σ_i C(c_i, i + 1) over its cells in
    /// increasing order.
    fn rank(&self, mask: u32) -> usize {
        let mut r = 0;
        let mut rest = mask;
        let mut i = 0;
        while rest != 0 {
            let cell = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            i += 1;
            r += self.binom[cell][i];
        }
        r
    }

    fn unrank(&self, p: usize, mut r: usize) -> u32 {
        let mut mask = 0u32;
        let mut top = self.cells();
        for i in (1..=p).rev() {
            let mut cell = top - 1;
            while self.binom[cell][i] > r {
                cell -= 1;
            }
            r -= self.binom[cell][i];
            mask |= 1 << cell;
            top = cell;
        }
        mask
    }

    fn fill_row(&self, p: usize, r: usize, row: &mut [f64], prev: &[f64]) {
        let c = self.cap + 1;
        let mask = self.unrank(p, r);
        let mut cells = [0usize; 25];
        let mut rest = mask;
        for slot in cells.iter_mut().take(p) {
            *slot = rest.trailing_zeros() as usize;
            rest &= rest - 1;
        }
        // Removing the j-th cell keeps the ranks of the cells below it and
        // lowers the index of each cell above it by one.
        let mut suffix = [0usize; 26];
        for j in (0..p).rev() {
            suffix[j] = suffix[j + 1]
                + if j + 1 < p {
                    self.binom[cells[j + 1]][j + 1]
                } else {
                    0
                };
        }
        let mut prefix = 0usize;
        for j in 0..p {
            let d = manhattan(self.width, p, cells[j]);
            if d <= self.cap {
                let base = (prefix + suffix[j]) * c;
                for (out, src) in row[d..].iter_mut().zip(&prev[base..base + c - d]) {
                    *out += src;
                }
            }
            prefix += self.binom[cells[j]][j + 1];
        }
    }

    fn row(&self, mask: u32) -> &[f64] {
        let c = self.cap + 1;
        let i = self.offsets[mask.count_ones() as usize] + self.rank(mask);
        &self.table[i * c..(i + 1) * c]
    }

    fn full(&self) -> u32 {
        ((1u64 << self.cells()) - 1) as u32
    }

    fn count(&self, m: usize) -> f64 {
        assert!(
            m <= self.cap,
            "Manhattan distance {m} above table cap {}",
            self.cap
        );
        (0..self.cells())
            .map(|blank| self.row(self.full() ^ (1 << blank))[m])
            .sum()
    }

    /// Uniform placement with Manhattan distance `m`: `out[cell] = tile`, 0 for
    /// the blank. Panics if the level is empty.
    fn sample(&self, m: usize, rng: &mut Rng) -> Vec<u8> {
        let total = self.count(m);
        assert!(total > 0.0, "no placement has Manhattan distance {m}");
        let cells = self.cells();
        let mut out = vec![0u8; cells];
        let blank = pick(
            (0..cells).map(|b| (b, self.row(self.full() ^ (1 << b))[m])),
            total,
            rng,
        );
        let mut mask = self.full() ^ (1 << blank);
        let mut j = m;
        for tile in (1..cells).rev() {
            let here = self.row(mask)[j];
            let bits = mask;
            let candidates = (0..cells)
                .filter(move |&c| bits >> c & 1 == 1)
                .filter_map(|cell| {
                    let d = manhattan(self.width, tile, cell);
                    (d <= j).then(|| (cell, self.row(mask ^ (1 << cell))[j - d]))
                });
            let cell = pick(candidates, here, rng);
            out[cell] = tile as u8;
            mask ^= 1 << cell;
            j -= manhattan(self.width, tile, cell);
        }
        debug_assert_eq!(j, 0);
        out
    }
}

/// Draw an item with probability weight / total. Floating-point rounding can
/// leave the draw past the last weight; it then takes the last nonzero item.
fn pick(items: impl Iterator<Item = (usize, f64)>, total: f64, rng: &mut Rng) -> usize {
    let mut target = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * total;
    let mut last = None;
    for (item, weight) in items {
        if weight <= 0.0 {
            continue;
        }
        last = Some(item);
        if target < weight {
            return item;
        }
        target -= weight;
    }
    last.expect("a level with nonzero count has a nonzero branch")
}

/// Placement counts and uniform sampling by Manhattan distance on the
/// 24-puzzle, for distances up to a cap. The table holds (2^25 − 1)·(cap + 1)
/// `f64`s: 10.2 GiB at cap 40.
pub struct MdLevels {
    levels: Levels,
}

impl MdLevels {
    pub fn build(cap: u8) -> MdLevels {
        MdLevels {
            levels: Levels::build(W, cap as usize),
        }
    }

    pub fn cap(&self) -> u8 {
        self.levels.cap as u8
    }

    pub fn table_bytes(&self) -> usize {
        self.levels.table.len() * std::mem::size_of::<f64>()
    }

    /// Placements (any blank cell, either parity) with Manhattan distance `m`.
    pub fn count(&self, m: u8) -> f64 {
        self.levels.count(m as usize)
    }

    /// A uniform placement with Manhattan distance `m`, solvable or not.
    pub fn sample(&self, m: u8, rng: &mut Rng) -> State {
        let cells = self.levels.sample(m as usize, rng);
        let mut s = [0u8; N_CELLS];
        s.copy_from_slice(&cells);
        State(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Manhattan-distance histogram of all placements with distance ≤ cap, by
    /// depth-first assignment of tiles within the remaining budget.
    fn enumerate(width: usize, cap: usize) -> Vec<u64> {
        fn go(width: usize, tile: usize, used: u32, spent: usize, cap: usize, hist: &mut [u64]) {
            let cells = width * width;
            if tile == cells {
                hist[spent] += 1;
                return;
            }
            for cell in 0..cells {
                let d = manhattan(width, tile, cell);
                if used >> cell & 1 == 0 && spent + d <= cap {
                    go(width, tile + 1, used | 1 << cell, spent + d, cap, hist);
                }
            }
        }
        let mut hist = vec![0u64; cap + 1];
        go(width, 1, 0, 0, cap, &mut hist);
        hist
    }

    #[test]
    fn rank_and_unrank_are_inverse_on_every_subset() {
        for width in [2, 3, 4] {
            let levels = Levels::build(width, 0);
            let cells = width * width;
            let mut seen: Vec<Vec<bool>> = (0..=cells)
                .map(|p| vec![false; levels.binom[cells][p]])
                .collect();
            for mask in 0u32..1 << cells {
                let p = mask.count_ones() as usize;
                let r = levels.rank(mask);
                assert!(!seen[p][r], "rank collision");
                seen[p][r] = true;
                assert_eq!(levels.unrank(p, r), mask);
            }
        }
    }

    #[test]
    fn counts_match_enumeration_on_3x3_and_4x4() {
        let levels = Levels::build(3, 20);
        for (m, &n) in enumerate(3, 20).iter().enumerate() {
            assert_eq!(levels.count(m), n as f64, "3x3, m = {m}");
        }
        let levels = Levels::build(4, 9);
        for (m, &n) in enumerate(4, 9).iter().enumerate() {
            assert_eq!(levels.count(m), n as f64, "4x4, m = {m}");
        }
    }

    #[test]
    fn samples_are_uniform_within_a_level() {
        let levels = Levels::build(3, 8);
        let m = 6;
        let n = levels.count(m) as usize;
        let draws = 200 * n;
        let mut rng = Rng::stream(7, 7, 7);
        let mut freq: HashMap<Vec<u8>, u64> = HashMap::new();
        for _ in 0..draws {
            let cells = levels.sample(m, &mut rng);
            let md: usize = (0..9)
                .filter(|&c| cells[c] != 0)
                .map(|c| manhattan(3, cells[c] as usize, c))
                .sum();
            assert_eq!(md, m);
            *freq.entry(cells).or_default() += 1;
        }
        assert_eq!(freq.len(), n, "every placement is reached");
        // Chi-square with n − 1 degrees of freedom: mean n − 1, sd sqrt(2(n − 1));
        // 6 sd is a false-failure rate below 1e-8.
        let e = draws as f64 / n as f64;
        let chi2: f64 = freq.values().map(|&f| (f as f64 - e).powi(2) / e).sum();
        let df = (n - 1) as f64;
        assert!(chi2 < df + 6.0 * (2.0 * df).sqrt(), "chi2 {chi2}, df {df}");
    }

    #[test]
    #[ignore = "builds a 1.0 GiB table; ~2 s; cargo test md_levels_24 -- --ignored"]
    fn md_levels_24_small_levels_match_enumeration() {
        use crate::puzzle24::search::{Heuristic, ManhattanHeuristic};
        let levels = MdLevels::build(3);
        for (m, &n) in enumerate(W, 3).iter().enumerate() {
            assert_eq!(levels.count(m as u8), n as f64, "m = {m}");
        }
        let mut rng = Rng::stream(8, 8, 8);
        for m in 0..=3 {
            for _ in 0..100 {
                assert_eq!(ManhattanHeuristic.h(&levels.sample(m, &mut rng)), m);
            }
        }
    }
}
