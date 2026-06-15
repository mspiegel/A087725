//! Patterns, projected states, and ranking for 24-puzzle additive PDBs.
//!
//! A [`Pattern`] is a subset of the tile values `{1..=24}` packed into a `u32`
//! bitmask. A [`ProjectedState`] is a 24-puzzle state with non-pattern tile
//! values replaced by [`ANON`]; we track the positions of the pattern tiles and
//! the blank (the latter is needed during BFS to enumerate moves but is *not*
//! part of the standard no-blank PDB index).
//!
//! This is the **standard additive PDB** machinery (Korf & Felner 2002): the
//! PDB is indexed by the positions of the `k` pattern tiles only; the stored
//! value is the minimum, over all blank positions, of the number of pattern-tile
//! moves to bring those tiles home. Moves swapping the blank with a non-pattern
//! ("anon") tile cost 0; moves swapping with a pattern tile cost 1.
//!
//! It serves two roles for the 24-puzzle work: (1) a directly usable Korf-max
//! heuristic, and (2) the admissibility oracle for the zero-aware PDBs — the
//! zero-aware value is pointwise ≥ this one and ≤ the true distance.
//!
//! Two ranks are exposed on [`ProjectedState`]:
//! - [`rank`](ProjectedState::rank) — the `k` pattern-tile positions only.
//!   Size `P(25, k)`. The **PDB index** at query time.
//! - [`rank_with_blank`](ProjectedState::rank_with_blank) — adds the blank as a
//!   `(k+1)`-th element. Size `P(25, k+1)`. The BFS-visited key during build.

use crate::puzzle24::state::{Move, MoveSet, State, GOAL, N_CELLS, W};

/// Sentinel tile value for non-pattern tiles in a [`ProjectedState`].
pub const ANON: u8 = 0xFF;

/// Subset of the 24-puzzle's tile values `{1..=24}`, as a `u32` bitmask
/// (bit `i` set ⇔ tile `i` in the pattern; bit 0, the blank, is never set).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pattern(pub u32);

impl Pattern {
    /// Build a pattern from tile values, which must be unique and in `1..=24`.
    pub fn new(tiles: &[u8]) -> Self {
        let mut bits = 0u32;
        for &t in tiles {
            assert!((1..=24).contains(&t), "tile {} out of range 1..=24", t);
            let mask = 1u32 << t;
            assert_eq!(bits & mask, 0, "tile {} appears more than once", t);
            bits |= mask;
        }
        Pattern(bits)
    }

    pub fn empty() -> Self {
        Pattern(0)
    }

    pub fn contains(self, tile: u8) -> bool {
        debug_assert!(tile <= 24);
        self.0 & (1u32 << tile) != 0
    }

    /// Number of tiles in the pattern (0..=24).
    pub fn size(self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Iterate the pattern's tile values in ascending order.
    pub fn iter(self) -> PatternIter {
        PatternIter { bits: self.0 & !1, idx: 1 }
    }

    /// PDB storage size: `P(25, k)`. The empty pattern is `1`.
    pub fn num_projected_states(self) -> u64 {
        let k = self.size() as u64;
        let mut n = 1u64;
        for i in 0..k {
            n *= (N_CELLS as u64) - i;
        }
        n
    }

    /// BFS-visited state space size: `P(25, k+1) = (25-k) · num_projected_states`.
    pub fn num_bfs_states(self) -> u64 {
        self.num_projected_states() * (N_CELLS as u64 - self.size() as u64)
    }

    /// `true` iff `self` and `other` share no tile values (combinable additively).
    pub fn is_disjoint(self, other: Pattern) -> bool {
        self.0 & other.0 == 0
    }
}

pub struct PatternIter {
    bits: u32,
    idx: u8,
}

impl Iterator for PatternIter {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        while self.idx <= 24 {
            let i = self.idx;
            self.idx += 1;
            if self.bits & (1u32 << i) != 0 {
                return Some(i);
            }
        }
        None
    }
}

/// A [`State`] projected through a pattern: non-pattern tile values become
/// [`ANON`]. The blank's and each pattern tile's positions are tracked.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProjectedState {
    /// Row-major board: pattern tile values, the blank (`0`), and [`ANON`].
    pub cells: [u8; N_CELLS],
    /// `pos_of[t]` is the current cell of tile `t`. Defined for `t == 0` (blank)
    /// and `t` in the pattern; other entries are undefined.
    pos_of: [u8; N_CELLS],
}

impl ProjectedState {
    /// Project `s` through `pattern`.
    pub fn from_state(s: &State, pattern: Pattern) -> Self {
        let mut cells = [0u8; N_CELLS];
        let mut pos_of = [0u8; N_CELLS];
        for i in 0..N_CELLS {
            let v = s.0[i];
            if v == 0 {
                cells[i] = 0;
                pos_of[0] = i as u8;
            } else if pattern.contains(v) {
                cells[i] = v;
                pos_of[v as usize] = i as u8;
            } else {
                cells[i] = ANON;
            }
        }
        ProjectedState { cells, pos_of }
    }

    /// Projection of [`GOAL`] through `pattern`.
    pub fn goal(pattern: Pattern) -> Self {
        Self::from_state(&GOAL, pattern)
    }

    /// Build directly from a projection array (`0` = blank, [`ANON`] = filler,
    /// otherwise a pattern tile value), recomputing `pos_of`. Used by the
    /// zero-aware region BFS, which constructs successor projections by hand.
    pub fn from_projection(cells: [u8; N_CELLS]) -> Self {
        let mut pos_of = [0u8; N_CELLS];
        for (c, &v) in cells.iter().enumerate() {
            if v == 0 {
                pos_of[0] = c as u8;
            } else if v != ANON {
                pos_of[v as usize] = c as u8;
            }
        }
        ProjectedState { cells, pos_of }
    }

    #[inline]
    pub fn blank_pos(&self) -> u8 {
        self.pos_of[0]
    }

    pub fn legal_moves(&self) -> MoveSet {
        let b = self.blank_pos() as usize;
        let row = b / W;
        let col = b % W;
        let mut moves = MoveSet::empty();
        if row > 0 {
            moves.insert(Move::Up);
        }
        if row < W - 1 {
            moves.insert(Move::Down);
        }
        if col > 0 {
            moves.insert(Move::Left);
        }
        if col < W - 1 {
            moves.insert(Move::Right);
        }
        moves
    }

    /// Apply `m`. Returns the new projected state and the projected-edge cost:
    /// `1` if the blank swapped with a pattern tile, `0` for an anon filler.
    pub fn apply(&self, m: Move) -> (Self, u8) {
        let b = self.blank_pos() as usize;
        let (nr, nc) = step(b, m);
        let n = nr * W + nc;
        let swapped = self.cells[n];
        let cost = if swapped == ANON { 0 } else { 1 };
        let mut cells = self.cells;
        let mut pos_of = self.pos_of;
        cells.swap(b, n);
        pos_of[0] = n as u8;
        if swapped != ANON {
            pos_of[swapped as usize] = b as u8;
        }
        (ProjectedState { cells, pos_of }, cost)
    }

    /// PDB index in `[0, pattern.num_projected_states())` — pattern-tile
    /// positions only, ascending tile-value order, O(k) via `pos_of`.
    #[inline]
    pub fn rank(&self, pattern: Pattern) -> u64 {
        let mut available: u32 = (1u32 << N_CELLS) - 1; // bits 0..=24
        let mut rank: u64 = 0;
        let mut radix: u64 = N_CELLS as u64;
        for tile in pattern.iter() {
            let pos = self.pos_of[tile as usize] as u32;
            let lo = (1u32 << pos) - 1;
            let d = (available & lo).count_ones() as u64;
            rank = rank * radix + d;
            available &= !(1u32 << pos);
            radix -= 1;
        }
        rank
    }

    /// BFS-visited index in `[0, pattern.num_bfs_states())` — like [`rank`] but
    /// with the blank appended as a `(k+1)`-th element.
    pub fn rank_with_blank(&self, pattern: Pattern) -> u64 {
        let mut available: u32 = (1u32 << N_CELLS) - 1;
        let mut rank: u64 = 0;
        let mut radix: u64 = N_CELLS as u64;
        for tile in pattern.iter() {
            let pos = self.pos_of[tile as usize] as u32;
            let lo = (1u32 << pos) - 1;
            let d = (available & lo).count_ones() as u64;
            rank = rank * radix + d;
            available &= !(1u32 << pos);
            radix -= 1;
        }
        let bp = self.blank_pos() as u32;
        let lo = (1u32 << bp) - 1;
        let d = (available & lo).count_ones() as u64;
        rank = rank * radix + d;
        rank
    }
}

/// `(row, col)` of the blank's destination after move `m` from cell `b`.
#[inline]
fn step(b: usize, m: Move) -> (usize, usize) {
    let br = b / W;
    let bc = b % W;
    match m {
        Move::Up => {
            debug_assert!(br > 0);
            (br - 1, bc)
        }
        Move::Down => {
            debug_assert!(br < W - 1);
            (br + 1, bc)
        }
        Move::Left => {
            debug_assert!(bc > 0);
            (br, bc - 1)
        }
        Move::Right => {
            debug_assert!(bc < W - 1);
            (br, bc + 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_construction_and_membership() {
        let p = Pattern::new(&[1, 3, 5, 24]);
        assert_eq!(p.size(), 4);
        for t in 0..=24u8 {
            assert_eq!(p.contains(t), [1, 3, 5, 24].contains(&t));
        }
    }

    #[test]
    fn pattern_iterates_ascending_without_blank() {
        let p = Pattern::new(&[5, 1, 24, 3, 8]);
        assert_eq!(p.iter().collect::<Vec<_>>(), vec![1, 3, 5, 8, 24]);
        assert!(!p.iter().any(|t| t == 0));
    }

    #[test]
    #[should_panic]
    fn duplicate_tile_panics() {
        let _ = Pattern::new(&[1, 2, 1]);
    }

    #[test]
    #[should_panic]
    fn out_of_range_tile_panics() {
        let _ = Pattern::new(&[25]);
    }

    #[test]
    fn projected_state_sizes() {
        // 6-tile additive (no-blank) PDB: P(25,6) = 127,512,000.
        let p6 = Pattern::new(&[1, 2, 3, 6, 7, 8]);
        assert_eq!(p6.num_projected_states(), 127_512_000);
        // BFS space adds the blank: P(25,7) = 2,422,728,000.
        assert_eq!(p6.num_bfs_states(), 2_422_728_000);
        assert_eq!(Pattern::new(&[1]).num_projected_states(), 25);
        assert_eq!(Pattern::new(&[1, 2]).num_projected_states(), 25 * 24);
    }

    #[test]
    fn korf_6666_partition_is_disjoint_and_covers() {
        let a = Pattern::new(&[1, 2, 3, 6, 7, 8]);
        let b = Pattern::new(&[4, 5, 9, 10, 14, 15]);
        let c = Pattern::new(&[11, 12, 16, 17, 21, 22]);
        let d = Pattern::new(&[13, 18, 19, 20, 23, 24]);
        assert!(a.is_disjoint(b) && a.is_disjoint(c) && a.is_disjoint(d));
        assert!(b.is_disjoint(c) && b.is_disjoint(d) && c.is_disjoint(d));
        // Union covers all 24 tiles exactly once.
        let union = a.0 | b.0 | c.0 | d.0;
        assert_eq!(union, ((1u32 << 25) - 1) & !1, "partition must cover tiles 1..=24");
    }

    #[test]
    fn projection_keeps_pattern_drops_others() {
        let p = Pattern::new(&[1, 3, 24]);
        let proj = ProjectedState::from_state(&GOAL, p);
        assert_eq!(proj.cells[0], 1);
        assert_eq!(proj.cells[1], ANON);
        assert_eq!(proj.cells[2], 3);
        assert_eq!(proj.cells[23], 24);
        assert_eq!(proj.cells[24], 0);
        assert_eq!(proj.pos_of[1], 0);
        assert_eq!(proj.pos_of[3], 2);
        assert_eq!(proj.pos_of[24], 23);
        assert_eq!(proj.pos_of[0], 24);
    }

    #[test]
    fn goal_projection_ranks_in_range() {
        for tiles in [&[1u8][..], &[1, 2, 3, 6, 7, 8], &[13, 18, 19, 20, 23, 24]] {
            let p = Pattern::new(tiles);
            let g = ProjectedState::goal(p);
            assert!(g.rank(p) < p.num_projected_states());
            assert!(g.rank_with_blank(p) < p.num_bfs_states());
        }
    }

    #[test]
    fn apply_cost_zero_for_anon_one_for_pattern() {
        // Pattern excludes tile 20 (at pos 19, above blank) but includes tile 24
        // (at pos 23, left of blank).
        let p = Pattern::new(&[1, 2, 3, 24]);
        let proj = ProjectedState::from_state(&GOAL, p);
        let (_, c_up) = proj.apply(Move::Up); // swaps pos 19 = tile 20 (anon)
        let (_, c_left) = proj.apply(Move::Left); // swaps pos 23 = tile 24 (pattern)
        assert_eq!(c_up, 0);
        assert_eq!(c_left, 1);
    }

    #[test]
    fn rank_ignores_blank_position() {
        // Two projections with identical pattern-tile positions but different
        // blank positions share a PDB rank but differ in rank_with_blank.
        let p = Pattern::new(&[1, 7, 13]);
        // Build by moving only ANON tiles / blank in GOAL projection.
        let proj = ProjectedState::goal(p);
        let (a, _) = proj.apply(Move::Up); // blank 24→19 (tile 20 anon), pattern fixed
        let (b, _) = a.apply(Move::Up); // blank 19→14 (tile 15 anon), pattern fixed
        assert_eq!(a.pos_of[1], b.pos_of[1]);
        assert_eq!(a.pos_of[7], b.pos_of[7]);
        assert_eq!(a.pos_of[13], b.pos_of[13]);
        assert_ne!(a.blank_pos(), b.blank_pos());
        assert_eq!(a.rank(p), b.rank(p));
        assert_ne!(a.rank_with_blank(p), b.rank_with_blank(p));
    }

    #[test]
    fn incremental_rank_matches_reprojection() {
        let p = Pattern::new(&[3, 7, 11, 15, 19, 23]);
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut proj = ProjectedState::from_state(&s, p);
        for _ in 0..2000 {
            assert_eq!(proj.rank(p), ProjectedState::from_state(&s, p).rank(p));
            assert_eq!(
                proj.rank_with_blank(p),
                ProjectedState::from_state(&s, p).rank_with_blank(p)
            );
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            s = s.apply(m);
            proj = proj.apply(m).0;
        }
    }
}
