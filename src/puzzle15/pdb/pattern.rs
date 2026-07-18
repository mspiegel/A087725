//! Patterns, projected states, and ranking for 15-puzzle pattern databases.
//!
//! A [`Pattern`] is a subset of the 15-puzzle's tile values `{1..=15}` packed
//! into a `u32` bitmask. A [`ProjectedState`] is a 15-puzzle state with
//! non-pattern tile values replaced by the [`ANON`] sentinel; we track the
//! positions of pattern tiles and the blank (the latter is needed during BFS
//! to enumerate legal moves but is *not* part of the PDB index).
//!
//! # Ranking conventions
//!
//! We follow Korf (1997): the PDB is indexed by the positions of the `k`
//! pattern tiles only — **not** by the blank's position. For a configuration
//! of pattern tiles, the stored PDB value is the *minimum* number of moves to
//! bring those tiles home, taken over all possible blank positions. This is
//! still an admissible lower bound on the full puzzle's distance.
//!
//! Two ranks are exposed on [`ProjectedState`]:
//!
//! - [`rank`](ProjectedState::rank) — uses only the `k` pattern-tile positions
//!   (mixed-radix bases `(16, 15, …, 16-k+1)`). Size `P(16, k) = 16!/(16-k)!`.
//!   This is the **PDB index** used at IDA\* query time.
//! - [`rank_with_blank`](ProjectedState::rank_with_blank) — includes the
//!   blank as a `(k+1)`-th element. Size `P(16, k+1)`. Used only during BFS
//!   construction, as the BFS-visited tracker (different blank positions for
//!   the same pattern-tile config are distinct BFS nodes).
//!
//! For the canonical Korf 7-8 partition:
//! - 7-tile pattern: PDB size `P(16, 7) = 57,657,600`; BFS space `P(16, 8) = 518,918,400`.
//! - 8-tile pattern: PDB size `P(16, 8) = 518,918,400`; BFS space `P(16, 9) = 4,151,347,200`.
//!
//! # Hot-path optimization
//!
//! [`ProjectedState`] carries a `pos_of: [u8; 16]` auxiliary array mapping
//! each tracked value (blank + pattern tiles) to its current cell, updated
//! incrementally inside [`apply`](ProjectedState::apply). This makes the
//! PDB-index `rank` `O(k)` instead of the `O(k · 16)` linear scan the puzzle8
//! code uses. At `k = 8` and billions of IDA\* node expansions per antipode,
//! the saving is material.

use crate::puzzle15::state::{Move, MoveSet, State, GOAL, N_CELLS, W};

/// Sentinel tile value for non-pattern tiles in a [`ProjectedState`].
///
/// Chosen as `0xFF` so it's distinct from real tile values `0..=15` and easy
/// to recognise in debug dumps.
pub const ANON: u8 = 0xFF;

/// Subset of the 15-puzzle's tile values `{1..=15}`.
///
/// Stored as a `u32` bitmask: bit `i` set ⇔ tile `i` is in the pattern. Bit
/// `0` (the blank) is never set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pattern(pub u32);

impl Pattern {
    /// Build a pattern from a slice of tile values. Tile values must be in
    /// `1..=15` and unique; the order in `tiles` does not matter (the pattern
    /// is a set).
    pub fn new(tiles: &[u8]) -> Self {
        let mut bits = 0u32;
        for &t in tiles {
            assert!((1..=15).contains(&t), "tile {t} out of range 1..=15");
            let mask = 1u32 << t;
            assert_eq!(bits & mask, 0, "tile {t} appears more than once");
            bits |= mask;
        }
        Pattern(bits)
    }

    pub fn empty() -> Self {
        Pattern(0)
    }

    pub fn contains(self, tile: u8) -> bool {
        debug_assert!(tile <= 15);
        self.0 & (1u32 << tile) != 0
    }

    /// Number of tiles in the pattern (0..=15).
    pub fn size(self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Iterate the pattern's tile values in ascending order.
    pub fn iter(self) -> PatternIter {
        PatternIter { bits: self.0 & !1, idx: 1 }
    }

    /// PDB storage size: `P(16, k) = 16 · 15 · … · (16-k+1)` where `k` is the
    /// pattern size. The empty pattern is `1`.
    pub fn num_projected_states(self) -> u64 {
        let k = self.size() as u64;
        let mut n = 1u64;
        for i in 0..k {
            n *= (N_CELLS as u64) - i;
        }
        n
    }

    /// BFS-visited state space size: `P(16, k+1) = 16 · 15 · … · (16-k)`.
    /// This is `N_CELLS · num_projected_states()`, but presented in its own
    /// name to make build-vs-storage distinction visible.
    pub fn num_bfs_states(self) -> u64 {
        self.num_projected_states() * (N_CELLS as u64 - self.size() as u64)
    }

    /// `true` iff `self` and `other` share no tile values. Disjoint patterns
    /// can be combined additively into an admissible heuristic (Korf & Felner
    /// 2002).
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
        while self.idx <= 15 {
            let i = self.idx;
            self.idx += 1;
            if self.bits & (1u32 << i) != 0 {
                return Some(i);
            }
        }
        None
    }
}

/// A [`State`] projected through a pattern: non-pattern tile values are
/// replaced by [`ANON`]. The position of the blank and of each pattern tile
/// is preserved in `cells` and indexed in `pos_of`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProjectedState {
    /// Row-major board: pattern tile values, the blank (`0`), and [`ANON`]
    /// elsewhere.
    pub cells: [u8; N_CELLS],
    /// `pos_of[t]` is the current cell index of tile value `t`. Only defined
    /// for `t == 0` (blank) and `t` in the pattern; other entries are
    /// undefined and must not be read.
    pos_of: [u8; N_CELLS],
}

impl ProjectedState {
    /// Project `s` through `pattern`: keep pattern tile values and the blank;
    /// replace everything else with [`ANON`]. Computes `pos_of` for the
    /// tracked tiles.
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
                // pos_of[v] is undefined for non-pattern tiles.
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

    /// Cell holding tile value `v` (or the blank for `v == 0`), via the tracked
    /// `pos_of` map — O(1), no board scan.
    #[inline]
    pub fn pos_of(&self, v: u8) -> u8 {
        self.pos_of[v as usize]
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

    /// Apply `m`. Returns the new projected state and the projected-edge
    /// cost: `1` if the move swapped the blank with a pattern tile, `0` if
    /// the swap involved an anonymous filler.
    pub fn apply(&self, m: Move) -> (Self, u8) {
        let b = self.blank_pos() as usize;
        let br = b / W;
        let bc = b % W;
        let (nr, nc) = match m {
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
        };
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

    /// In-place variant of [`apply`](Self::apply) for make/unmake search: mutate
    /// this board by `m` without allocating a new one (no `cells`/`pos_of` copy).
    ///
    /// Since `apply` is an involution under move inversion, `apply_in_place(m)`
    /// followed by `apply_in_place(m.inverse())` restores the original — that is
    /// how a mutable-context search undoes a move on backtrack. The projected
    /// edge cost is discarded (the heuristic value doesn't use it).
    #[inline]
    pub fn apply_in_place(&mut self, m: Move) {
        let b = self.pos_of[0] as usize;
        let br = b / W;
        let bc = b % W;
        let (nr, nc) = match m {
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
        };
        let n = nr * W + nc;
        let swapped = self.cells[n];
        self.cells.swap(b, n);
        self.pos_of[0] = n as u8;
        if swapped != ANON {
            self.pos_of[swapped as usize] = b as u8;
        }
    }

    /// PDB index in `[0, pattern.num_projected_states())`.
    ///
    /// Visits the `k` pattern tiles in ascending tile-value order, recording
    /// each one's position among the still-available cells. The blank is
    /// **not** included — the PDB is indexed only by pattern-tile positions.
    /// Uses the precomputed `pos_of` table for O(k) lookups.
    #[inline]
    pub fn rank(&self, pattern: Pattern) -> u64 {
        let mut available: u32 = (1u32 << N_CELLS) - 1; // bits 0..=15 set
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

    /// BFS-visited index in `[0, pattern.num_bfs_states())`.
    ///
    /// Like [`rank`](Self::rank) but additionally encodes the blank's position
    /// as a `(k+1)`-th element. Used as the visited-set key during PDB
    /// construction so that two states with identical pattern-tile positions
    /// but distinct blank positions are tracked as separate BFS nodes.
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

        // Blank as the (k+1)-th element.
        let bp = self.blank_pos() as u32;
        let lo = (1u32 << bp) - 1;
        let d = (available & lo).count_ones() as u64;
        rank = rank * radix + d;

        rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_construction_and_membership() {
        let p = Pattern::new(&[1, 3, 5, 15]);
        assert_eq!(p.size(), 4);
        for t in 0..=15u8 {
            assert_eq!(p.contains(t), [1, 3, 5, 15].contains(&t));
        }
    }

    #[test]
    fn pattern_iterates_in_ascending_order() {
        let p = Pattern::new(&[5, 1, 15, 3, 8]);
        let tiles: Vec<u8> = p.iter().collect();
        assert_eq!(tiles, vec![1, 3, 5, 8, 15]);
    }

    #[test]
    fn pattern_blank_never_in_iter() {
        let p = Pattern::new(&[1, 2, 3]);
        assert!(!p.contains(0));
        assert!(!p.iter().any(|t| t == 0));
    }

    #[test]
    fn empty_pattern() {
        let p = Pattern::empty();
        assert_eq!(p.size(), 0);
        assert_eq!(p.iter().count(), 0);
        // No pattern tiles: one PDB entry (the empty configuration). BFS
        // space sees the blank in 16 positions.
        assert_eq!(p.num_projected_states(), 1);
        assert_eq!(p.num_bfs_states(), 16);
    }

    #[test]
    #[should_panic]
    fn duplicate_tile_panics() {
        let _ = Pattern::new(&[1, 2, 1]);
    }

    #[test]
    #[should_panic]
    fn out_of_range_tile_panics() {
        let _ = Pattern::new(&[16]);
    }

    #[test]
    fn num_projected_states_korf_partition_sizes() {
        // Korf P7: 16·15·14·13·12·11·10 = 57_657_600.
        let p7 = Pattern::new(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(p7.num_projected_states(), 57_657_600);
        // Korf P8: ·9 = 518_918_400.
        let p8 = Pattern::new(&[8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(p8.num_projected_states(), 518_918_400);
        // BFS state spaces include the blank.
        assert_eq!(p7.num_bfs_states(), 518_918_400);
        assert_eq!(p8.num_bfs_states(), 4_151_347_200);
    }

    #[test]
    fn num_projected_states_small_cases() {
        assert_eq!(Pattern::new(&[1]).num_projected_states(), 16);
        assert_eq!(Pattern::new(&[1, 2]).num_projected_states(), 16 * 15);
        assert_eq!(Pattern::new(&[1, 2, 3, 4]).num_projected_states(), 16 * 15 * 14 * 13);
    }

    #[test]
    fn disjoint_check() {
        let a = Pattern::new(&[1, 2, 3, 4, 5, 6, 7]);
        let b = Pattern::new(&[8, 9, 10, 11, 12, 13, 14, 15]);
        let c = Pattern::new(&[5, 6, 7]);
        assert!(a.is_disjoint(b));
        assert!(b.is_disjoint(a));
        assert!(!a.is_disjoint(c));
        assert!(!c.is_disjoint(a));
        assert!(b.is_disjoint(c));
    }

    #[test]
    fn projection_keeps_pattern_tiles_drops_others() {
        let p = Pattern::new(&[1, 3, 15]);
        let proj = ProjectedState::from_state(&GOAL, p);
        assert_eq!(proj.cells[0], 1);
        assert_eq!(proj.cells[1], ANON);
        assert_eq!(proj.cells[2], 3);
        assert_eq!(proj.cells[3], ANON);
        assert_eq!(proj.cells[14], 15);
        assert_eq!(proj.cells[15], 0);
        assert_eq!(proj.pos_of[1], 0);
        assert_eq!(proj.pos_of[3], 2);
        assert_eq!(proj.pos_of[15], 14);
        assert_eq!(proj.pos_of[0], 15);
    }

    #[test]
    fn goal_projection_ranks_in_range() {
        for tiles in [
            &[1u8][..],
            &[1, 2],
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[1, 2, 3, 4, 5, 6, 7],
            &[8, 9, 10, 11, 12, 13, 14, 15],
        ] {
            let p = Pattern::new(tiles);
            let goal_proj = ProjectedState::goal(p);
            let r = goal_proj.rank(p);
            assert!(r < p.num_projected_states(),
                "rank {} >= {} for {:?}", r, p.num_projected_states(), tiles);
            let r_full = goal_proj.rank_with_blank(p);
            assert!(r_full < p.num_bfs_states(),
                "rank_with_blank {} >= {} for {:?}",
                r_full, p.num_bfs_states(), tiles);
        }
    }

    #[test]
    fn rank_ignores_blank_position() {
        // Two projected states with identical pattern-tile positions but
        // different blank positions must produce identical PDB ranks. This is
        // the defining property of the no-blank PDB.
        let p = Pattern::new(&[1, 5, 10]);
        let s1 = State([1, 0, 7, 12, 4, 5, 11, 8, 2, 3, 10, 13, 6, 14, 15, 9]);
        // s1 has 1 at pos 0, 5 at pos 5, 10 at pos 10; blank at pos 1.
        let s2 = State([1, 7, 12, 4, 5, 11, 8, 2, 0, 3, 10, 13, 6, 14, 15, 9]);
        // s2 has 1 at pos 0, 5 at pos 4 (different!), so this won't share.
        // Construct one that DOES share pattern positions but differs in blank.
        let s3 = State([1, 14, 7, 12, 4, 5, 11, 8, 2, 3, 10, 13, 6, 0, 15, 9]);
        // s3 has 1@0, 5@5, 10@10; blank at 13.
        let p1 = ProjectedState::from_state(&s1, p);
        let p3 = ProjectedState::from_state(&s3, p);
        assert_eq!(p1.pos_of[1], 0);
        assert_eq!(p1.pos_of[5], 5);
        assert_eq!(p1.pos_of[10], 10);
        assert_eq!(p3.pos_of[1], 0);
        assert_eq!(p3.pos_of[5], 5);
        assert_eq!(p3.pos_of[10], 10);
        assert_ne!(p1.pos_of[0], p3.pos_of[0]); // different blank positions
        assert_eq!(p1.rank(p), p3.rank(p));     // same PDB index
        assert_ne!(p1.rank_with_blank(p), p3.rank_with_blank(p));
        let _ = (p1, p3, s2);
    }

    #[test]
    fn apply_zero_cost_when_swapping_anon() {
        // Pattern {1..11, 13, 15}: tile 12 is ANON, tile 15 is pattern.
        let p = Pattern::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]);
        let proj = ProjectedState::from_state(&GOAL, p);
        let (_, c_up) = proj.apply(Move::Up);    // swap with pos 11 = tile 12 (ANON)
        let (_, c_left) = proj.apply(Move::Left); // swap with pos 14 = tile 15 (pattern)
        assert_eq!(c_up, 0, "swap with ANON must be free");
        assert_eq!(c_left, 1, "swap with pattern tile must cost 1");
    }

    #[test]
    fn apply_then_inverse_is_identity_on_projected() {
        let p = Pattern::new(&[1, 2, 3]);
        let proj = ProjectedState::from_state(&GOAL, p);
        for m in Move::ALL {
            if proj.legal_moves().contains(m) {
                let (after, c1) = proj.apply(m);
                let (back, c2) = after.apply(m.inverse());
                assert_eq!(back, proj);
                assert_eq!(c1, c2, "move and inverse should have the same cost");
            }
        }
    }

    #[test]
    fn pos_of_is_consistent_after_apply() {
        let p = Pattern::new(&[1, 6, 11]);
        let mut proj = ProjectedState::from_state(&GOAL, p);
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        for i in 0u32..500 {
            assert_eq!(proj.cells[proj.pos_of[0] as usize], 0);
            for t in p.iter() {
                assert_eq!(proj.cells[proj.pos_of[t as usize] as usize], t);
            }
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if proj.legal_moves().contains(m) {
                    proj = proj.apply(m).0;
                    break;
                }
            }
        }
    }

    #[test]
    fn rank_with_blank_bijective_on_small_pattern_bfs() {
        // For k=2, the BFS visited space is 16·15·14 = 3360 states. Enumerate
        // them and verify rank_with_blank is a bijection into [0, 3360).
        use std::collections::{HashSet, VecDeque};
        let p = Pattern::new(&[1, 2]);
        let goal_proj = ProjectedState::goal(p);
        let mut seen_cells: HashSet<([u8; N_CELLS], u8)> = HashSet::new();
        let mut ranks: HashSet<u64> = HashSet::new();
        let mut q: VecDeque<ProjectedState> = VecDeque::new();
        q.push_back(goal_proj);
        seen_cells.insert((goal_proj.cells, goal_proj.blank_pos()));
        ranks.insert(goal_proj.rank_with_blank(p));
        while let Some(s) = q.pop_front() {
            for m in s.legal_moves().iter() {
                let (n, _) = s.apply(m);
                let key = (n.cells, n.blank_pos());
                if seen_cells.contains(&key) {
                    continue;
                }
                let r = n.rank_with_blank(p);
                assert!(r < p.num_bfs_states(),
                    "rank_with_blank {} >= {}", r, p.num_bfs_states());
                assert!(ranks.insert(r), "duplicate rank_with_blank {r}");
                seen_cells.insert(key);
                q.push_back(n);
            }
        }
        assert!(seen_cells.len() <= p.num_bfs_states() as usize);
    }

    #[test]
    fn pdb_rank_after_apply_matches_reprojection() {
        // Incremental rank (via pos_of) must match a fresh projection from
        // the full state.
        let p = Pattern::new(&[3, 7, 11, 15]);
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        let mut s = GOAL;
        let mut proj = ProjectedState::from_state(&s, p);
        for i in 0u32..500 {
            assert_eq!(proj.rank(p), ProjectedState::from_state(&s, p).rank(p));
            assert_eq!(
                proj.rank_with_blank(p),
                ProjectedState::from_state(&s, p).rank_with_blank(p)
            );
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    proj = proj.apply(m).0;
                    break;
                }
            }
        }
    }
}
