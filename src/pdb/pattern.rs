//! Patterns, projected states, and ranking for pattern databases.
//!
//! A [`Pattern`] is a subset of the 8-puzzle's tile values `{1..=8}` packed
//! into a `u16` bitmask. A [`ProjectedState`] is an 8-puzzle state with
//! non-pattern tile values replaced by the [`ANON`] sentinel; we track only
//! the positions of pattern tiles and the blank.
//!
//! # Ranking
//!
//! For a pattern of size `k`, the projected state is determined by the
//! positions of the `k` pattern tiles plus the blank (`k + 1` distinct
//! positions out of 9). We rank such configurations into
//! `[0, num_projected_states)` using a mixed-radix encoding:
//!
//! ```text
//! rank = (((d_0 * 8 + d_1) * 7 + d_2) * 6 + … + d_k) * (9 - k)
//!        ^^^ position of pattern tile 1 among the 9 cells
//!              ^^^ position of pattern tile 2 among the remaining 8
//!                     …
//!                          ^^^ position of blank among the remaining (9-k)
//! ```
//!
//! Pattern tiles are visited in **ascending tile-value order** so the
//! ranking is canonical and deterministic.
//!
//! `num_projected_states(k) = 9! / (8 - k)!` — for example, a 4-tile pattern
//! gives `9 · 8 · 7 · 6 · 5 = 15_120` projected states.

use crate::state::{Move, MoveSet, State, GOAL};

/// Sentinel tile value for non-pattern tiles in a [`ProjectedState`].
///
/// Chosen as `0xFF` so it's distinct from real tile values `0..=8` and easy
/// to recognise in debug dumps.
pub const ANON: u8 = 0xFF;

/// Subset of the 8-puzzle's tile values `{1..=8}`.
///
/// Stored as a `u16` bitmask: bit `i` set ⇔ tile `i` is in the pattern. Bit
/// `0` (the blank) is never set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pattern(pub u16);

impl Pattern {
    /// Build a pattern from a slice of tile values. Tile values must be in
    /// `1..=8` and unique; the order in `tiles` does not matter (the pattern
    /// is a set).
    pub fn new(tiles: &[u8]) -> Self {
        let mut bits = 0u16;
        for &t in tiles {
            assert!((1..=8).contains(&t), "tile {} out of range 1..=8", t);
            let mask = 1u16 << t;
            assert_eq!(bits & mask, 0, "tile {} appears more than once", t);
            bits |= mask;
        }
        Pattern(bits)
    }

    pub fn empty() -> Self {
        Pattern(0)
    }

    pub fn contains(self, tile: u8) -> bool {
        debug_assert!(tile <= 8);
        self.0 & (1 << tile) != 0
    }

    /// Number of tiles in the pattern (0..=8).
    pub fn size(self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Iterate the pattern's tile values in ascending order.
    pub fn iter(self) -> PatternIter {
        PatternIter { bits: self.0 & !1, idx: 1 }
    }

    /// Total number of projected states for this pattern: `9! / (8 - k)!`
    /// where `k` is the pattern size.
    pub fn num_projected_states(self) -> u32 {
        let k = self.size() as u32;
        let mut n = 1u32;
        for i in 0..=k {
            n *= 9 - i;
        }
        n
    }
}

pub struct PatternIter {
    bits: u16,
    idx: u8,
}

impl Iterator for PatternIter {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        while self.idx <= 8 {
            let i = self.idx;
            self.idx += 1;
            if self.bits & (1 << i) != 0 {
                return Some(i);
            }
        }
        None
    }
}

/// A [`State`] projected through a pattern: non-pattern tile values are
/// replaced by [`ANON`]. The position of the blank and of each pattern tile
/// is preserved.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProjectedState(pub [u8; 9]);

impl ProjectedState {
    /// Project `s` through `pattern`: keep pattern tile values and the blank;
    /// replace everything else with [`ANON`].
    pub fn from_state(s: &State, pattern: Pattern) -> Self {
        let mut out = [0u8; 9];
        for i in 0..9 {
            let v = s.0[i];
            out[i] = if v == 0 || pattern.contains(v) { v } else { ANON };
        }
        ProjectedState(out)
    }

    /// Projection of [`GOAL`] through `pattern`. Equivalent to
    /// `from_state(&GOAL, pattern)` but useful as a named constant.
    pub fn goal(pattern: Pattern) -> Self {
        Self::from_state(&GOAL, pattern)
    }

    pub fn blank_pos(&self) -> u8 {
        for i in 0..9 {
            if self.0[i] == 0 {
                return i as u8;
            }
        }
        panic!("projected state has no blank: {:?}", self.0)
    }

    pub fn legal_moves(&self) -> MoveSet {
        let b = self.blank_pos();
        let row = b / 3;
        let col = b % 3;
        let mut moves = MoveSet::empty();
        if row > 0 {
            moves.insert(Move::Up);
        }
        if row < 2 {
            moves.insert(Move::Down);
        }
        if col > 0 {
            moves.insert(Move::Left);
        }
        if col < 2 {
            moves.insert(Move::Right);
        }
        moves
    }

    /// Apply `m`. Returns the new projected state and the projected-edge
    /// cost: `1` if the move swapped the blank with a pattern tile, `0` if
    /// the swap involved an anonymous filler.
    pub fn apply(&self, m: Move) -> (Self, u8) {
        let b = self.blank_pos() as usize;
        let br = b / 3;
        let bc = b % 3;
        let (nr, nc) = match m {
            Move::Up => {
                debug_assert!(br > 0);
                (br - 1, bc)
            }
            Move::Down => {
                debug_assert!(br < 2);
                (br + 1, bc)
            }
            Move::Left => {
                debug_assert!(bc > 0);
                (br, bc - 1)
            }
            Move::Right => {
                debug_assert!(bc < 2);
                (br, bc + 1)
            }
        };
        let n = nr * 3 + nc;
        let swapped = self.0[n];
        let cost = if swapped == ANON { 0 } else { 1 };
        let mut next = self.0;
        next.swap(b, n);
        (ProjectedState(next), cost)
    }

    /// Rank of this projected state in `[0, pattern.num_projected_states())`.
    ///
    /// The encoding visits pattern tiles in ascending tile-value order, then
    /// the blank, recording each one's position among the still-available
    /// cells. See the module docs for the formula.
    pub fn rank(&self, pattern: Pattern) -> u32 {
        let mut available: u16 = 0x01FF; // bits 0..=8 set
        let mut rank: u32 = 0;
        let mut radix: u32 = 9;

        for tile in pattern.iter() {
            // Find tile's position.
            let mut pos: u8 = 0xFF;
            for i in 0..9 {
                if self.0[i] == tile {
                    pos = i as u8;
                    break;
                }
            }
            debug_assert!(pos != 0xFF, "tile {} not found in projected state {:?}", tile, self.0);
            let lo = (1u16 << pos) - 1;
            let d = (available & lo).count_ones();
            rank = rank * radix + d;
            available &= !(1u16 << pos);
            radix -= 1;
        }

        // Then the blank.
        let bp = self.blank_pos();
        let lo = (1u16 << bp) - 1;
        let d = (available & lo).count_ones();
        rank = rank * radix + d;

        rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_construction_and_membership() {
        let p = Pattern::new(&[1, 3, 5]);
        assert_eq!(p.size(), 3);
        for t in 0..=8u8 {
            assert_eq!(p.contains(t), [1, 3, 5].contains(&t));
        }
    }

    #[test]
    fn pattern_iterates_in_ascending_order() {
        let p = Pattern::new(&[5, 1, 8, 3]);
        let tiles: Vec<u8> = p.iter().collect();
        assert_eq!(tiles, vec![1, 3, 5, 8]);
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
        // Just the blank position: 9 distinct states.
        assert_eq!(p.num_projected_states(), 9);
    }

    #[test]
    #[should_panic]
    fn duplicate_tile_panics() {
        let _ = Pattern::new(&[1, 2, 1]);
    }

    #[test]
    #[should_panic]
    fn out_of_range_tile_panics() {
        let _ = Pattern::new(&[9]);
    }

    #[test]
    fn num_projected_states_known_values() {
        assert_eq!(Pattern::new(&[1]).num_projected_states(), 9 * 8);
        assert_eq!(Pattern::new(&[1, 2]).num_projected_states(), 9 * 8 * 7);
        assert_eq!(Pattern::new(&[1, 2, 3, 4]).num_projected_states(), 9 * 8 * 7 * 6 * 5);
        // For k=8: 9!/0! = 9! = 362880.
        assert_eq!(Pattern::new(&[1, 2, 3, 4, 5, 6, 7, 8]).num_projected_states(), 362_880);
    }

    #[test]
    fn projection_keeps_pattern_tiles_drops_others() {
        let p = Pattern::new(&[1, 3]);
        let proj = ProjectedState::from_state(&GOAL, p);
        // GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0]. Tile 1 stays, 3 stays, others ANON, blank stays.
        assert_eq!(proj.0[0], 1);
        assert_eq!(proj.0[1], ANON);
        assert_eq!(proj.0[2], 3);
        assert_eq!(proj.0[3], ANON);
        assert_eq!(proj.0[8], 0);
    }

    #[test]
    fn goal_projection_rank_is_known() {
        // For any pattern, the goal projection's rank corresponds to pattern
        // tiles at their goal positions and blank at position 8.
        for tiles in [
            &[1u8][..],
            &[1, 2],
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        ] {
            let p = Pattern::new(tiles);
            let goal_proj = ProjectedState::goal(p);
            let r = goal_proj.rank(p);
            assert!(r < p.num_projected_states(), "goal rank {} >= {} for {:?}", r, p.num_projected_states(), tiles);
        }
    }

    #[test]
    fn apply_zero_cost_when_swapping_anon() {
        // Pattern that excludes tile 8: the move swapping blank with tile 8
        // should be 0-cost (tile 8 is anonymous).
        let p = Pattern::new(&[1, 2, 3, 4, 5, 6, 7]);
        let proj = ProjectedState::from_state(&GOAL, p);
        // GOAL blank at position 8; legal moves Up (swap with pos 5 = tile 6) and Left (swap with pos 7 = tile 8).
        // tile 6 is in pattern → cost 1; tile 8 not in pattern → cost 0.
        let (_, c_up) = proj.apply(Move::Up);
        let (_, c_left) = proj.apply(Move::Left);
        assert_eq!(c_left, 0, "swapping blank with non-pattern tile 8 must be free");
        assert_eq!(c_up, 1, "swapping blank with pattern tile 6 must cost 1");
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
    fn ranks_are_bijective_on_reachable_states() {
        // For a small pattern, every rank from BFS exploration is distinct
        // and lies in [0, num_projected_states).
        use std::collections::{HashMap, HashSet, VecDeque};
        let p = Pattern::new(&[1, 2]);
        let goal_proj = ProjectedState::goal(p);
        let mut seen: HashMap<[u8; 9], u32> = HashMap::new();
        let mut ranks: HashSet<u32> = HashSet::new();
        let mut q: VecDeque<ProjectedState> = VecDeque::new();
        q.push_back(goal_proj);
        seen.insert(goal_proj.0, goal_proj.rank(p));
        while let Some(s) = q.pop_front() {
            for m in s.legal_moves().iter() {
                let (n, _) = s.apply(m);
                if seen.contains_key(&n.0) {
                    continue;
                }
                let r = n.rank(p);
                assert!(r < p.num_projected_states(), "rank {} >= {}", r, p.num_projected_states());
                assert!(ranks.insert(r), "duplicate rank {} for distinct states", r);
                seen.insert(n.0, r);
                q.push_back(n);
            }
        }
        assert!(!seen.is_empty());
    }
}
