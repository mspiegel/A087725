//! Admissible heuristics for IDA*.
//!
//! A heuristic `h` is *admissible* iff `h(s) ≤ dist(s, GOAL)` for every state.
//! Admissibility is required for IDA* to return optimal solutions. Both
//! heuristics below are admissible: Manhattan distance is a classic lower
//! bound (each tile must move at least its Manhattan distance to its goal),
//! and the exact table is trivially admissible (equal to true distance).

use crate::bfs::DistanceTable;
use crate::state::State;

/// Admissible estimate of distance from `s` to [`crate::state::GOAL`].
pub trait Heuristic {
    fn h(&self, s: &State) -> u8;
}

/// Sum of Manhattan distances from each non-blank tile to its goal position.
///
/// In the goal `1 2 3 / 4 5 6 / 7 8 _`, tile `k ∈ 1..=8` sits at position
/// `k - 1` (row-major).
pub struct ManhattanHeuristic;

impl Heuristic for ManhattanHeuristic {
    fn h(&self, s: &State) -> u8 {
        let mut total: u32 = 0;
        for pos in 0..9u8 {
            let tile = s.0[pos as usize];
            if tile == 0 {
                continue;
            }
            let goal_pos = tile - 1;
            let cur_row = (pos / 3) as i32;
            let cur_col = (pos % 3) as i32;
            let goal_row = (goal_pos / 3) as i32;
            let goal_col = (goal_pos % 3) as i32;
            total += (cur_row - goal_row).unsigned_abs();
            total += (cur_col - goal_col).unsigned_abs();
        }
        total as u8
    }
}

/// Exact distance via the precomputed [`DistanceTable`]. Returns the true
/// optimal cost; an IDA* run with this heuristic expands no surplus nodes.
pub struct TableHeuristic<'a> {
    table: &'a DistanceTable,
}

impl<'a> TableHeuristic<'a> {
    pub fn new(table: &'a DistanceTable) -> Self {
        Self { table }
    }
}

impl<'a> Heuristic for TableHeuristic<'a> {
    fn h(&self, s: &State) -> u8 {
        self.table.dist(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GOAL;

    #[test]
    fn manhattan_of_goal_is_zero() {
        assert_eq!(ManhattanHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn manhattan_is_admissible_on_full_state_space() {
        // For every solvable state, Manhattan must be <= true distance.
        let t = DistanceTable::build();
        let h = ManhattanHeuristic;
        for r in 0..crate::state::N_STATES {
            let s = crate::rank::unrank(r);
            let est = h.h(&s);
            let truth = t.dist(&s);
            assert!(
                est <= truth,
                "Manhattan({:?}) = {} > dist = {}",
                s.0,
                est,
                truth
            );
        }
    }

    #[test]
    fn table_heuristic_equals_dist() {
        let t = DistanceTable::build();
        let h = TableHeuristic::new(&t);
        // Spot-check a few states.
        assert_eq!(h.h(&GOAL), 0);
        let antipodes = t.antipodes();
        for s in &antipodes {
            assert_eq!(h.h(s), crate::state::DIAMETER);
        }
    }
}
