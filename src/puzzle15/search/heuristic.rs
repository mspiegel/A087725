//! Admissible heuristics for the 15-puzzle IDA\*.
//!
//! A heuristic `h` is *admissible* iff `h(s) ≤ dist(s, GOAL)` for every state.
//! Admissibility is required for IDA\* to return optimal solutions. Manhattan
//! distance is a classic lower bound — each tile must move at least its
//! Manhattan distance to its goal position — and is the baseline used while
//! the PDB engine is being built up.

use crate::puzzle15::state::{State, N_CELLS, W};

/// Admissible estimate of distance from `s` to [`crate::puzzle15::state::GOAL`].
pub trait Heuristic {
    fn h(&self, s: &State) -> u8;
}

/// Blanket implementation: any reference to a [`Heuristic`] is also a
/// [`Heuristic`]. Lets us pass borrowed PDBs to combinators like
/// `MaxHeuristic` without taking ownership.
impl<H: Heuristic + ?Sized> Heuristic for &H {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        (**self).h(s)
    }
}

/// Sum of Manhattan distances from each non-blank tile to its goal position.
///
/// In the goal `1 2 3 4 / 5 6 7 8 / 9 10 11 12 / 13 14 15 _`, tile
/// `k ∈ 1..=15` sits at position `k - 1` (row-major).
pub struct ManhattanHeuristic;

impl Heuristic for ManhattanHeuristic {
    fn h(&self, s: &State) -> u8 {
        let mut total: u32 = 0;
        for pos in 0..N_CELLS {
            let tile = s.0[pos];
            if tile == 0 {
                continue;
            }
            let goal_pos = (tile - 1) as usize;
            let cur_row = (pos / W) as i32;
            let cur_col = (pos % W) as i32;
            let goal_row = (goal_pos / W) as i32;
            let goal_col = (goal_pos % W) as i32;
            total += (cur_row - goal_row).unsigned_abs();
            total += (cur_col - goal_col).unsigned_abs();
        }
        total as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::state::{GOAL, Move};

    #[test]
    fn manhattan_of_goal_is_zero() {
        assert_eq!(ManhattanHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn manhattan_changes_by_at_most_1_per_move() {
        // Single move: the moved tile shifts by exactly 1 cell. Its Manhattan
        // term changes by ±1; everything else is fixed. So |Δh| = 1.
        let mut s = GOAL;
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        for i in 0u32..500 {
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    let h_before = ManhattanHeuristic.h(&s);
                    let after = s.apply(m);
                    let h_after = ManhattanHeuristic.h(&after);
                    let diff = (h_after as i16) - (h_before as i16);
                    assert!(diff == 1 || diff == -1, "Δh = {diff} not ±1");
                    s = after;
                    break;
                }
            }
        }
    }

    #[test]
    fn manhattan_admissible_on_shallow_bfs() {
        // BFS from GOAL out to depth 10; for every reached state h(s) ≤ true
        // distance.
        let table = crate::puzzle15::search::tests_util::bfs_distances(10);
        for (state, &truth) in &table {
            let est = ManhattanHeuristic.h(&State(*state));
            assert!(est <= truth, "h({state:?}) = {est} > true {truth}");
        }
    }
}
