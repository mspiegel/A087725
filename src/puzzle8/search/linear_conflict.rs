//! Manhattan distance + Linear Conflict (Hansson, Mayer, Yung 1992).
//!
//! For each row r, consider the tiles already in their goal row r and look at
//! the order in which they currently appear (left → right). If two such tiles
//! are out of goal-column order, at least one of them must temporarily leave
//! the row to let the other pass — two extra moves. The admissible LC bonus
//! per row is `2 * (k - LIS)` where `k` is the count of home-row tiles in that
//! row and `LIS` is the longest strictly-increasing subsequence of their goal
//! columns. Equivalently, `k - LIS` is the minimum number of tiles to remove
//! so the remaining home-row tiles are sorted by goal column. Same for each
//! column.
//!
//! Counting *pairs* in conflict instead of (k − LIS) would over-count and
//! break admissibility — three pairwise-conflicting home-row tiles share a
//! single forced eviction, not three.
//!
//! Always at least as tight as `ManhattanHeuristic` (LC ≥ 0). Admissible
//! because every conflict really does force at least two extra moves.

use super::Heuristic;
use crate::puzzle8::state::State;

const W: usize = 3;

/// Manhattan distance plus the linear-conflict correction. Admissible.
pub struct LinearConflictHeuristic;

impl Heuristic for LinearConflictHeuristic {
    fn h(&self, s: &State) -> u8 {
        let mut total: u32 = 0;

        // Manhattan term.
        for pos in 0..(W * W) {
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

        // Row conflicts: for each row, count home-row tiles in order of
        // current column; collect their goal columns; add 2 * (k - LIS).
        for r in 0..W {
            let mut home_cols = [0u8; W];
            let mut n = 0usize;
            for c in 0..W {
                let tile = s.0[r * W + c];
                if tile == 0 {
                    continue;
                }
                let goal_pos = (tile - 1) as usize;
                if goal_pos / W == r {
                    home_cols[n] = (goal_pos % W) as u8;
                    n += 1;
                }
            }
            total += 2 * lc_removals(&home_cols[..n]) as u32;
        }

        // Column conflicts: same shape, transposed.
        for c in 0..W {
            let mut home_rows = [0u8; W];
            let mut n = 0usize;
            for r in 0..W {
                let tile = s.0[r * W + c];
                if tile == 0 {
                    continue;
                }
                let goal_pos = (tile - 1) as usize;
                if goal_pos % W == c {
                    home_rows[n] = (goal_pos / W) as u8;
                    n += 1;
                }
            }
            total += 2 * lc_removals(&home_rows[..n]) as u32;
        }

        // 8-puzzle diameter is 31; the clamp is defensive against future
        // accidents only.
        total.min(u8::MAX as u32) as u8
    }
}

/// Minimum number of elements to remove so that the remaining sequence is
/// strictly increasing. Equivalently, `len(arr) - LIS(arr)`. O(n²) DP; with
/// `n ≤ 3` for the 8-puzzle (at most one row/column at a time) this is
/// trivially fast.
fn lc_removals(arr: &[u8]) -> usize {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mut lis = [1u8; W];
    let mut best = 1u8;
    for i in 1..n {
        for j in 0..i {
            if arr[j] < arr[i] && lis[j] + 1 > lis[i] {
                lis[i] = lis[j] + 1;
            }
        }
        if lis[i] > best {
            best = lis[i];
        }
    }
    n - best as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::search::ManhattanHeuristic;
    use crate::puzzle8::state::{State, GOAL};

    #[test]
    fn lc_of_goal_is_zero() {
        assert_eq!(LinearConflictHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn lc_dominates_manhattan_on_a_few_states() {
        // Standard property: LC ≥ MD for every state.
        // We don't have BFS-truth here; we'll prove admissibility exhaustively
        // in the integration test.
        let states = [
            GOAL,
            State([2, 1, 3, 4, 5, 6, 7, 8, 0]),
            State([3, 2, 1, 4, 5, 6, 7, 8, 0]),
            State([1, 2, 3, 4, 5, 6, 8, 7, 0]),
        ];
        for s in states {
            assert!(
                LinearConflictHeuristic.h(&s) >= ManhattanHeuristic.h(&s),
                "LC < MD for {:?}",
                s.0
            );
        }
    }

    #[test]
    fn lc_detects_simple_row_swap() {
        // Swap tiles 1 and 2 in row 0: both are home-row but reversed.
        // Manhattan = 1 + 1 = 2 (each is one cell off). LC adds 2 → total 4.
        let s = State([2, 1, 3, 4, 5, 6, 7, 8, 0]);
        assert_eq!(ManhattanHeuristic.h(&s), 2);
        assert_eq!(LinearConflictHeuristic.h(&s), 4);
    }

    #[test]
    fn lc_detects_simple_column_swap() {
        // Swap tiles 1 and 4 in column 0: both home-column but reversed.
        // Manhattan = 1 + 1 = 2. LC adds 2 → 4.
        let s = State([4, 2, 3, 1, 5, 6, 7, 8, 0]);
        assert_eq!(ManhattanHeuristic.h(&s), 2);
        assert_eq!(LinearConflictHeuristic.h(&s), 4);
    }

    #[test]
    fn lc_no_phantom_conflict_when_tiles_out_of_goal_row() {
        // Cycle every tile one cell forward — no tile is in its goal row or
        // goal column, so LC adds nothing and LC == MD.
        let s = State([0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(LinearConflictHeuristic.h(&s), ManhattanHeuristic.h(&s));
    }

    #[test]
    fn lc_removals_lis_correct_on_small_cases() {
        assert_eq!(lc_removals(&[]), 0);
        assert_eq!(lc_removals(&[0]), 0);
        assert_eq!(lc_removals(&[0, 1]), 0);
        assert_eq!(lc_removals(&[1, 0]), 1);
        assert_eq!(lc_removals(&[0, 1, 2]), 0);
        assert_eq!(lc_removals(&[2, 0, 1]), 1); // LIS = [0,1], remove 2
        assert_eq!(lc_removals(&[2, 1, 0]), 2); // LIS = [0], remove 2
    }
}
