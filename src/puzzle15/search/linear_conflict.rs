//! Manhattan distance + Linear Conflict for the 15-puzzle (Hansson, Mayer,
//! Yung 1992). See `crate::puzzle8::search::linear_conflict` for the
//! derivation; the 4×4 version is the same algorithm with `W = 4` and up to
//! four home-row tiles per line.

use super::Heuristic;
use crate::puzzle15::state::{State, W};

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

        // Row conflicts: 2 * (count of home-row tiles - LIS by goal column).
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

        // 15-puzzle diameter is 80; clamp is defensive.
        total.min(u8::MAX as u32) as u8
    }
}

/// `len(arr) - LIS(arr)` — the minimum number of elements to remove so the
/// remaining sequence is strictly increasing. O(n²) DP; `n ≤ 4`.
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
    use crate::puzzle15::search::ManhattanHeuristic;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::state::{GOAL, State};

    #[test]
    fn lc_of_goal_is_zero() {
        assert_eq!(LinearConflictHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn lc_detects_row_swap() {
        // Swap tiles 1 and 2 in row 0: 2 1 3 4 / ... → MD = 2, LC adds 2 → 4.
        let mut s = GOAL.0;
        s.swap(0, 1);
        let s = State(s);
        assert_eq!(ManhattanHeuristic.h(&s), 2);
        assert_eq!(LinearConflictHeuristic.h(&s), 4);
    }

    #[test]
    fn lc_detects_column_swap() {
        // Swap tiles 1 and 5 in column 0: row 0 starts with 5, row 1 with 1.
        let mut s = GOAL.0;
        s.swap(0, 4);
        let s = State(s);
        assert_eq!(ManhattanHeuristic.h(&s), 2);
        assert_eq!(LinearConflictHeuristic.h(&s), 4);
    }

    #[test]
    fn lc_dominates_manhattan_on_shallow_bfs() {
        let truth = bfs_distances(8);
        for (raw, &true_dist) in &truth {
            let lc = LinearConflictHeuristic.h(&State(*raw));
            let md = ManhattanHeuristic.h(&State(*raw));
            assert!(lc >= md, "LC {} < MD {} for {:?}", lc, md, raw);
            assert!(lc <= true_dist, "LC {} > truth {} for {:?}", lc, true_dist, raw);
        }
    }

    #[test]
    fn lc_removals_lis_correct_on_small_cases() {
        assert_eq!(lc_removals(&[]), 0);
        assert_eq!(lc_removals(&[0]), 0);
        assert_eq!(lc_removals(&[0, 1, 2, 3]), 0);
        assert_eq!(lc_removals(&[3, 2, 1, 0]), 3);  // LIS = [0], remove 3
        assert_eq!(lc_removals(&[1, 3, 2, 0]), 2);  // LIS = [1, 3] or [1, 2], remove 2
    }
}
