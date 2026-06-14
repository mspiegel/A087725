//! Walking Distance for the 15-puzzle (Ken'ichiro Takahashi).
//!
//! See `crate::puzzle8::search::walking_distance` for the abstraction. The
//! 4×4 row-WD state is a 4×4 matrix `M[r][g]` = "non-blank tiles in row r
//! whose goal row is g," plus the blank's row index ∈ {0..3}. Column-WD is
//! the same shape with rows and columns swapped.
//!
//! The reachable WD-state space is small enough (~25,000 per axis) to hold
//! the full BFS distance table in a `HashMap<u64, u8>` keyed on packed
//! `(matrix, blank-axis-index)`. By symmetry of the 4×4 goal, row-WD and
//! column-WD share the same table.

use super::Heuristic;
use crate::puzzle15::state::{State, W};
use std::collections::HashMap;
use std::sync::OnceLock;

/// A 4×4 row-distribution matrix.
type WdMatrix = [[u8; W]; W];

/// Pack `(matrix, blank-axis-index)` into a u64 key. Each cell holds a value
/// ≤ 4 (fits in 3 bits); the axis index is ≤ 3 (fits in 2 bits). Total used
/// bits: 16 * 3 + 2 = 50, well within u64.
#[inline]
fn pack(m: &WdMatrix, blank_idx: u8) -> u64 {
    let mut k: u64 = blank_idx as u64;
    for r in 0..W {
        for c in 0..W {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

/// Goal row-distribution matrix for the 15-puzzle (also serves as the column-
/// distribution goal by symmetry). Tiles 1–4 have goal row 0; 5–8 row 1;
/// 9–12 row 2; 13–15 row 3; the blank (row 3) is excluded.
fn goal_matrix() -> WdMatrix {
    let mut m = [[0u8; W]; W];
    m[0][0] = 4;
    m[1][1] = 4;
    m[2][2] = 4;
    m[3][3] = 3;
    m
}

const GOAL_BLANK_IDX: u8 = 3;

/// BFS from the goal WD-state, collecting every reachable matrix-and-blank
/// combination with its distance to goal.
fn build_table() -> HashMap<u64, u8> {
    let goal_m = goal_matrix();
    let goal_key = pack(&goal_m, GOAL_BLANK_IDX);

    let mut table: HashMap<u64, u8> = HashMap::with_capacity(64 * 1024);
    table.insert(goal_key, 0);

    let mut frontier: Vec<(WdMatrix, u8)> = vec![(goal_m, GOAL_BLANK_IDX)];
    let mut depth: u8 = 0;

    while !frontier.is_empty() {
        let mut next: Vec<(WdMatrix, u8)> = Vec::new();
        let next_depth = depth + 1;
        for (m, br) in &frontier {
            // Blank axis-index decreases: a tile from axis (br - 1) crosses
            // into axis br, picking any non-empty goal-axis slot.
            if *br > 0 {
                let from = (*br - 1) as usize;
                let to = *br as usize;
                for g in 0..W {
                    if m[from][g] > 0 {
                        let mut m2 = *m;
                        m2[from][g] -= 1;
                        m2[to][g] += 1;
                        let new_br = *br - 1;
                        let key = pack(&m2, new_br);
                        if !table.contains_key(&key) {
                            table.insert(key, next_depth);
                            next.push((m2, new_br));
                        }
                    }
                }
            }
            // Blank axis-index increases: symmetric.
            if (*br as usize) < W - 1 {
                let from = (*br + 1) as usize;
                let to = *br as usize;
                for g in 0..W {
                    if m[from][g] > 0 {
                        let mut m2 = *m;
                        m2[from][g] -= 1;
                        m2[to][g] += 1;
                        let new_br = *br + 1;
                        let key = pack(&m2, new_br);
                        if !table.contains_key(&key) {
                            table.insert(key, next_depth);
                            next.push((m2, new_br));
                        }
                    }
                }
            }
        }
        depth = next_depth;
        frontier = next;
    }
    table
}

fn table() -> &'static HashMap<u64, u8> {
    static T: OnceLock<HashMap<u64, u8>> = OnceLock::new();
    T.get_or_init(build_table)
}

/// `WD_row(s) + WD_col(s)`. Admissible.
pub struct WalkingDistanceHeuristic;

impl WalkingDistanceHeuristic {
    /// Force the lookup table to be built. Optional — `h` will build on first
    /// call regardless. Useful in tests to amortize table construction.
    pub fn warm_up() {
        let _ = table();
    }

    /// Size of the reachable WD-state space, for diagnostics.
    pub fn table_size() -> usize {
        table().len()
    }
}

impl Heuristic for WalkingDistanceHeuristic {
    fn h(&self, s: &State) -> u8 {
        let mut m_row = [[0u8; W]; W];
        let mut m_col = [[0u8; W]; W];
        let mut br: u8 = 0;
        let mut bc: u8 = 0;
        for pos in 0..(W * W) {
            let tile = s.0[pos];
            let r = (pos / W) as u8;
            let c = (pos % W) as u8;
            if tile == 0 {
                br = r;
                bc = c;
                continue;
            }
            let goal_pos = (tile - 1) as usize;
            let gr = goal_pos / W;
            let gc = goal_pos % W;
            m_row[r as usize][gr] += 1;
            m_col[c as usize][gc] += 1;
        }
        let t = table();
        let h_row = *t
            .get(&pack(&m_row, br))
            .expect("row-WD state must be reachable from goal");
        let h_col = *t
            .get(&pack(&m_col, bc))
            .expect("col-WD state must be reachable from goal");
        h_row + h_col
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::search::ManhattanHeuristic;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::state::{GOAL, Move, State};

    #[test]
    fn wd_of_goal_is_zero() {
        assert_eq!(WalkingDistanceHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn wd_after_single_vertical_move() {
        // Blank Up from goal: one tile crosses a row → row-WD = 1, col-WD = 0.
        let s = GOAL.apply(Move::Up);
        assert_eq!(WalkingDistanceHeuristic.h(&s), 1);
    }

    #[test]
    fn wd_after_single_horizontal_move() {
        // Blank Left from goal: one tile crosses a column → col-WD = 1.
        let s = GOAL.apply(Move::Left);
        assert_eq!(WalkingDistanceHeuristic.h(&s), 1);
    }

    #[test]
    fn wd_admissible_on_shallow_bfs() {
        let truth = bfs_distances(8);
        for (raw, &true_dist) in &truth {
            let est = WalkingDistanceHeuristic.h(&State(*raw));
            assert!(est <= true_dist, "WD {} > truth {} for {:?}", est, true_dist, raw);
        }
    }

    #[test]
    fn wd_can_exceed_manhattan() {
        // WD is supposed to be tighter than Manhattan on average. Look for
        // *any* state in shallow BFS where WD strictly exceeds MD — proves
        // the table isn't degenerate.
        let truth = bfs_distances(8);
        let mut found_strict = false;
        for raw in truth.keys() {
            let wd = WalkingDistanceHeuristic.h(&State(*raw));
            let md = ManhattanHeuristic.h(&State(*raw));
            if wd > md {
                found_strict = true;
                break;
            }
        }
        assert!(found_strict, "WD never exceeded Manhattan on shallow BFS");
    }

    #[test]
    fn table_size_is_around_25k() {
        // Standard 4x4 row-WD state count is 24,964. We expect ours to be in
        // that ballpark (within ~10x — the goal layout is slightly
        // asymmetric due to the blank).
        let n = WalkingDistanceHeuristic::table_size();
        assert!(
            (1_000..100_000).contains(&n),
            "unexpected WD table size {} (expected ~25k)",
            n
        );
    }
}
