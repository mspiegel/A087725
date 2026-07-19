//! Walking Distance (Ken'ichiro Takahashi).
//!
//! An admissible heuristic computed in an *abstraction* of the puzzle. Drop
//! tile identity within rows: each tile is labeled only by its goal row. Now
//! "horizontal moves" of the blank are free — they don't change anyone's row
//! membership — and only vertical moves matter. The optimal cost in this
//! abstracted puzzle is a lower bound on the real cost (relaxation), and we
//! pre-compute it by a small BFS over the abstracted state space.
//!
//! Concretely, the **row-WD** state is a 3×3 matrix `M_row[r][g]` =
//! "how many non-blank tiles whose goal row is `g` currently sit in row `r`",
//! plus the blank's row index. Column-WD is the same with rows and columns
//! swapped. `WD(s) = WD_row(s) + WD_col(s)`.
//!
//! For the 3×3 (8-puzzle) board, row-WD and column-WD have isomorphic state
//! spaces — the WD goal-state and transitions have the same shape — so a
//! single BFS table serves both.

use super::Heuristic;
use crate::puzzle8::state::State;
use std::collections::HashMap;
use std::sync::OnceLock;

const W: usize = 3;

/// A 3×3 row-distribution matrix.
type WdMatrix = [[u8; W]; W];

/// Pack `(matrix, blank-axis-index)` into a single u32 key. Each cell holds a
/// value ≤ 3 (fits in 2 bits); the blank index is ≤ 2 (fits in 2 bits).
#[inline]
fn pack(m: &WdMatrix, blank_idx: u8) -> u32 {
    let mut k: u32 = blank_idx as u32;
    for r in 0..W {
        for c in 0..W {
            k = (k << 2) | (m[r][c] as u32);
        }
    }
    k
}

/// Goal row-distribution matrix for the 8-puzzle (also serves as the column-
/// distribution goal by symmetry). Tiles 1–3 have goal row 0; 4–6 row 1;
/// 7–8 row 2; the blank (row 2) is excluded.
fn goal_matrix() -> WdMatrix {
    let mut m = [[0u8; W]; W];
    m[0][0] = 3;
    m[1][1] = 3;
    m[2][2] = 2;
    m
}

/// BFS from the goal WD-state, enumerating every reachable matrix-and-blank
/// combination and its shortest-path distance to goal. Distances are u8;
/// `WD ≤ DIAMETER = 31` so this is safe.
fn build_table() -> HashMap<u32, u8> {
    let goal_m = goal_matrix();
    let goal_blank: u8 = 2;
    let goal_key = pack(&goal_m, goal_blank);

    let mut table: HashMap<u32, u8> = HashMap::with_capacity(512);
    table.insert(goal_key, 0);

    let mut frontier: Vec<(WdMatrix, u8)> = vec![(goal_m, goal_blank)];
    let mut depth: u8 = 0;

    while !frontier.is_empty() {
        let mut next: Vec<(WdMatrix, u8)> = Vec::new();
        let next_depth = depth + 1;
        for (m, br) in &frontier {
            // Blank moves Up: a tile from row (br-1) moves down into row br.
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
                        if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key) {
                            e.insert(next_depth);
                            next.push((m2, new_br));
                        }
                    }
                }
            }
            // Blank moves Down: a tile from row (br+1) moves up into row br.
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
                        if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key) {
                            e.insert(next_depth);
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

fn table() -> &'static HashMap<u32, u8> {
    static T: OnceLock<HashMap<u32, u8>> = OnceLock::new();
    T.get_or_init(build_table)
}

/// `WD_row(s) + WD_col(s)`. Admissible — each component is the optimal cost
/// in a row- or column-abstraction of the puzzle, both of which are
/// relaxations of the real puzzle.
pub struct WalkingDistanceHeuristic;

impl WalkingDistanceHeuristic {
    /// Force the lookup table to be built. Optional — `h` will build on first
    /// call regardless.
    pub fn warm_up() {
        let _ = table();
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
        // Row and column abstractions share the same table by symmetry of the
        // 3×3 goal layout.
        let h_row = *t
            .get(&pack(&m_row, br))
            .expect("row-WD state must be reachable");
        let h_col = *t
            .get(&pack(&m_col, bc))
            .expect("col-WD state must be reachable");
        h_row + h_col
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::state::{Move, GOAL};

    #[test]
    fn wd_of_goal_is_zero() {
        assert_eq!(WalkingDistanceHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn wd_table_contains_goal_with_distance_zero() {
        let t = table();
        let key = pack(&goal_matrix(), 2);
        assert_eq!(t.get(&key).copied(), Some(0));
    }

    #[test]
    fn wd_after_single_vertical_move() {
        // From GOAL (blank at pos 8, row 2), apply Up → blank moves to row 1,
        // tile 6 (goal row 1) moves to row 2. Row-WD becomes:
        //   M_row[1][1] = 2 (lost one tile-with-goal-row-1)
        //   M_row[2][1] = 1 (gained one tile-with-goal-row-1)
        //   M_row[2][2] = 2 → still 2 (tiles 7, 8 untouched)
        //   blank_row = 1
        // This WD-state is at distance 1 from goal (one BFS step away).
        // Column-WD is unaffected by a vertical move.
        let s = GOAL.apply(Move::Up);
        // WD-row should be 1 (one vertical move out of goal).
        // WD-col should be 0 (no column changes).
        let h = WalkingDistanceHeuristic.h(&s);
        assert_eq!(h, 1, "WD after one Up should be 1; got {h}");
    }

    #[test]
    fn wd_after_single_horizontal_move() {
        // From GOAL, apply Left → blank moves to col 1, tile 8 moves to col 2.
        // Row-WD unchanged (no row crossings). Column-WD increases by 1.
        let s = GOAL.apply(Move::Left);
        let h = WalkingDistanceHeuristic.h(&s);
        assert_eq!(h, 1, "WD after one Left should be 1; got {h}");
    }

    #[test]
    fn wd_grows_along_a_walk() {
        // Take a few moves, WD never exceeds the depth.
        let mut s = GOAL;
        let path = [
            Move::Up,
            Move::Left,
            Move::Down,
            Move::Right,
            Move::Up,
            Move::Up,
        ];
        let mut depth: u8 = 0;
        for m in path {
            if s.legal_moves().contains(m) {
                s = s.apply(m);
                depth += 1;
                let h = WalkingDistanceHeuristic.h(&s);
                assert!(h <= depth, "WD {h} > depth {depth} after {depth} moves");
            }
        }
    }
}
