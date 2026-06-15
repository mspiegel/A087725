//! Admissible heuristics for the 24-puzzle IDA\*.
//!
//! A heuristic `h` is *admissible* iff `h(s) ≤ dist(s, GOAL)` for every state.
//! Admissibility is required for IDA\* to return optimal solutions. Manhattan
//! distance is the classic lower bound and the baseline while the PDB engine is
//! built up. All 24-puzzle costs fit `u8`: the diameter is ≤ 205 and the
//! maximum total Manhattan distance is ≤ 192, both below 255.

use crate::puzzle24::state::{State, N_CELLS, W};

/// Admissible estimate of distance from `s` to [`crate::puzzle24::state::GOAL`].
pub trait Heuristic {
    fn h(&self, s: &State) -> u8;
}

/// Blanket impl: any reference to a [`Heuristic`] is also a [`Heuristic`], so
/// borrowed PDBs can be passed to combinators like `MaxHeuristic`.
impl<'a, H: Heuristic + ?Sized> Heuristic for &'a H {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        (**self).h(s)
    }
}

/// Sum of Manhattan distances from each non-blank tile to its goal position.
/// Tile `k ∈ 1..=24` sits at position `k - 1` (row-major) in the goal.
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

/// Incremental Manhattan heuristic: maintains the running distance in O(1) per
/// move. A single move slides exactly one tile by one cell, so the tile that
/// moved into the parent's blank cell changes its Manhattan term by ±1 and
/// nothing else does. Used to validate the [`IncHeuristic`] driver and as the
/// template for the zero-aware incremental PDB evaluator.
///
/// [`IncHeuristic`]: crate::puzzle24::search::idastar::IncHeuristic
pub struct IncManhattan;

/// Manhattan distance of a single tile value at a given cell.
#[inline]
fn tile_manhattan(tile: u8, pos: usize) -> u8 {
    let goal_pos = (tile - 1) as usize;
    let dr = (pos / W) as i32 - (goal_pos / W) as i32;
    let dc = (pos % W) as i32 - (goal_pos % W) as i32;
    (dr.unsigned_abs() + dc.unsigned_abs()) as u8
}

impl crate::puzzle24::search::idastar::IncHeuristic for IncManhattan {
    type Ctx = u8; // running Manhattan distance

    fn root(&self, s: &State) -> (u8, u8) {
        let h = ManhattanHeuristic.h(s);
        (h, h)
    }

    fn advance(&self, parent: &u8, child: &State, m: crate::puzzle24::state::Move) -> (u8, u8) {
        // The blank moved in direction `m` to `nb`; the displaced tile now sits
        // at `b` (the parent's blank cell) = `nb` stepped back by `m`.
        let nb = child.blank_pos() as usize;
        let b = match m {
            crate::puzzle24::state::Move::Up => nb + W,
            crate::puzzle24::state::Move::Down => nb - W,
            crate::puzzle24::state::Move::Left => nb + 1,
            crate::puzzle24::state::Move::Right => nb - 1,
        };
        let tile = child.0[b];
        // term went from manhattan(tile, nb) [its parent cell] to manhattan(tile, b).
        let delta = tile_manhattan(tile, b) as i16 - tile_manhattan(tile, nb) as i16;
        let h = (*parent as i16 + delta) as u8;
        (h, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::{Move, GOAL};

    #[test]
    fn manhattan_of_goal_is_zero() {
        assert_eq!(ManhattanHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn manhattan_changes_by_at_most_1_per_move() {
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        for i in 0u32..500 {
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    let h_before = ManhattanHeuristic.h(&s);
                    let after = s.apply(m);
                    let h_after = ManhattanHeuristic.h(&after);
                    let diff = (h_after as i16) - (h_before as i16);
                    assert!(diff == 1 || diff == -1, "Δh = {} not ±1", diff);
                    s = after;
                    break;
                }
            }
        }
    }

    #[test]
    fn manhattan_admissible_on_shallow_bfs() {
        let table = crate::puzzle24::search::tests_util::bfs_distances(10);
        for (state, &truth) in &table {
            let est = ManhattanHeuristic.h(&State(*state));
            assert!(est <= truth, "h({:?}) = {} > true {}", state, est, truth);
        }
    }
}
