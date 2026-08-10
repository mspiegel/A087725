//! Manhattan distance + Linear Conflict for the 24-puzzle (Hansson, Mayer,
//! Yung 1992). Ported from `crate::puzzle15::search::linear_conflict`; the 5×5
//! version is the same algorithm with `W = 5` and up to five home-row tiles per
//! line.
//!
//! **Difference from the 15-puzzle port:** the 4×4 version precomputes a 16-bit
//! lookup table keyed on a row/column's packed 4-tile contents (4 bits/tile).
//! At 5×5 a line packs five tiles needing 5 bits each → a `2^25`-entry table per
//! line, far too large. So this port **drops the LUT** and recomputes each dirty
//! line directly: a 5-cell scan collecting home-line goal coordinates, then an
//! `O(n²)` LIS on `n ≤ 5`. A single slide dirties at most three lines (two rows +
//! one column for a vertical slide; two columns + one row for a horizontal
//! slide), so per-node cost is three short scans regardless.

use super::{Heuristic, IncHeuristic, SearchStats};
use crate::puzzle24::state::{Move, State, W};

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
            total += row_pen_direct(s, r) as u32;
        }

        // Column conflicts: same shape, transposed.
        for c in 0..W {
            total += col_pen_direct(s, c) as u32;
        }

        // 24-puzzle diameter is ≤ 205; clamp is defensive.
        total.min(u8::MAX as u32) as u8
    }
}

/// Incremental Manhattan + Linear-Conflict. Same admissible heuristic as
/// [`LinearConflictHeuristic`]; only the per-node cost differs.
///
/// A single slide moves exactly one tile by one cell, so the heuristic update is
/// localized: Manhattan changes by ±1 (signed by the moved tile's goal-distance
/// delta), and at most **three** of the ten row/column "home-tile" penalties are
/// dirty (two rows + one column for a vertical slide; two columns + one row for
/// a horizontal slide). All other seven lines are byte-identical to the parent's,
/// so we copy them verbatim and recompute only the dirty ones — each a 5-cell
/// scan plus an LIS on ≤5 elements.
///
/// The blank cell is carried in [`LcCtx`] so [`advance`](IncHeuristic::advance)
/// never has to rescan `child` for it; the moved tile's `(from, to)` is then
/// `(child.blank, parent.blank)` (the tile filled the cell the blank just left).
pub struct LinearConflictInc;

/// Per-node state for [`LinearConflictInc`]. `Copy` so the IDA* recursion can
/// thread it by value with implicit backtracking.
#[derive(Clone, Copy, Debug)]
pub struct LcCtx {
    /// Sum of tile-to-goal Manhattan distances (≤ 192, never overflows `u8`).
    manhattan: u8,
    /// `2 * lc_removals` per row — already doubled, so summing gives the LC bonus.
    row_pen: [u8; W],
    /// Same as `row_pen`, transposed.
    col_pen: [u8; W],
    /// Position of the blank in this node's state.
    blank: u8,
}

#[inline]
fn manhattan_of(tile: u8, pos: usize) -> i32 {
    if tile == 0 {
        return 0;
    }
    let g = (tile - 1) as usize;
    let dr = (pos / W) as i32 - (g / W) as i32;
    let dc = (pos % W) as i32 - (g % W) as i32;
    dr.abs() + dc.abs()
}

/// Linear-conflict penalty (`2 * lc_removals`) for row `r`, computed directly
/// from the board — the 5×5 analogue of the 15-puzzle's LUT read. Scans the five
/// cells of the row, keeps the goal columns of tiles already in their home row,
/// and returns twice the minimum number that must yield to order the rest.
#[inline]
fn row_pen_direct(s: &State, r: usize) -> u8 {
    let off = r * W;
    let mut goals = [0u8; W];
    let mut n = 0usize;
    for c in 0..W {
        let tile = s.0[off + c];
        if tile == 0 {
            continue;
        }
        let goal_pos = (tile - 1) as usize;
        if goal_pos / W == r {
            goals[n] = (goal_pos % W) as u8;
            n += 1;
        }
    }
    2 * lc_removals(&goals[..n]) as u8
}

/// Linear-conflict penalty for column `c`, transpose of [`row_pen_direct`].
#[inline]
fn col_pen_direct(s: &State, c: usize) -> u8 {
    let mut goals = [0u8; W];
    let mut n = 0usize;
    for r in 0..W {
        let tile = s.0[r * W + c];
        if tile == 0 {
            continue;
        }
        let goal_pos = (tile - 1) as usize;
        if goal_pos % W == c {
            goals[n] = (goal_pos / W) as u8;
            n += 1;
        }
    }
    2 * lc_removals(&goals[..n]) as u8
}

#[inline]
fn ctx_h(ctx: &LcCtx) -> u8 {
    let mut total = ctx.manhattan as u32;
    for i in 0..W {
        total += ctx.row_pen[i] as u32 + ctx.col_pen[i] as u32;
    }
    total.min(u8::MAX as u32) as u8
}

impl IncHeuristic for LinearConflictInc {
    type Ctx = LcCtx;

    fn root(&self, s: &State, _stats: &mut SearchStats) -> (u8, LcCtx) {
        let mut manhattan: u32 = 0;
        let mut blank: u8 = 0;
        for p in 0..(W * W) {
            let t = s.0[p];
            if t == 0 {
                blank = p as u8;
                continue;
            }
            manhattan += manhattan_of(t, p) as u32;
        }
        let mut row_pen = [0u8; W];
        let mut col_pen = [0u8; W];
        for i in 0..W {
            row_pen[i] = row_pen_direct(s, i);
            col_pen[i] = col_pen_direct(s, i);
        }
        let ctx = LcCtx {
            manhattan: manhattan as u8,
            row_pen,
            col_pen,
            blank,
        };
        (ctx_h(&ctx), ctx)
    }

    fn advance(
        &self,
        parent: &LcCtx,
        child: &State,
        m: Move,
        _stats: &mut SearchStats,
    ) -> (u8, LcCtx) {
        // After the move: the blank moved by `delta(m)`, the tile that filled
        // the blank's old cell is what just moved. Hence:
        //   from = blank_new (where the tile was, in the parent)
        //   to   = blank_old (where the tile is now, in the child)
        let delta: i32 = match m {
            Move::Up => -(W as i32),
            Move::Down => W as i32,
            Move::Left => -1,
            Move::Right => 1,
        };
        let from_cell = (parent.blank as i32 + delta) as usize;
        let to_cell = parent.blank as usize;
        let tile = child.0[to_cell];
        debug_assert_ne!(tile, 0, "moved tile cannot be the blank");

        let dmd = manhattan_of(tile, to_cell) - manhattan_of(tile, from_cell);
        let manhattan = (parent.manhattan as i32 + dmd) as u8;

        let mut row_pen = parent.row_pen;
        let mut col_pen = parent.col_pen;
        let rf = from_cell / W;
        let rt = to_cell / W;
        let cf = from_cell % W;
        let ct = to_cell % W;
        if rf != rt {
            // Vertical slide: tile changed row but kept column.
            row_pen[rf] = row_pen_direct(child, rf);
            row_pen[rt] = row_pen_direct(child, rt);
            col_pen[cf] = col_pen_direct(child, cf); // cf == ct
        } else {
            // Horizontal slide: tile changed column but kept row.
            col_pen[cf] = col_pen_direct(child, cf);
            col_pen[ct] = col_pen_direct(child, ct);
            row_pen[rf] = row_pen_direct(child, rf); // rf == rt
        }

        let ctx = LcCtx {
            manhattan,
            row_pen,
            col_pen,
            blank: from_cell as u8,
        };
        (ctx_h(&ctx), ctx)
    }
}

/// `len(arr) - LIS(arr)` — the minimum number of elements to remove so the
/// remaining sequence is strictly increasing. O(n²) DP; `n ≤ 5`.
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
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::ManhattanHeuristic;
    use crate::puzzle24::state::{State, GOAL};

    #[test]
    fn lc_of_goal_is_zero() {
        assert_eq!(LinearConflictHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn lc_detects_row_swap() {
        // Swap tiles 1 and 2 in row 0: 2 1 3 4 5 / ... → MD = 2, LC adds 2 → 4.
        let mut s = GOAL.0;
        s.swap(0, 1);
        let s = State(s);
        assert_eq!(ManhattanHeuristic.h(&s), 2);
        assert_eq!(LinearConflictHeuristic.h(&s), 4);
    }

    #[test]
    fn lc_detects_column_swap() {
        // Swap tiles 1 and 6 in column 0 (positions 0 and 5 on the 5-wide
        // board): row 0 starts with 6, row 1 with 1. Both home column 0, in each
        // other's way → MD = 2, LC adds 2 → 4.
        let mut s = GOAL.0;
        s.swap(0, 5);
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
            assert!(lc >= md, "LC {lc} < MD {md} for {raw:?}");
            assert!(lc <= true_dist, "LC {lc} > truth {true_dist} for {raw:?}");
        }
    }

    /// Incremental LC's `root` must match the from-scratch LC on every state.
    #[test]
    fn lc_inc_root_matches_scratch_on_shallow_bfs() {
        let truth = bfs_distances(9);
        let mut stats = SearchStats::default();
        for raw in truth.keys() {
            let s = State(*raw);
            let h_scratch = LinearConflictHeuristic.h(&s);
            let (h_inc, _) = IncHeuristic::root(&LinearConflictInc, &s, &mut stats);
            assert_eq!(h_inc, h_scratch, "root mismatch for {raw:?}");
        }
    }

    /// Incremental LC's `advance` must agree with a fresh `root` at every step
    /// of a long random walk — the workhorse correctness test for any
    /// incremental update.
    #[test]
    fn lc_inc_advance_matches_fresh_root_random_walk() {
        let mut rng: u64 = 0x0C0D_EFAC_EBAD_BEEF;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (mut h, mut ctx) = IncHeuristic::root(&LinearConflictInc, &s, &mut stats);
        assert_eq!(h, 0);
        for step in 0..2000 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = LinearConflictInc.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&LinearConflictInc, &ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "advance vs root diverged at step {} (move {:?}, state {:?})",
                step, m, ns.0
            );
            assert_eq!(
                h_adv,
                LinearConflictHeuristic.h(&ns),
                "advance vs scratch LC diverged at step {step}"
            );
            s = ns;
            h = h_adv;
            ctx = ctx_adv;
        }
        let _ = h;
    }

    #[test]
    fn lc_removals_lis_correct_on_small_cases() {
        assert_eq!(lc_removals(&[]), 0);
        assert_eq!(lc_removals(&[0]), 0);
        assert_eq!(lc_removals(&[0, 1, 2, 3]), 0);
        assert_eq!(lc_removals(&[3, 2, 1, 0]), 3); // LIS = [0], remove 3
        assert_eq!(lc_removals(&[1, 3, 2, 0]), 2); // LIS = [1, 3] or [1, 2], remove 2
        assert_eq!(lc_removals(&[0, 1, 2, 3, 4]), 0); // full 5-cell line, no conflict
        assert_eq!(lc_removals(&[4, 3, 2, 1, 0]), 4); // fully reversed 5-cell line
    }
}
