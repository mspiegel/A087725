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

use super::{Heuristic, IncHeuristic, SearchStats};
use crate::puzzle15::state::{Move, State, W};
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
                        if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key) {
                            e.insert(next_depth);
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

/// Incremental Walking Distance: same admissible heuristic as
/// [`WalkingDistanceHeuristic`], advanced in O(1) per node.
///
/// Key invariant: a 15-puzzle slide moves one tile by one cell, vertically OR
/// horizontally. The row-WD axis only sees changes when a tile crosses a row
/// boundary (vertical slide); the column-WD axis only on horizontal slides.
/// So per node, **exactly one** of the two matrices, blanks, and `h` halves
/// changes — the other half is carried forward verbatim.
///
/// For the affected axis we do two `+/-1` updates on the matrix entries, a
/// fresh `pack`, and a single hashmap lookup — replacing today's 16-cell
/// rebuild + 2 packs + 2 lookups. `WalkingDistanceHeuristic::warm_up()` must
/// have been called first (same as for the non-incremental version).
pub struct WalkingDistanceInc;

/// Per-node state for [`WalkingDistanceInc`]. The two matrices, the blank's
/// row and column, and both axis-`h` values pre-computed. `Copy` so the IDA*
/// recursion can thread it by value.
#[derive(Clone, Copy, Debug)]
pub struct WdCtx {
    m_row: WdMatrix,
    m_col: WdMatrix,
    br: u8,
    bc: u8,
    h_row: u8,
    h_col: u8,
}

impl IncHeuristic for WalkingDistanceInc {
    type Ctx = WdCtx;

    fn root(&self, s: &State, _stats: &mut SearchStats) -> (u8, WdCtx) {
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
            m_row[r as usize][goal_pos / W] += 1;
            m_col[c as usize][goal_pos % W] += 1;
        }
        let t = table();
        let h_row = *t
            .get(&pack(&m_row, br))
            .expect("row-WD state must be reachable from goal");
        let h_col = *t
            .get(&pack(&m_col, bc))
            .expect("col-WD state must be reachable from goal");
        (
            h_row + h_col,
            WdCtx {
                m_row,
                m_col,
                br,
                bc,
                h_row,
                h_col,
            },
        )
    }

    fn advance(
        &self,
        parent: &WdCtx,
        child: &State,
        m: Move,
        _stats: &mut SearchStats,
    ) -> (u8, WdCtx) {
        #[cfg(feature = "verifier-stats")]
        {
            _stats.wd_advances += 1;
        }
        // The moved tile's (from, to) cells (see LinearConflictInc for the
        // identity): from = blank_new, to = blank_old. The parent carries the
        // blank's old (br, bc), so blank_old is reconstructed without rescanning.
        let parent_blank = (parent.br as usize) * W + parent.bc as usize;
        let delta: i32 = match m {
            Move::Up => -(W as i32),
            Move::Down => W as i32,
            Move::Left => -1,
            Move::Right => 1,
        };
        let from_cell = (parent_blank as i32 + delta) as usize;
        let to_cell = parent_blank;
        let tile = child.0[to_cell];
        debug_assert_ne!(tile, 0, "moved tile cannot be the blank");
        let goal_pos = (tile - 1) as usize;
        let gr = goal_pos / W;
        let gc = goal_pos % W;

        let rf = from_cell / W;
        let rt = to_cell / W;
        let cf = from_cell % W;
        let ct = to_cell % W;
        let new_br = rf as u8;
        let new_bc = cf as u8;
        let t = table();

        if rf != rt {
            // Vertical slide: row-WD matrix changes, blank-row changes; column
            // axis (m_col, bc, h_col) is byte-identical to parent.
            let mut m_row = parent.m_row;
            m_row[rf][gr] -= 1;
            m_row[rt][gr] += 1;
            let h_row = *t
                .get(&pack(&m_row, new_br))
                .expect("row-WD state must be reachable from goal");
            let ctx = WdCtx {
                m_row,
                m_col: parent.m_col,
                br: new_br,
                bc: new_bc,
                h_row,
                h_col: parent.h_col,
            };
            (h_row + parent.h_col, ctx)
        } else {
            // Horizontal slide: column-WD matrix changes, blank-col changes;
            // row axis is byte-identical.
            let mut m_col = parent.m_col;
            m_col[cf][gc] -= 1;
            m_col[ct][gc] += 1;
            let h_col = *t
                .get(&pack(&m_col, new_bc))
                .expect("col-WD state must be reachable from goal");
            let ctx = WdCtx {
                m_row: parent.m_row,
                m_col,
                br: new_br,
                bc: new_bc,
                h_row: parent.h_row,
                h_col,
            };
            (parent.h_row + h_col, ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::search::ManhattanHeuristic;
    use crate::puzzle15::state::{Move, State, GOAL};

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
            assert!(est <= true_dist, "WD {est} > truth {true_dist} for {raw:?}");
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

    /// Incremental WD `root` must match the from-scratch WD on every state.
    #[test]
    fn wd_inc_root_matches_scratch_on_shallow_bfs() {
        WalkingDistanceHeuristic::warm_up();
        let truth = bfs_distances(10);
        let mut stats = SearchStats::default();
        for raw in truth.keys() {
            let s = State(*raw);
            let scratch = WalkingDistanceHeuristic.h(&s);
            let (inc, _) = IncHeuristic::root(&WalkingDistanceInc, &s, &mut stats);
            assert_eq!(inc, scratch, "root mismatch for {raw:?}");
        }
    }

    /// `advance` must agree with a fresh `root` at every step of a long random
    /// walk. Also verifies it matches the from-scratch WD at every step.
    #[test]
    fn wd_inc_advance_matches_fresh_root_random_walk() {
        WalkingDistanceHeuristic::warm_up();
        let mut rng: u64 = 0xBADD_C0FE_F1F0_BEEF;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (_, mut ctx) = IncHeuristic::root(&WalkingDistanceInc, &s, &mut stats);
        for step in 0..2000 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = WalkingDistanceInc.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&WalkingDistanceInc, &ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "advance vs root diverged at step {} (move {:?}, state {:?})",
                step, m, ns.0
            );
            assert_eq!(
                h_adv,
                WalkingDistanceHeuristic.h(&ns),
                "advance vs scratch WD diverged at step {step}"
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    #[test]
    fn table_size_is_around_25k() {
        // Standard 4x4 row-WD state count is 24,964. We expect ours to be in
        // that ballpark (within ~10x — the goal layout is slightly
        // asymmetric due to the blank).
        let n = WalkingDistanceHeuristic::table_size();
        assert!(
            (1_000..100_000).contains(&n),
            "unexpected WD table size {n} (expected ~25k)"
        );
    }

    /// Fast smoke for WD-driving-a-search: incremental WD (`WalkingDistanceInc`)
    /// must guide IDA\* to an *optimal* solution on every board within true
    /// distance ≤ 10 of GOAL. This exercises the same invariant as the (gated,
    /// slow) 24-puzzle `wd_mut_idastar_matches_copy_length_and_nodes` — that the
    /// WD heuristic is admissible enough to drive a correct optimal search — but
    /// on the 15-puzzle's tiny ~25k-state table it runs in a fraction of a second.
    #[test]
    fn wd_inc_idastar_solves_optimally_on_shallow_ball() {
        use crate::puzzle15::search::idastar::idastar_inc_with_stats;
        let table = bfs_distances(10);
        for (raw, &truth) in &table {
            let s = State(*raw);
            let (sol, _stats) = idastar_inc_with_stats(&s, &WalkingDistanceInc);
            let sol = sol.expect("WD-guided IDA* should solve");
            assert_eq!(
                sol.len() as u8,
                truth,
                "WD IDA* returned len {} but true dist is {} for {:?}",
                sol.len(),
                truth,
                raw
            );
            let mut cur = s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL, "WD solution doesn't reach goal from {raw:?}");
        }
    }
}
