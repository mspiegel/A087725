//! Walking Distance for the 24-puzzle (Ken'ichiro Takahashi). Ported from
//! `crate::puzzle15::search::walking_distance` to the 5×5 board.
//!
//! The row-WD state is a 5×5 matrix `M[r][g]` = "non-blank tiles in row r whose
//! goal row is g," plus the blank's row index ∈ {0..4}. Column-WD is the same
//! shape with rows and columns swapped. By symmetry of the 5×5 goal, row-WD and
//! column-WD share one BFS distance table.
//!
//! **Differences from the 15-puzzle port:**
//!
//! - *State count.* The reachable 5×5 row-WD space is **65,650,495** states
//!   (measured), vs the 4×4's 24,964 — five orders of magnitude bigger. The BFS
//!   that fills the table is a one-off but takes ~25–30 s and the resident
//!   `HashMap` is ~1.2 GiB, so [`WalkingDistanceHeuristic::warm_up`] should be
//!   called once at startup and the heuristic reused across many solves (the
//!   ladder/Phase-2 harnesses do exactly this).
//! - *Key packing.* A naïve pack of all 25 cells (3 bits each) + blank axis is
//!   78 bits and would need a `u128`. We instead drop **each row's last column**:
//!   given the blank axis-index, row `r`'s margin is known (5, except the blank's
//!   row which is 4), so `m[r][W-1] = margin(r) − Σ_{c<W-1} m[r][c]` is derivable.
//!   That leaves `5·4 = 20` cells (3 bits each) + the 3-bit blank index = **63
//!   bits → fits a `u64`**, halving both the key width and the table footprint.

use super::{Heuristic, IncHeuristic, IncHeuristicMut, SearchStats};
use crate::puzzle24::state::{Move, State, W};
use std::collections::HashMap;
use std::sync::OnceLock;

/// A 5×5 row-distribution matrix.
type WdMatrix = [[u8; W]; W];

/// Pack `(matrix, blank-axis-index)` into a u64 key. Each stored cell holds a
/// value ≤ 5 (3 bits); we omit the last column of every row (derivable from the
/// blank-aware row margin), leaving `5·4 = 20` cells + the 3-bit blank index =
/// 63 bits — within u64. Injective on reachable WD-states because the blank
/// index (carried in the key) pins down every omitted cell.
#[inline]
fn pack(m: &WdMatrix, blank_idx: u8) -> u64 {
    let mut k: u64 = blank_idx as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

/// Goal row-distribution matrix for the 24-puzzle (also the column-distribution
/// goal by symmetry). Tiles 1–5 have goal row 0; 6–10 row 1; 11–15 row 2; 16–20
/// row 3; 21–24 row 4 (the blank, also row 4, is excluded → only 4 there).
fn goal_matrix() -> WdMatrix {
    let mut m = [[0u8; W]; W];
    m[0][0] = 5;
    m[1][1] = 5;
    m[2][2] = 5;
    m[3][3] = 5;
    m[4][4] = 4;
    m
}

const GOAL_BLANK_IDX: u8 = 4;

/// BFS from the goal WD-state, collecting every reachable matrix-and-blank
/// combination with its distance to goal.
fn build_table() -> HashMap<u64, u8> {
    let goal_m = goal_matrix();
    let goal_key = pack(&goal_m, GOAL_BLANK_IDX);

    // Reserve for the full reachable set (65,650,495) up front so the BFS never
    // pays for an incremental rehash.
    let mut table: HashMap<u64, u8> = HashMap::with_capacity(66_000_000);
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
    /// call regardless. The 5×5 BFS is heavier than the 4×4 one, so warming up
    /// at startup keeps the first solve from paying for it.
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
/// Key invariant: a slide moves one tile by one cell, vertically OR
/// horizontally. The row-WD axis only sees changes when a tile crosses a row
/// boundary (vertical slide); the column-WD axis only on horizontal slides.
/// So per node, **exactly one** of the two matrices, blanks, and `h` halves
/// changes — the other half is carried forward verbatim.
///
/// For the affected axis we do two `+/-1` updates on the matrix entries, a fresh
/// `pack`, and a single hashmap lookup. `WalkingDistanceHeuristic::warm_up()`
/// must have been called first (same as for the non-incremental version).
pub struct WalkingDistanceInc;

/// Per-node state for [`WalkingDistanceInc`]. The two matrices, the blank's row
/// and column, and both axis-`h` values pre-computed. `Copy` so the IDA*
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
        (h_row + h_col, WdCtx { m_row, m_col, br, bc, h_row, h_col })
    }

    fn advance(
        &self,
        parent: &WdCtx,
        child: &State,
        m: Move,
        _stats: &mut SearchStats,
    ) -> (u8, WdCtx) {
        #[cfg(feature = "verifier-stats")]
        { _stats.wd_advances += 1; }
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

/// One undo frame for the make/unmake path: which axis moved (`vertical` ⇒ the
/// row matrix), the two matrix cells that changed (`i -= 1`, `j += 1` at goal
/// index `g`), and the pre-move scalars to restore. Reversing is a `±1` flip of
/// those two cells plus a scalar copy — no `pack`/table lookup needed.
struct WdUndo {
    vertical: bool,
    i: u8,
    j: u8,
    g: u8,
    br: u8,
    bc: u8,
    h_row: u8,
    h_col: u8,
}

/// Make/unmake context for [`WalkingDistanceInc`]. Unlike the `Copy`
/// [`IncHeuristic`] path — which rebuilds a whole [`WdCtx`] (two 5×5 matrices)
/// per node — this mutates one matrix cell-pair in place and records a tiny undo
/// frame, so backtracking is a `±1` flip instead of a 54-byte copy.
pub struct WdMutCtx {
    cur: WdCtx,
    undo: Vec<WdUndo>,
}

impl IncHeuristicMut for WalkingDistanceInc {
    type Ctx = WdMutCtx;

    fn root(&self, s: &State) -> (u8, WdMutCtx) {
        let mut stats = SearchStats::default();
        let (h, cur) = <Self as IncHeuristic>::root(self, s, &mut stats);
        (h, WdMutCtx { cur, undo: Vec::with_capacity(220) })
    }

    fn make(&self, ctx: &mut WdMutCtx, child: &State, m: Move) -> u8 {
        let c = &mut ctx.cur;
        let parent_blank = (c.br as usize) * W + c.bc as usize;
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

        // Snapshot the scalars before mutating (matrix reversal is derived below).
        let mut undo = WdUndo {
            vertical: rf != rt,
            i: 0,
            j: 0,
            g: 0,
            br: c.br,
            bc: c.bc,
            h_row: c.h_row,
            h_col: c.h_col,
        };

        if rf != rt {
            // Vertical slide: row-WD matrix changes; column axis untouched.
            c.m_row[rf][gr] -= 1;
            c.m_row[rt][gr] += 1;
            undo.i = rf as u8;
            undo.j = rt as u8;
            undo.g = gr as u8;
            c.h_row = *t
                .get(&pack(&c.m_row, new_br))
                .expect("row-WD state must be reachable from goal");
        } else {
            // Horizontal slide: column-WD matrix changes; row axis untouched.
            c.m_col[cf][gc] -= 1;
            c.m_col[ct][gc] += 1;
            undo.i = cf as u8;
            undo.j = ct as u8;
            undo.g = gc as u8;
            c.h_col = *t
                .get(&pack(&c.m_col, new_bc))
                .expect("col-WD state must be reachable from goal");
        }
        c.br = new_br;
        c.bc = new_bc;
        ctx.undo.push(undo);
        c.h_row + c.h_col
    }

    fn unmake(&self, ctx: &mut WdMutCtx, _m: Move) {
        let u = ctx.undo.pop().expect("unmake without matching make");
        let c = &mut ctx.cur;
        // Reverse the cell-pair flip (make did `i -= 1`, `j += 1`).
        if u.vertical {
            c.m_row[u.i as usize][u.g as usize] += 1;
            c.m_row[u.j as usize][u.g as usize] -= 1;
        } else {
            c.m_col[u.i as usize][u.g as usize] += 1;
            c.m_col[u.j as usize][u.g as usize] -= 1;
        }
        c.br = u.br;
        c.bc = u.bc;
        c.h_row = u.h_row;
        c.h_col = u.h_col;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::ManhattanHeuristic;
    use crate::puzzle24::state::{Move, State, GOAL};

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
        // WD is supposed to be tighter than Manhattan on average. Look for *any*
        // state in shallow BFS where WD strictly exceeds MD — proves the table
        // isn't degenerate.
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

    /// `pack`/`unpack` round-trip is implicit in lookups; here we check the key
    /// is injective on a few hand matrices (no 3-bit overflow at value 5).
    #[test]
    fn pack_is_injective_on_full_rows() {
        let goal = goal_matrix();
        let mut other = goal;
        other[0][0] = 4;
        other[1][0] = 1;
        assert_ne!(pack(&goal, GOAL_BLANK_IDX), pack(&other, GOAL_BLANK_IDX));
        // A cell value of 5 must not bleed into the neighbouring field.
        assert_eq!(goal[0][0], 5);
        assert_ne!(pack(&goal, 0), pack(&goal, 1));
    }

    /// Incremental WD `root` must match the from-scratch WD on every state.
    #[test]
    fn wd_inc_root_matches_scratch_on_shallow_bfs() {
        WalkingDistanceHeuristic::warm_up();
        let truth = bfs_distances(9);
        let mut stats = SearchStats::default();
        for (raw, _) in &truth {
            let s = State(*raw);
            let scratch = WalkingDistanceHeuristic.h(&s);
            let (inc, _) = IncHeuristic::root(&WalkingDistanceInc, &s, &mut stats);
            assert_eq!(inc, scratch, "root mismatch for {:?}", raw);
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
                "advance vs scratch WD diverged at step {}",
                step
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// The 5×5 row-WD reachable state space has a fixed, measured size. Pin it
    /// exactly: any change means the BFS transition rule or goal margins drifted.
    /// (4×4 reference: 24,964.)
    #[test]
    fn table_size_matches_measured() {
        const WD24_STATES: usize = 65_650_495;
        let n = WalkingDistanceHeuristic::table_size();
        assert_eq!(n, WD24_STATES, "WD table size drifted from the measured value");
        println!("24-puzzle WD table size: {} states", n);
    }

    /// The make/unmake driver must expand the exact same nodes and return the
    /// same optimal length as the copy driver — WD's fine-grained undo (±1 matrix
    /// flip + scalar restore) must reproduce the copy path's per-node state.
    #[test]
    fn wd_mut_idastar_matches_copy_length_and_nodes() {
        use crate::puzzle24::search::{idastar_inc_mut_with_stats, idastar_inc_with_stats};
        let mut rng: u64 = 0x51ED_270B_2E07_9AA1;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..5 {
            let mut s = GOAL;
            for _ in 0..18 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (c, cs) = idastar_inc_with_stats(&s, &WalkingDistanceInc);
            let (m, ms) = idastar_inc_mut_with_stats(&s, &WalkingDistanceInc);
            assert_eq!(
                c.expect("copy no sol").len(),
                m.expect("mut no sol").len(),
                "WD mut/copy optimal length differs"
            );
            assert_eq!(cs.nodes, ms.nodes, "WD mut/copy node count differs");
        }
    }
}
