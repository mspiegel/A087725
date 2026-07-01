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

/// Incremental max of two [`IncHeuristic`]s — the incremental analogue of
/// [`MaxHeuristic`](crate::puzzle24::pdb::MaxHeuristic). The per-node context is
/// the pair of sub-contexts (both `Copy`, so the tuple is `Copy` and threads
/// through the recursion). Admissible whenever both components are.
///
/// Used by the per-board *selective* solver to run a cheap `max(LinearConflict,
/// WalkingDistance)` on boards where the (expensive) zero-aware PDB term adds
/// nothing at the root — see `examples/ladder24.rs`.
pub struct MaxInc<A, B> {
    a: A,
    b: B,
}

impl<A, B> MaxInc<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> crate::puzzle24::search::idastar::IncHeuristic for MaxInc<A, B>
where
    A: crate::puzzle24::search::idastar::IncHeuristic,
    B: crate::puzzle24::search::idastar::IncHeuristic,
{
    type Ctx = (A::Ctx, B::Ctx);

    #[inline]
    fn root(
        &self,
        s: &State,
        stats: &mut crate::puzzle24::search::SearchStats,
    ) -> (u8, Self::Ctx) {
        let (ha, ca) = self.a.root(s, stats);
        let (hb, cb) = self.b.root(s, stats);
        (ha.max(hb), (ca, cb))
    }

    #[inline]
    fn advance(
        &self,
        parent: &Self::Ctx,
        child: &State,
        m: crate::puzzle24::state::Move,
        stats: &mut crate::puzzle24::search::SearchStats,
    ) -> (u8, Self::Ctx) {
        let (ha, ca) = self.a.advance(&parent.0, child, m, stats);
        let (hb, cb) = self.b.advance(&parent.1, child, m, stats);
        (ha.max(hb), (ca, cb))
    }
}

impl<A, B> crate::puzzle24::search::idastar::IncHeuristicMut for MaxInc<A, B>
where
    A: crate::puzzle24::search::idastar::IncHeuristicMut,
    B: crate::puzzle24::search::idastar::IncHeuristicMut,
{
    type Ctx = (A::Ctx, B::Ctx);

    #[inline]
    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let (ha, ca) = self.a.root(s);
        let (hb, cb) = self.b.root(s);
        (ha.max(hb), (ca, cb))
    }

    #[inline]
    fn make(&self, ctx: &mut Self::Ctx, child: &State, m: crate::puzzle24::state::Move) -> u8 {
        let ha = self.a.make(&mut ctx.0, child, m);
        let hb = self.b.make(&mut ctx.1, child, m);
        ha.max(hb)
    }

    #[inline]
    fn unmake(&self, ctx: &mut Self::Ctx, m: crate::puzzle24::state::Move) {
        self.a.unmake(&mut ctx.0, m);
        self.b.unmake(&mut ctx.1, m);
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

    fn root(&self, s: &State, _stats: &mut crate::puzzle24::search::SearchStats) -> (u8, u8) {
        let h = ManhattanHeuristic.h(s);
        (h, h)
    }

    fn advance(
        &self,
        parent: &u8,
        child: &State,
        m: crate::puzzle24::state::Move,
        _stats: &mut crate::puzzle24::search::SearchStats,
    ) -> (u8, u8) {
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

/// Make/unmake context for [`IncManhattan`]: the running distance plus a LIFO
/// stack of per-move deltas so [`unmake`](crate::puzzle24::search::idastar::IncHeuristicMut::unmake)
/// can reverse without the child state. (Manhattan's `Copy` context is a single
/// byte, so this exists only to satisfy the trait for use as a `MaxInc`/select
/// component — there is no copy to save.)
pub struct ManhattanMutCtx {
    h: u8,
    undo: Vec<i8>,
}

impl crate::puzzle24::search::idastar::IncHeuristicMut for IncManhattan {
    type Ctx = ManhattanMutCtx;

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let h = ManhattanHeuristic.h(s);
        (h, ManhattanMutCtx { h, undo: Vec::with_capacity(220) })
    }

    fn make(&self, ctx: &mut Self::Ctx, child: &State, m: crate::puzzle24::state::Move) -> u8 {
        let nb = child.blank_pos() as usize;
        let b = match m {
            crate::puzzle24::state::Move::Up => nb + W,
            crate::puzzle24::state::Move::Down => nb - W,
            crate::puzzle24::state::Move::Left => nb + 1,
            crate::puzzle24::state::Move::Right => nb - 1,
        };
        let tile = child.0[b];
        let delta = tile_manhattan(tile, b) as i16 - tile_manhattan(tile, nb) as i16;
        ctx.h = (ctx.h as i16 + delta) as u8;
        ctx.undo.push(delta as i8);
        ctx.h
    }

    fn unmake(&self, ctx: &mut Self::Ctx, _m: crate::puzzle24::state::Move) {
        let delta = ctx.undo.pop().expect("unmake without matching make");
        ctx.h = (ctx.h as i16 - delta as i16) as u8;
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

    /// `MaxInc(LC, WD)` must equal `max(LC, WD)` at the root and stay in sync via
    /// `advance` over a long walk (the combinator just maxes the two terms).
    #[test]
    fn maxinc_lc_wd_matches_scratch_max_random_walk() {
        use crate::puzzle24::search::idastar::IncHeuristic;
        use crate::puzzle24::search::{
            LinearConflictHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
            WalkingDistanceInc,
        };
        WalkingDistanceHeuristic::warm_up();
        let mx = MaxInc::new(LinearConflictInc, WalkingDistanceInc);
        let mut rng: u64 = 0x5AFE_C0DE_1234_9999;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (h0, mut ctx) = mx.root(&s, &mut stats);
        assert_eq!(h0, 0);
        for step in 0..2000 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = mx.advance(&ctx, &ns, m, &mut stats);
            let scratch = LinearConflictHeuristic.h(&ns).max(WalkingDistanceHeuristic.h(&ns));
            assert_eq!(h_adv, scratch, "MaxInc advance vs scratch diverged at step {}", step);
            let (h_fresh, _) = mx.root(&ns, &mut stats);
            assert_eq!(h_adv, h_fresh, "MaxInc advance vs fresh root diverged at step {}", step);
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// Manhattan make/unmake driver must match the copy driver (length + nodes).
    /// Its `undo` delta stack must reverse each move exactly.
    #[test]
    fn manhattan_mut_idastar_matches_copy_length_and_nodes() {
        use crate::puzzle24::search::{idastar_inc_mut_with_stats, idastar_inc_with_stats};
        let mut rng: u64 = 0x2B3C_4D5E_6F70_8191;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        for _ in 0..5 {
            let mut s = GOAL;
            for _ in 0..12 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (c, cs) = idastar_inc_with_stats(&s, &IncManhattan);
            let (m, ms) = idastar_inc_mut_with_stats(&s, &IncManhattan);
            assert_eq!(c.expect("copy").len(), m.expect("mut").len(), "Manhattan length differs");
            assert_eq!(cs.nodes, ms.nodes, "Manhattan node count differs");
        }
    }

    /// `MaxInc` make/unmake composition (here `max(Manhattan, LC)`) must match the
    /// copy driver — verifies both members' mut paths compose correctly.
    #[test]
    fn maxinc_mut_idastar_matches_copy_length_and_nodes() {
        use crate::puzzle24::search::{idastar_inc_mut_with_stats, idastar_inc_with_stats};
        use crate::puzzle24::search::LinearConflictInc;
        let mx = MaxInc::new(IncManhattan, LinearConflictInc);
        let mut rng: u64 = 0x9F1E_2D3C_4B5A_6978;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        for _ in 0..5 {
            let mut s = GOAL;
            for _ in 0..14 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (c, cs) = idastar_inc_with_stats(&s, &mx);
            let (m, ms) = idastar_inc_mut_with_stats(&s, &mx);
            assert_eq!(c.expect("copy").len(), m.expect("mut").len(), "MaxInc length differs");
            assert_eq!(cs.nodes, ms.nodes, "MaxInc node count differs");
        }
    }
}
