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
impl<H: Heuristic + ?Sized> Heuristic for &H {
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

impl<A, B> crate::puzzle24::search::recursive::IncHeuristic for MaxInc<A, B>
where
    A: crate::puzzle24::search::recursive::IncHeuristic,
    B: crate::puzzle24::search::recursive::IncHeuristic,
{
    type Ctx = (A::Ctx, B::Ctx);

    #[inline]
    fn root(&self, s: &State, stats: &mut crate::puzzle24::search::SearchStats) -> (u8, Self::Ctx) {
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

/// Variadic max of ≥1 incremental heuristics: the ergonomic form of hand-nesting
/// [`MaxInc::new`]. `max_inc!(a, b, c)` expands to `MaxInc::new(a, MaxInc::new(b,
/// c))` — a right-associated fold, so the component types may all differ (each
/// `MaxInc` layer just maxes its two children). A real `fn` can't be variadic
/// over heterogeneous heuristic types, hence the macro; it is pure type
/// composition with **zero** runtime cost, and the result implements whichever of
/// [`IncHeuristic`](crate::puzzle24::search::IncHeuristic) /
/// [`IncHeuristicMut`](crate::puzzle24::search::IncHeuristicMut) all components
/// do. `max_inc!(a)` is just `a`. A trailing comma is accepted.
///
/// ```ignore
/// // max(cWD, k6 zPDB, three-group k8 zPDB), no manual nesting:
/// let h = max_inc!(Cwd::new(), z_k6, z_k8);
/// ```
#[macro_export]
macro_rules! max_inc {
    ($h:expr $(,)?) => { $h };
    ($h:expr, $($rest:expr),+ $(,)?) => {
        $crate::puzzle24::search::MaxInc::new($h, $crate::max_inc!($($rest),+))
    };
}

/// Lipschitz-deferred variant of [`MaxInc`]: `A` is the cheap component, `B`
/// the expensive one. **`B` must be 1-Lipschitz per move** (`|h(child) −
/// h(parent)| ≤ 1` — true for WD/cWD/zero-aware PDBs, whose projected edges
/// cost ≤ 1; NOT for LinearConflict, which can jump by more than 1).
///
/// Instead of advancing `B` on every child, moves are pushed onto a pending
/// stack. With `v` = `B`'s value at its last applied state and `k` pending
/// moves, the child's true `h_b` lies in `[v−k−1, v+k+1]`, so:
///
/// - `B` can prune the child only if `v + k + 1 > budget`. Only then is `B`
///   caught up (pending moves applied in order) and probed exactly — the same
///   prune decision the eager [`MaxInc`] makes, so the search tree and node
///   counts are **identical**; only `B`'s probes collapse to the minimal set
///   (the near-frontier region where `B` is competitive with `A`).
/// - Otherwise the move is deferred and `max(h_a, v − k − 1)` is returned —
///   admissible by the Lipschitz bound. Deferred moves popped on backtrack
///   before a catch-up are *never* applied to `B` at all.
///
/// [`make_bounded`]: crate::puzzle24::search::recursive::IncHeuristicMut::make_bounded
pub struct LazyMaxInc<A, B> {
    a: A,
    b: B,
}

impl<A, B> LazyMaxInc<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

/// Eager copy-context impl so [`LazyMaxInc`] satisfies drivers requiring both
/// bounds (the parallel splitter's frontier expansion is cold — laziness only
/// matters in the hot make/unmake recursion).
impl<A, B> crate::puzzle24::search::recursive::IncHeuristic for LazyMaxInc<A, B>
where
    A: crate::puzzle24::search::recursive::IncHeuristic,
    B: crate::puzzle24::search::recursive::IncHeuristic,
{
    type Ctx = (A::Ctx, B::Ctx);

    #[inline]
    fn root(&self, s: &State, stats: &mut crate::puzzle24::search::SearchStats) -> (u8, Self::Ctx) {
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

/// Incremental Manhattan heuristic: maintains the running distance in O(1) per
/// move. A single move slides exactly one tile by one cell, so the tile that
/// moved into the parent's blank cell changes its Manhattan term by ±1 and
/// nothing else does. Used to validate the [`IncHeuristic`] driver and as the
/// template for the zero-aware incremental PDB evaluator.
///
/// [`IncHeuristic`]: crate::puzzle24::search::recursive::IncHeuristic
pub struct IncManhattan;

/// Manhattan distance of a single tile value at a given cell.
#[inline]
fn tile_manhattan(tile: u8, pos: usize) -> u8 {
    let goal_pos = (tile - 1) as usize;
    let dr = (pos / W) as i32 - (goal_pos / W) as i32;
    let dc = (pos % W) as i32 - (goal_pos % W) as i32;
    (dr.unsigned_abs() + dc.unsigned_abs()) as u8
}

impl crate::puzzle24::search::recursive::IncHeuristic for IncManhattan {
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
                    assert!(diff == 1 || diff == -1, "Δh = {diff} not ±1");
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
            assert!(est <= truth, "h({state:?}) = {est} > true {truth}");
        }
    }

    /// `MaxInc(LC, WD)` must equal `max(LC, WD)` at the root and stay in sync via
    /// `advance` over a long walk (the combinator just maxes the two terms).
    #[test]
    fn maxinc_lc_wd_matches_scratch_max_random_walk() {
        use crate::puzzle24::search::recursive::IncHeuristic;
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
            let scratch = LinearConflictHeuristic
                .h(&ns)
                .max(WalkingDistanceHeuristic.h(&ns));
            assert_eq!(
                h_adv, scratch,
                "MaxInc advance vs scratch diverged at step {step}"
            );
            let (h_fresh, _) = mx.root(&ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "MaxInc advance vs fresh root diverged at step {step}"
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// `max_inc!` is exactly the right-nested `MaxInc` and reports the true max of
    /// its components. Single-arg is the identity; a trailing comma is accepted.
    #[test]
    fn max_inc_macro_folds_and_matches_nested() {
        use crate::max_inc;
        use crate::puzzle24::search::{IncHeuristic, LinearConflictHeuristic, LinearConflictInc};
        // Three distinct component types, to exercise the heterogeneous fold.
        let macro_h = max_inc!(IncManhattan, IncManhattan, LinearConflictInc);
        let nested_h = MaxInc::new(IncManhattan, MaxInc::new(IncManhattan, LinearConflictInc));
        // Single-arg identity + trailing comma both parse.
        let solo = max_inc!(IncManhattan,);

        let mut rng: u64 = 0x1357_9BDF_2468_ACE0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..200 {
            let mut s = GOAL;
            for _ in 0..16 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let mut st = crate::puzzle24::search::SearchStats::default();
            let hm = macro_h.root(&s, &mut st).0;
            let hn = nested_h.root(&s, &mut st).0;
            let expect = ManhattanHeuristic.h(&s).max(LinearConflictHeuristic.h(&s));
            assert_eq!(hm, hn, "macro fold != hand-nested MaxInc");
            assert_eq!(hm, expect, "macro fold != scratch max");
            assert_eq!(
                solo.root(&s, &mut st).0,
                ManhattanHeuristic.h(&s),
                "solo != identity"
            );
        }
    }
}
