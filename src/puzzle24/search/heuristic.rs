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

    /// Forward the cheap child lower-bound (e.g. cWD's neighbor-prune) so it is
    /// retained through the combiner. Each side's `child_h_lb` bounds only its own
    /// term, but `max(a, b) ≥ a ≥ a_lb` (and symmetrically for `b`), so the max of
    /// whichever bounds exist is an admissible LB on the combined child `h`. `None`
    /// means "no cheap bound available" (NOT 0), so we keep the other side's real
    /// bound rather than clamping the pre-prune to a useless 0.
    #[inline]
    fn child_h_lb(
        &self,
        ctx: &Self::Ctx,
        s: &State,
        blank: u8,
        m: crate::puzzle24::state::Move,
    ) -> Option<u8> {
        match (
            self.a.child_h_lb(&ctx.0, s, blank, m),
            self.b.child_h_lb(&ctx.1, s, blank, m),
        ) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }
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
/// [`make_bounded`]: crate::puzzle24::search::idastar::IncHeuristicMut::make_bounded
pub struct LazyMaxInc<A, B> {
    a: A,
    b: B,
}

impl<A, B> LazyMaxInc<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

/// Per-level record for [`LazyMaxCtx`]: was this level's move applied to `B`
/// (carrying `B`'s pre-move value for restore) or deferred?
enum LazyTag {
    Pending,
    Applied(u8),
}

/// Context for [`LazyMaxInc`]: the two sub-contexts plus the deferral state.
/// `pending` always corresponds to the top `pending.len()` entries of `tags`
/// (catch-up flips them all to `Applied` before anything new is pushed).
pub struct LazyMaxCtx<CA, CB> {
    a: CA,
    b: CB,
    /// `B`'s exact value at its currently-applied state.
    b_val: u8,
    /// Moves made in the search but not yet applied to `B` (with the post-move
    /// state, for `B` impls that read it).
    pending: Vec<(crate::puzzle24::state::Move, State)>,
    tags: Vec<LazyTag>,
}

impl<A, B> LazyMaxInc<A, B>
where
    A: crate::puzzle24::search::idastar::IncHeuristicMut,
    B: crate::puzzle24::search::idastar::IncHeuristicMut,
{
    /// Apply all pending moves to `B` in order, flipping their tags to
    /// `Applied` so backtracking unmakes them properly.
    fn catch_up(&self, ctx: &mut LazyMaxCtx<A::Ctx, B::Ctx>) {
        let k = ctx.pending.len();
        let base = ctx.tags.len() - k;
        for (j, (pm, ps)) in ctx.pending.drain(..).enumerate() {
            let prev = ctx.b_val;
            ctx.b_val = self.b.make(&mut ctx.b, &ps, pm);
            debug_assert!(
                (ctx.b_val as i16 - prev as i16).abs() <= 1,
                "B is not 1-Lipschitz: {} -> {}",
                prev,
                ctx.b_val
            );
            ctx.tags[base + j] = LazyTag::Applied(prev);
        }
    }
}

impl<A, B> crate::puzzle24::search::idastar::IncHeuristicMut for LazyMaxInc<A, B>
where
    A: crate::puzzle24::search::idastar::IncHeuristicMut,
    B: crate::puzzle24::search::idastar::IncHeuristicMut,
{
    type Ctx = LazyMaxCtx<A::Ctx, B::Ctx>;

    #[inline]
    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let (ha, ca) = self.a.root(s);
        let (hb, cb) = self.b.root(s);
        (
            ha.max(hb),
            LazyMaxCtx {
                a: ca,
                b: cb,
                b_val: hb,
                pending: Vec::with_capacity(220),
                tags: Vec::with_capacity(220),
            },
        )
    }

    /// Exact value on demand (catch-up + probe).
    #[inline]
    fn make(&self, ctx: &mut Self::Ctx, child: &State, m: crate::puzzle24::state::Move) -> u8 {
        let ha = self.a.make(&mut ctx.a, child, m);
        self.catch_up(ctx);
        let prev = ctx.b_val;
        let hb = self.b.make(&mut ctx.b, child, m);
        debug_assert!((hb as i16 - prev as i16).abs() <= 1, "B is not 1-Lipschitz");
        ctx.b_val = hb;
        ctx.tags.push(LazyTag::Applied(prev));
        ha.max(hb)
    }

    #[inline]
    fn make_bounded(
        &self,
        ctx: &mut Self::Ctx,
        child: &State,
        m: crate::puzzle24::state::Move,
        budget: u8,
    ) -> u8 {
        let ha = self.a.make(&mut ctx.a, child, m);
        // A alone prunes: B's value is irrelevant (the child is pruned at first
        // touch and the pending move pops right back off on the unmake). No
        // Lipschitz argument needed — `ha` is admissible.
        if ha > budget {
            ctx.pending.push((m, *child));
            ctx.tags.push(LazyTag::Pending);
            return ha;
        }
        let k = ctx.pending.len();
        // Lipschitz upper bound on the child's h_b; if it can't exceed the
        // budget, B can't prune here either.
        if ctx.b_val as usize + k + 1 <= budget as usize {
            ctx.pending.push((m, *child));
            ctx.tags.push(LazyTag::Pending);
            // Admissible: true h_b ≥ b_val − k − 1 (may saturate to 0).
            return ha.max(ctx.b_val.saturating_sub(k as u8 + 1));
        }
        self.catch_up(ctx);
        let prev = ctx.b_val;
        let hb = self.b.make(&mut ctx.b, child, m);
        debug_assert!((hb as i16 - prev as i16).abs() <= 1, "B is not 1-Lipschitz");
        ctx.b_val = hb;
        ctx.tags.push(LazyTag::Applied(prev));
        ha.max(hb)
    }

    #[inline]
    fn unmake(&self, ctx: &mut Self::Ctx, m: crate::puzzle24::state::Move) {
        self.a.unmake(&mut ctx.a, m);
        match ctx.tags.pop().expect("unmake without matching make") {
            LazyTag::Pending => {
                let (pm, _) = ctx.pending.pop().expect("pending/tags out of sync");
                debug_assert!(pm == m, "pending move mismatch");
            }
            LazyTag::Applied(prev) => {
                self.b.unmake(&mut ctx.b, m);
                ctx.b_val = prev;
            }
        }
    }

    /// Same forwarding as [`MaxInc::child_h_lb`].
    #[inline]
    fn child_h_lb(
        &self,
        ctx: &Self::Ctx,
        s: &State,
        blank: u8,
        m: crate::puzzle24::state::Move,
    ) -> Option<u8> {
        match (
            self.a.child_h_lb(&ctx.a, s, blank, m),
            self.b.child_h_lb(&ctx.b, s, blank, m),
        ) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }
}

/// Eager copy-context impl so [`LazyMaxInc`] satisfies drivers requiring both
/// bounds (the parallel splitter's frontier expansion is cold — laziness only
/// matters in the hot make/unmake recursion).
impl<A, B> crate::puzzle24::search::idastar::IncHeuristic for LazyMaxInc<A, B>
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

    /// `MaxInc::child_h_lb` forwards the sub-heuristics' cheap child bounds (this
    /// is what retains cWD's neighbor-prune through the combiner): max of whichever
    /// exist, `None` treated as "no info" (not 0). Since `max(a,b) ≥ a ≥ a_lb` (and
    /// symmetrically), the forwarded value stays an admissible LB on the combined h.
    #[test]
    fn maxinc_child_h_lb_forwards_and_stays_admissible() {
        use crate::puzzle24::search::idastar::IncHeuristicMut;
        use crate::puzzle24::state::State;

        // Stub with a fixed child_h_lb and a fixed make value.
        struct Stub {
            lb: Option<u8>,
            h: u8,
        }
        impl IncHeuristicMut for Stub {
            type Ctx = ();
            fn root(&self, _s: &State) -> (u8, ()) {
                (self.h, ())
            }
            fn make(&self, _c: &mut (), _child: &State, _m: Move) -> u8 {
                self.h
            }
            fn unmake(&self, _c: &mut (), _m: Move) {}
            fn child_h_lb(&self, _c: &(), _s: &State, _b: u8, _m: Move) -> Option<u8> {
                self.lb
            }
        }

        let ctx = ((), ());
        // (a_lb, b_lb, expected forwarded lb)
        let cases = [
            (Some(3u8), Some(7u8), Some(7u8)), // both present -> max
            (Some(5), None, Some(5)),          // only a
            (None, Some(4), Some(4)),          // only b
            (None, None, None),                // neither -> no prune
        ];
        for (la, lb, expect) in cases {
            // make value 9 ≥ every lb, so the admissibility invariant (lb ≤ h) holds.
            let mx = MaxInc::new(Stub { lb: la, h: 9 }, Stub { lb, h: 9 });
            let got = mx.child_h_lb(&ctx, &GOAL, 24, Move::Up);
            assert_eq!(got, expect, "forward({la:?}, {lb:?})");
            if let Some(v) = got {
                assert!(v <= 9, "forwarded lb {v} > combined make 9 (inadmissible)");
            }
        }
    }

    /// `LazyMaxInc` must be *node-identical* to the eager `MaxInc` (`B` is only
    /// probed when its Lipschitz envelope admits a prune, and then it is exact),
    /// while performing strictly fewer expensive `B` makes. Verified end-to-end
    /// on real bounded searches with `A` = IncManhattan (cheap/weak) and `B` = a
    /// call-counting WalkingDistance (expensive, dominant, 1-Lipschitz) — the
    /// case where a wrong deferral would change the search tree.
    #[test]
    fn lazy_maxinc_node_identical_to_eager_with_fewer_b_probes() {
        use crate::puzzle24::search::idastar::{
            idastar_inc_mut_bounded_with_stats, BoundedOutcome, IncHeuristicMut,
        };
        use crate::puzzle24::search::{WalkingDistanceHeuristic, WalkingDistanceInc};
        use crate::puzzle24::state::GOAL;
        use std::cell::Cell;

        struct Counted<'c, H> {
            inner: H,
            makes: &'c Cell<u64>,
        }
        impl<'c, H: IncHeuristicMut> IncHeuristicMut for Counted<'c, H> {
            type Ctx = H::Ctx;
            fn root(&self, s: &State) -> (u8, Self::Ctx) {
                self.inner.root(s)
            }
            fn make(&self, ctx: &mut Self::Ctx, child: &State, m: Move) -> u8 {
                self.makes.set(self.makes.get() + 1);
                self.inner.make(ctx, child, m)
            }
            fn unmake(&self, ctx: &mut Self::Ctx, m: Move) {
                self.inner.unmake(ctx, m)
            }
            fn child_h_lb(&self, ctx: &Self::Ctx, s: &State, b: u8, m: Move) -> Option<u8> {
                self.inner.child_h_lb(ctx, s, b, m)
            }
        }

        // Deterministic scramble (no immediate undo).
        fn scramble(seed: u64, steps: u32) -> State {
            let mut s = GOAL;
            let mut last: Option<Move> = None;
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            for _ in 0..steps {
                let legal: Vec<Move> = s
                    .legal_moves()
                    .iter()
                    .filter(|&m| last.map_or(true, |p| m != p.inverse()))
                    .collect();
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let m = legal[(x as usize) % legal.len()];
                s = s.apply(m);
                last = Some(m);
            }
            s
        }

        WalkingDistanceHeuristic::warm_up();
        // Config 1 — dominant A (WD), weak B (Manhattan): the real deployment
        // shape (cWD drives the bound, the zPDB sits below it). Deferral must
        // fire, i.e. strictly fewer B makes, with an identical tree.
        // Config 2 — weak A, dominant B: probes fire on nearly every child; a
        // wrong deferral or stale `b_val` would change the tree.
        let (mut def_eager_b, mut def_lazy_b) = (0u64, 0u64);
        for seed in 0..12u64 {
            let start = scramble(seed, 24);
            for b_dominant in [false, true] {
                let eb = Cell::new(0u64);
                let lb = Cell::new(0u64);
                let run = |lazy_b_makes: &Cell<u64>, lazy: bool| {
                    let counted = |c| Counted { inner: IncManhattan, makes: c };
                    // Same (A,B) pair for eager and lazy; only the combinator differs.
                    if b_dominant {
                        let a = IncManhattan;
                        let b = Counted { inner: WalkingDistanceInc, makes: lazy_b_makes };
                        if lazy {
                            idastar_inc_mut_bounded_with_stats(&start, &LazyMaxInc::new(a, b), 40)
                        } else {
                            idastar_inc_mut_bounded_with_stats(&start, &MaxInc::new(a, b), 40)
                        }
                    } else {
                        let a = WalkingDistanceInc;
                        let b = counted(lazy_b_makes);
                        if lazy {
                            idastar_inc_mut_bounded_with_stats(&start, &LazyMaxInc::new(a, b), 40)
                        } else {
                            idastar_inc_mut_bounded_with_stats(&start, &MaxInc::new(a, b), 40)
                        }
                    }
                };
                let (oe, se) = run(&eb, false);
                let (ol, sl) = run(&lb, true);
                let len = |o: &BoundedOutcome| match o {
                    BoundedOutcome::Solved(p) => p.len(),
                    _ => panic!("scramble should solve within bound 40"),
                };
                assert_eq!(len(&oe), len(&ol), "seed {seed} bdom {b_dominant}: length differs");
                assert_eq!(se.nodes, sl.nodes, "seed {seed} bdom {b_dominant}: tree differs");
                assert!(lb.get() <= eb.get(), "seed {seed} bdom {b_dominant}: lazy did MORE B makes");
                if !b_dominant {
                    def_eager_b += eb.get();
                    def_lazy_b += lb.get();
                }
            }
        }
        // In the deployment-shaped config the deferral must actually fire.
        assert!(
            def_lazy_b < def_eager_b,
            "lazy B makes {def_lazy_b} not fewer than eager {def_eager_b} — deferral never fired"
        );
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
