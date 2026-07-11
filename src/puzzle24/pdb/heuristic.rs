//! Heuristic combinators that plug additive PDBs into the 24-puzzle IDA\*.
//!
//! - [`PdbHeuristic`]: single PDB lookup.
//! - [`AdditivePdbHeuristic`]: sum over a *disjoint* pattern partition
//!   (Korf & Felner 2002); disjointness enforced at construction.
//! - [`ReflectedHeuristic`]: queries the wrapped heuristic on the diagonal
//!   reflection (a goal-preserving isomorphism, so still admissible).
//! - [`MaxHeuristic`]: max of two admissibles.
//! - [`KorfPdbInc`]: the incremental Korf-max evaluator —
//!   `max(additive(s), additive(reflect(s)))` advanced one move per node.
//!
//! All 24-puzzle costs fit `u8` (diameter ≤ 205 < 255); the clamps are defensive.

use super::db::PatternDb;
use super::pattern::{Pattern, ProjectedState};
use super::zdb::ZPatternDb;
use crate::puzzle24::search::{Heuristic, IncHeuristic, IncHeuristicMut, SearchStats};
use crate::puzzle24::state::{Move, State};
use crate::puzzle24::symmetry::{reflect, transpose_move};

/// Single-PDB heuristic.
pub struct PdbHeuristic<'a> {
    db: &'a PatternDb,
}

impl<'a> PdbHeuristic<'a> {
    pub fn new(db: &'a PatternDb) -> Self {
        Self { db }
    }
}

impl<'a> Heuristic for PdbHeuristic<'a> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        self.db.h(s)
    }
}

/// Sum-of-PDBs heuristic over a disjoint pattern partition.
pub struct AdditivePdbHeuristic<'a> {
    dbs: &'a [PatternDb],
}

impl<'a> AdditivePdbHeuristic<'a> {
    /// Verifies pairwise disjointness at construction.
    pub fn new(dbs: &'a [PatternDb]) -> Self {
        let mut union: u32 = 0;
        for db in dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "additive PDB patterns must be disjoint");
            union |= p;
        }
        Self { dbs }
    }

    pub fn coverage(&self) -> Pattern {
        let mut union: u32 = 0;
        for db in self.dbs {
            union |= db.pattern().0;
        }
        Pattern(union)
    }

    pub fn bytes_stored(&self) -> usize {
        self.dbs.iter().map(|db| db.bytes_stored()).sum()
    }
}

impl<'a> Heuristic for AdditivePdbHeuristic<'a> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        let mut sum: u32 = 0;
        for db in self.dbs {
            sum += db.h(s) as u32;
        }
        sum.min(u8::MAX as u32) as u8
    }
}

/// Wraps a [`Heuristic`] to evaluate on the diagonal reflection of the input.
pub struct ReflectedHeuristic<H: Heuristic> {
    inner: H,
}

impl<H: Heuristic> ReflectedHeuristic<H> {
    pub fn new(inner: H) -> Self {
        Self { inner }
    }
}

impl<H: Heuristic> Heuristic for ReflectedHeuristic<H> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        self.inner.h(&reflect(s))
    }
}

/// Max of two admissible heuristics.
pub struct MaxHeuristic<A: Heuristic, B: Heuristic> {
    a: A,
    b: B,
}

impl<A: Heuristic, B: Heuristic> MaxHeuristic<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Heuristic, B: Heuristic> Heuristic for MaxHeuristic<A, B> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        self.a.h(s).max(self.b.h(s))
    }
}

/// Incremental Korf-max heuristic: `max(additive(s), additive(reflect(s)))` over
/// `N` pairwise-disjoint PDBs. The per-node [`KorfCtx`] keeps each PDB's
/// projected board in the normal and reflected views; [`advance`] slides the
/// normal view by `m` and the reflected view by [`transpose_move`], avoiding a
/// full re-projection + `reflect` per node. Value is identical to the scratch
/// `MaxHeuristic`-of-`AdditivePdbHeuristic` composition.
///
/// [`advance`]: IncHeuristic::advance
pub struct KorfPdbInc<'a, const N: usize> {
    patterns: [Pattern; N],
    dist: [&'a [u8]; N],
}

/// Per-node context for [`KorfPdbInc`]: each PDB's projected board in the normal
/// and reflected views.
#[derive(Clone, Copy)]
pub struct KorfCtx<const N: usize> {
    normal: [ProjectedState; N],
    reflected: [ProjectedState; N],
}

impl<'a, const N: usize> KorfPdbInc<'a, N> {
    /// Build from `N` PDBs; panics unless the patterns are pairwise disjoint.
    pub fn new(dbs: [&'a PatternDb; N]) -> Self {
        let mut union: u32 = 0;
        for db in &dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "additive PDB patterns must be disjoint");
            union |= p;
        }
        Self {
            patterns: std::array::from_fn(|i| dbs[i].pattern()),
            dist: std::array::from_fn(|i| dbs[i].raw()),
        }
    }

    #[inline]
    fn value(&self, ctx: &KorfCtx<N>) -> u8 {
        let mut hn: u32 = 0;
        let mut hr: u32 = 0;
        for i in 0..N {
            let p = self.patterns[i];
            let d = self.dist[i];
            let rn = ctx.normal[i].rank(p) as usize;
            let rr = ctx.reflected[i].rank(p) as usize;
            debug_assert!(rn < d.len() && rr < d.len());
            // SAFETY: rank() ∈ [0, num_projected_states) == d.len() by construction.
            hn += unsafe { *d.get_unchecked(rn) } as u32;
            hr += unsafe { *d.get_unchecked(rr) } as u32;
        }
        hn.max(hr).min(u8::MAX as u32) as u8
    }
}

impl<'a, const N: usize> IncHeuristic for KorfPdbInc<'a, N> {
    type Ctx = KorfCtx<N>;

    fn root(&self, s: &State, _stats: &mut SearchStats) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let ctx = KorfCtx {
            normal: std::array::from_fn(|i| ProjectedState::from_state(s, self.patterns[i])),
            reflected: std::array::from_fn(|i| ProjectedState::from_state(&rs, self.patterns[i])),
        };
        (self.value(&ctx), ctx)
    }

    fn advance(
        &self,
        parent: &Self::Ctx,
        _child: &State,
        m: Move,
        _stats: &mut SearchStats,
    ) -> (u8, Self::Ctx) {
        let tm = transpose_move(m);
        let ctx = KorfCtx {
            normal: std::array::from_fn(|i| parent.normal[i].apply(m).0),
            reflected: std::array::from_fn(|i| parent.reflected[i].apply(tm).0),
        };
        (self.value(&ctx), ctx)
    }
}

impl<'a, const N: usize> IncHeuristicMut for KorfPdbInc<'a, N> {
    // Reuses [`KorfCtx`]: the projections are slid in place (self-inverse), and
    // `value` recomputes `h` from them each `make` — the same work the copy path
    // does, minus the per-node projection-array copy. No undo record needed.
    type Ctx = KorfCtx<N>;

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let mut stats = SearchStats::default();
        <Self as IncHeuristic>::root(self, s, &mut stats)
    }

    fn make(&self, ctx: &mut Self::Ctx, _child: &State, m: Move) -> u8 {
        let tm = transpose_move(m);
        for i in 0..N {
            ctx.normal[i].apply_in_place(m);
            ctx.reflected[i].apply_in_place(tm);
        }
        self.value(ctx)
    }

    fn unmake(&self, ctx: &mut Self::Ctx, m: Move) {
        let im = m.inverse();
        let tim = transpose_move(im);
        for i in 0..N {
            ctx.normal[i].apply_in_place(im);
            ctx.reflected[i].apply_in_place(tim);
        }
    }
}

/// Incremental zero-aware-PDB Korf-max heuristic:
/// `max(zpdb(s), zpdb(reflect(s)))` over `N` pairwise-disjoint ZPDBs. Per-node
/// context tracks each PDB's projected board and running absolute `h`, in both
/// the normal and reflected views.
///
/// On every puzzle move `m`, for each PDB and each view, the projected state
/// slides ([`transpose_move`]`(m)` for the reflected view) and reports a
/// projected-edge cost: `0` if the blank swapped with an *anon* filler, `1` if
/// it swapped with a pattern tile. That cost predicts **exactly** whether the
/// abstract `(m, p, r)` index moves:
/// - cost `0` — the blank wandered inside its region; the index, and hence `h`,
///   are unchanged, so the (expensive) [`ZpdbLayout::rank`] is skipped entirely;
/// - cost `1` — a unit ZPDB edge; rank the new projection once and read the new
///   `h` from the 1-bit codec's [`ZPatternDb::diff_lookup`] in O(1).
///
/// Since the PDBs are pairwise disjoint, a single moved tile is a pattern tile
/// for at most one PDB, so at most one PDB recomputes a rank per node — the rest
/// carry through. (Equivalence to the old "rank both, compare indices" form is
/// guarded by [`tests::zpdb_cost_predicts_index_change`].)
///
/// At the search root, each view's `h` is seeded via O(h)
/// [`ZPatternDb::cold_lookup`]. The diagonal symmetry fixes the goal-blank
/// corner (`σ(24) = 24` so `τ(0) = 0`), so the paper's Eq. 2 reflected
/// zero-swap collapses to the standard reflection — the same wrapper we use
/// for [`KorfPdbInc`].
///
/// Admissibility: each ZPDB value lower-bounds the true distance; the sum
/// over disjoint patterns is admissible; max of normal and reflected is
/// admissible because the diagonal symmetry preserves distance to GOAL.
pub struct ZpdbInc<'a, const N: usize> {
    dbs: [&'a ZPatternDb; N],
}

#[derive(Clone, Copy)]
pub struct ZpdbCtx<const N: usize> {
    normal: [ProjectedState; N],
    reflected: [ProjectedState; N],
    n_h: [u8; N],
    r_h: [u8; N],
}

impl<'a, const N: usize> ZpdbInc<'a, N> {
    /// Build from `N` ZPDBs; panics unless the patterns are pairwise disjoint.
    pub fn new(dbs: [&'a ZPatternDb; N]) -> Self {
        let mut union: u32 = 0;
        for db in &dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "ZPDB patterns must be disjoint");
            union |= p;
        }
        Self { dbs }
    }

    #[inline]
    fn value(&self, ctx: &ZpdbCtx<N>) -> u8 {
        Self::korf_max(&ctx.n_h, &ctx.r_h)
    }

    /// Korf-max over the two views: `max(Σ normal_h, Σ reflected_h)`, saturating.
    /// Shared by the copy ([`IncHeuristic`]) and make/unmake ([`IncHeuristicMut`])
    /// contexts, which carry the same per-PDB `h` arrays.
    #[inline]
    fn korf_max(n_h: &[u8; N], r_h: &[u8; N]) -> u8 {
        let mut hn: u32 = 0;
        let mut hr: u32 = 0;
        for i in 0..N {
            hn += n_h[i] as u32;
            hr += r_h[i] as u32;
        }
        hn.max(hr).min(u8::MAX as u32) as u8
    }
}

impl<'a, const N: usize> IncHeuristic for ZpdbInc<'a, N> {
    type Ctx = ZpdbCtx<N>;

    fn root(&self, s: &State, _stats: &mut SearchStats) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let mut ctx = ZpdbCtx {
            normal: std::array::from_fn(|i| {
                ProjectedState::from_state(s, self.dbs[i].pattern())
            }),
            reflected: std::array::from_fn(|i| {
                ProjectedState::from_state(&rs, self.dbs[i].pattern())
            }),
            n_h: [0u8; N],
            r_h: [0u8; N],
        };
        for i in 0..N {
            let db = self.dbs[i];
            ctx.n_h[i] = db.cold_lookup_proj(&ctx.normal[i]);
            ctx.r_h[i] = db.cold_lookup_proj(&ctx.reflected[i]);
        }
        (self.value(&ctx), ctx)
    }

    fn advance(
        &self,
        parent: &Self::Ctx,
        _child: &State,
        m: Move,
        _stats: &mut SearchStats,
    ) -> (u8, Self::Ctx) {
        #[cfg(feature = "verifier-stats")]
        {
            _stats.zpdb_advances += 1;
            _stats.proj_applies += 2 * N as u64;
        }
        let tm = transpose_move(m);
        let mut ctx = *parent;
        for i in 0..N {
            let db = self.dbs[i];

            // Normal view: slide the projection. The projected-edge cost is `1`
            // iff a pattern tile swapped with the blank, which is *exactly* when
            // the (m,p,r) index moves — so only then do we re-rank and look up
            // the new `h`. A cost-`0` anon swap leaves index and `h` untouched
            // (ctx already inherits parent's `n_h[i]` via `*parent`).
            let (np, n_cost) = parent.normal[i].apply(m);
            if n_cost != 0 {
                #[cfg(feature = "verifier-stats")]
                { _stats.zpdb_rank_calls += 1; }
                let n_idx = db.layout().rank(&np, db.pattern());
                ctx.n_h[i] = db.diff_lookup(n_idx, parent.n_h[i]);
            }
            ctx.normal[i] = np;

            // Reflected view (same logic under transpose_move(m)).
            let (rp, r_cost) = parent.reflected[i].apply(tm);
            if r_cost != 0 {
                #[cfg(feature = "verifier-stats")]
                { _stats.zpdb_rank_calls += 1; }
                let r_idx = db.layout().rank(&rp, db.pattern());
                ctx.r_h[i] = db.diff_lookup(r_idx, parent.r_h[i]);
            }
            ctx.reflected[i] = rp;
        }
        (self.value(&ctx), ctx)
    }
}

/// Make/unmake context for [`ZpdbInc`] — the same projections and per-PDB `h`
/// arrays as [`ZpdbCtx`], plus a LIFO `undo` stack. Each [`make`](IncHeuristicMut::make)
/// mutates the projections in place (no array copy) and pushes the pre-move `h`
/// snapshot; each [`unmake`](IncHeuristicMut::unmake) pops it and slides the
/// projections back with the inverse move. The projections are self-undoing, so
/// only the derived `h` values (2·N bytes) are saved per node — versus the full
/// `~102·N`-byte context copy the [`IncHeuristic`] path makes per node.
pub struct ZpdbCtxMut<const N: usize> {
    normal: [ProjectedState; N],
    reflected: [ProjectedState; N],
    n_h: [u8; N],
    r_h: [u8; N],
    undo: Vec<([u8; N], [u8; N])>,
}

impl<'a, const N: usize> IncHeuristicMut for ZpdbInc<'a, N> {
    type Ctx = ZpdbCtxMut<N>;

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let mut ctx = ZpdbCtxMut {
            normal: std::array::from_fn(|i| {
                ProjectedState::from_state(s, self.dbs[i].pattern())
            }),
            reflected: std::array::from_fn(|i| {
                ProjectedState::from_state(&rs, self.dbs[i].pattern())
            }),
            n_h: [0u8; N],
            r_h: [0u8; N],
            undo: Vec::with_capacity(220),
        };
        for i in 0..N {
            let db = self.dbs[i];
            ctx.n_h[i] = db.cold_lookup_proj(&ctx.normal[i]);
            ctx.r_h[i] = db.cold_lookup_proj(&ctx.reflected[i]);
        }
        (Self::korf_max(&ctx.n_h, &ctx.r_h), ctx)
    }

    // Projection-based: the raw `child` state is not needed — the projections
    // are maintained incrementally.
    fn make(&self, ctx: &mut Self::Ctx, _child: &State, m: Move) -> u8 {
        // Snapshot the pre-move `h` values so `unmake` can restore them; the
        // projections themselves are restored by the inverse slide, no save.
        ctx.undo.push((ctx.n_h, ctx.r_h));
        let tm = transpose_move(m);
        for i in 0..N {
            let db = self.dbs[i];
            // Normal view: cost `1` iff a pattern tile swapped with the blank,
            // which is exactly when the (m,p,r) index — and thus `h` — moves.
            let n_cost = ctx.normal[i].apply_in_place(m);
            if n_cost != 0 {
                let n_idx = db.layout().rank(&ctx.normal[i], db.pattern());
                ctx.n_h[i] = db.diff_lookup(n_idx, ctx.n_h[i]);
            }
            // Reflected view (same logic under transpose_move(m)).
            let r_cost = ctx.reflected[i].apply_in_place(tm);
            if r_cost != 0 {
                let r_idx = db.layout().rank(&ctx.reflected[i], db.pattern());
                ctx.r_h[i] = db.diff_lookup(r_idx, ctx.r_h[i]);
            }
        }
        Self::korf_max(&ctx.n_h, &ctx.r_h)
    }

    fn unmake(&self, ctx: &mut Self::Ctx, m: Move) {
        let (n_h, r_h) = ctx.undo.pop().expect("unmake without matching make");
        ctx.n_h = n_h;
        ctx.r_h = r_h;
        let tm = transpose_move(m);
        for i in 0..N {
            ctx.normal[i].apply_in_place(m.inverse());
            ctx.reflected[i].apply_in_place(tm.inverse());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::pdb::pattern::Pattern;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::ManhattanHeuristic;
    use crate::puzzle24::state::{State, GOAL};

    #[test]
    fn pdb_heuristic_zero_on_goal() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        assert_eq!(PdbHeuristic::new(&pdb).h(&GOAL), 0);
    }

    #[test]
    fn additive_sums_components() {
        let dbs = vec![
            PatternDb::build(Pattern::new(&[1, 2, 3])),
            PatternDb::build(Pattern::new(&[7, 8, 9])),
        ];
        let h = AdditivePdbHeuristic::new(&dbs);
        assert_eq!(h.h(&GOAL), 0);
        let s = GOAL.apply(Move::Up).apply(Move::Left);
        assert_eq!(h.h(&s), dbs[0].h(&s) + dbs[1].h(&s));
    }

    #[test]
    #[should_panic(expected = "disjoint")]
    fn additive_rejects_overlap() {
        let dbs = vec![
            PatternDb::build(Pattern::new(&[1, 2, 3])),
            PatternDb::build(Pattern::new(&[3, 4, 5])),
        ];
        let _ = AdditivePdbHeuristic::new(&dbs);
    }

    #[test]
    fn korf_max_admissible_on_shallow_bfs() {
        let dbs = vec![
            PatternDb::build(Pattern::new(&[1, 2, 3])),
            PatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let h_add = AdditivePdbHeuristic::new(&dbs);
        let h_refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs));
        let h_korf = MaxHeuristic::new(MaxHeuristic::new(&h_add, &h_refl), ManhattanHeuristic);
        let truth = bfs_distances(8);
        for (raw, &true_dist) in &truth {
            let est = h_korf.h(&State(*raw));
            assert!(est <= true_dist, "korf h {} > true {} for {:?}", est, true_dist, raw);
        }
    }

    #[test]
    fn korf_inc_root_matches_scratch() {
        let dbs = [
            PatternDb::build(Pattern::new(&[1, 2, 3])),
            PatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let h_add = AdditivePdbHeuristic::new(&dbs);
        let h_refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs));
        let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);
        let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);

        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        for _ in 0..3000 {
            let (h_inc, _) = IncHeuristic::root(&inc, &s, &mut stats);
            assert_eq!(h_inc, h_korf.h(&s), "inc.root != scratch korf at {:?}", s.0);
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            s = s.apply(opts[(next() as usize) % opts.len()]);
        }
    }

    // ---------- ZpdbInc -------------------------------------------------

    #[test]
    fn zpdb_inc_root_at_goal_is_zero() {
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let inc = ZpdbInc::new([&dbs[0], &dbs[1]]);
        let mut stats = SearchStats::default();
        let (h, _) = IncHeuristic::root(&inc, &GOAL, &mut stats);
        assert_eq!(h, 0);
    }

    /// ZPDB root must dominate the additive PDB root (admissibility oracle).
    /// `zpdb(s) ≥ additive(s)` pointwise, so the sum-of-zpdbs and its
    /// max-with-reflection both dominate the sum-of-additives.
    #[test]
    fn zpdb_inc_root_geq_additive_inc_root_random_walk() {
        let patterns = [Pattern::new(&[1, 2, 3]), Pattern::new(&[6, 7, 8])];
        let zdbs = [
            ZPatternDb::build(patterns[0]),
            ZPatternDb::build(patterns[1]),
        ];
        let adbs = [
            PatternDb::build(patterns[0]),
            PatternDb::build(patterns[1]),
        ];
        let zinc = ZpdbInc::new([&zdbs[0], &zdbs[1]]);
        let kinc = KorfPdbInc::new([&adbs[0], &adbs[1]]);

        let mut rng: u64 = 0xBADC_AB1E_C0FF_EE00;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        for _ in 0..1500 {
            let (h_z, _) = IncHeuristic::root(&zinc, &s, &mut stats);
            let (h_k, _) = IncHeuristic::root(&kinc, &s, &mut stats);
            assert!(
                h_z >= h_k,
                "ZPDB root {} < additive Korf-max root {} at {:?}",
                h_z,
                h_k,
                s.0
            );
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            s = s.apply(opts[(next() as usize) % opts.len()]);
        }
    }

    /// Incremental advance must agree with a fresh cold-root at every step
    /// along a random walk — the single most important hot-path invariant.
    #[test]
    fn zpdb_inc_advance_matches_fresh_root_random_walk() {
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let inc = ZpdbInc::new([&dbs[0], &dbs[1]]);

        let mut rng: u64 = 0xFEED_FACE_C0DE_BAD0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (_, mut ctx) = IncHeuristic::root(&inc, &s, &mut stats);
        for i in 0..1500 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = inc.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&inc, &ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "advance diverged at step {} (state {:?}, move {:?})",
                i, ns.0, m
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// The incremental `advance`'s core invariant: the projected-edge cost from
    /// `ProjectedState::apply` (0 = anon swap, 1 = pattern-tile swap) predicts
    /// EXACTLY whether the abstract `(m,p,r)` rank index changes. `advance`
    /// relies on this to skip the `rank` recompute on cost-0 edges. Checked over
    /// a random walk on the 5x5 board at k=3 and the production k=6, using only
    /// the cheap `ZpdbLayout` combinatorics — no PDB table build needed.
    #[test]
    fn zpdb_cost_predicts_index_change() {
        use crate::puzzle24::pdb::zpdb::ZpdbLayout;
        for pattern in [Pattern::new(&[1, 2, 3]), Pattern::new(&[1, 2, 3, 6, 7, 8])] {
            let layout = ZpdbLayout::new(pattern);
            let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut s = GOAL;
            for _ in 0..1500 {
                let parent = ProjectedState::from_state(&s, pattern);
                let pidx = layout.rank(&parent, pattern);
                for m in s.legal_moves().iter() {
                    let (child, cost) = parent.apply(m);
                    let cidx = layout.rank(&child, pattern);
                    assert_eq!(
                        cost == 0,
                        cidx == pidx,
                        "cost {} but {}->{} for pattern {:?}, move {:?}, state {:?}",
                        cost, pidx, cidx, pattern, m, s.0,
                    );
                }
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
        }
    }

    /// ZPDB-driven IDA\* must find the same optimal-length solution as the
    /// additive Korf-max-driven IDA\* on shallow scrambles — they're both
    /// admissible, so both yield optima.
    #[test]
    fn zpdb_inc_idastar_finds_optima_on_shallow_scrambles() {
        use crate::puzzle24::search::{idastar_inc_with_stats, ManhattanHeuristic};
        let zdbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let zinc = ZpdbInc::new([&zdbs[0], &zdbs[1]]);

        // Walk a few small scrambles and verify IDA* returns a solution of
        // ManhattanHeuristic-admissible length (Manhattan ≤ true).
        let mut rng: u64 = 0xC0DE_F00D_DEAD_BEEF;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..5 {
            let mut s = GOAL;
            for _ in 0..10 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (sol_z, _) = idastar_inc_with_stats(&s, &zinc);
            let sol_z = sol_z.expect("ZPDB IDA* found no solution");
            assert!(
                ManhattanHeuristic.h(&s) as usize <= sol_z.len(),
                "Manhattan bound violated by ZPDB IDA* (Δ {} vs {})",
                ManhattanHeuristic.h(&s),
                sol_z.len()
            );
            // And that the path actually reaches GOAL.
            let mut cur = s;
            for &m in &sol_z {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "ZPDB IDA* solution did not reach GOAL");
        }
    }

    // ---------- KorfPdbInc ----------------------------------------------

    #[test]
    #[ignore = "builds two 4-tile blank-cell PDBs (~14s); run with `cargo test -- --ignored`. \
                The faster k=3 sibling covers advance==reprojection on every run."]
    fn korf_inc_advance_matches_reprojection() {
        let dbs = [
            PatternDb::build(Pattern::new(&[2, 5, 8, 11])),
            PatternDb::build(Pattern::new(&[1, 3, 6, 9])),
        ];
        let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);
        let mut rng: u64 = 0x0F0F_0F0F_1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (_, mut ctx) = IncHeuristic::root(&inc, &s, &mut stats);
        for i in 0..3000 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = inc.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&inc, &ns, &mut stats);
            assert_eq!(h_adv, h_fresh, "advance diverged at step {}", i);
            s = ns;
            ctx = ctx_adv;
        }
    }

    // ---------- ZpdbInc make/unmake (IncHeuristicMut) --------------------

    /// The make/unmake context must track the copy context step-for-step: after
    /// a `make(m)`, `h` equals a fresh cold-root; after the matching `unmake(m)`,
    /// the context restores exactly (verified by re-making a *different* sibling
    /// move and comparing to that sibling's fresh root). Exercises the `undo`
    /// stack across nested make/unmake, the crux of the in-place path.
    #[test]
    fn zpdb_mut_make_unmake_matches_fresh_root_random_walk() {
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let inc = ZpdbInc::new([&dbs[0], &dbs[1]]);

        let mut rng: u64 = 0xDEAD_BEEF_1357_9BDF;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let (_, mut ctx) = IncHeuristicMut::root(&inc, &s);
        let mut stats = SearchStats::default();
        for i in 0..1500 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            // Try EVERY legal move via make/unmake without advancing: each must
            // match that child's fresh root, and each unmake must restore ctx so
            // the next sibling starts clean.
            for &m in &opts {
                let child = s.apply(m);
                let h_make = inc.make(&mut ctx, &child, m);
                let (h_fresh, _) = IncHeuristic::root(&inc, &child, &mut stats);
                assert_eq!(
                    h_make, h_fresh,
                    "make diverged from fresh root at step {} move {:?}",
                    i, m
                );
                inc.unmake(&mut ctx, m);
            }
            // After trying all siblings, ctx must still match the parent root.
            let (h_parent, _) = IncHeuristic::root(&inc, &s, &mut stats);
            assert_eq!(
                ZpdbInc::korf_max(&ctx.n_h, &ctx.r_h),
                h_parent,
                "ctx not restored to parent after sibling sweep at step {}",
                i
            );
            // Advance for real via make (no matching unmake — descend).
            let m = opts[(next() as usize) % opts.len()];
            let child = s.apply(m);
            inc.make(&mut ctx, &child, m);
            s = child;
        }
    }

    /// The whole point of make/unmake: identical *search*, cheaper per node. The
    /// mut driver must return the same optimal length AND expand the exact same
    /// number of nodes as the copy driver on the same scrambles.
    #[test]
    fn zpdb_mut_idastar_matches_copy_length_and_node_count() {
        use crate::puzzle24::search::{idastar_inc_mut_with_stats, idastar_inc_with_stats};
        let zdbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let zinc = ZpdbInc::new([&zdbs[0], &zdbs[1]]);

        let mut rng: u64 = 0x0BAD_F00D_5EED_1234;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..8 {
            let mut s = GOAL;
            for _ in 0..14 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (sol_copy, st_copy) = idastar_inc_with_stats(&s, &zinc);
            let (sol_mut, st_mut) = idastar_inc_mut_with_stats(&s, &zinc);
            let lc = sol_copy.expect("copy IDA* found no solution").len();
            let lm = sol_mut.expect("mut IDA* found no solution").len();
            assert_eq!(lc, lm, "optimal length differs (copy {} vs mut {})", lc, lm);
            assert_eq!(
                st_copy.nodes, st_mut.nodes,
                "node count differs (copy {} vs mut {})",
                st_copy.nodes, st_mut.nodes
            );
        }
    }

    /// The parallel make/unmake driver must match the parallel copy driver. Two
    /// invariants, mirroring the copy-driver's own parallel tests:
    ///   * solving (`u8::MAX`): same optimal length. (Node counts are *not*
    ///     comparable on a solving run — the shared found-flag early-exit races
    ///     at the check granularity, so either driver varies run-to-run.)
    ///   * bounded exhaust (`optimal-1`): the found-flag stays inert, so both
    ///     drivers prove the same LB and expand the *exact* same node count.
    #[cfg(feature = "parallel")]
    #[test]
    fn zpdb_parallel_mut_matches_parallel_copy() {
        use crate::puzzle24::search::{
            idastar_inc_bounded_parallel, idastar_inc_bounded_parallel_mut, LadderOutcome,
        };
        let zdbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let zinc = ZpdbInc::new([&zdbs[0], &zdbs[1]]);

        let mut rng: u64 = 0x1122_3344_5566_7788;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..6 {
            let mut s = GOAL;
            for _ in 0..16 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            // Solving: same optimal length.
            let truth = match idastar_inc_bounded_parallel(&s, &zinc, u8::MAX, None).0 {
                LadderOutcome::Solved(p) => p.len(),
                o => panic!("copy solve unexpected: {:?}", o),
            };
            match idastar_inc_bounded_parallel_mut(&s, &zinc, u8::MAX, None, false).0 {
                LadderOutcome::Solved(pm) => assert_eq!(pm.len(), truth, "solved length differs"),
                o => panic!("mut solve unexpected: {:?}", o),
            }
            // Bounded exhaust at optimal-1: identical proven LB and node count.
            if truth == 0 {
                continue;
            }
            let mb = (truth - 1) as u8;
            let (oc, st_copy) = idastar_inc_bounded_parallel(&s, &zinc, mb, None);
            let (om, st_mut) = idastar_inc_bounded_parallel_mut(&s, &zinc, mb, None, false);
            match (oc, om) {
                (LadderOutcome::ProvedAtLeast(a), LadderOutcome::ProvedAtLeast(b)) => {
                    assert_eq!(a, b, "proven LB differs (copy {} vs mut {})", a, b);
                }
                (a, b) => panic!("expected both ProvedAtLeast, got {:?} / {:?}", a, b),
            }
            assert_eq!(
                st_copy.nodes, st_mut.nodes,
                "exhaust node count differs (copy {} vs mut {})",
                st_copy.nodes, st_mut.nodes
            );
        }
    }

    /// KorfPdbInc (additive Korf-max) make/unmake must match the copy driver:
    /// same optimal length and node count. Uses cheap k=3 PDBs.
    #[test]
    fn korf_mut_idastar_matches_copy_length_and_nodes() {
        use crate::puzzle24::search::{idastar_inc_mut_with_stats, idastar_inc_with_stats};
        let dbs = [
            PatternDb::build(Pattern::new(&[1, 2, 3])),
            PatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);
        let mut rng: u64 = 0x3141_5926_5358_9793;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        for _ in 0..6 {
            let mut s = GOAL;
            for _ in 0..12 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (c, cs) = idastar_inc_with_stats(&s, &inc);
            let (m, ms) = idastar_inc_mut_with_stats(&s, &inc);
            assert_eq!(c.expect("copy").len(), m.expect("mut").len(), "Korf length differs");
            assert_eq!(cs.nodes, ms.nodes, "Korf node count differs");
        }
    }
}
