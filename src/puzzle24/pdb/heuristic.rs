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
use crate::puzzle24::search::{Heuristic, IncHeuristic};
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

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let ctx = KorfCtx {
            normal: std::array::from_fn(|i| ProjectedState::from_state(s, self.patterns[i])),
            reflected: std::array::from_fn(|i| ProjectedState::from_state(&rs, self.patterns[i])),
        };
        (self.value(&ctx), ctx)
    }

    fn advance(&self, parent: &Self::Ctx, _child: &State, m: Move) -> (u8, Self::Ctx) {
        let tm = transpose_move(m);
        let ctx = KorfCtx {
            normal: std::array::from_fn(|i| parent.normal[i].apply(m).0),
            reflected: std::array::from_fn(|i| parent.reflected[i].apply(tm).0),
        };
        (self.value(&ctx), ctx)
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
        let mut hn: u32 = 0;
        let mut hr: u32 = 0;
        for i in 0..N {
            hn += ctx.n_h[i] as u32;
            hr += ctx.r_h[i] as u32;
        }
        hn.max(hr).min(u8::MAX as u32) as u8
    }
}

impl<'a, const N: usize> IncHeuristic for ZpdbInc<'a, N> {
    type Ctx = ZpdbCtx<N>;

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
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

    fn advance(&self, parent: &Self::Ctx, _child: &State, m: Move) -> (u8, Self::Ctx) {
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
                let n_idx = db.layout().rank(&np, db.pattern());
                ctx.n_h[i] = db.diff_lookup(n_idx, parent.n_h[i]);
            }
            ctx.normal[i] = np;

            // Reflected view (same logic under transpose_move(m)).
            let (rp, r_cost) = parent.reflected[i].apply(tm);
            if r_cost != 0 {
                let r_idx = db.layout().rank(&rp, db.pattern());
                ctx.r_h[i] = db.diff_lookup(r_idx, parent.r_h[i]);
            }
            ctx.reflected[i] = rp;
        }
        (self.value(&ctx), ctx)
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
        for _ in 0..3000 {
            let (h_inc, _) = IncHeuristic::root(&inc, &s);
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
        let (h, _) = IncHeuristic::root(&inc, &GOAL);
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
        for _ in 0..1500 {
            let (h_z, _) = IncHeuristic::root(&zinc, &s);
            let (h_k, _) = IncHeuristic::root(&kinc, &s);
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
        let (_, mut ctx) = IncHeuristic::root(&inc, &s);
        for i in 0..1500 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = inc.advance(&ctx, &ns, m);
            let (h_fresh, _) = IncHeuristic::root(&inc, &ns);
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
        let (_, mut ctx) = IncHeuristic::root(&inc, &s);
        for i in 0..3000 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = inc.advance(&ctx, &ns, m);
            let (h_fresh, _) = IncHeuristic::root(&inc, &ns);
            assert_eq!(h_adv, h_fresh, "advance diverged at step {}", i);
            s = ns;
            ctx = ctx_adv;
        }
    }
}
