//! Zero-aware PDB heuristics for the 15-puzzle IDA\*. Ported from
//! `puzzle24::pdb::heuristic`'s `ZpdbInc`/`ZpdbCtx`, plus a non-incremental
//! [`ZpdbHeuristic`] and an additive/max combiner.
//!
//! - [`ZpdbHeuristic`]: single zero-aware PDB lookup (implements [`Heuristic`]).
//! - [`AdditiveZpdbHeuristic`]: sum over a disjoint pattern partition.
//! - [`ZpdbInc`]: incremental Korf-max evaluator —
//!   `max(zpdb(s), zpdb(reflect(s)))` advanced one move per node — implements
//!   [`IncHeuristic`].
//!
//! Admissibility: each ZPDB value lower-bounds the true distance (it is a BFS
//! over a contracted quotient of the real move graph) and pointwise dominates
//! the blank-agnostic additive PDB; the sum over disjoint patterns is
//! admissible; max of the normal and diagonal-reflected views is admissible
//! because the reflection preserves distance to GOAL.

use super::pattern::{Pattern, ProjectedState};
use super::zdb::ZPatternDb;
use crate::puzzle15::search::{
    Heuristic, IncHeuristic, LcCtx, LinearConflictInc, SearchStats, WalkingDistanceInc, WdCtx,
};
use crate::puzzle15::state::{Move, State};
use crate::puzzle15::symmetry::{reflect, transpose_move};

/// Single zero-aware-PDB heuristic.
pub struct ZpdbHeuristic<'a> {
    db: &'a ZPatternDb,
}

impl<'a> ZpdbHeuristic<'a> {
    pub fn new(db: &'a ZPatternDb) -> Self {
        Self { db }
    }
}

impl<'a> Heuristic for ZpdbHeuristic<'a> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        self.db.cold_lookup(s)
    }
}

/// Sum-of-ZPDBs heuristic over a disjoint pattern partition.
pub struct AdditiveZpdbHeuristic<'a> {
    dbs: &'a [ZPatternDb],
}

impl<'a> AdditiveZpdbHeuristic<'a> {
    /// Verifies pairwise disjointness at construction.
    pub fn new(dbs: &'a [ZPatternDb]) -> Self {
        let mut union: u32 = 0;
        for db in dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "additive ZPDB patterns must be disjoint");
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
}

impl<'a> Heuristic for AdditiveZpdbHeuristic<'a> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        let mut sum: u32 = 0;
        for db in self.dbs {
            sum += db.cold_lookup(s) as u32;
        }
        sum.min(u8::MAX as u32) as u8
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
///   `h` from [`ZPatternDb::diff_lookup`] in O(1).
///
/// Since the PDBs are pairwise disjoint, a single moved tile is a pattern tile
/// for at most one PDB, so at most one PDB recomputes a rank per node — the rest
/// carry through. (Equivalence to the old "rank both, compare indices" form is
/// guarded by [`tests::zpdb_cost_predicts_index_change`].)
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

    fn root(&self, s: &State, _stats: &mut SearchStats) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let mut ctx = ZpdbCtx {
            normal: std::array::from_fn(|i| ProjectedState::from_state(s, self.dbs[i].pattern())),
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

/// Incremental "zpdb-plus": [`ZpdbInc`] (zero-aware Korf-max) maxed per node
/// with the classical Linear-Conflict and Walking-Distance heuristics — the
/// drop-in incremental replacement for the additive `korf-plus` the enumeration
/// verifier used.
///
/// Because each zero-aware ZPDB pointwise dominates its blank-agnostic additive
/// PDB, this **dominates the additive korf-plus** (identical LC/WD terms, a
/// `≥` PDB component), so a verifier driven by it never expands more nodes than
/// before. All three components are advanced incrementally per node:
/// the ZPDB component in O(k) via projected-edge cost, Linear Conflict in
/// O(1) by recomputing only the at-most-three dirty row/column penalties, and
/// Walking Distance in O(1) by updating only the axis that changed (the
/// other axis's matrix and `h` are byte-identical to parent's).
///
/// `WalkingDistanceHeuristic::warm_up()` must have been called first; the
/// per-node `h` stays `u8` since the 15-puzzle diameter is 80.
pub struct ZpdbPlusInc<'a, const N: usize> {
    zpdb: ZpdbInc<'a, N>,
    lc: LinearConflictInc,
    wd: WalkingDistanceInc,
}

/// Per-node context for [`ZpdbPlusInc`] — the three sub-heuristics' contexts.
/// `Copy` so the search threads it through recursion.
#[derive(Clone, Copy)]
pub struct ZpdbPlusCtx<const N: usize> {
    zpdb: ZpdbCtx<N>,
    lc: LcCtx,
    wd: WdCtx,
}

impl<'a, const N: usize> ZpdbPlusInc<'a, N> {
    /// Build from `N` pairwise-disjoint ZPDBs (same contract as [`ZpdbInc::new`]).
    pub fn new(dbs: [&'a ZPatternDb; N]) -> Self {
        Self {
            zpdb: ZpdbInc::new(dbs),
            lc: LinearConflictInc,
            wd: WalkingDistanceInc,
        }
    }
}

impl<'a, const N: usize> IncHeuristic for ZpdbPlusInc<'a, N> {
    type Ctx = ZpdbPlusCtx<N>;

    #[inline]
    fn root(&self, s: &State, stats: &mut SearchStats) -> (u8, Self::Ctx) {
        let (h_zpdb, zctx) = self.zpdb.root(s, stats);
        let (h_lc, lctx) = self.lc.root(s, stats);
        let (h_wd, wctx) = self.wd.root(s, stats);
        let h = h_zpdb.max(h_lc).max(h_wd);
        (
            h,
            ZpdbPlusCtx {
                zpdb: zctx,
                lc: lctx,
                wd: wctx,
            },
        )
    }

    #[inline]
    fn advance(
        &self,
        parent: &Self::Ctx,
        child: &State,
        m: Move,
        stats: &mut SearchStats,
    ) -> (u8, Self::Ctx) {
        let (h_zpdb, zctx) = self.zpdb.advance(&parent.zpdb, child, m, stats);
        let (h_lc, lctx) = self.lc.advance(&parent.lc, child, m, stats);
        let (h_wd, wctx) = self.wd.advance(&parent.wd, child, m, stats);
        let h = h_zpdb.max(h_lc).max(h_wd);
        (
            h,
            ZpdbPlusCtx {
                zpdb: zctx,
                lc: lctx,
                wd: wctx,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::pdb::db::PatternDb;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::state::{State, GOAL};

    #[test]
    fn zpdb_heuristic_zero_on_goal() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        assert_eq!(ZpdbHeuristic::new(&zdb).h(&GOAL), 0);
    }

    /// Admissibility vs truth: zpdb(s) <= true_distance(s) over bfs_distances(10).
    #[test]
    fn zpdb_admissible_vs_truth_k3() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        let h = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(10);
        let mut violations = 0;
        for (raw, &td) in &truth {
            if h.h(&State(*raw)) > td {
                violations += 1;
            }
        }
        assert_eq!(violations, 0, "{violations} admissibility violations");
    }

    #[test]
    fn zpdb_admissible_vs_truth_k4() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let h = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(10);
        let mut violations = 0;
        for (raw, &td) in &truth {
            if h.h(&State(*raw)) > td {
                violations += 1;
            }
        }
        assert_eq!(violations, 0, "{violations} admissibility violations");
    }

    /// Tighter-or-equal than blank-agnostic: zpdb(s) >= additive.h(s) pointwise.
    #[test]
    fn zpdb_dominates_blank_agnostic_k3() {
        let pattern = Pattern::new(&[1, 2, 3]);
        let zdb = ZPatternDb::build(pattern);
        let pdb = PatternDb::build(pattern);
        let zh = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(10);
        let mut strictly_greater = 0usize;
        let mut total = 0usize;
        for raw in truth.keys() {
            let s = State(*raw);
            let z = zh.h(&s);
            let a = pdb.h(&s);
            assert!(z >= a, "zpdb {z} < blank-agnostic {a} for {raw:?}");
            if z > a {
                strictly_greater += 1;
            }
            total += 1;
        }
        println!(
            "zpdb strictly tighter than blank-agnostic on {}/{} states ({:.1}%)",
            strictly_greater,
            total,
            100.0 * strictly_greater as f64 / total as f64
        );
    }

    /// Incremental advance must agree with a fresh root at every step.
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
                "advance diverged at step {} (state {:?})",
                i, ns.0
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// The incremental `advance`'s core invariant: the projected-edge cost from
    /// `ProjectedState::apply` (0 = anon swap, 1 = pattern-tile swap) predicts
    /// EXACTLY whether the abstract `(m,p,r)` rank index changes. `advance`
    /// relies on this to skip the expensive `rank` recompute on cost-0 edges, so
    /// if it ever broke the heuristic would silently go wrong. Checked over a
    /// long random walk at k up to 8 (the production size) using only the cheap
    /// `ZpdbLayout` combinatorics — no PDB table build needed.
    #[test]
    fn zpdb_cost_predicts_index_change() {
        use crate::puzzle15::pdb::zpdb::ZpdbLayout;
        for pattern in [
            Pattern::new(&[1, 2, 3]),
            Pattern::new(&[6, 7, 8]),
            Pattern::new(&[1, 2, 3, 4, 5, 6, 7, 8]),
            Pattern::new(&[8, 9, 10, 11, 12, 13, 14, 15]),
        ] {
            let layout = ZpdbLayout::new(pattern);
            let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut s = GOAL;
            for _ in 0..4000 {
                let parent = ProjectedState::from_state(&s, pattern);
                let pidx = layout.rank(&parent, pattern);
                // Every legal edge: cost == 0  <=>  index unchanged.
                for m in s.legal_moves().iter() {
                    let (child, cost) = parent.apply(m);
                    let cidx = layout.rank(&child, pattern);
                    assert_eq!(
                        cost == 0,
                        cidx == pidx,
                        "cost {} but {}->{} for pattern {:?}, move {:?}, state {:?}",
                        cost,
                        pidx,
                        cidx,
                        pattern,
                        m,
                        s.0,
                    );
                }
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
        }
    }

    /// IDA* with the zero-aware heuristic returns optimal-length solutions.
    #[test]
    fn zpdb_inc_idastar_optimal_on_known_depths() {
        use crate::puzzle15::search::idastar_inc_with_stats;
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3, 4])),
            ZPatternDb::build(Pattern::new(&[5, 6, 7, 8])),
        ];
        let inc = ZpdbInc::new([&dbs[0], &dbs[1]]);
        let truth = bfs_distances(12);
        // Sample ~20 states of varying known depth.
        let mut samples: Vec<(State, u8)> = truth
            .iter()
            .map(|(raw, &d)| (State(*raw), d))
            .filter(|(_, d)| *d >= 8)
            .collect();
        samples.sort_by_key(|(s, _)| s.0);
        samples.truncate(20);
        assert!(
            samples.len() >= 10,
            "not enough deep samples ({})",
            samples.len()
        );
        for (s, true_d) in &samples {
            let (sol, _) = idastar_inc_with_stats(s, &inc);
            let sol = sol.expect("ZPDB IDA* found no solution");
            assert_eq!(sol.len() as u8, *true_d, "non-optimal for {:?}", s.0);
            let mut cur = *s;
            for &m in &sol {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "solution did not reach GOAL");
        }
    }

    /// `ZpdbPlusInc` must pointwise dominate the additive korf-plus it replaces
    /// in the enumeration verifier — `max(zpdb, refl-zpdb, LC, WD)` ≥
    /// `max(add, refl-add, LC, WD)` — so the verifier can never expand MORE
    /// nodes than before. Also stays admissible (≤ true distance).
    #[test]
    fn zpdb_plus_inc_dominates_korf_plus_pointwise() {
        use crate::puzzle15::pdb::heuristic::{AdditivePdbHeuristic, ReflectedHeuristic};
        use crate::puzzle15::search::{LinearConflictHeuristic, WalkingDistanceHeuristic};

        let pats = [Pattern::new(&[1, 2, 3, 4]), Pattern::new(&[5, 6, 7, 8])];
        let zdbs = [ZPatternDb::build(pats[0]), ZPatternDb::build(pats[1])];
        let adbs = [PatternDb::build(pats[0]), PatternDb::build(pats[1])];
        let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);

        WalkingDistanceHeuristic::warm_up();
        let add = AdditivePdbHeuristic::new(&adbs);
        let refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&adbs));

        let mut stats = SearchStats::default();
        for (raw, &td) in &bfs_distances(12) {
            let s = State(*raw);
            let hz = IncHeuristic::root(&zplus, &s, &mut stats).0;
            let korf_plus = add
                .h(&s)
                .max(refl.h(&s))
                .max(LinearConflictHeuristic.h(&s))
                .max(WalkingDistanceHeuristic.h(&s));
            assert!(
                hz >= korf_plus,
                "zpdb-plus {hz} < korf-plus {korf_plus} at {raw:?}"
            );
            assert!(
                hz <= td,
                "zpdb-plus {hz} inadmissible (true {td}) at {raw:?}"
            );
        }
    }
}
