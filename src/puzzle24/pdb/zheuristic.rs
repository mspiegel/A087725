//! Up-to-date zero-aware PDB heuristics for the 24-puzzle IDA\*, ported from
//! `crate::puzzle15::pdb::zheuristic`.
//!
//! The incremental Korf-max evaluator [`ZpdbInc`] and its context already live
//! in [`super::heuristic`] (it was the original 24-puzzle port that `puzzle15`
//! later grew the "plus" stack from); this module reuses it and adds the pieces
//! `puzzle24` was missing:
//!
//! - [`ZpdbHeuristic`]: single zero-aware PDB lookup (implements [`Heuristic`]),
//!   for candidate ranking via O(h) cold lookups.
//! - [`AdditiveZpdbHeuristic`]: sum over a disjoint pattern partition.
//! - [`ZpdbPlusInc`]: the 1-bit-incremental `max(zpdb, refl-zpdb, MD+LC, WD)` —
//!   the drop-in incremental replacement for the additive `korf-plus`, and the
//!   strongest admissible heuristic available for deep-board search.
//!
//! Admissibility: each ZPDB value lower-bounds the true distance (it is a BFS
//! over a contracted quotient of the real move graph) and pointwise dominates
//! the blank-agnostic additive PDB; the sum over disjoint patterns is
//! admissible; max of the normal and diagonal-reflected views is admissible
//! because the reflection preserves distance to GOAL; LC and WD are independently
//! admissible, so their max with the ZPDB term is too.

use super::heuristic::{ZpdbCtx, ZpdbCtxMut, ZpdbInc};
use super::pattern::Pattern;
use super::zdb::ZPatternDb;
use crate::puzzle24::search::{
    Heuristic, IncHeuristic, IncHeuristicMut, LcCtx, LcMutCtx, LinearConflictInc, SearchStats,
    WalkingDistanceInc, WdCtx, WdMutCtx,
};
use crate::puzzle24::state::{Move, State};

/// Single zero-aware-PDB heuristic. Each evaluation is one O(h)
/// [`ZPatternDb::cold_lookup`] — use for candidate ranking, not the IDA\* hot
/// path (there, drive [`ZpdbInc`]/[`ZpdbPlusInc`] incrementally instead).
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

/// Sum-of-ZPDBs heuristic over a disjoint pattern partition. Admissible by
/// Korf-Felner additivity; each term pointwise dominates its blank-agnostic
/// additive PDB, so this dominates the plain additive heuristic.
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

/// Incremental "zpdb-plus": [`ZpdbInc`] (zero-aware Korf-max) maxed per node with
/// the classical Linear-Conflict and Walking-Distance heuristics — the drop-in
/// incremental replacement for the additive `korf-plus`.
///
/// Because each zero-aware ZPDB pointwise dominates its blank-agnostic additive
/// PDB, this **dominates the additive korf-plus** (identical LC/WD terms, a `≥`
/// PDB component), so a verifier driven by it never expands more nodes than
/// before. All three components advance incrementally per node: the ZPDB
/// component in O(k) via projected-edge cost, Linear Conflict by recomputing only
/// the at-most-three dirty row/column penalties, and Walking Distance by updating
/// only the axis that changed (the other axis is byte-identical to the parent's).
///
/// [`WalkingDistanceHeuristic::warm_up`] should be called first; the per-node `h`
/// stays `u8` since the 24-puzzle diameter is ≤ 205.
///
/// [`WalkingDistanceHeuristic::warm_up`]: crate::puzzle24::search::WalkingDistanceHeuristic::warm_up
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
        // UFCS: `ZpdbInc`/`LinearConflictInc`/`WalkingDistanceInc` implement both
        // `IncHeuristic` and `IncHeuristicMut`, whose `root`s collide under method
        // syntax.
        let (h_zpdb, zctx) = IncHeuristic::root(&self.zpdb, s, stats);
        let (h_lc, lctx) = IncHeuristic::root(&self.lc, s, stats);
        let (h_wd, wctx) = IncHeuristic::root(&self.wd, s, stats);
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

/// Make/unmake context for [`ZpdbPlusInc`]: the three components' mutable
/// contexts, each maintained in place (see [`ZpdbCtxMut`], [`LcMutCtx`],
/// [`WdMutCtx`]). The zpdb component is the copy-heavy one, so this stack is
/// where make/unmake pays off most for the combined heuristic.
pub struct ZpdbPlusMutCtx<const N: usize> {
    zpdb: ZpdbCtxMut<N>,
    lc: LcMutCtx,
    wd: WdMutCtx,
}

impl<'a, const N: usize> IncHeuristicMut for ZpdbPlusInc<'a, N> {
    type Ctx = ZpdbPlusMutCtx<N>;

    #[inline]
    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let (h_zpdb, zctx) = IncHeuristicMut::root(&self.zpdb, s);
        let (h_lc, lctx) = IncHeuristicMut::root(&self.lc, s);
        let (h_wd, wctx) = IncHeuristicMut::root(&self.wd, s);
        let h = h_zpdb.max(h_lc).max(h_wd);
        (
            h,
            ZpdbPlusMutCtx {
                zpdb: zctx,
                lc: lctx,
                wd: wctx,
            },
        )
    }

    #[inline]
    fn make(&self, ctx: &mut Self::Ctx, child: &State, m: Move) -> u8 {
        let h_zpdb = self.zpdb.make(&mut ctx.zpdb, child, m);
        let h_lc = self.lc.make(&mut ctx.lc, child, m);
        let h_wd = self.wd.make(&mut ctx.wd, child, m);
        h_zpdb.max(h_lc).max(h_wd)
    }

    #[inline]
    fn unmake(&self, ctx: &mut Self::Ctx, m: Move) {
        self.zpdb.unmake(&mut ctx.zpdb, m);
        self.lc.unmake(&mut ctx.lc, m);
        self.wd.unmake(&mut ctx.wd, m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::pdb::db::PatternDb;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::state::{State, GOAL};

    #[test]
    fn zpdb_heuristic_zero_on_goal() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        assert_eq!(ZpdbHeuristic::new(&zdb).h(&GOAL), 0);
    }

    /// Admissibility vs truth: zpdb(s) <= true_distance(s) over bfs_distances(9).
    #[test]
    fn zpdb_admissible_vs_truth_k3() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        let h = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(9);
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
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3, 6]));
        let h = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(9);
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
        let truth = bfs_distances(9);
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

    #[test]
    fn additive_zpdb_sums_components_and_rejects_overlap() {
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let h = AdditiveZpdbHeuristic::new(&dbs);
        assert_eq!(h.h(&GOAL), 0);
        let s = GOAL.apply(Move::Up).apply(Move::Left).apply(Move::Up);
        assert_eq!(h.h(&s), dbs[0].cold_lookup(&s) + dbs[1].cold_lookup(&s));
    }

    #[test]
    #[should_panic(expected = "disjoint")]
    fn additive_zpdb_rejects_overlap() {
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[3, 4, 5])),
        ];
        let _ = AdditiveZpdbHeuristic::new(&dbs);
    }

    /// `ZpdbPlusInc::advance` must agree with a fresh `root` at every step of a
    /// random walk — the core hot-path invariant for the combined heuristic.
    #[test]
    fn zpdb_plus_advance_matches_fresh_root_random_walk() {
        use crate::puzzle24::search::WalkingDistanceHeuristic;
        WalkingDistanceHeuristic::warm_up();
        let zdbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);

        let mut rng: u64 = 0xF00D_BABE_1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (_, mut ctx) = IncHeuristic::root(&zplus, &s, &mut stats);
        for i in 0..1500 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = zplus.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&zplus, &ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "advance diverged at step {} (state {:?})",
                i, ns.0
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// `ZpdbPlusInc` must pointwise dominate the additive korf-plus it replaces —
    /// `max(zpdb, refl-zpdb, LC, WD) ≥ max(add, refl-add, LC, WD)` — so a
    /// verifier driven by it never expands MORE nodes than before. Also stays
    /// admissible (≤ true distance).
    #[test]
    fn zpdb_plus_inc_dominates_korf_plus_pointwise() {
        use crate::puzzle24::pdb::heuristic::{AdditivePdbHeuristic, ReflectedHeuristic};
        use crate::puzzle24::search::{LinearConflictHeuristic, WalkingDistanceHeuristic};

        let pats = [Pattern::new(&[1, 2, 3]), Pattern::new(&[6, 7, 8])];
        let zdbs = [ZPatternDb::build(pats[0]), ZPatternDb::build(pats[1])];
        let adbs = [PatternDb::build(pats[0]), PatternDb::build(pats[1])];
        let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);

        WalkingDistanceHeuristic::warm_up();
        let add = AdditivePdbHeuristic::new(&adbs);
        let refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&adbs));

        let mut stats = SearchStats::default();
        for (raw, &td) in &bfs_distances(9) {
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

    /// IDA\* driven by `ZpdbPlusInc` returns optimal-length solutions on shallow
    /// scrambles and the path reaches GOAL.
    #[test]
    fn zpdb_plus_idastar_optimal_on_known_depths() {
        use crate::puzzle24::search::{idastar_inc_with_stats, WalkingDistanceHeuristic};
        WalkingDistanceHeuristic::warm_up();
        let zdbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3, 6])),
            ZPatternDb::build(Pattern::new(&[4, 5, 9, 10])),
        ];
        let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);
        let truth = bfs_distances(10);
        let mut samples: Vec<(State, u8)> = truth
            .iter()
            .map(|(raw, &d)| (State(*raw), d))
            .filter(|(_, d)| *d >= 7)
            .collect();
        samples.sort_by_key(|(s, _)| s.0);
        samples.truncate(20);
        assert!(
            samples.len() >= 10,
            "not enough deep samples ({})",
            samples.len()
        );
        for (s, true_d) in &samples {
            let (sol, _) = idastar_inc_with_stats(s, &zplus);
            let sol = sol.expect("ZpdbPlus IDA* found no solution");
            assert_eq!(sol.len() as u8, *true_d, "non-optimal for {:?}", s.0);
            let mut cur = *s;
            for &m in &sol {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "solution did not reach GOAL");
        }
    }

    /// ZpdbPlus make/unmake (composing zpdb + LC + WD mutable contexts) must
    /// match the copy driver on length and node count — the composite whose
    /// zpdb component gains the most from make/unmake.
    #[test]
    fn zpdb_plus_mut_idastar_matches_copy_length_and_nodes() {
        use crate::puzzle24::search::{idastar_inc_with_stats, Search, WalkingDistanceHeuristic};
        WalkingDistanceHeuristic::warm_up();
        let zdbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3, 6])),
            ZPatternDb::build(Pattern::new(&[4, 5, 9, 10])),
        ];
        let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);
        let mut rng: u64 = 0x1618_0339_8875_2440;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..5 {
            let mut s = GOAL;
            for _ in 0..16 {
                let opts: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(opts[(next() as usize) % opts.len()]);
            }
            let (c, cs) = idastar_inc_with_stats(&s, &zplus);
            let (m, ms) = Search::new(&s, &zplus).solve_with_stats();
            assert_eq!(
                c.expect("copy").len(),
                m.expect("mut").len(),
                "ZpdbPlus length differs"
            );
            assert_eq!(cs.nodes, ms.nodes, "ZpdbPlus node count differs");
        }
    }
}
