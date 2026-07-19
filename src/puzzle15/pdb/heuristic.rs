//! Heuristic combinators that plug PDBs into the 15-puzzle IDA\*.
//!
//! - [`PdbHeuristic`]: single PDB lookup.
//! - [`AdditivePdbHeuristic`]: sum of PDB values across a *disjoint*
//!   pattern partition. Disjointness is enforced at construction.
//! - [`ReflectedHeuristic`]: wraps any [`Heuristic`] and queries it on the
//!   diagonal reflection of the input state. The reflection is a
//!   goal-preserving isomorphism, so the wrapped heuristic remains
//!   admissible.
//! - [`MaxHeuristic`]: max of two admissibles is admissible. Used to combine
//!   the normal and reflected views into Korf's tightest heuristic at half
//!   the storage budget — instead of building separately stored reflected
//!   PDB files, we reflect the state on demand.

use super::db::PatternDb;
use super::pattern::{Pattern, ProjectedState};
use crate::puzzle15::search::{Heuristic, IncHeuristic, IncHeuristicMut, SearchStats};
use crate::puzzle15::state::{Move, State};
use crate::puzzle15::symmetry::{reflect, transpose_move};

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
///
/// The patterns of the supplied [`PatternDb`]s must be pairwise disjoint
/// (Korf & Felner 2002); otherwise the sum could double-count moves that
/// affect tiles in multiple patterns, breaking admissibility. Construction
/// verifies disjointness; runtime queries are a sum of per-PDB lookups.
pub struct AdditivePdbHeuristic<'a> {
    dbs: &'a [PatternDb],
}

impl<'a> AdditivePdbHeuristic<'a> {
    /// Verifies disjointness at construction. Patterns sharing any tile is a
    /// programmer error and panics.
    pub fn new(dbs: &'a [PatternDb]) -> Self {
        let mut union: u32 = 0;
        for db in dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "additive PDB patterns must be disjoint");
            union |= p;
        }
        Self { dbs }
    }

    /// Union of all patterns in this composition.
    pub fn coverage(&self) -> Pattern {
        let mut union: u32 = 0;
        for db in self.dbs {
            union |= db.pattern().0;
        }
        Pattern(union)
    }

    /// Total bytes stored across all component PDBs.
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
        // 15-puzzle diameter is 80 ≪ 255, so the clamp is defensive.
        sum.min(u8::MAX as u32) as u8
    }
}

/// Wraps a [`Heuristic`] so that `h(s)` is evaluated on the diagonal
/// reflection of `s`. Admissible because reflection is a goal-preserving
/// isomorphism: `dist(s) = dist(reflect(s))`, so any admissible bound on
/// `reflect(s)` is also an admissible bound on `s`.
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

/// Max of two admissible heuristics. Always admissible (max of lower bounds
/// is a lower bound).
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
        let ha = self.a.h(s);
        let hb = self.b.h(s);
        ha.max(hb)
    }
}

/// Incremental form of the Korf heuristic — `max(additive(s),
/// additive(reflect(s)))` over a disjoint, fixed-size PDB partition.
///
/// Holds `N` pairwise-disjoint PDBs (the Korf 7-8 partition is `N = 2`). The
/// per-node [`KorfCtx`] keeps each PDB's projected board in both the normal and
/// reflected views; [`advance`](IncHeuristic::advance) slides those projections
/// by one move — the normal view by the move itself, the reflected view by
/// [`transpose_move`] — instead of re-projecting the full board and recomputing
/// [`reflect`] at every node. The value is byte-for-byte identical to the
/// [`MaxHeuristic`]-of-[`AdditivePdbHeuristic`] composition; only the per-node
/// cost differs.
pub struct KorfPdbInc<'a, const N: usize> {
    /// Each PDB's pattern (cached so `value`/`root` don't re-read it).
    patterns: [Pattern; N],
    /// Each PDB's distance slice, cached once. Avoids recomputing
    /// `Storage::dist()` (`&mmap[HEADER..]`, a length-checked re-slice) on every
    /// lookup — profiling showed that re-slice running 4× per node.
    dist: [&'a [u8]; N],
}

/// Per-node context for [`KorfPdbInc`]: each PDB's projected board in the normal
/// and reflected views. `Copy` so the search threads it by value.
#[derive(Clone, Copy)]
pub struct KorfCtx<const N: usize> {
    normal: [ProjectedState; N],
    reflected: [ProjectedState; N],
}

impl<'a, const N: usize> KorfPdbInc<'a, N> {
    /// Build from `N` PDBs. Panics unless the patterns are pairwise disjoint
    /// (required for additive admissibility, Korf & Felner 2002).
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
            // SAFETY: `rank()` is in `[0, num_projected_states)`, and
            // `d.len() == num_projected_states` by the PDB construction
            // invariant (asserted in `PatternDb::from_dist` / `load*`). So both
            // indices are in bounds; this elides the per-lookup bounds check
            // profiling flagged in the hot path.
            debug_assert!(rn < d.len() && rr < d.len());
            hn += unsafe { *d.get_unchecked(rn) } as u32;
            hr += unsafe { *d.get_unchecked(rr) } as u32;
        }
        // 15-puzzle diameter 80 ≪ 255; clamp is defensive.
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

/// Make/unmake variant — same heuristic value, but mutates the projected boards
/// in place instead of copying a fresh context per node. A/B against the `Copy`
/// path with `korf100_bench --mut`.
impl<'a, const N: usize> IncHeuristicMut for KorfPdbInc<'a, N> {
    type Ctx = KorfCtx<N>;

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let ctx = KorfCtx {
            normal: std::array::from_fn(|i| ProjectedState::from_state(s, self.patterns[i])),
            reflected: std::array::from_fn(|i| ProjectedState::from_state(&rs, self.patterns[i])),
        };
        (self.value(&ctx), ctx)
    }

    fn make(&self, ctx: &mut Self::Ctx, m: Move) -> u8 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::pdb::pattern::Pattern;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::search::ManhattanHeuristic;
    use crate::puzzle15::state::{State, GOAL};

    #[test]
    fn pdb_heuristic_returns_zero_on_goal() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let h = PdbHeuristic::new(&pdb);
        assert_eq!(h.h(&GOAL), 0);
    }

    #[test]
    fn additive_returns_sum_of_components() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let b = PatternDb::build(Pattern::new(&[7, 8, 9]));
        let dbs = vec![a, b];
        let h = AdditivePdbHeuristic::new(&dbs);
        // For GOAL, both components return 0.
        assert_eq!(h.h(&GOAL), 0);
        // For a non-goal state, sum equals individual.
        let s = GOAL
            .apply(crate::puzzle15::state::Move::Up)
            .apply(crate::puzzle15::state::Move::Left);
        let expected = dbs[0].h(&s) + dbs[1].h(&s);
        assert_eq!(h.h(&s), expected);
    }

    #[test]
    #[should_panic(expected = "disjoint")]
    fn additive_rejects_overlapping_patterns() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let b = PatternDb::build(Pattern::new(&[3, 4, 5]));
        let dbs = vec![a, b];
        let _ = AdditivePdbHeuristic::new(&dbs);
    }

    #[test]
    fn coverage_reports_union_of_patterns() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let b = PatternDb::build(Pattern::new(&[5, 6, 7]));
        let dbs = vec![a, b];
        let h = AdditivePdbHeuristic::new(&dbs);
        let cov = h.coverage();
        for t in 1..=15u8 {
            assert_eq!(cov.contains(t), [1, 2, 3, 5, 6, 7].contains(&t));
        }
    }

    #[test]
    fn reflected_heuristic_admissible_on_shallow_bfs() {
        // For each state at known true distance ≤ 10, the reflected heuristic
        // value must be ≤ true distance.
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let base = PdbHeuristic::new(&pdb);
        let refl = ReflectedHeuristic::new(base);
        let truth = bfs_distances(10);
        for (raw, &true_dist) in &truth {
            let est = refl.h(&State(*raw));
            assert!(
                est <= true_dist,
                "reflected h({est}) > true dist {true_dist}"
            );
        }
    }

    #[test]
    fn reflected_of_goal_is_zero() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let base = PdbHeuristic::new(&pdb);
        let refl = ReflectedHeuristic::new(base);
        assert_eq!(refl.h(&GOAL), 0);
    }

    #[test]
    fn max_heuristic_is_admissible() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let h_pdb = PdbHeuristic::new(&pdb);
        let h_max = MaxHeuristic::new(h_pdb, ManhattanHeuristic);
        let truth = bfs_distances(10);
        for (raw, &true_dist) in &truth {
            let est = h_max.h(&State(*raw));
            assert!(
                est <= true_dist,
                "max h {est} > true {true_dist} for {raw:?}"
            );
        }
    }

    #[test]
    fn max_picks_larger_of_two() {
        // Construct a state where Manhattan < PDB so max returns PDB.
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let h_pdb = PdbHeuristic::new(&pdb);
        let h_man = ManhattanHeuristic;
        let h_max = MaxHeuristic::new(&h_pdb, &h_man);
        let s = GOAL.apply(crate::puzzle15::state::Move::Up);
        let est_max = h_max.h(&s);
        let est_pdb = h_pdb.h(&s);
        let est_man = h_man.h(&s);
        assert_eq!(est_max, est_pdb.max(est_man));
    }

    #[test]
    fn korf_style_max_of_additive_and_reflected_admissible() {
        // Build a small disjoint partition and check the standard Korf-style
        // max(additive(p), reflected(additive(p))) composition.
        let a = PatternDb::build(Pattern::new(&[1, 2]));
        let b = PatternDb::build(Pattern::new(&[3, 4]));
        let dbs = vec![a, b];
        // We have to construct the additive heuristic by reference.
        let h_add = AdditivePdbHeuristic::new(&dbs);
        let h_refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs));
        let h_korf = MaxHeuristic::new(&h_add, &h_refl);
        let truth = bfs_distances(10);
        for (raw, &true_dist) in &truth {
            let est = h_korf.h(&State(*raw));
            assert!(
                est <= true_dist,
                "korf h {est} > true {true_dist} for {raw:?}"
            );
        }
    }

    #[test]
    fn korf_inc_root_matches_scratch_combinator() {
        // KorfPdbInc::root must equal max(additive, reflected(additive)) — the
        // exact scratch heuristic it replaces — at every state along a walk.
        let dbs = [
            PatternDb::build(Pattern::new(&[1, 2, 3, 4])),
            PatternDb::build(Pattern::new(&[5, 6, 7, 8])),
        ];
        let h_add = AdditivePdbHeuristic::new(&dbs);
        let h_refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs));
        let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);
        let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);

        let pseudo = |i: u32| Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize];
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        for i in 0u32..3000 {
            let (h_inc, _) = IncHeuristic::root(&inc, &s, &mut stats);
            assert_eq!(h_inc, h_korf.h(&s), "inc.root != scratch korf at {:?}", s.0);
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    #[test]
    fn korf_inc_advance_matches_fresh_reprojection() {
        // The incremental advance (normal view by m, reflected view by the
        // transposed move) must match a fresh root() reprojection at every step
        // — the property that makes the incremental search exact.
        let dbs = [
            PatternDb::build(Pattern::new(&[2, 5, 8, 11, 14])),
            PatternDb::build(Pattern::new(&[1, 3, 6, 9])),
        ];
        let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);

        let pseudo = |i: u32| Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize];
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (_, mut ctx) = IncHeuristic::root(&inc, &s, &mut stats);
        for i in 0u32..3000 {
            let mut chosen = None;
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    chosen = Some(m);
                    break;
                }
            }
            let m = chosen.unwrap();
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = inc.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&inc, &ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "advance diverged from reprojection at step {i}"
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    #[test]
    fn korf_inc_mut_matches_immutable_path() {
        // The make/unmake search must produce identical results (path, length,
        // node count) to the Copy-context search with the same evaluator —
        // validates make/unmake correctness and undo-on-backtrack.
        use crate::puzzle15::search::{idastar_inc_mut_with_stats, idastar_inc_with_stats};
        let dbs = [
            PatternDb::build(Pattern::new(&[1, 2, 3, 4])),
            PatternDb::build(Pattern::new(&[5, 6, 7, 8])),
        ];
        let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);
        let pseudo = |i: u32| Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize];
        let mut s = GOAL;
        for seed in 0..12u32 {
            let (a_sol, a_st) = idastar_inc_with_stats(&s, &inc);
            let (b_sol, b_st) = idastar_inc_mut_with_stats(&s, &inc);
            assert_eq!(a_sol, b_sol, "paths differ at depth {seed}");
            assert_eq!(a_st.nodes, b_st.nodes, "node counts differ at depth {seed}");
            for k in 0..4 {
                let m = pseudo(seed.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }
}
