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

    #[test]
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
