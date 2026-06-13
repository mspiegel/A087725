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
use super::pattern::Pattern;
use crate::puzzle15::search::Heuristic;
use crate::puzzle15::state::State;
use crate::puzzle15::symmetry::reflect;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::pdb::pattern::Pattern;
    use crate::puzzle15::search::ManhattanHeuristic;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::state::{GOAL, State};

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
        let s = GOAL.apply(crate::puzzle15::state::Move::Up).apply(crate::puzzle15::state::Move::Left);
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
            assert!(est <= true_dist, "reflected h({}) > true dist {}", est, true_dist);
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
            assert!(est <= true_dist, "max h {} > true {} for {:?}", est, true_dist, raw);
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
            assert!(est <= true_dist, "korf h {} > true {} for {:?}", est, true_dist, raw);
        }
    }

}
