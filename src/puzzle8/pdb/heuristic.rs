//! [`PdbHeuristic`] and [`AdditivePdbHeuristic`] — plug PDBs into
//! [`crate::puzzle8::search::idastar`] via the [`crate::puzzle8::search::Heuristic`] trait.
//!
//! `AdditivePdbHeuristic` requires patterns to be pairwise disjoint
//! (Korf & Felner 2002): every full-puzzle move advances at most one
//! pattern, so the sum of pattern distances remains admissible. Construction
//! verifies this; runtime queries are a sum of per-PDB lookups.

use super::db::PatternDb;
use super::pattern::Pattern;
use crate::puzzle8::search::Heuristic;
use crate::puzzle8::state::State;

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
    fn h(&self, s: &State) -> u8 {
        self.db.h(s)
    }
}

/// Sum-of-PDBs heuristic over a disjoint pattern partition.
///
/// The patterns of the supplied [`PatternDb`]s must be pairwise disjoint;
/// otherwise the sum could double-count moves that affect tiles in multiple
/// patterns, breaking admissibility.
pub struct AdditivePdbHeuristic<'a> {
    dbs: &'a [PatternDb],
}

impl<'a> AdditivePdbHeuristic<'a> {
    /// Verifies disjointness at construction. Patterns sharing any tile is a
    /// programmer error and panics.
    pub fn new(dbs: &'a [PatternDb]) -> Self {
        let mut union: u16 = 0;
        for db in dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "additive PDB patterns must be disjoint");
            union |= p;
        }
        Self { dbs }
    }

    /// Union of all patterns in this composition.
    pub fn coverage(&self) -> Pattern {
        let mut union: u16 = 0;
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
    fn h(&self, s: &State) -> u8 {
        let mut sum: u32 = 0;
        for db in self.dbs {
            sum += db.h(s) as u32;
        }
        // For the 8-puzzle the sum is bounded by 31 < 256, so the clamp is
        // never exercised in practice, but it's the correct defensive form.
        sum.min(u8::MAX as u32) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::bfs::DistanceTable;
    use crate::puzzle8::pdb::pattern::Pattern;
    use crate::puzzle8::state::{State, GOAL};

    #[test]
    fn pdb_heuristic_returns_zero_on_goal() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let h = PdbHeuristic::new(&pdb);
        assert_eq!(h.h(&GOAL), 0);
    }

    #[test]
    fn additive_returns_sum_of_components() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let b = PatternDb::build(Pattern::new(&[5, 6, 7, 8]));
        let dbs = vec![a, b];
        let h = AdditivePdbHeuristic::new(&dbs);
        let s = State([8, 6, 7, 2, 5, 4, 3, 0, 1]); // antipode 1
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
    fn additive_sum_admissible_on_antipodes() {
        let table = DistanceTable::build();
        let a = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let b = PatternDb::build(Pattern::new(&[5, 6, 7, 8]));
        let dbs = vec![a, b];
        let h = AdditivePdbHeuristic::new(&dbs);
        for ap in table.antipodes() {
            let est = h.h(&ap);
            let truth = table.dist(&ap);
            assert!(
                est <= truth,
                "additive PDB inadmissible at antipode {:?}: h={}, truth={}",
                ap.0,
                est,
                truth
            );
        }
    }

    #[test]
    fn coverage_reports_union_of_patterns() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let b = PatternDb::build(Pattern::new(&[5, 6, 7]));
        let dbs = vec![a, b];
        let h = AdditivePdbHeuristic::new(&dbs);
        let cov = h.coverage();
        for t in 1..=8u8 {
            assert_eq!(cov.contains(t), [1, 2, 3, 5, 6, 7].contains(&t));
        }
    }
}
