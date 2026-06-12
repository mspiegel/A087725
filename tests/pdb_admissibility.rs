//! Exhaustive admissibility test: for every implemented PDB partition,
//! the heuristic is `≤ true distance` on every solvable 8-puzzle state.
//!
//! Admissibility is the contract: a heuristic that ever exceeds the true
//! distance breaks IDA*'s optimality guarantee. We check it on the full
//! state space so any inadmissibility (anywhere) fails this test.

use puzzle8::bfs::DistanceTable;
use puzzle8::pdb::{AdditivePdbHeuristic, PatternDb, PdbHeuristic};
use puzzle8::pdb::pattern::Pattern;
use puzzle8::rank::unrank;
use puzzle8::search::Heuristic;
use puzzle8::state::N_STATES;

fn assert_admissible<H: Heuristic>(h: &H, table: &DistanceTable, label: &str) {
    for r in 0..N_STATES {
        let s = unrank(r);
        let est = h.h(&s);
        let truth = table.dist(&s);
        assert!(
            est <= truth,
            "{}: h({:?}) = {} > truth = {}",
            label,
            s.0,
            est,
            truth
        );
    }
}

#[test]
fn single_pdbs_admissible() {
    let table = DistanceTable::build();
    let patterns: &[&[u8]] = &[
        &[1],
        &[1, 2],
        &[1, 2, 3],
        &[1, 2, 3, 4],
        &[5, 6, 7, 8],
        &[1, 2, 3, 4, 5],
        &[4, 5, 6, 7, 8],
        &[1, 2, 3, 4, 5, 6, 7],
    ];
    for tiles in patterns {
        let pdb = PatternDb::build(Pattern::new(tiles));
        let h = PdbHeuristic::new(&pdb);
        assert_admissible(&h, &table, &format!("PDB{:?}", tiles));
    }
}

#[test]
fn additive_pairs_admissible() {
    let table = DistanceTable::build();
    let pairs: &[(&[u8], &[u8])] = &[
        (&[1, 2, 3, 4], &[5, 6, 7, 8]),
        (&[1, 2, 3], &[4, 5, 6, 7, 8]),
        (&[1, 2], &[3, 4, 5, 6, 7, 8]),
        (&[1, 2, 3, 4, 5], &[6, 7, 8]),
    ];
    for (left, right) in pairs {
        let dbs = vec![
            PatternDb::build(Pattern::new(left)),
            PatternDb::build(Pattern::new(right)),
        ];
        let h = AdditivePdbHeuristic::new(&dbs);
        assert_admissible(&h, &table, &format!("Add({:?}, {:?})", left, right));
    }
}

#[test]
fn additive_triple_admissible() {
    let table = DistanceTable::build();
    let dbs = vec![
        PatternDb::build(Pattern::new(&[1, 2, 3])),
        PatternDb::build(Pattern::new(&[4, 5])),
        PatternDb::build(Pattern::new(&[6, 7, 8])),
    ];
    let h = AdditivePdbHeuristic::new(&dbs);
    assert_admissible(&h, &table, "Add(1-3, 4-5, 6-8)");
}
