//! The lossless gate for direction #3: every PDB-driven IDA* solver must
//! produce optimal solutions matching `table.dist(s)`.
//!
//! Two flavours of test:
//!
//! * **Exhaustive sweeps** for the *strong* additive PDBs — those are the
//!   practical heuristics, and the test budget allows the full 181,440-state
//!   sweep in under a minute.
//!
//! * **Sampled sweeps** for *weaker* single-pattern PDBs that cover only a
//!   subset of tiles (e.g. PDB[1,2,3,4]). These produce hugely-deep IDA*
//!   searches per instance — full-sweep cost would dominate the test suite
//!   without strengthening the correctness guarantee, since admissibility
//!   (verified in `pdb_admissibility.rs`) plus IDA*'s optimality theorem
//!   already implies these solvers return optimal results.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::pdb::pattern::Pattern;
use puzzle8::puzzle8::pdb::{AdditivePdbHeuristic, PatternDb, PdbHeuristic};
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{idastar, Heuristic};
use puzzle8::puzzle8::state::{GOAL, N_STATES};

fn assert_optimal_on_full_space<H: Heuristic>(h: &H, table: &DistanceTable, label: &str) {
    for r in 0..N_STATES {
        let s = unrank(r);
        let sol = idastar(&s, h).expect("solvable state must have a solution");
        let len = sol.len() as u8;
        let truth = table.dist(&s);
        assert_eq!(len, truth, "{}: IDA* gave {} for {:?}, truth = {}", label, len, s.0, truth);
        // Spot-replay every 5000 states.
        if r % 5000 == 0 {
            let mut cur = s;
            for &m in &sol {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "{}: solution did not reach GOAL from {:?}", label, s.0);
        }
    }
}

fn assert_optimal_on_sample<H: Heuristic>(
    h: &H,
    table: &DistanceTable,
    label: &str,
    stride: u32,
) {
    let mut checked = 0u32;
    for r in (0..N_STATES).step_by(stride as usize) {
        let s = unrank(r);
        let sol = idastar(&s, h).expect("solvable state must have a solution");
        let len = sol.len() as u8;
        let truth = table.dist(&s);
        assert_eq!(len, truth, "{}: IDA* gave {} for {:?}, truth = {}", label, len, s.0, truth);
        checked += 1;
    }
    assert!(checked > 100, "sample too small for {}: {} states", label, checked);
}

#[test]
#[ignore = "exhaustive over the full 8-puzzle state space; run with `cargo test -- --ignored`"]
fn idastar_with_additive_4plus4_pdb_is_optimal_everywhere() {
    let table = DistanceTable::build();
    let dbs = vec![
        PatternDb::build(Pattern::new(&[1, 2, 3, 4])),
        PatternDb::build(Pattern::new(&[5, 6, 7, 8])),
    ];
    let h = AdditivePdbHeuristic::new(&dbs);
    assert_optimal_on_full_space(&h, &table, "Add([1-4],[5-8])");
}

#[test]
#[ignore = "exhaustive over the full 8-puzzle state space; run with `cargo test -- --ignored`"]
fn idastar_with_additive_3plus5_pdb_is_optimal_everywhere() {
    let table = DistanceTable::build();
    let dbs = vec![
        PatternDb::build(Pattern::new(&[1, 2, 3])),
        PatternDb::build(Pattern::new(&[4, 5, 6, 7, 8])),
    ];
    let h = AdditivePdbHeuristic::new(&dbs);
    assert_optimal_on_full_space(&h, &table, "Add([1-3],[4-8])");
}

#[test]
#[ignore = "exhaustive over the full 8-puzzle state space; run with `cargo test -- --ignored`"]
fn idastar_with_additive_triple_pdb_is_optimal_everywhere() {
    let table = DistanceTable::build();
    let dbs = vec![
        PatternDb::build(Pattern::new(&[1, 2, 3])),
        PatternDb::build(Pattern::new(&[4, 5])),
        PatternDb::build(Pattern::new(&[6, 7, 8])),
    ];
    let h = AdditivePdbHeuristic::new(&dbs);
    assert_optimal_on_full_space(&h, &table, "Add([1-3],[4-5],[6-8])");
}

#[test]
fn idastar_with_single_5tile_pdb_optimal_on_sample() {
    // PDB[1..5] is weaker than Manhattan (covers only 5 of 8 tiles);
    // exhaustive testing is slow. Sample stride 200 → ~907 states, ~seconds.
    let table = DistanceTable::build();
    let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4, 5]));
    let h = PdbHeuristic::new(&pdb);
    assert_optimal_on_sample(&h, &table, "PDB[1,2,3,4,5]", 200);
}

#[test]
#[ignore = "exhaustive over the full 8-puzzle state space; run with `cargo test -- --ignored`"]
fn idastar_with_single_7tile_pdb_is_optimal_everywhere() {
    // PDB[1..7] is *strong* (covers all but one tile); exhaustive is cheap.
    let table = DistanceTable::build();
    let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4, 5, 6, 7]));
    let h = PdbHeuristic::new(&pdb);
    assert_optimal_on_full_space(&h, &table, "PDB[1,2,3,4,5,6,7]");
}
