//! Exhaustive admissibility test for Linear Conflict on the 8-puzzle.
//!
//! For every one of the 181,440 solvable states:
//!   * `LC(s) ≤ table.dist(s)` (admissibility — required for IDA\* optimality)
//!   * `LC(s) ≥ Manhattan(s)`  (LC is supposed to tighten, never loosen)

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{Heuristic, LinearConflictHeuristic, ManhattanHeuristic};
use puzzle8::puzzle8::state::N_STATES;

#[test]
fn lc_is_admissible_on_full_state_space() {
    let table = DistanceTable::build();
    let h = LinearConflictHeuristic;
    for r in 0..N_STATES {
        let s = unrank(r);
        let est = h.h(&s);
        let truth = table.dist(&s);
        assert!(
            est <= truth,
            "LC({:?}) = {} > truth = {}",
            s.0,
            est,
            truth
        );
    }
}

#[test]
fn lc_dominates_manhattan_on_full_state_space() {
    let h_lc = LinearConflictHeuristic;
    let h_md = ManhattanHeuristic;
    for r in 0..N_STATES {
        let s = unrank(r);
        let lc = h_lc.h(&s);
        let md = h_md.h(&s);
        assert!(
            lc >= md,
            "LC({:?}) = {} < MD = {} (LC should never loosen MD)",
            s.0,
            lc,
            md
        );
    }
}
