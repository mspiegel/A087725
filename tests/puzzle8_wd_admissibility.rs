//! Exhaustive admissibility test for Walking Distance on the 8-puzzle.
//!
//! For every solvable state, `WD_row(s) + WD_col(s) ≤ table.dist(s)`.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle8::state::N_STATES;

#[test]
fn wd_is_admissible_on_full_state_space() {
    let table = DistanceTable::build();
    let h = WalkingDistanceHeuristic;
    WalkingDistanceHeuristic::warm_up();
    for r in 0..N_STATES {
        let s = unrank(r);
        let est = h.h(&s);
        let truth = table.dist(&s);
        assert!(
            est <= truth,
            "WD({:?}) = {} > truth = {}",
            s.0,
            est,
            truth
        );
    }
}
