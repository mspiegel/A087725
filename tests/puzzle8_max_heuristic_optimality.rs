//! IDA\* with `max(Manhattan, LC, WD)` returns the optimal solution length on
//! every solvable 8-puzzle state.
//!
//! This combines admissibility (verified separately in the LC and WD
//! admissibility tests) with IDA\*'s optimality theorem — if any of those
//! contracts are broken, the solution length here will diverge from
//! `table.dist(s)`.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{
    idastar, LinearConflictHeuristic, ManhattanHeuristic, MaxHeuristic,
    WalkingDistanceHeuristic,
};
use puzzle8::puzzle8::state::{GOAL, N_STATES};

#[test]
fn idastar_with_max_of_md_lc_wd_is_optimal_on_full_state_space() {
    let table = DistanceTable::build();
    WalkingDistanceHeuristic::warm_up();

    // max(MD, LC, WD) via nested binary combinators.
    let h_lc_wd = MaxHeuristic::new(LinearConflictHeuristic, WalkingDistanceHeuristic);
    let h = MaxHeuristic::new(ManhattanHeuristic, h_lc_wd);

    for r in 0..N_STATES {
        let s = unrank(r);
        let sol = idastar(&s, &h).expect("solvable state must have a solution");
        let len = sol.len() as u8;
        let truth = table.dist(&s);
        assert_eq!(
            len, truth,
            "IDA*[max(MD,LC,WD)] gave {} for {:?}, truth = {}",
            len, s.0, truth
        );
        // Spot-replay every 5000 states to catch corrupted move sequences.
        if r % 5000 == 0 {
            let mut cur = s;
            for &m in &sol {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "solution did not reach GOAL from {:?}", s.0);
        }
    }
}
