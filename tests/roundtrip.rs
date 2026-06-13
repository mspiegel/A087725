//! Exhaustive round-trip: for every solvable state, IDA*+Manhattan must
//! return a path of optimal length matching the precomputed distance table.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{idastar, ManhattanHeuristic};
use puzzle8::puzzle8::state::{GOAL, N_STATES};

#[test]
fn idastar_manhattan_matches_table_distance_on_full_state_space() {
    let t = DistanceTable::build();
    let h = ManhattanHeuristic;

    let mut mismatches = 0u32;
    for r in 0..N_STATES {
        let s = unrank(r);
        let truth = t.dist(&s);
        let sol = idastar(&s, &h).expect("solvable state must have a solution");
        let len = sol.len() as u8;
        if len != truth {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("mismatch at {:?}: IDA*={}, table={}", s.0, len, truth);
            }
        }
        // Spot-verify by replaying the solution.
        if r % 1000 == 0 {
            let mut cur = s;
            for &m in &sol {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "solution does not reach GOAL from {:?}", s.0);
        }
    }
    assert_eq!(mismatches, 0, "{} states had non-optimal IDA*+Manhattan solutions", mismatches);
}
