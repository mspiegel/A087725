//! Integration test: 24-puzzle Walking Distance.
//!
//! Against exhaustive shallow BFS, WD is admissible (`WD ≤ truth`); driving IDA*
//! with the incremental WD recovers the exact BFS distance and replays to GOAL.
//! Builds the (heavy, ~65.65M-state) WD table once.
//!
//! Run with `cargo test --release --test puzzle24_walking_distance --features parallel,mmap`.

use std::collections::HashMap;

use puzzle8::puzzle24::search::{
    idastar_inc, Heuristic, WalkingDistanceHeuristic, WalkingDistanceInc,
};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

fn bfs(depth_limit: u8) -> HashMap<[u8; N_CELLS], u8> {
    let mut dist: HashMap<[u8; N_CELLS], u8> = HashMap::new();
    dist.insert(GOAL.0, 0);
    let mut frontier = vec![GOAL];
    let mut depth = 0u8;
    while !frontier.is_empty() && depth < depth_limit {
        let nd = depth + 1;
        let mut next = Vec::new();
        for s in &frontier {
            for m in s.legal_moves().iter() {
                let s2 = s.apply(m);
                if !dist.contains_key(&s2.0) {
                    dist.insert(s2.0, nd);
                    next.push(s2);
                }
            }
        }
        depth = nd;
        frontier = next;
    }
    dist
}

fn reaches_goal(start: &State, sol: &[Move]) -> bool {
    let mut cur = *start;
    for m in sol {
        cur = cur.apply(*m);
    }
    cur == GOAL
}

#[test]
fn wd_admissible_on_shallow_states() {
    WalkingDistanceHeuristic::warm_up();
    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let wd = WalkingDistanceHeuristic.h(&s);
        assert!(wd <= d, "WD {} > truth {} for {:?}", wd, d, raw);
    }
}

#[test]
fn wd_inc_solver_optimal_on_shallow_states() {
    WalkingDistanceHeuristic::warm_up();
    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let sol = idastar_inc(&s, &WalkingDistanceInc).expect("solvable");
        assert_eq!(sol.len() as u8, d, "WD-inc non-optimal for {:?}", raw);
        assert!(reaches_goal(&s, &sol));
    }
}
