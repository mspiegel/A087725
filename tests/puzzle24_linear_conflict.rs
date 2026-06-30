//! Integration test: 24-puzzle Manhattan + Linear Conflict.
//!
//! Against exhaustive shallow BFS, LC is admissible (`LC ≤ truth`) and dominates
//! Manhattan (`LC ≥ MD`); driving IDA* with the incremental LC recovers the exact
//! BFS distance and replays to GOAL.
//!
//! Run with `cargo test --test puzzle24_linear_conflict --features parallel,mmap`.

use std::collections::HashMap;

use puzzle8::puzzle24::search::{
    idastar_inc, Heuristic, LinearConflictHeuristic, LinearConflictInc, ManhattanHeuristic,
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
fn lc_admissible_and_dominates_manhattan() {
    let truth = bfs(9);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let lc = LinearConflictHeuristic.h(&s);
        let md = ManhattanHeuristic.h(&s);
        assert!(lc >= md, "LC {} < MD {} for {:?}", lc, md, raw);
        assert!(lc <= d, "LC {} > truth {} for {:?}", lc, d, raw);
    }
}

#[test]
fn lc_inc_solver_optimal_on_shallow_states() {
    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let sol = idastar_inc(&s, &LinearConflictInc).expect("solvable");
        assert_eq!(sol.len() as u8, d, "LC-inc non-optimal for {:?}", raw);
        assert!(reaches_goal(&s, &sol));
    }
}
