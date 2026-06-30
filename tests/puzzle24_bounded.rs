//! Integration test for the 24-puzzle bounded / lower-bound IDA* mode.
//!
//! Against exhaustive shallow BFS: a bound equal to the true depth must recover
//! the optimal solution (and replay to GOAL); a bound one below it must prove
//! `depth >= truth` exactly; and a bound below the root heuristic proves
//! `depth >= h0` with no node expansion.
//!
//! Run with `cargo test --test puzzle24_bounded --features parallel,mmap`.

use std::collections::HashMap;

use puzzle8::puzzle24::search::{
    idastar_inc_bounded_with_stats, BoundedOutcome, IncManhattan, ManhattanHeuristic,
};
use puzzle8::puzzle24::search::Heuristic;
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
fn bound_equal_to_depth_solves_optimally() {
    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let (outcome, _) = idastar_inc_bounded_with_stats(&s, &IncManhattan, d);
        match outcome {
            BoundedOutcome::Solved(sol) => {
                assert_eq!(sol.len() as u8, d, "non-optimal for {:?}", raw);
                assert!(reaches_goal(&s, &sol), "path doesn't reach GOAL for {:?}", raw);
            }
            other => panic!("expected Solved for {:?}, got {:?}", raw, other),
        }
    }
}

#[test]
fn bound_below_depth_proves_lower_bound() {
    let truth = bfs(8);
    for (raw, &d) in &truth {
        if d == 0 {
            continue;
        }
        let s = State(*raw);
        let (outcome, _) = idastar_inc_bounded_with_stats(&s, &IncManhattan, d - 1);
        assert_eq!(
            outcome,
            BoundedOutcome::ProvedAtLeast(d),
            "expected ProvedAtLeast({}) for {:?}",
            d,
            raw
        );
    }
}

#[test]
fn bound_below_root_h_proves_without_search() {
    let truth = bfs(8);
    for (raw, _) in &truth {
        let s = State(*raw);
        let h0 = ManhattanHeuristic.h(&s);
        if h0 < 2 {
            continue;
        }
        let (outcome, stats) = idastar_inc_bounded_with_stats(&s, &IncManhattan, h0 - 1);
        assert_eq!(outcome, BoundedOutcome::ProvedAtLeast(h0), "for {:?}", raw);
        assert_eq!(stats.nodes, 0, "no nodes should expand below root h for {:?}", raw);
    }
}
