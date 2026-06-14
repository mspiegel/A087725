//! Admissibility of Linear Conflict on the 15-puzzle over a shallow BFS truth
//! table. The full state space (10.46 × 10¹²) is too large for an exhaustive
//! sweep — BFS to depth 10 reaches ~30k states, enough to catch
//! implementation bugs while keeping the test fast.

use puzzle8::puzzle15::search::{Heuristic, LinearConflictHeuristic, ManhattanHeuristic};
use puzzle8::puzzle15::state::{State, GOAL, N_CELLS};

use std::collections::{HashMap, VecDeque};

fn bfs_distances(depth_limit: u8) -> HashMap<[u8; N_CELLS], u8> {
    let mut dist: HashMap<[u8; N_CELLS], u8> = HashMap::new();
    dist.insert(GOAL.0, 0);
    let mut frontier: VecDeque<State> = VecDeque::new();
    frontier.push_back(GOAL);
    while let Some(s) = frontier.pop_front() {
        let d = dist[&s.0];
        if d >= depth_limit {
            continue;
        }
        for m in s.legal_moves().iter() {
            let n = s.apply(m);
            if !dist.contains_key(&n.0) {
                dist.insert(n.0, d + 1);
                frontier.push_back(n);
            }
        }
    }
    dist
}

#[test]
fn lc_is_admissible_and_dominates_manhattan_on_shallow_bfs() {
    let truth = bfs_distances(10);
    let lc = LinearConflictHeuristic;
    let md = ManhattanHeuristic;
    for (raw, &true_dist) in &truth {
        let est_lc = lc.h(&State(*raw));
        let est_md = md.h(&State(*raw));
        assert!(
            est_lc <= true_dist,
            "LC({:?}) = {} > truth = {}",
            raw,
            est_lc,
            true_dist
        );
        assert!(
            est_lc >= est_md,
            "LC({:?}) = {} < MD = {} (LC should never loosen MD)",
            raw,
            est_lc,
            est_md
        );
    }
}
