//! Admissibility of Walking Distance on the 15-puzzle over a shallow BFS
//! truth table.

use puzzle8::puzzle15::search::{Heuristic, WalkingDistanceHeuristic};
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
fn wd_is_admissible_on_shallow_bfs() {
    WalkingDistanceHeuristic::warm_up();
    let truth = bfs_distances(10);
    let h = WalkingDistanceHeuristic;
    for (raw, &true_dist) in &truth {
        let est = h.h(&State(*raw));
        assert!(
            est <= true_dist,
            "WD({:?}) = {} > truth = {}",
            raw,
            est,
            true_dist
        );
    }
}
