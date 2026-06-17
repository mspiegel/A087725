//! Optimal search algorithms over [`crate::puzzle15::state::State`].

pub mod heuristic;
pub mod idastar;
pub mod linear_conflict;
pub mod walking_distance;

pub use heuristic::{Heuristic, ManhattanHeuristic};
pub use idastar::{
    idastar, idastar_inc_mut_with_stats, idastar_inc_with_stats, idastar_with_stats,
    IncHeuristic, IncHeuristicMut, SearchStats,
};
pub use linear_conflict::{LcCtx, LinearConflictHeuristic, LinearConflictInc};
pub use walking_distance::WalkingDistanceHeuristic;

/// Test-only helpers shared across `heuristic` and `idastar` tests.
#[cfg(test)]
pub mod tests_util {
    use crate::puzzle15::state::{State, GOAL, N_CELLS};
    use std::collections::HashMap;

    /// BFS forward from [`GOAL`] to `depth_limit`, returning a map from every
    /// reached state to its true distance.
    ///
    /// Use only for shallow depths (≤ ~12) in tests — frontier grows roughly
    /// exponentially.
    pub fn bfs_distances(depth_limit: u8) -> HashMap<[u8; N_CELLS], u8> {
        let mut dist: HashMap<[u8; N_CELLS], u8> = HashMap::new();
        dist.insert(GOAL.0, 0);
        let mut frontier: Vec<State> = vec![GOAL];
        let mut depth: u8 = 0;
        while !frontier.is_empty() && depth < depth_limit {
            let next_depth = depth + 1;
            let mut next: Vec<State> = Vec::with_capacity(frontier.len() * 3);
            for s in &frontier {
                for m in s.legal_moves().iter() {
                    let s_next = s.apply(m);
                    if !dist.contains_key(&s_next.0) {
                        dist.insert(s_next.0, next_depth);
                        next.push(s_next);
                    }
                }
            }
            depth = next_depth;
            frontier = next;
        }
        dist
    }
}
