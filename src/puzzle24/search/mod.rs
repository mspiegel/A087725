//! Optimal search algorithms over [`crate::puzzle24::state::State`].

pub mod cwd;
pub mod cwd_lm;
pub mod cwd_lm1l;
pub mod cwd_lm_joint;
pub mod flat;
// Only the `cwd-table-tests` gate consumes the frozen oracle.
#[cfg(all(test, feature = "cwd-table-tests"))]
mod flat_oracle;
pub mod heuristic;
pub mod idastar;
pub mod linear_conflict;
pub mod move_dfa;
pub mod outcome;
pub mod walking_distance;

pub use cwd::{load_cwd_overlay, Cwd, CwdOverlay, CwdScratch};
pub use heuristic::{Heuristic, IncManhattan, ManhattanHeuristic, MaxInc};
pub use idastar::{
    idastar, idastar_inc, idastar_inc_bounded_telemetry, idastar_inc_bounded_with_stats,
    idastar_inc_ladder, idastar_inc_with_stats, idastar_with_stats, IncHeuristic, LadderOutcome,
};
pub use linear_conflict::{LcCtx, LinearConflictHeuristic, LinearConflictInc};
pub use move_dfa::{MoveDfa, MovePruner, NullPruner};
pub use outcome::{BoundedOutcome, SearchStats};
pub use walking_distance::{
    build_full_table, load_dist_table, save_dist_table, WalkingDistanceHeuristic,
    WalkingDistanceInc, WalkingDistanceTo, WdBuild, WdCtx, WdLoadError, WdTable, WdTableSource,
    FULL_WD_ENTRIES, WD_KIND_FULL, WD_TABLE_MAGIC, WD_TABLE_VERSION,
};

/// Test-only helpers shared across the search tests.
#[cfg(test)]
pub mod tests_util {
    use crate::puzzle24::state::{State, GOAL, N_CELLS};
    use std::collections::HashMap;

    /// BFS forward from [`GOAL`] to `depth_limit`, returning a map from every
    /// reached state to its true distance. Use only for shallow depths in
    /// tests — the 24-puzzle frontier grows ~3× per layer.
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
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(s_next.0) {
                        e.insert(next_depth);
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
