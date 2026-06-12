//! Backward BFS in projected space, producing a PDB distance table.
//!
//! The 8-puzzle's projected move graph has edge weights `{0, 1}`: a move
//! that swaps the blank with an anonymous (non-pattern) tile is free
//! (cost 0); a move that swaps the blank with a pattern tile costs 1. We
//! compute, for every projected state, the minimum-cost path back to the
//! projected goal.
//!
//! # Algorithm
//!
//! Depth-by-depth processing. At each integer depth `d`:
//!
//! 1. **0-cost closure** — starting from all projected states already known
//!    to be at depth `d`, repeatedly follow 0-cost edges, marking newly
//!    reached states as also at depth `d`.
//! 2. **1-cost transitions** — from every state at depth `d`, take each
//!    1-cost edge to find new states at depth `d + 1`.
//!
//! Repeat until no new states are discovered. This is equivalent to 0/1
//! BFS but easier to reason about than the deque variant.
//!
//! Determinism: move iteration order is fixed by `MoveSet::iter()` (always
//! `Up, Down, Left, Right`), and frontier processing order is insertion
//! order. Two runs produce byte-identical distance tables.

use super::pattern::{Pattern, ProjectedState};
use crate::state::GOAL;

/// Sentinel used during construction for "not yet visited."
pub const UNVISITED: u8 = u8::MAX;

/// Backward BFS over the projected state space of `pattern`, returning the
/// distance vector indexed by `ProjectedState::rank`.
///
/// The returned vector has length `pattern.num_projected_states()`. Entries
/// for projected states unreachable from the projected goal remain
/// [`UNVISITED`] (a non-issue when querying via [`crate::pdb::PdbHeuristic`]
/// from a solvable full-puzzle state — its projection is always reachable).
pub fn build(pattern: Pattern) -> Vec<u8> {
    let n = pattern.num_projected_states() as usize;
    let mut dist: Vec<u8> = vec![UNVISITED; n];

    let goal_proj = ProjectedState::from_state(&GOAL, pattern);
    let goal_rank = goal_proj.rank(pattern) as usize;
    dist[goal_rank] = 0;

    // `frontier_d` holds all states at the current depth `d`. Each iteration
    // of the outer loop advances `d` by one.
    let mut frontier_d: Vec<ProjectedState> = vec![goal_proj];
    let mut depth: u8 = 0;

    loop {
        // (1) 0-cost closure within depth `d`. Use frontier_d as the seed and
        // expand 0-cost edges until no new states are reachable.
        let mut work: Vec<ProjectedState> = frontier_d.clone();
        while let Some(s) = work.pop() {
            for m in s.legal_moves().iter() {
                let (s_next, cost) = s.apply(m);
                if cost != 0 {
                    continue;
                }
                let r = s_next.rank(pattern) as usize;
                if dist[r] == UNVISITED {
                    dist[r] = depth;
                    work.push(s_next);
                    frontier_d.push(s_next);
                }
            }
        }

        // (2) 1-cost transitions. Take every 1-cost edge from depth-`d`
        // states to find new depth-`d+1` states.
        let next_depth = depth.checked_add(1).expect("depth overflow > 254");
        let mut frontier_next: Vec<ProjectedState> = Vec::new();
        for s in &frontier_d {
            for m in s.legal_moves().iter() {
                let (s_next, cost) = s.apply(m);
                if cost != 1 {
                    continue;
                }
                let r = s_next.rank(pattern) as usize;
                if dist[r] == UNVISITED {
                    dist[r] = next_depth;
                    frontier_next.push(s_next);
                }
            }
        }

        if frontier_next.is_empty() {
            break;
        }
        depth = next_depth;
        frontier_d = frontier_next;
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfs::DistanceTable;
    use crate::pdb::pattern::Pattern;
    use crate::rank::unrank;
    use crate::state::{N_STATES, State};

    #[test]
    fn empty_pattern_distance_is_zero_everywhere_reachable() {
        // With no pattern tiles, every move is 0-cost, so all 9 projected
        // states (one per blank position) are at distance 0.
        let p = Pattern::empty();
        let dist = build(p);
        assert_eq!(dist.len(), 9);
        let goal_proj = ProjectedState::from_state(&crate::state::GOAL, p);
        // The goal's projection is one of them, at distance 0.
        assert_eq!(dist[goal_proj.rank(p) as usize], 0);
        // All reachable projected states should be at distance 0 (blank can
        // reach every cell in the 3x3 grid through 0-cost moves).
        let reachable: u32 = dist.iter().filter(|&&d| d != UNVISITED).count() as u32;
        assert_eq!(reachable, 9, "blank reaches every position via 0-cost moves");
    }

    #[test]
    fn single_tile_pattern_distance_at_goal_is_zero() {
        let p = Pattern::new(&[1]);
        let dist = build(p);
        let goal_proj = ProjectedState::from_state(&crate::state::GOAL, p);
        assert_eq!(dist[goal_proj.rank(p) as usize], 0);
    }

    #[test]
    fn pdb_is_admissible_for_small_pattern() {
        // For a pattern of any size, h_pdb(s) <= dist(s, GOAL).
        let table = DistanceTable::build();
        for tiles in [&[1u8, 2][..], &[1, 2, 3], &[5, 6, 7, 8]] {
            let p = Pattern::new(tiles);
            let pdb = build(p);
            for r in 0..N_STATES {
                let s = unrank(r);
                let proj = ProjectedState::from_state(&s, p);
                let pr = proj.rank(p) as usize;
                let h = pdb[pr];
                assert!(h != UNVISITED, "projection of solvable state {:?} is unreachable", s.0);
                let truth = table.dist(&s);
                assert!(h <= truth, "PDB({:?}) inadmissible for {:?}: h={}, truth={}", tiles, s.0, h, truth);
            }
        }
    }

    #[test]
    fn full_pattern_pdb_equals_full_distance_table_on_reachable() {
        // With all 8 tiles in the pattern, every move costs 1, so the
        // projected distance equals the full puzzle distance — exactly.
        let p = Pattern::new(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let pdb = build(p);
        let table = DistanceTable::build();
        for r in 0..N_STATES {
            let s = unrank(r);
            let proj = ProjectedState::from_state(&s, p);
            let pr = proj.rank(p) as usize;
            assert_eq!(pdb[pr], table.dist(&s), "mismatch at {:?}", s.0);
        }
    }

    #[test]
    fn build_is_deterministic() {
        let p = Pattern::new(&[1, 2, 3, 4]);
        let a = build(p);
        let b = build(p);
        assert_eq!(a, b);
    }

    #[test]
    fn pdb_distance_consistent_with_neighbor_transitions() {
        // For every reachable projected state at depth d, at least one
        // legal move leads to a projected state at depth (d - move_cost).
        let p = Pattern::new(&[1, 2, 3, 4]);
        let dist = build(p);
        // We exhaustively enumerate reachable states by BFS again.
        use std::collections::HashSet;
        let goal_proj = ProjectedState::from_state(&crate::state::GOAL, p);
        let mut visited: HashSet<[u8; 9]> = HashSet::new();
        let mut frontier: Vec<ProjectedState> = vec![goal_proj];
        visited.insert(goal_proj.0);
        while let Some(s) = frontier.pop() {
            let d_s = dist[s.rank(p) as usize];
            assert_ne!(d_s, UNVISITED);
            let mut found_better = false;
            for m in s.legal_moves().iter() {
                let (s_next, cost) = s.apply(m);
                let d_n = dist[s_next.rank(p) as usize];
                if d_n == UNVISITED {
                    continue;
                }
                if visited.insert(s_next.0) {
                    frontier.push(s_next);
                }
                // d_s <= d_n + cost (otherwise the BFS missed an opportunity)
                assert!(d_s <= d_n + cost, "d_s={} > d_n={} + cost={}", d_s, d_n, cost);
                if d_n + cost == d_s {
                    found_better = true;
                }
            }
            if d_s > 0 {
                assert!(found_better, "no neighbor saturates distance at projected state {:?}", s.0);
            }
        }
        // Sanity: we explored more than just the goal.
        assert!(visited.len() > 1);
        // Cross-check a sample full-state projection.
        let table = DistanceTable::build();
        let sample = State([8, 6, 7, 2, 5, 4, 3, 0, 1]); // antipode 1
        let proj = ProjectedState::from_state(&sample, p);
        let h = dist[proj.rank(p) as usize];
        assert!(h <= table.dist(&sample));
    }
}
