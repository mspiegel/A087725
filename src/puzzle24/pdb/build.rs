//! 0/1 BFS in projected space, producing a 24-puzzle additive PDB.
//!
//! The projected move graph has edge weights `{0, 1}`: a move swapping the
//! blank with a non-pattern (anon) tile is free; swapping with a pattern tile
//! costs 1. We compute, for every pattern-tile configuration, the minimum-cost
//! path back to the projected goal over all blank positions (the no-blank PDB
//! convention).
//!
//! Two arrays: `bfs_dist` of length `P(25, k+1)` keyed by `rank_with_blank`
//! (BFS-visited tracker), and `pdb` of length `P(25, k)` keyed by `rank` (the
//! output). Depth-by-depth: at depth `d`, take the 0-cost closure, then 1-cost
//! transitions to depth `d+1`. Move order is fixed, so two runs produce
//! byte-identical output.

use super::pattern::{Pattern, ProjectedState};

/// Sentinel used during construction for "not yet visited."
pub const UNVISITED: u8 = u8::MAX;

/// 0/1 BFS over the projected state space of `pattern`. Returns the PDB distance
/// vector indexed by [`ProjectedState::rank`], length `num_projected_states`.
pub fn build(pattern: Pattern) -> Vec<u8> {
    let pdb_size = pattern.num_projected_states() as usize;
    let bfs_size = pattern.num_bfs_states() as usize;
    let mut pdb: Vec<u8> = vec![UNVISITED; pdb_size];
    let mut bfs_dist: Vec<u8> = vec![UNVISITED; bfs_size];

    let goal_proj = ProjectedState::goal(pattern);
    bfs_dist[goal_proj.rank_with_blank(pattern) as usize] = 0;
    pdb[goal_proj.rank(pattern) as usize] = 0;

    let mut frontier_d: Vec<ProjectedState> = vec![goal_proj];
    let mut depth: u8 = 0;

    loop {
        // (1) 0-cost closure within depth `d`. Process `frontier_d` in place by
        // index, appending newly-found states — no `work` clone (OPTIMIZATION.md
        // #9). All closure states share `depth`, so traversal order does not
        // affect the output arrays.
        let mut i = 0;
        while i < frontier_d.len() {
            let s = frontier_d[i];
            i += 1;
            for m in s.legal_moves().iter() {
                let (s_next, cost) = s.apply(m);
                if cost != 0 {
                    continue;
                }
                let bfs_r = s_next.rank_with_blank(pattern) as usize;
                if bfs_dist[bfs_r] == UNVISITED {
                    bfs_dist[bfs_r] = depth;
                    let pdb_r = s_next.rank(pattern) as usize;
                    if pdb[pdb_r] == UNVISITED {
                        pdb[pdb_r] = depth;
                    }
                    frontier_d.push(s_next);
                }
            }
        }

        // (2) 1-cost transitions to depth d+1.
        let next_depth = depth.checked_add(1).expect("PDB depth overflowed u8");
        let mut frontier_next: Vec<ProjectedState> = Vec::new();
        for s in &frontier_d {
            for m in s.legal_moves().iter() {
                let (s_next, cost) = s.apply(m);
                if cost != 1 {
                    continue;
                }
                let bfs_r = s_next.rank_with_blank(pattern) as usize;
                if bfs_dist[bfs_r] == UNVISITED {
                    bfs_dist[bfs_r] = next_depth;
                    let pdb_r = s_next.rank(pattern) as usize;
                    if pdb[pdb_r] == UNVISITED {
                        pdb[pdb_r] = next_depth;
                    }
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

    pdb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::pdb::pattern::Pattern;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::state::State;

    #[test]
    fn empty_pattern_distance_is_zero() {
        let pdb = build(Pattern::empty());
        assert_eq!(pdb.len(), 1);
        assert_eq!(pdb[0], 0);
    }

    #[test]
    fn pdb_at_goal_is_zero() {
        for tiles in [&[1u8][..], &[1, 7, 13], &[2, 3, 4, 5]] {
            let p = Pattern::new(tiles);
            let pdb = build(p);
            assert_eq!(pdb[ProjectedState::goal(p).rank(p) as usize], 0);
        }
    }

    #[test]
    fn small_pattern_admissible_against_shallow_bfs() {
        // For each single-tile pattern, the PDB value must lower-bound the true
        // full-puzzle distance for any reached state (verified to depth 7).
        let truth = bfs_distances(7);
        for k in [1u8, 7, 13, 24] {
            let p = Pattern::new(&[k]);
            let pdb = build(p);
            for (raw, &true_dist) in &truth {
                let proj = ProjectedState::from_state(&State(*raw), p);
                let h = pdb[proj.rank(p) as usize];
                assert_ne!(h, UNVISITED, "gap for tile {k} at {raw:?}");
                assert!(h <= true_dist, "PDB({k})={h} > true {true_dist}");
            }
        }
    }

    #[test]
    fn two_tile_pattern_admissible() {
        let truth = bfs_distances(7);
        let p = Pattern::new(&[1, 2]);
        let pdb = build(p);
        for (raw, &true_dist) in &truth {
            let proj = ProjectedState::from_state(&State(*raw), p);
            let h = pdb[proj.rank(p) as usize];
            assert!(h <= true_dist, "h {h} > true {true_dist}");
        }
    }

    #[test]
    fn no_gaps_on_small_pattern() {
        // The projected graph is connected, so every P(25,k) config is reached.
        let p = Pattern::new(&[1, 2]);
        let pdb = build(p);
        assert_eq!(pdb.iter().filter(|&&d| d == UNVISITED).count(), 0);
    }

    #[test]
    fn build_is_deterministic() {
        let p = Pattern::new(&[1, 2, 3]);
        assert_eq!(build(p), build(p));
    }

    #[test]
    fn pdb_size_matches_capacity() {
        let p = Pattern::new(&[1, 2, 3]);
        assert_eq!(build(p).len() as u64, p.num_projected_states());
    }
}

#[cfg(feature = "parallel")]
pub use parallel::build_parallel;

#[cfg(feature = "parallel")]
mod parallel {
    //! Multi-threaded 0/1 BFS: 0-cost closure single-threaded per layer, 1-cost
    //! transitions fanned across rayon. Output is byte-identical to [`build`].

    use super::{Pattern, ProjectedState, UNVISITED};
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    const CHUNK: usize = 4096;

    /// Multi-threaded 0/1 BFS. Byte-identical to [`super::build`].
    pub fn build_parallel(pattern: Pattern) -> Vec<u8> {
        let pdb_size = pattern.num_projected_states() as usize;
        let bfs_size = pattern.num_bfs_states() as usize;
        let bfs_dist: Vec<AtomicU8> = (0..bfs_size).map(|_| AtomicU8::new(UNVISITED)).collect();
        let pdb: Vec<AtomicU8> = (0..pdb_size).map(|_| AtomicU8::new(UNVISITED)).collect();

        let goal_proj = ProjectedState::goal(pattern);
        bfs_dist[goal_proj.rank_with_blank(pattern) as usize].store(0, Ordering::Relaxed);
        pdb[goal_proj.rank(pattern) as usize].store(0, Ordering::Relaxed);

        let mut frontier_d: Vec<ProjectedState> = vec![goal_proj];
        let mut depth: u8 = 0;

        loop {
            // (1) 0-cost closure within depth `d` — single-threaded, in place by
            // index (OPTIMIZATION.md #9: no `work` clone).
            let mut i = 0;
            while i < frontier_d.len() {
                let s = frontier_d[i];
                i += 1;
                for m in s.legal_moves().iter() {
                    let (s_next, cost) = s.apply(m);
                    if cost != 0 {
                        continue;
                    }
                    let bfs_r = s_next.rank_with_blank(pattern) as usize;
                    let prev = bfs_dist[bfs_r].fetch_min(depth, Ordering::Relaxed);
                    if prev == UNVISITED {
                        let pdb_r = s_next.rank(pattern) as usize;
                        pdb[pdb_r].fetch_min(depth, Ordering::Relaxed);
                        frontier_d.push(s_next);
                    }
                }
            }

            // (2) 1-cost transitions to depth+1 — parallel.
            let next_depth = depth.checked_add(1).expect("PDB depth overflowed u8");
            let frontier_next: Vec<ProjectedState> = frontier_d
                .par_chunks(CHUNK)
                .flat_map_iter(|chunk| {
                    let mut local_new: Vec<ProjectedState> = Vec::new();
                    for s in chunk {
                        for m in s.legal_moves().iter() {
                            let (s_next, cost) = s.apply(m);
                            if cost != 1 {
                                continue;
                            }
                            let bfs_r = s_next.rank_with_blank(pattern) as usize;
                            let prev = bfs_dist[bfs_r].fetch_min(next_depth, Ordering::Relaxed);
                            if prev == UNVISITED {
                                let pdb_r = s_next.rank(pattern) as usize;
                                pdb[pdb_r].fetch_min(next_depth, Ordering::Relaxed);
                                local_new.push(s_next);
                            }
                        }
                    }
                    local_new.into_iter()
                })
                .collect();

            if frontier_next.is_empty() {
                break;
            }
            depth = next_depth;
            frontier_d = frontier_next;
        }

        pdb.into_iter().map(|a| a.into_inner()).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parallel_matches_sequential_k2() {
            let p = Pattern::new(&[1, 2]);
            assert_eq!(super::super::build(p), build_parallel(p));
        }

        #[test]
        fn parallel_matches_sequential_k3() {
            let p = Pattern::new(&[5, 12, 20]);
            assert_eq!(super::super::build(p), build_parallel(p));
        }

        #[test]
        fn parallel_is_deterministic() {
            let p = Pattern::new(&[1, 2, 3]);
            assert_eq!(build_parallel(p), build_parallel(p));
        }
    }
}
