//! Single-threaded 0/1 BFS in projected space, producing a 15-puzzle PDB.
//!
//! The 15-puzzle's projected move graph has edge weights `{0, 1}`: a move
//! that swaps the blank with an anonymous (non-pattern) tile is free
//! (cost 0); a move that swaps the blank with a pattern tile costs 1. We
//! compute, for every pattern-tile configuration, the minimum-cost path back
//! to the projected goal — taken over all possible blank positions, per the
//! no-blank PDB convention.
//!
//! # Algorithm
//!
//! Depth-by-depth processing. At each integer depth `d`:
//!
//! 1. **0-cost closure** — starting from all (pattern_pos, blank_pos)
//!    projected states already known to be at depth `d`, repeatedly follow
//!    0-cost edges, marking newly reached BFS-states as also at depth `d`.
//!    Each newly reached state's pattern_rank lower-bounds the PDB output.
//! 2. **1-cost transitions** — from every depth-`d` BFS-state, take each
//!    1-cost edge to find new depth-`(d + 1)` BFS-states.
//!
//! Repeat until no new BFS-states are discovered.
//!
//! Two arrays:
//!
//! - `bfs_dist: Vec<u8>` of length `P(16, k+1)`, indexed by
//!   [`ProjectedState::rank_with_blank`]. Tracks BFS visits.
//! - `pdb: Vec<u8>` of length `P(16, k)`, indexed by [`ProjectedState::rank`].
//!   Output PDB; stores the minimum depth at which any
//!   `(pattern_pos, *)` was visited.
//!
//! Since BFS visits depths in increasing order, the *first* time any
//! `(pattern_pos, *)` is discovered fixes `pdb[pattern_rank]` to the correct
//! minimum.
//!
//! Determinism: move iteration order is fixed by [`MoveSet::iter`], frontier
//! order is insertion order. Two runs produce byte-identical output.
//!
//! [`ProjectedState::rank`]: super::pattern::ProjectedState::rank
//! [`ProjectedState::rank_with_blank`]: super::pattern::ProjectedState::rank_with_blank
//! [`MoveSet::iter`]: crate::puzzle15::state::MoveSet::iter

use super::pattern::{Pattern, ProjectedState};

/// Sentinel used during construction for "not yet visited."
pub const UNVISITED: u8 = u8::MAX;

/// 0/1 BFS over the projected state space of `pattern`, returning the PDB
/// distance vector indexed by [`ProjectedState::rank`].
///
/// The returned vector has length `pattern.num_projected_states()`. Entries
/// for pattern-tile configurations unreachable from the goal projection remain
/// [`UNVISITED`].
///
/// # Memory
///
/// Allocates a transient `Vec<u8>` of length `pattern.num_bfs_states()` for
/// BFS bookkeeping. For Korf P7 that's ~519 MB; for Korf P8, ~4.15 GB. The
/// transient is dropped before return.
///
/// [`ProjectedState::rank`]: super::pattern::ProjectedState::rank
pub fn build(pattern: Pattern) -> Vec<u8> {
    let pdb_size = pattern.num_projected_states() as usize;
    let bfs_size = pattern.num_bfs_states() as usize;
    let mut pdb: Vec<u8> = vec![UNVISITED; pdb_size];
    let mut bfs_dist: Vec<u8> = vec![UNVISITED; bfs_size];

    let goal_proj = ProjectedState::goal(pattern);
    let goal_bfs_rank = goal_proj.rank_with_blank(pattern) as usize;
    let goal_pdb_rank = goal_proj.rank(pattern) as usize;
    bfs_dist[goal_bfs_rank] = 0;
    pdb[goal_pdb_rank] = 0;

    let mut frontier_d: Vec<ProjectedState> = vec![goal_proj];
    let mut depth: u8 = 0;

    loop {
        // (1) 0-cost closure within depth `d`.
        let mut work: Vec<ProjectedState> = frontier_d.clone();
        while let Some(s) = work.pop() {
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
                    work.push(s_next);
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
    use crate::puzzle15::pdb::pattern::Pattern;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::state::State;

    #[test]
    fn empty_pattern_distance_is_zero_everywhere() {
        // Single-entry PDB. The only "pattern configuration" trivially has
        // zero pattern tiles to move home → distance 0.
        let pdb = build(Pattern::empty());
        assert_eq!(pdb.len(), 1);
        assert_eq!(pdb[0], 0);
    }

    #[test]
    fn single_tile_pattern_admissible_against_shallow_bfs() {
        // For each single-tile pattern {k}, the PDB entry must be ≤ the true
        // distance for any state with that tile configured at any position.
        // Verify against the full-puzzle BFS to depth 8.
        let truth = bfs_distances(8);
        for k in 1u8..=15 {
            let p = Pattern::new(&[k]);
            let pdb = build(p);
            // For each reached state, get its pattern-only rank and compare.
            for (raw, &true_dist) in &truth {
                let proj = crate::puzzle15::pdb::pattern::ProjectedState::from_state(
                    &State(*raw),
                    p,
                );
                let h = pdb[proj.rank(p) as usize];
                assert_ne!(h, UNVISITED, "PDB has gap for {:?} pattern {}", raw, k);
                assert!(
                    h <= true_dist,
                    "PDB({}) = {} > true {} for state {:?}",
                    k, h, true_dist, raw
                );
            }
        }
    }

    #[test]
    fn pdb_at_goal_pattern_is_zero() {
        // For any pattern, the goal projection's PDB entry must be 0.
        for tiles in [
            &[1u8][..],
            &[1, 5, 10],
            &[2, 3, 4, 5],
        ] {
            let p = Pattern::new(tiles);
            let pdb = build(p);
            let goal_proj = ProjectedState::goal(p);
            let r = goal_proj.rank(p) as usize;
            assert_eq!(pdb[r], 0, "goal projection should have distance 0 for {:?}", tiles);
        }
    }

    #[test]
    fn pdb_is_deterministic_across_runs() {
        // Build the same small pattern twice; outputs must be byte-identical.
        let p = Pattern::new(&[1, 2, 3]);
        let a = build(p);
        let b = build(p);
        assert_eq!(a, b, "PDB build is not deterministic");
    }

    #[test]
    fn no_gaps_on_small_pattern() {
        // For a small pattern, every (pattern_pos) reachable from goal must
        // be filled. Since the projected graph is connected (the blank can
        // reach any cell), all P(16, k) pattern configurations are reachable
        // — so no UNVISITED entries.
        let p = Pattern::new(&[1, 2]);
        let pdb = build(p);
        let unvisited = pdb.iter().filter(|&&d| d == UNVISITED).count();
        assert_eq!(unvisited, 0, "PDB has {} gaps for pattern {{1,2}}", unvisited);
    }

    #[test]
    fn pdb_size_matches_pattern_capacity() {
        let p = Pattern::new(&[1, 2, 3, 4]);
        let pdb = build(p);
        assert_eq!(pdb.len() as u64, p.num_projected_states());
    }
}

#[cfg(feature = "parallel")]
mod parallel {
    //! Multi-threaded 0/1 BFS in projected space.
    //!
    //! Layer-synchronous parallel BFS:
    //! - **0-cost closure** is single-threaded within a layer (small relative
    //!   to the 1-cost work; gains little from threading).
    //! - **1-cost transitions** fan out across rayon: each thread takes a
    //!   chunk of the current frontier, expands it, and emits thread-local
    //!   newly-discovered states. The merge into the next frontier happens
    //!   at the chunk boundary.
    //!
    //! Race-condition handling: both the BFS-visited array (sized
    //! `num_bfs_states`) and the output PDB (sized `num_projected_states`)
    //! are `Vec<AtomicU8>`. We use `fetch_min(d, Relaxed)` for both. Only the
    //! thread whose `fetch_min` flipped a BFS cell from `UNVISITED` (= `0xFF`)
    //! to a real depth pushes that state into the next frontier — this
    //! prevents duplicate frontier entries without a deduplication pass.
    //!
    //! Determinism: thread interleaving can vary, but the *value* written at
    //! each cell is layer-determined (the min depth) — so the PDB output is
    //! byte-identical across runs, matching the sequential
    //! [`build`](super::build) output exactly.

    use super::{Pattern, ProjectedState, UNVISITED};
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    const CHUNK: usize = 4096;

    /// Multi-threaded 0/1 BFS. Returns a `Vec<u8>` byte-identical to
    /// [`build`](super::build)'s output for the same pattern.
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
            // (1) 0-cost closure within depth `d` — single-threaded.
            let mut work = frontier_d.clone();
            while let Some(s) = work.pop() {
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
                        work.push(s_next);
                        frontier_d.push(s_next);
                    }
                }
            }

            // (2) 1-cost transitions to depth+1 — PARALLEL.
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

        // Convert AtomicU8 → u8. AtomicU8 is repr(transparent) over u8 — the
        // byte layout is identical — but we use into_inner() for clarity
        // rather than relying on a transmute.
        pdb.into_iter().map(|a| a.into_inner()).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::puzzle15::pdb::pattern::Pattern;

        /// The critical gate: parallel build must produce byte-identical
        /// output to the sequential build.
        #[test]
        fn parallel_matches_sequential_k3() {
            let p = Pattern::new(&[1, 2, 3]);
            let seq = super::super::build(p);
            let par = build_parallel(p);
            assert_eq!(seq, par);
        }

        #[test]
        fn parallel_matches_sequential_k4() {
            let p = Pattern::new(&[1, 4, 7, 10]);
            let seq = super::super::build(p);
            let par = build_parallel(p);
            assert_eq!(seq, par);
        }

        #[test]
        fn parallel_matches_sequential_k5() {
            let p = Pattern::new(&[2, 5, 8, 11, 14]);
            let seq = super::super::build(p);
            let par = build_parallel(p);
            assert_eq!(seq, par);
        }

        #[test]
        fn parallel_pdb_at_goal_is_zero() {
            let p = Pattern::new(&[1, 2, 3, 4]);
            let pdb = build_parallel(p);
            let goal_proj = ProjectedState::goal(p);
            assert_eq!(pdb[goal_proj.rank(p) as usize], 0);
        }

        #[test]
        fn parallel_build_is_deterministic_across_runs() {
            let p = Pattern::new(&[1, 2, 3, 4]);
            let a = build_parallel(p);
            let b = build_parallel(p);
            assert_eq!(a, b);
        }
    }
}

#[cfg(feature = "parallel")]
pub use parallel::build_parallel;
