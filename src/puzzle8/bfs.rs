//! Backward breadth-first search from [`GOAL`] producing the full distance
//! table.
//!
//! Because the 8-puzzle's move graph is undirected (every move has an inverse,
//! see [`Move::inverse`]), a BFS forward from `GOAL` visits every state at its
//! optimal distance from `GOAL`. We therefore call this "backward" search by
//! convention — the resulting table maps each state to its distance *to* the
//! goal, which is what the solver consumes.

use crate::puzzle8::rank::rank;
use crate::puzzle8::state::{State, GOAL, N_STATES};

/// Sentinel for states whose distance has not been written. Since the diameter
/// is 31 < 255, any value > 31 is "not visited."
pub const UNVISITED: u8 = u8::MAX;

/// Dense distance table indexed by [`rank`].
#[derive(Clone)]
pub struct DistanceTable {
    data: Vec<u8>,
}

impl DistanceTable {
    /// Allocate an empty table (all entries [`UNVISITED`]). Mostly useful for
    /// tests.
    pub fn new_empty() -> Self {
        Self {
            data: vec![UNVISITED; N_STATES as usize],
        }
    }

    /// Build the full distance table by backward BFS from [`GOAL`].
    pub fn build() -> Self {
        let mut table = Self::new_empty();
        table.data[rank(&GOAL) as usize] = 0;

        let mut frontier: Vec<State> = Vec::with_capacity(4);
        frontier.push(GOAL);
        let mut depth: u8 = 0;

        while !frontier.is_empty() {
            let mut next: Vec<State> = Vec::with_capacity(frontier.len() * 2);
            let next_depth = depth + 1;
            for s in &frontier {
                for m in s.legal_moves().iter() {
                    let s_next = s.apply(m);
                    let idx = rank(&s_next) as usize;
                    if table.data[idx] == UNVISITED {
                        table.data[idx] = next_depth;
                        next.push(s_next);
                    }
                }
            }
            depth = next_depth;
            frontier = next;
        }

        table
    }

    /// Distance from `s` to [`GOAL`].
    pub fn dist(&self, s: &State) -> u8 {
        self.data[rank(s) as usize]
    }

    /// Distance for a raw rank value.
    pub fn dist_of_rank(&self, r: u32) -> u8 {
        self.data[r as usize]
    }

    /// Read-only view of the raw bytes (length [`N_STATES`]).
    pub fn raw(&self) -> &[u8] {
        &self.data
    }

    /// Mutable raw view, for loading from disk.
    pub(crate) fn raw_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Number of visited (reachable) states. Should equal [`N_STATES`] after a
    /// successful [`build`](Self::build).
    pub fn visited_count(&self) -> u32 {
        self.data.iter().filter(|&&d| d != UNVISITED).count() as u32
    }

    /// Diameter: the maximum optimal distance over all visited states.
    pub fn diameter(&self) -> u8 {
        self.data
            .iter()
            .copied()
            .filter(|&d| d != UNVISITED)
            .max()
            .unwrap_or(0)
    }

    /// All states at distance [`diameter`](Self::diameter) from [`GOAL`].
    pub fn antipodes(&self) -> Vec<State> {
        let d_max = self.diameter();
        let mut out = Vec::new();
        for r in 0..N_STATES {
            if self.data[r as usize] == d_max {
                out.push(crate::puzzle8::rank::unrank(r));
            }
        }
        out
    }

    /// Histogram of optimal distances: `histogram()[d]` = number of states at
    /// distance `d`. Length is `diameter() + 1`.
    pub fn histogram(&self) -> Vec<u32> {
        let d_max = self.diameter();
        let mut hist = vec![0u32; (d_max as usize) + 1];
        for &d in &self.data {
            if d != UNVISITED {
                hist[d as usize] += 1;
            }
        }
        hist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::state::DIAMETER;

    fn build_once() -> DistanceTable {
        DistanceTable::build()
    }

    #[test]
    fn bfs_visits_every_solvable_state() {
        let t = build_once();
        assert_eq!(t.visited_count(), N_STATES);
    }

    #[test]
    fn diameter_matches_published() {
        let t = build_once();
        assert_eq!(t.diameter(), DIAMETER);
    }

    #[test]
    fn goal_distance_is_zero() {
        let t = build_once();
        assert_eq!(t.dist(&GOAL), 0);
    }

    #[test]
    fn exactly_two_antipodes() {
        let t = build_once();
        let a = t.antipodes();
        assert_eq!(a.len(), 2, "expected 2 antipodes, got {}: {:?}", a.len(), a);
    }

    #[test]
    fn known_antipodes_have_max_distance() {
        let t = build_once();
        // From Reinefeld 1993 / OEIS A087725 commentary:
        //   8 6 7        6 4 7
        //   2 5 4   and  8 5 .
        //   3 . 1        3 2 1
        let a1 = State([8, 6, 7, 2, 5, 4, 3, 0, 1]);
        let a2 = State([6, 4, 7, 8, 5, 0, 3, 2, 1]);
        assert!(a1.is_solvable());
        assert!(a2.is_solvable());
        assert_eq!(t.dist(&a1), DIAMETER, "a1 should be at distance {DIAMETER}");
        assert_eq!(t.dist(&a2), DIAMETER, "a2 should be at distance {DIAMETER}");

        let antipodes = t.antipodes();
        let arr: Vec<[u8; 9]> = antipodes.iter().map(|s| s.0).collect();
        assert!(arr.contains(&a1.0), "a1 not in antipode set: {arr:?}");
        assert!(arr.contains(&a2.0), "a2 not in antipode set: {arr:?}");
    }

    #[test]
    fn histogram_sums_to_n_states() {
        let t = build_once();
        let h = t.histogram();
        assert_eq!(h.len(), DIAMETER as usize + 1);
        let total: u32 = h.iter().sum();
        assert_eq!(total, N_STATES);
        // Single state at depth 0 (the goal).
        assert_eq!(h[0], 1);
        // Exactly 2 antipodes at depth 31.
        assert_eq!(h[DIAMETER as usize], 2);
    }

    #[test]
    fn distance_is_consistent_with_neighbor_distances() {
        // For every state s, dist(s) must equal 1 + min(dist(neighbor)).
        let t = build_once();
        for r in 0..N_STATES {
            let s = crate::puzzle8::rank::unrank(r);
            let d = t.dist(&s);
            if s == GOAL {
                continue;
            }
            let min_neighbor = s
                .legal_moves()
                .iter()
                .map(|m| t.dist(&s.apply(m)))
                .min()
                .unwrap();
            assert_eq!(
                d,
                min_neighbor + 1,
                "dist({:?}) = {} but min neighbor dist = {}",
                s.0,
                d,
                min_neighbor
            );
        }
    }
}
