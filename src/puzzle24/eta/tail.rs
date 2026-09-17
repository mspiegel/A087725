//! The tail stratum `V_≥L = { v : d(v, GOAL) ≥ L }`, sampled uniformly.
//!
//! Almost every state lies in the tail, so a uniform draw is accepted unless it
//! is proven closer than `L`. Unlike a `k`-step walk endpoint, a uniform state
//! carries no parity relation to `L`, so the proof must exhaust threshold
//! `L − 1` ([`at_least`]).

use crate::puzzle24::eta::rng::Rng;
use crate::puzzle24::search::{idastar_inc_bounded_with_stats, BoundedOutcome, IncHeuristic};
use crate::puzzle24::state::State;

/// A uniform random solvable state: Fisher–Yates over the 25 cells, then, if
/// the tile permutation is odd, swap the tiles in the first two non-blank
/// cells. For a fixed blank cell that swap pairs odd and even permutations one
/// to one, so the result is uniform over solvable states.
pub fn uniform_solvable(rng: &mut Rng) -> State {
    let mut cells: [u8; 25] = std::array::from_fn(|i| i as u8);
    for i in (1..25).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        cells.swap(i, j);
    }
    let mut s = State(cells);
    if !s.is_solvable() {
        let mut non_blank = (0..25).filter(|&i| s.0[i] != 0);
        let (a, c) = (non_blank.next().unwrap(), non_blank.next().unwrap());
        s.0.swap(a, c);
    }
    s
}

/// `true` iff `s` is proven at least `min_distance` moves from GOAL, by a
/// bounded IDA\* with heuristic `e` exhausting threshold `min_distance − 1`.
pub fn at_least<E: IncHeuristic>(s: &State, min_distance: u8, e: &E) -> bool {
    if min_distance == 0 {
        return true;
    }
    match idastar_inc_bounded_with_stats(s, e, min_distance - 1).0 {
        BoundedOutcome::Solved(_) => false,
        BoundedOutcome::ProvedAtLeast(k) => k >= min_distance,
        other => panic!("bounded search on a solvable state returned {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::IncManhattan;
    use crate::puzzle24::state::{Move, GOAL};

    #[test]
    fn uniform_solvable_states_are_solvable_permutations_with_uniform_blank() {
        let mut rng = Rng::stream(4, 4, 4);
        let n = 100_000;
        let mut blank_counts = [0u64; 25];
        for _ in 0..n {
            let s = uniform_solvable(&mut rng);
            assert!(s.is_solvable());
            let mut seen = [false; 25];
            for &t in &s.0 {
                assert!(!seen[t as usize], "repeated tile");
                seen[t as usize] = true;
            }
            blank_counts[s.blank_pos() as usize] += 1;
        }
        // Chi-square with 24 degrees of freedom; 99.9% critical value 51.2.
        let e = n as f64 / 25.0;
        let chi2: f64 = blank_counts
            .iter()
            .map(|&c| (c as f64 - e).powi(2) / e)
            .sum();
        assert!(chi2 < 51.2, "blank position chi2 {chi2}");
    }

    /// Against true distances, including thresholds of both parities: a board
    /// at distance d is at least m away iff d >= m.
    #[test]
    fn at_least_matches_true_distances_at_every_threshold() {
        const K: u8 = 12;
        let dist = bfs_distances(K);
        let mut rng = Rng::stream(6, 6, 6);
        let mut checked = 0;
        for _ in 0..400 {
            let steps = rng.below(K as u64 + 1) as usize;
            let mut s = GOAL;
            for _ in 0..steps {
                let moves: Vec<Move> = s.legal_moves().iter().collect();
                s = s.apply(moves[rng.below(moves.len() as u64) as usize]);
            }
            let d = dist[&s.0];
            for m in 0..=K {
                assert_eq!(at_least(&s, m, &IncManhattan), d >= m, "d = {d}, m = {m}");
                checked += 1;
            }
        }
        assert!(checked > 0);
    }
}
