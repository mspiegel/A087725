//! Sphere-stratum attempts: walk `k` steps from GOAL, then decide whether the
//! end board lies at distance exactly `k`.
//!
//! The walk shows `d(v) ≤ k`, and `d(v) ≡ k (mod 2)` because every move changes
//! the blank's colour. So `d(v) = k` iff no solution of length `≤ k − 2`
//! exists, which one bounded IDA\* run with threshold cap `k − 2` decides
//! ([`reject_shorter`]). Rejected walks dominate at large `k`, and this search
//! is roughly `b²` cheaper than a full threshold-`k` iteration.

use crate::puzzle24::eta::rng::Rng;
use crate::puzzle24::eta::walker::Walker;
use crate::puzzle24::search::{idastar_inc_bounded_with_stats, BoundedOutcome, IncHeuristic};
use crate::puzzle24::state::{Move, State};

/// Result of a single sphere attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attempt {
    /// The walk dead-ended before `k` steps.
    DeadEnd,
    /// The walk ended at a board closer than `k`.
    Rejected,
    /// The walk ended at a board at distance exactly `k`; `walk_prob` is the
    /// probability of the specific path taken (a lower bound on `P(v)`).
    Accepted { board: State, walk_prob: f64 },
}

/// `true` iff `v` (reached by a `k`-step walk) has a solution shorter than `k`.
/// Also returns the search's node count.
pub fn reject_shorter<E: IncHeuristic>(v: &State, k: u32, e: &E) -> (bool, u64) {
    if k < 2 {
        return (false, 0);
    }
    let cap = u8::try_from(k - 2).expect("sphere depth exceeds u8 search bounds");
    let (outcome, stats) = idastar_inc_bounded_with_stats(v, e, cap);
    match outcome {
        BoundedOutcome::Solved(_) => (true, stats.nodes),
        BoundedOutcome::ProvedAtLeast(at_least) => {
            debug_assert!(at_least as u32 >= k, "parity: {at_least} < {k}");
            (false, stats.nodes)
        }
        other => panic!("bounded search on a walk endpoint returned {other:?}"),
    }
}

/// Walk `k` steps with `rng` and classify the end board with [`reject_shorter`]
/// under heuristic `e`. `path` receives the walk's moves. Returns the attempt
/// and the verification node count.
pub fn attempt<E: IncHeuristic>(
    walker: &Walker,
    k: u32,
    rng: &mut Rng,
    e: &E,
    path: &mut Vec<Move>,
) -> (Attempt, u64) {
    attempt_with(walker, k, rng, path, |v, k| reject_shorter(v, k, e))
}

/// [`attempt`] with a caller-supplied rejection test `reject(v, k) ->
/// (has a solution shorter than k, nodes)`, for verifiers outside the
/// [`IncHeuristic`] family.
pub fn attempt_with<F>(
    walker: &Walker,
    k: u32,
    rng: &mut Rng,
    path: &mut Vec<Move>,
    mut reject: F,
) -> (Attempt, u64)
where
    F: FnMut(&State, u32) -> (bool, u64),
{
    let Some((board, walk_prob)) = walker.walk(k, rng, path) else {
        return (Attempt::DeadEnd, 0);
    };
    let (shorter, nodes) = reject(&board, k);
    if shorter {
        (Attempt::Rejected, nodes)
    } else {
        (Attempt::Accepted { board, walk_prob }, nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::move_dfa::MoveDfa;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::IncManhattan;
    use crate::puzzle24::state::GOAL;

    const K: u32 = 12;

    #[test]
    fn walker_attempts_agree_with_true_distances() {
        let dist = bfs_distances(K as u8);
        let dfa = MoveDfa::build_default();
        let walker = Walker::new(&dfa, true);
        let mut path = Vec::new();
        for k in [2, 5, 8, 11, K] {
            let mut rng = Rng::stream(9, k as u64, 0);
            for _ in 0..300 {
                let (a, _) = attempt(&walker, k, &mut rng, &IncManhattan, &mut path);
                let end = path.iter().fold(GOAL, |s, &m| s.apply(m));
                match a {
                    Attempt::DeadEnd => panic!("moribund walker dead-ended"),
                    Attempt::Rejected => assert!(dist[&end.0] < k as u8),
                    Attempt::Accepted { board, walk_prob } => {
                        assert_eq!(board, end);
                        assert_eq!(dist[&board.0], k as u8);
                        assert_eq!(walker.path_probability(&path), walk_prob);
                    }
                }
            }
        }
    }

    /// The DFA walker almost never overshoots at these depths, so rejection is
    /// exercised on plain non-reversing walks, which often do.
    #[test]
    fn reject_shorter_matches_true_distances() {
        let dist = bfs_distances(K as u8);
        let mut rng = Rng::stream(11, 0, 0);
        let (mut shorter, mut exact) = (0, 0);
        for _ in 0..2000 {
            let k = 2 + rng.below(K as u64 - 1) as u32;
            let mut s = GOAL;
            let mut last: Option<Move> = None;
            for _ in 0..k {
                let moves: Vec<Move> = s
                    .legal_moves()
                    .iter()
                    .filter(|&m| last != Some(m.inverse()))
                    .collect();
                let m = moves[rng.below(moves.len() as u64) as usize];
                s = s.apply(m);
                last = Some(m);
            }
            let (rej, _) = reject_shorter(&s, k, &IncManhattan);
            assert_eq!(rej, dist[&s.0] < k as u8, "k={k} d={}", dist[&s.0]);
            if rej {
                shorter += 1;
            } else {
                exact += 1;
            }
        }
        // Fixed seed: 35 of the 2000 walks overshoot.
        assert!(
            shorter >= 20 && exact >= 100,
            "{shorter} shorter, {exact} exact"
        );
    }
}
