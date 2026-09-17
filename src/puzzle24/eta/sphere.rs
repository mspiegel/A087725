//! Sphere-stratum attempts: walk `k` steps from GOAL, then decide whether the
//! end board lies at distance exactly `k`.
//!
//! The walk shows `d(v) ≤ k`, and `d(v) ≡ k (mod 2)` because every move changes
//! the blank's colour. So `d(v) = k` iff no solution of length `≤ k − 2`
//! exists, which one bounded IDA\* run with threshold cap `k − 2` decides
//! ([`reject_shorter`]). Rejected walks dominate at large `k`, and this search
//! is roughly `b²` cheaper than a full threshold-`k` iteration.
//!
//! **Checkpoints.** Every prefix of a shortest path is itself a shortest path,
//! so a walk whose first `j` steps already reach a board closer than `j` can
//! never be accepted. Testing prefixes at depths `k − c, k − 2c, …` rejects most
//! doomed walks at a fraction of the depth-`k` search cost. Which walks are
//! accepted, and with what probability, is unchanged: a checkpoint only stops
//! walks the final test would reject.

use crate::puzzle24::eta::rng::Rng;
use crate::puzzle24::eta::walker::{WalkEnd, Walker};
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
/// under heuristic `e`, checking prefixes every `check_every` steps back from
/// `k` (0 = final board only). `path` receives the walk's moves. Returns the
/// attempt and the total verification node count.
pub fn attempt<E: IncHeuristic>(
    walker: &Walker,
    k: u32,
    check_every: u32,
    rng: &mut Rng,
    e: &E,
    path: &mut Vec<Move>,
) -> (Attempt, u64) {
    let mut nodes = 0u64;
    let end = walker.walk_with(k, rng, path, |s, j| {
        if check_every == 0 || (k - j) % check_every != 0 {
            return false;
        }
        let (shorter, n) = reject_shorter(s, j, e);
        nodes += n;
        shorter
    });
    match end {
        WalkEnd::DeadEnd => (Attempt::DeadEnd, nodes),
        WalkEnd::Stopped => (Attempt::Rejected, nodes),
        WalkEnd::Done(board, walk_prob) => {
            let (shorter, n) = reject_shorter(&board, k, e);
            nodes += n;
            if shorter {
                (Attempt::Rejected, nodes)
            } else {
                (Attempt::Accepted { board, walk_prob }, nodes)
            }
        }
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
        let walker = Walker::new(&dfa);
        let mut path = Vec::new();
        for k in [2, 5, 8, 11, K] {
            let mut rng = Rng::stream(9, k as u64, 0);
            for _ in 0..300 {
                let (a, _) = attempt(&walker, k, 0, &mut rng, &IncManhattan, &mut path);
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

    /// Checkpoints change cost, not outcomes: with a fresh RNG stream per
    /// attempt, every checkpoint spacing accepts the same boards with the same
    /// probabilities, and rejects (or dead-ends) the rest.
    #[test]
    fn checkpoints_do_not_change_outcomes() {
        let dfa = MoveDfa::build_default();
        let mut rejected = 0;
        let walker = Walker::new(&dfa);
        let mut path = Vec::new();
        for k in [20, 26] {
            for i in 0..300 {
                let run = |every: u32, path: &mut Vec<Move>| {
                    let mut rng = Rng::stream(13, k as u64, i);
                    attempt(&walker, k, every, &mut rng, &IncManhattan, path).0
                };
                let base = run(0, &mut path);
                for every in [1, 3, 7] {
                    let got = run(every, &mut path);
                    match base {
                        Attempt::Accepted { .. } => assert_eq!(got, base),
                        Attempt::Rejected => assert_eq!(got, Attempt::Rejected),
                        Attempt::DeadEnd => {
                            assert!(matches!(got, Attempt::DeadEnd | Attempt::Rejected))
                        }
                    }
                }
                if base == Attempt::Rejected {
                    rejected += 1;
                }
            }
        }
        assert!(rejected > 0, "no rejections exercised");
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
