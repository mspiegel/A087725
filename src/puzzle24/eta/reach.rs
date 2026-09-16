//! Reach probability `P(v)`: the probability that a `k`-step walk from GOAL
//! under the [`Walker`] rule ends at a board `v` with `d(v) = k`.
//!
//! `P(v)` sums, over every allowed `k`-step path from GOAL to `v`, the product
//! of the per-step move probabilities along the path
//! ([`Walker::move_weights`]). Every such path is a shortest path, so it stays
//! inside the *interval* of boards `u` with `d(GOAL, u) + d(u, v) = k`. The
//! computation never enumerates paths:
//!
//! 1. **Backward search** from `v` with threshold `k` and an admissible
//!    heuristic, memoised by `(board, depth)`, marks every board on some
//!    length-`k` path from `v` to GOAL. A board `u` reached at depth `g` from
//!    which GOAL is reached in `k − g` more moves has `d(u) = k − g` exactly
//!    (the triangle inequality with `d(v) = k`), so each interval board carries
//!    one distance from GOAL.
//! 2. **Forward pass** from GOAL, layer by layer over `(board, walker node)`:
//!    each state splits its probability over the moves the walker allows with
//!    `k − j` steps left, in proportion to their weights, keeping only moves
//!    into interval boards at
//!    distance `j + 1`. The mass that reaches `v` at layer `k` is `P(v)`.
//!
//! Admissibility is all the backward search needs: a node on a length-`k`
//! solution path at depth `g` has `h ≤ d ≤ k − g`, so the bound never cuts it.

use std::collections::HashMap;

use crate::puzzle24::eta::layers::pack;
use crate::puzzle24::eta::walker::Walker;
use crate::puzzle24::search::{IncHeuristic, SearchStats};
use crate::puzzle24::state::{Move, State, GOAL};

/// Result of [`reach_probability`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reach {
    /// `P(v)`.
    pub prob: f64,
    /// Boards on some length-`k` path between GOAL and `v`, both included.
    pub interval: usize,
    /// Largest number of `(board, walker node)` states in one forward layer.
    pub max_layer_states: usize,
    /// Nodes expanded by the backward search.
    pub nodes: u64,
}

struct Backward<'a, E: IncHeuristic> {
    e: &'a E,
    k: u8,
    /// `(board, depth)` → reaches GOAL in exactly `k − depth` moves.
    memo: HashMap<(u128, u8), bool>,
    /// Interval board → distance from GOAL.
    interval: HashMap<u128, u8>,
    stats: SearchStats,
}

impl<E: IncHeuristic> Backward<'_, E> {
    fn search(
        &mut self,
        s: &State,
        blank: u8,
        ctx: E::Ctx,
        h: u8,
        g: u8,
        last: Option<Move>,
    ) -> bool {
        self.stats.nodes += 1;
        if g.saturating_add(h) > self.k {
            return false;
        }
        if s == &GOAL {
            debug_assert_eq!(g, self.k, "walk endpoint closer to GOAL than k");
            self.interval.insert(pack(s), 0);
            return g == self.k;
        }
        let key = (pack(s), g);
        if let Some(&ok) = self.memo.get(&key) {
            return ok;
        }
        let mut ok = false;
        for m in State::legal_moves_at(blank).iter() {
            if last == Some(m.inverse()) {
                continue;
            }
            let (child, cb) = s.apply_at(m, blank);
            let (ch, cctx) = self.e.advance(&ctx, &child, m, &mut self.stats);
            ok |= self.search(&child, cb, cctx, ch, g + 1, Some(m));
        }
        self.memo.insert(key, ok);
        if ok {
            self.interval.insert(key.0, self.k - g);
        }
        ok
    }
}

/// `P(v)` for a board `v` at distance exactly `k` from GOAL, under `walker`,
/// using heuristic `e` for the backward search. The caller must have
/// established `d(v) = k` (for example with
/// [`reject_shorter`](crate::puzzle24::eta::sphere::reject_shorter)).
pub fn reach_probability<E: IncHeuristic>(walker: &Walker, v: &State, k: u32, e: &E) -> Reach {
    let k8 = u8::try_from(k).expect("sphere depth exceeds u8 search bounds");
    let mut stats = SearchStats::default();
    let (h0, ctx0) = e.root(v, &mut stats);
    let mut back = Backward {
        e,
        k: k8,
        memo: HashMap::new(),
        interval: HashMap::new(),
        stats,
    };
    let found = back.search(v, v.blank_pos(), ctx0, h0, 0, None);
    assert!(found, "no length-{k} path from the board to GOAL");
    let nodes = back.stats.nodes;
    let interval = back.interval;
    drop(back.memo);

    let target = pack(v);
    let mut layer: HashMap<(u128, u32), (State, u8, f64)> = HashMap::new();
    layer.insert((pack(&GOAL), walker.root()), (GOAL, GOAL.blank_pos(), 1.0));
    let mut max_layer_states = 1;
    for j in 0..k {
        let mut next: HashMap<(u128, u32), (State, u8, f64)> = HashMap::new();
        for (&(_, node), &(s, blank, p)) in &layer {
            let probs = walker.move_probs(node, k - j, &s, blank);
            for m in Move::ALL {
                if probs[m as usize] == 0.0 {
                    continue;
                }
                let (child, cb) = s.apply_at(m, blank);
                let key = pack(&child);
                if interval.get(&key) != Some(&((j + 1) as u8)) {
                    continue;
                }
                let share = p * probs[m as usize];
                next.entry((key, walker.next(node, m)))
                    .or_insert((child, cb, 0.0))
                    .2 += share;
            }
        }
        max_layer_states = max_layer_states.max(next.len());
        layer = next;
    }
    let prob = layer
        .iter()
        .filter(|((key, _), _)| *key == target)
        .map(|(_, &(_, _, p))| p)
        .sum();
    Reach {
        prob,
        interval: interval.len(),
        max_layer_states,
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::eta::estimate::Accum;
    use crate::puzzle24::eta::layers::for_each_layer;
    use crate::puzzle24::eta::layers::unpack;
    use crate::puzzle24::eta::rng::Rng;
    use crate::puzzle24::eta::sphere::{attempt, Attempt};
    use crate::puzzle24::eta::walker::test_support::enumerate;
    use crate::puzzle24::eta::walker::Choice;
    use crate::puzzle24::eta::weights::branching_factor;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::{Heuristic, IncManhattan, ManhattanHeuristic, MoveDfa};

    fn dfa() -> &'static MoveDfa {
        static DFA: std::sync::OnceLock<MoveDfa> = std::sync::OnceLock::new();
        DFA.get_or_init(MoveDfa::build_default)
    }

    /// `P(v)` equals the summed probability of all enumerated walks ending at
    /// `v`, for every board at distance `k`.
    #[test]
    fn reach_probability_matches_exhaustive_walk_enumeration() {
        const K: u32 = 11;
        let dist = bfs_distances(K as u8);
        for (moribund, choice, tilt) in [
            (false, Choice::Uniform, 0.0),
            (true, Choice::Uniform, 0.0),
            (false, Choice::Lookahead, 0.0),
            (true, Choice::Lookahead, 0.0),
            (true, Choice::Lookahead, 1.0),
        ] {
            let walker = Walker::new(dfa(), moribund)
                .with_choice(choice)
                .with_md_tilt(tilt);
            for k in [7, 9, K] {
                let (ends, _) = enumerate(&walker, k);
                let mut checked = 0;
                for (board, &p_enum) in &ends {
                    if dist.get(board) != Some(&(k as u8)) {
                        continue;
                    }
                    let r = reach_probability(&walker, &State(*board), k, &IncManhattan);
                    let rel = (r.prob - p_enum).abs() / p_enum;
                    assert!(
                        rel < 1e-12,
                        "moribund={moribund} choice={choice:?} tilt={tilt} k={k}: {} vs {p_enum}",
                        r.prob
                    );
                    assert!(r.interval > k as usize, "interval smaller than a path");
                    checked += 1;
                }
                assert!(checked > 0);
            }
        }
    }

    /// Horvitz–Thompson over sampled walks recovers the exact size of V_12 and
    /// the exact Σ b^−MD over it, within four standard errors.
    #[test]
    fn horvitz_thompson_recovers_exact_sphere_totals() {
        const K: u32 = 12;
        let b = branching_factor();
        let mut exact_size = 0u64;
        let mut exact_md = 0.0f64;
        for_each_layer(K as usize, |k, layer| {
            if k == K as usize {
                exact_size = layer.len() as u64;
                exact_md = layer
                    .iter()
                    .map(|&key| b.powi(-(ManhattanHeuristic.h(&unpack(key)) as i32)))
                    .sum();
            }
        });
        for (choice, tilt) in [
            (Choice::Uniform, 0.0),
            (Choice::Lookahead, 0.0),
            (Choice::Lookahead, 1.0),
        ] {
            let walker = Walker::new(dfa(), true)
                .with_choice(choice)
                .with_md_tilt(tilt);
            let mut rng = Rng::stream(21, K as u64, 0);
            let mut path = Vec::new();
            let (mut size, mut md) = (Accum::default(), Accum::default());
            for _ in 0..6000 {
                match attempt(&walker, K, 0, &mut rng, &IncManhattan, &mut path).0 {
                    Attempt::Accepted { board, walk_prob } => {
                        let r = reach_probability(&walker, &board, K, &IncManhattan);
                        assert!(r.prob >= walk_prob * (1.0 - 1e-12));
                        size.add(1.0 / r.prob);
                        md.add(b.powi(-(ManhattanHeuristic.h(&board) as i32)) / r.prob);
                    }
                    _ => {
                        size.add_zeros(1);
                        md.add_zeros(1);
                    }
                }
            }
            let z_size = (size.mean() - exact_size as f64) / size.std_error();
            let z_md = (md.mean() - exact_md) / md.std_error();
            assert!(
                z_size.abs() < 4.0,
                "{choice:?} tilt {tilt}: size {} vs {exact_size} (z = {z_size:.2})",
                size.mean()
            );
            assert!(
                z_md.abs() < 4.0,
                "{choice:?} tilt {tilt}: MD sum {} vs {exact_md} (z = {z_md:.2})",
                md.mean()
            );
        }
    }
}
