//! The random walk's allowed-move rule `N′` for sphere sampling.
//!
//! A walk starts at [`GOAL`] and at each step picks uniformly among the moves
//! that are legal, do not undo the previous move, and are not pruned by a
//! Taylor–Korf move DFA ([`MoveDfa`], or [`LongMoveDfa`] for windows above 13).
//! The DFA prunes a move when some recent suffix of the
//! walk reaches the same board as a shorter or equal-length lexicographically
//! smaller sequence. The lexicographically smallest shortest path to any board
//! contains no such suffix, so every board at distance `k` stays reachable by a
//! walk of exactly `k` steps — the requirement for unbiased reweighting.
//!
//! **Moribund pruning** (Clausecker, ZIB Report 20-17, App. A) additionally
//! removes a move when every continuation from the resulting walker state
//! dead-ends before the walk's remaining steps run out. `doom(t)` is the number
//! of further steps after which every continuation from `t` has dead-ended, and a
//! move into `t` with `r` steps left after it is removed iff `r ≥ doom(t)`. A
//! walk that completes never enters such a state, so the set of completable
//! paths is unchanged; only the per-step choice counts change.
//!
//! The walker state is `(dfa state, blank cell, last move)`, interned into a
//! dense node table built once by breadth-first search from the root.
//!
//! [`MoveDfa`]: crate::puzzle24::search::MoveDfa
//! [`LongMoveDfa`]: crate::puzzle24::search::LongMoveDfa

use std::collections::HashMap;

use crate::puzzle24::eta::rng::Rng;
use crate::puzzle24::eta::weights::branching_factor;
use crate::puzzle24::search::move_dfa::MovePruner;
use crate::puzzle24::state::{Move, MoveSet, State, GOAL, W};

/// `last` value of the root node, which has no previous move.
const NO_LAST: u8 = 4;
/// `doom` value of a node from which some walk continues forever.
pub const DOOM_NEVER: u8 = u8::MAX;
/// Fixpoint rounds allowed for the doom table; each round extends the longest
/// resolved dead-end chain by at least one step, so this bounds `doom` at 254.
const DOOM_ROUNDS: usize = 256;

#[derive(Clone, Debug)]
struct Node {
    dfa: u32,
    blank: u8,
    /// Legal, non-reversing, DFA-unpruned moves.
    base: MoveSet,
    /// Child node per move code, for moves in `base`.
    next: [u32; 4],
    doom: u8,
}

/// How a [`Walker::walk_with`] walk ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WalkEnd {
    /// All `k` steps taken: end board and the path's probability.
    Done(State, f64),
    /// No allowed move before step `k`.
    DeadEnd,
    /// The stop callback ended the walk.
    Stopped,
}

/// How a walk chooses among the allowed moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Every allowed move equally likely.
    Uniform,
    /// Each allowed move weighted by the number of moves allowed after it
    /// (with one fewer step remaining). Walks entering branch-rich stretches
    /// are otherwise undersampled per path (records/eta24_probe.txt,
    /// Result 3). A move with no allowed continuation gets weight 0; such a
    /// path cannot complete, so every completable path keeps positive
    /// probability.
    Lookahead,
}

/// The compiled walker rule over the reachable `(dfa, blank, last)` states.
pub struct Walker {
    nodes: Vec<Node>,
    moribund: bool,
    choice: Choice,
    /// Weight multiplier for a move that raises Manhattan distance by one
    /// (`b^−λ`); a move that lowers it gets the reciprocal. `None` = no tilt.
    md_tilt: Option<f64>,
}

/// Manhattan distance of tile `tile` at cell `cell` from its goal cell.
fn tile_md(tile: u8, cell: u8) -> i32 {
    let (g, w) = (tile as i32 - 1, W as i32);
    let c = cell as i32;
    (c / w - g / w).abs() + (c % w - g % w).abs()
}

/// Change in Manhattan distance when the blank at `blank` takes move `m`
/// on `board`: the tile beside the blank slides into the blank's cell. Always
/// `+1` or `−1`.
fn md_delta(board: &State, blank: u8, m: Move) -> i32 {
    let nb = blank_after(blank, m);
    let tile = board.0[nb as usize];
    tile_md(tile, blank) - tile_md(tile, nb)
}

fn blank_after(blank: u8, m: Move) -> u8 {
    let w = W as u8;
    match m {
        Move::Up => blank - w,
        Move::Down => blank + w,
        Move::Left => blank - 1,
        Move::Right => blank + 1,
    }
}

impl Walker {
    /// Build the node table from a move-pruning DFA ([`MoveDfa`] or
    /// [`LongMoveDfa`]), rooted at [`GOAL`].
    pub fn new<P: MovePruner<St = u32>>(dfa: &P, moribund: bool) -> Walker {
        let root_blank = GOAL.blank_pos();
        let root_key = (dfa.root_state(root_blank), root_blank, NO_LAST);
        let mut index: HashMap<(u32, u8, u8), u32> = HashMap::new();
        index.insert(root_key, 0);
        let mut keys = vec![root_key];
        let mut nodes: Vec<Node> = Vec::new();
        let mut head = 0;
        while head < keys.len() {
            let (st, blank, last) = keys[head];
            head += 1;
            let mut base = State::legal_moves_at(blank);
            for m in Move::ALL {
                if dfa.is_pruned(st, m) {
                    base.0 &= !(1u8 << m as u8);
                }
            }
            if last != NO_LAST {
                base.0 &= !(1u8 << Move::ALL[last as usize].inverse() as u8);
            }
            let mut next = [u32::MAX; 4];
            for m in base.iter() {
                let key = (dfa.advance(st, m), blank_after(blank, m), m as u8);
                let id = *index.entry(key).or_insert_with(|| {
                    keys.push(key);
                    (keys.len() - 1) as u32
                });
                next[m as usize] = id;
            }
            nodes.push(Node {
                dfa: st,
                blank,
                base,
                next,
                doom: DOOM_NEVER,
            });
        }
        let mut walker = Walker {
            nodes,
            moribund,
            choice: Choice::Uniform,
            md_tilt: None,
        };
        walker.resolve_doom();
        walker
    }

    fn resolve_doom(&mut self) {
        for _ in 0..DOOM_ROUNDS {
            let mut changed = false;
            for i in 0..self.nodes.len() {
                let node = &self.nodes[i];
                let doom = if node.base.is_empty() {
                    1
                } else {
                    let mut worst = 0u8;
                    let mut all_doomed = true;
                    for m in node.base.iter() {
                        let d = self.nodes[node.next[m as usize] as usize].doom;
                        if d == DOOM_NEVER {
                            all_doomed = false;
                            break;
                        }
                        worst = worst.max(d);
                    }
                    if all_doomed && worst < DOOM_NEVER - 1 {
                        worst + 1
                    } else {
                        DOOM_NEVER
                    }
                };
                if doom != self.nodes[i].doom {
                    self.nodes[i].doom = doom;
                    changed = true;
                }
            }
            if !changed {
                return;
            }
        }
        panic!("walker doom table did not converge in {DOOM_ROUNDS} rounds");
    }

    /// Root node (the walk's start at [`GOAL`]).
    pub fn root(&self) -> u32 {
        0
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn moribund(&self) -> bool {
        self.moribund
    }

    /// Use `choice` to pick among allowed moves.
    pub fn with_choice(mut self, choice: Choice) -> Walker {
        self.choice = choice;
        self
    }

    pub fn choice(&self) -> Choice {
        self.choice
    }

    /// Tilt move choice toward low Manhattan distance: on top of the
    /// [`Choice`] weight, a move that raises Manhattan distance is weighted
    /// `b^−λ` and one that lowers it `b^λ`, with `b` the branching factor.
    /// Along any path from GOAL the increments sum to the end board's
    /// Manhattan distance, so the tilt concentrates walks on boards where
    /// `b^−MD` is large. `λ = 0` removes the tilt.
    pub fn with_md_tilt(mut self, lambda: f64) -> Walker {
        self.md_tilt = (lambda != 0.0).then(|| branching_factor().powf(-lambda));
        self
    }

    /// Probability of each move at `node` with `remaining` steps left (this one
    /// included), on `board` with the blank at `blank`, indexed by move code;
    /// zero for moves not allowed. All zero at a dead end.
    pub fn move_probs(&self, node: u32, remaining: u32, board: &State, blank: u8) -> [f64; 4] {
        let weights = self.move_weights(node, remaining);
        let mut p = [0.0f64; 4];
        match self.md_tilt {
            None => {
                let total: u32 = weights.iter().sum();
                if total > 0 {
                    for (pi, &w) in p.iter_mut().zip(&weights) {
                        *pi = w as f64 / total as f64;
                    }
                }
            }
            Some(up) => {
                for m in Move::ALL {
                    let w = weights[m as usize];
                    if w > 0 {
                        let f = if md_delta(board, blank, m) > 0 {
                            up
                        } else {
                            1.0 / up
                        };
                        p[m as usize] = w as f64 * f;
                    }
                }
                let total: f64 = p.iter().sum();
                if total > 0.0 {
                    for x in &mut p {
                        *x /= total;
                    }
                }
            }
        }
        p
    }

    /// Integer weight of each move at `node` with `remaining` steps left (this
    /// one included), indexed by move code; zero for moves not allowed. A move
    /// is taken with probability `weight / sum of weights`.
    #[inline]
    pub fn move_weights(&self, node: u32, remaining: u32) -> [u32; 4] {
        let mut w = [0u32; 4];
        for m in self.allowed(node, remaining).iter() {
            w[m as usize] = match self.choice {
                Choice::Uniform => 1,
                Choice::Lookahead if remaining <= 1 => 1,
                Choice::Lookahead => self.allowed(self.next(node, m), remaining - 1).len(),
            };
        }
        w
    }

    /// Nodes whose every continuation eventually dead-ends.
    pub fn doomed_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.doom != DOOM_NEVER).count()
    }

    /// Blank cell at `node`.
    #[inline]
    pub fn blank(&self, node: u32) -> u8 {
        self.nodes[node as usize].blank
    }

    /// Move-DFA state at `node`.
    pub fn dfa_state(&self, node: u32) -> u32 {
        self.nodes[node as usize].dfa
    }

    /// Moves allowed at `node` when `remaining` steps (this one included) are
    /// still to be taken.
    #[inline]
    pub fn allowed(&self, node: u32, remaining: u32) -> MoveSet {
        let n = &self.nodes[node as usize];
        if !self.moribund {
            return n.base;
        }
        let after = remaining.saturating_sub(1);
        let mut mask = n.base;
        for m in n.base.iter() {
            let d = self.nodes[n.next[m as usize] as usize].doom;
            if d != DOOM_NEVER && after >= d as u32 {
                mask.0 &= !(1u8 << m as u8);
            }
        }
        mask
    }

    /// Node reached by taking allowed move `m` from `node`.
    #[inline]
    pub fn next(&self, node: u32, m: Move) -> u32 {
        let id = self.nodes[node as usize].next[m as usize];
        debug_assert_ne!(id, u32::MAX, "walker stepped along a disallowed move");
        id
    }

    /// Walk `k` steps from [`GOAL`], appending the moves to `path` (cleared
    /// first). Returns the end board and the walk's probability `Π 1/|N′ᵢ|`, or
    /// `None` if the walk dead-ends.
    pub fn walk(&self, k: u32, rng: &mut Rng, path: &mut Vec<Move>) -> Option<(State, f64)> {
        match self.walk_with(k, rng, path, |_, _| false) {
            WalkEnd::Done(s, p) => Some((s, p)),
            WalkEnd::DeadEnd => None,
            WalkEnd::Stopped => unreachable!("stop callback never fires"),
        }
    }

    /// [`walk`](Self::walk), calling `stop(board, steps_taken)` after each of the
    /// first `k − 1` steps; the walk ends early with [`WalkEnd::Stopped`] when
    /// it returns `true`. The random draws are identical to [`walk`](Self::walk)
    /// up to the stopping point.
    pub fn walk_with(
        &self,
        k: u32,
        rng: &mut Rng,
        path: &mut Vec<Move>,
        mut stop: impl FnMut(&State, u32) -> bool,
    ) -> WalkEnd {
        path.clear();
        let mut node = self.root();
        let mut s = GOAL;
        let mut blank = GOAL.blank_pos();
        let mut prob = 1.0f64;
        for step in 0..k {
            let mc = if self.md_tilt.is_none() {
                let weights = self.move_weights(node, k - step);
                let total: u32 = weights.iter().sum();
                if total == 0 {
                    return WalkEnd::DeadEnd;
                }
                // With uniform weights this is one draw below the allowed
                // count, then the pick-th allowed move in code order.
                let mut pick = rng.below(total as u64) as u32;
                let mc = (0..4)
                    .find(|&i| {
                        if pick < weights[i] {
                            true
                        } else {
                            pick -= weights[i];
                            false
                        }
                    })
                    .expect("pick < total");
                prob *= weights[mc] as f64 / total as f64;
                mc
            } else {
                let probs = self.move_probs(node, k - step, &s, blank);
                if probs.iter().all(|&x| x == 0.0) {
                    return WalkEnd::DeadEnd;
                }
                // Uniform in [0, 1) from the top 53 bits; the last move with
                // positive probability absorbs rounding.
                let u = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                let mut acc = 0.0;
                let last = (0..4).rev().find(|&i| probs[i] > 0.0).expect("a move");
                let mc = (0..4)
                    .find(|&i| {
                        acc += probs[i];
                        probs[i] > 0.0 && (u < acc || i == last)
                    })
                    .expect("u < 1");
                prob *= probs[mc];
                mc
            };
            let m = Move::ALL[mc];
            (s, blank) = s.apply_at(m, blank);
            node = self.next(node, m);
            path.push(m);
            if step + 1 < k && stop(&s, step + 1) {
                return WalkEnd::Stopped;
            }
        }
        WalkEnd::Done(s, prob)
    }

    /// Probability that a `moves.len()`-step walk takes exactly `moves`; zero if
    /// some move is not allowed where it is taken.
    pub fn path_probability(&self, moves: &[Move]) -> f64 {
        let k = moves.len() as u32;
        let mut node = self.root();
        let (mut s, mut blank) = (GOAL, GOAL.blank_pos());
        let mut prob = 1.0f64;
        for (step, &m) in moves.iter().enumerate() {
            let probs = self.move_probs(node, k - step as u32, &s, blank);
            if probs[m as usize] == 0.0 {
                return 0.0;
            }
            prob *= probs[m as usize];
            node = self.next(node, m);
            (s, blank) = s.apply_at(m, blank);
        }
        prob
    }
}

/// Exhaustive walk enumeration, shared by the walker and reach-probability tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::Walker;
    use crate::puzzle24::state::{Move, State, GOAL, N_CELLS};
    use std::collections::HashMap;

    /// Every allowed walk of `k` steps: endpoint → total probability of the
    /// walks ending there, and the probability of dead-ending.
    pub(crate) fn enumerate(w: &Walker, k: u32) -> (HashMap<[u8; N_CELLS], f64>, f64) {
        #[allow(clippy::too_many_arguments)]
        fn rec(
            w: &Walker,
            k: u32,
            step: u32,
            node: u32,
            s: State,
            blank: u8,
            prob: f64,
            out: &mut HashMap<[u8; N_CELLS], f64>,
            dead: &mut f64,
        ) {
            if step == k {
                *out.entry(s.0).or_insert(0.0) += prob;
                return;
            }
            let probs = w.move_probs(node, k - step, &s, blank);
            if probs.iter().all(|&x| x == 0.0) {
                *dead += prob;
                return;
            }
            for m in Move::ALL {
                if probs[m as usize] == 0.0 {
                    continue;
                }
                let p = prob * probs[m as usize];
                let (ns, nb) = s.apply_at(m, blank);
                rec(w, k, step + 1, w.next(node, m), ns, nb, p, out, dead);
            }
        }
        let mut out = HashMap::new();
        let mut dead = 0.0;
        let (root, blank) = (w.root(), GOAL.blank_pos());
        rec(w, k, 0, root, GOAL, blank, 1.0, &mut out, &mut dead);
        (out, dead)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::enumerate;
    use super::*;
    use crate::puzzle24::eta::layers::A090031;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::MoveDfa;

    fn dfa() -> &'static MoveDfa {
        static DFA: std::sync::OnceLock<MoveDfa> = std::sync::OnceLock::new();
        DFA.get_or_init(MoveDfa::build_default)
    }

    #[test]
    fn md_delta_matches_manhattan_difference() {
        use crate::puzzle24::search::{Heuristic, ManhattanHeuristic};
        let mut rng = Rng::stream(3, 1, 4);
        let (mut s, mut blank) = (GOAL, GOAL.blank_pos());
        for _ in 0..500 {
            let moves: Vec<Move> = State::legal_moves_at(blank).iter().collect();
            let m = moves[rng.below(moves.len() as u64) as usize];
            let (child, cb) = s.apply_at(m, blank);
            let want = ManhattanHeuristic.h(&child) as i32 - ManhattanHeuristic.h(&s) as i32;
            assert_eq!(md_delta(&s, blank, m), want);
            assert_eq!(want.abs(), 1);
            (s, blank) = (child, cb);
        }
    }

    #[test]
    fn dfa_state_determines_the_blank() {
        let w = Walker::new(dfa(), false);
        let mut blank_of: HashMap<u32, u8> = HashMap::new();
        for id in 0..w.node_count() as u32 {
            let (st, blank) = (w.dfa_state(id), w.blank(id));
            let b = *blank_of.entry(st).or_insert(blank);
            assert_eq!(b, blank, "DFA state {st} seen with blanks {b} and {blank}");
        }
    }

    #[test]
    fn every_board_at_distance_k_is_reachable_by_a_k_step_walk() {
        const K: u32 = 13;
        let dist = bfs_distances(K as u8);
        for (moribund, choice, tilt) in [
            (false, Choice::Uniform, 0.0),
            (true, Choice::Uniform, 0.0),
            (false, Choice::Lookahead, 0.0),
            (true, Choice::Lookahead, 0.0),
            (true, Choice::Lookahead, 1.0),
        ] {
            let w = Walker::new(dfa(), moribund)
                .with_choice(choice)
                .with_md_tilt(tilt);
            for k in [6, 9, 12, K] {
                let (ends, dead) = enumerate(&w, k);
                let at_k = ends
                    .keys()
                    .filter(|b| dist.get(*b) == Some(&(k as u8)))
                    .count();
                assert_eq!(
                    at_k as u64, A090031[k as usize],
                    "moribund={moribund} choice={choice:?} tilt={tilt} k={k}"
                );
                let total: f64 = ends.values().sum::<f64>() + dead;
                assert!((total - 1.0).abs() < 1e-12, "probability mass {total}");
                if moribund {
                    assert_eq!(dead, 0.0, "moribund walker dead-ended at k={k}");
                }
                // path_probability agrees with the enumeration on a sample path.
                let mut rng = Rng::stream(5, k as u64, moribund as u64);
                let mut path = Vec::new();
                if let Some((s, p)) = w.walk(k, &mut rng, &mut path) {
                    assert_eq!(w.path_probability(&path), p);
                    assert!(ends[&s.0] >= p * (1.0 - 1e-12));
                }
            }
        }
    }

    /// The same completeness check under the 15-move rule, through the depths
    /// where its longer dominated sequences (13–15 moves) first apply.
    #[test]
    #[ignore = "builds LongMoveDfa W=14 and BFS to depth 15; ~5 s in release (0.9 GB peak), minutes in debug; run with --release -- --ignored"]
    fn long_window_walker_reaches_every_board_at_distance_k() {
        const K: u32 = 15;
        let dfa = crate::puzzle24::search::LongMoveDfa::build(14);
        let dist = bfs_distances(K as u8);
        let w = Walker::new(&dfa, true);
        for k in [12, 13, 14, K] {
            let (ends, dead) = enumerate(&w, k);
            let at_k = ends
                .keys()
                .filter(|b| dist.get(*b) == Some(&(k as u8)))
                .count();
            assert_eq!(at_k as u64, A090031[k as usize], "k={k}");
            let total: f64 = ends.values().sum::<f64>() + dead;
            assert!((total - 1.0).abs() < 1e-12, "probability mass {total}");
            assert_eq!(dead, 0.0, "moribund walker dead-ended at k={k}");
        }
    }
}
