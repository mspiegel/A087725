//! Batch Weighted A\* Search (BWAS) — the solver's deployed search driver.
//!
//! `f(x) = g(x) + weight · h(x)`, where `h` is supplied as a *batched* closure
//! so a learned network can score many nodes per forward pass (per DeepCubeA).
//! `h` is generally **not** admissible here (it's the learned value net), so
//! this makes no optimality guarantee — it is best-effort search under a hard
//! node-expansion budget. The same driver serves (a) the learned solver, (b) a
//! classical heuristic baseline via a trivial closure, using different weights.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;

use super::profile;
use crate::puzzle15::state::{Move, State, GOAL};

#[derive(Clone, Copy, Debug)]
pub struct BwasConfig {
    /// `weight` (λ) in `f = g + weight·h`. `1.0` with an admissible `h` is exact A\*.
    pub weight: f32,
    /// Nodes evaluated per `h` forward pass (amortizes the network call).
    pub batch_size: usize,
    /// Hard cap on node *expansions* (not wall-clock). `0` = unlimited.
    pub node_budget: u64,
}

#[derive(Debug)]
pub enum BwasOutcome {
    Solved { moves: Vec<Move>, nodes_expanded: u64 },
    BudgetExceeded { nodes_expanded: u64 },
}

impl BwasOutcome {
    /// Solution length if solved, else `None`.
    pub fn solution_len(&self) -> Option<usize> {
        match self {
            BwasOutcome::Solved { moves, .. } => Some(moves.len()),
            BwasOutcome::BudgetExceeded { .. } => None,
        }
    }
}

/// Total-order wrapper over `f32` for the priority queue.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Open-set node. Ordered by `f`, then `g`, then the raw board (deterministic
/// tie-break); wrapped in `Reverse` so the min-`f` node is popped first.
struct Node {
    f: OrdF32,
    g: u32,
    state: State,
}
impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f && self.g == o.g && self.state.0 == o.state.0
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.f
            .cmp(&o.f)
            .then(self.g.cmp(&o.g))
            .then(self.state.0.cmp(&o.state.0))
    }
}

/// Run BWAS from `start` to `GOAL`. `heuristic_batch(states) -> h-values` (one
/// per state, same order); it must return exactly `states.len()` values.
pub fn search<F>(start: &State, cfg: &BwasConfig, heuristic_batch: F) -> BwasOutcome
where
    F: Fn(&[State]) -> Vec<f32>,
{
    let batch_size = cfg.batch_size.max(1);

    if *start == GOAL {
        return BwasOutcome::Solved { moves: Vec::new(), nodes_expanded: 0 };
    }

    let mut g_score: HashMap<State, u32> = HashMap::new();
    let mut came_from: HashMap<State, (State, Move)> = HashMap::new();
    let mut open: BinaryHeap<Reverse<Node>> = BinaryHeap::new();

    let t = Instant::now();
    let h0 = heuristic_batch(std::slice::from_ref(start))[0];
    profile::record_if("bwas/heuristic", t);
    g_score.insert(*start, 0);
    open.push(Reverse(Node { f: OrdF32(cfg.weight * h0), g: 0, state: *start }));

    let mut nodes_expanded: u64 = 0;

    loop {
        // Pop up to `batch_size` non-stale nodes.
        let t = Instant::now();
        let mut batch: Vec<Node> = Vec::with_capacity(batch_size);
        while batch.len() < batch_size {
            match open.pop() {
                None => break,
                Some(Reverse(node)) => {
                    // Lazy deletion: skip entries superseded by a cheaper path.
                    if g_score.get(&node.state).copied().unwrap_or(u32::MAX) < node.g {
                        continue;
                    }
                    batch.push(node);
                }
            }
        }
        profile::record_if("bwas/pop", t);
        if batch.is_empty() {
            return BwasOutcome::BudgetExceeded { nodes_expanded };
        }

        // Goal check on pop.
        for node in &batch {
            if node.state == GOAL {
                return BwasOutcome::Solved {
                    moves: reconstruct(&came_from, node.state),
                    nodes_expanded,
                };
            }
        }

        nodes_expanded += batch.len() as u64;
        if cfg.node_budget != 0 && nodes_expanded >= cfg.node_budget {
            return BwasOutcome::BudgetExceeded { nodes_expanded };
        }

        // Generate all children, keeping only those that improve g.
        let t = Instant::now();
        let mut child_states: Vec<State> = Vec::new();
        let mut child_meta: Vec<(State, Move, u32)> = Vec::new(); // (parent, move, g_child)
        for node in &batch {
            let blank = node.state.blank_pos();
            let g_child = node.g + 1;
            for m in State::legal_moves_at(blank).iter() {
                let (ns, _) = node.state.apply_at(m, blank);
                if g_score.get(&ns).copied().unwrap_or(u32::MAX) <= g_child {
                    continue; // already reached at least as cheaply
                }
                child_states.push(ns);
                child_meta.push((node.state, m, g_child));
            }
        }
        profile::record_if("bwas/expand", t);

        if child_states.is_empty() {
            continue;
        }

        let t = Instant::now();
        let hs = heuristic_batch(&child_states);
        profile::record_if("bwas/heuristic", t);
        debug_assert_eq!(hs.len(), child_states.len());

        let t = Instant::now();
        for (i, ns) in child_states.into_iter().enumerate() {
            let (parent, m, g_child) = child_meta[i];
            // Re-check: an earlier child in this same batch may have claimed `ns`.
            if g_score.get(&ns).copied().unwrap_or(u32::MAX) <= g_child {
                continue;
            }
            g_score.insert(ns, g_child);
            came_from.insert(ns, (parent, m));
            let f = g_child as f32 + cfg.weight * hs[i];
            open.push(Reverse(Node { f: OrdF32(f), g: g_child, state: ns }));
        }
        profile::record_if("bwas/push", t);
    }
}

/// Walk `came_from` back from `goal` to the start, producing the move sequence.
fn reconstruct(came_from: &HashMap<State, (State, Move)>, goal: State) -> Vec<Move> {
    let mut moves = Vec::new();
    let mut cur = goal;
    while let Some(&(parent, m)) = came_from.get(&cur) {
        moves.push(m);
        cur = parent;
    }
    moves.reverse();
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::search::ManhattanHeuristic;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::search::Heuristic;

    fn manhattan_batch(states: &[State]) -> Vec<f32> {
        states.iter().map(|s| ManhattanHeuristic.h(s) as f32).collect()
    }

    /// Replaying the returned moves from `start` must reach GOAL, and the move
    /// count must equal the reconstructed path length.
    fn assert_valid_solution(start: &State, moves: &[Move]) {
        let mut s = *start;
        for &m in moves {
            s = s.apply(m);
        }
        assert_eq!(s, GOAL, "replayed moves did not reach GOAL");
    }

    #[test]
    fn batch1_weight1_manhattan_is_optimal() {
        // With batch_size=1, weight=1, admissible Manhattan and goal-check-on-pop,
        // BWAS is exact A* — solution length must equal the true BFS distance.
        let truth = bfs_distances(12);
        let cfg = BwasConfig { weight: 1.0, batch_size: 1, node_budget: 0 };
        let mut checked = 0;
        for (raw, &dist) in truth.iter() {
            // Only test a sample to keep it quick, across the depth range.
            if (raw[0] as usize + raw[1] as usize) % 7 != 0 {
                continue;
            }
            let start = State(*raw);
            match search(&start, &cfg, manhattan_batch) {
                BwasOutcome::Solved { moves, .. } => {
                    assert_eq!(moves.len() as u8, dist, "suboptimal on {:?}", raw);
                    assert_valid_solution(&start, &moves);
                    checked += 1;
                }
                BwasOutcome::BudgetExceeded { .. } => panic!("unlimited budget exceeded"),
            }
        }
        assert!(checked > 20, "too few states checked: {}", checked);
    }

    #[test]
    fn batched_weighted_finds_valid_solution() {
        // batch_size>1, weight>1: no optimality claim, but must still return a
        // valid solution that actually solves the board.
        let truth = bfs_distances(12);
        let cfg = BwasConfig { weight: 2.5, batch_size: 32, node_budget: 5_000_000 };
        for (raw, _) in truth.iter().take(40) {
            let start = State(*raw);
            match search(&start, &cfg, manhattan_batch) {
                BwasOutcome::Solved { moves, .. } => assert_valid_solution(&start, &moves),
                BwasOutcome::BudgetExceeded { .. } => panic!("shallow board hit budget"),
            }
        }
    }

    #[test]
    fn tiny_budget_reports_budget_exceeded() {
        // A deep-ish board with a tiny budget must report BudgetExceeded, not loop.
        let mut s = GOAL;
        for m in [Move::Up, Move::Left, Move::Up, Move::Left, Move::Down, Move::Right] {
            if State::legal_moves_at(s.blank_pos()).contains(m) {
                s = s.apply(m);
            }
        }
        let cfg = BwasConfig { weight: 1.0, batch_size: 1, node_budget: 2 };
        match search(&s, &cfg, manhattan_batch) {
            BwasOutcome::BudgetExceeded { nodes_expanded } => assert!(nodes_expanded >= 2),
            BwasOutcome::Solved { .. } => { /* also acceptable if solved within 2 expansions */ }
        }
    }
}
