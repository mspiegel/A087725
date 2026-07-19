//! Fast **suboptimal** beam-search solver — the generator's reward baseline.
//!
//! A layer-by-layer beam search from `start` to `GOAL`, keeping the best `width`
//! survivors per layer as ranked by an admissible `Heuristic` (Walking Distance
//! by default). It returns an actual move sequence (suboptimal but valid), which
//! is what the GANCO regret compares the learned solver against
//! (`reward = learned_len − beam_len`). This is a standard, fast bounded-effort
//! search — the 24-puzzle has no exact solver past ~depth 50, so we cannot use an
//! optimal baseline. It is heuristic-agnostic: it only calls `Heuristic::h`.

use std::collections::HashSet;

use crate::puzzle24::search::Heuristic;
use crate::puzzle24::state::{Move, State, GOAL};

#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    /// Survivors kept per layer.
    pub width: usize,
    /// Hard cap on layers (search depth). Should exceed the diameter (~205).
    pub max_depth: usize,
    /// Hard cap on node expansions (0 = unlimited). Bounds worst-case effort.
    pub node_budget: u64,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self { width: 1000, max_depth: 220, node_budget: 2_000_000 }
    }
}

struct BeamNode {
    state: State,
    blank: u8,
    last: Option<Move>,
    path: Vec<Move>,
}

/// Beam search from `start` to `GOAL`. Returns the full move sequence (which,
/// replayed from `start`, reaches `GOAL`), or `None` if it exhausts `max_depth`
/// or `node_budget` without solving. `h` ranks survivors (lower = keep).
pub fn beam_search<H: Heuristic + ?Sized>(
    start: &State,
    h: &H,
    cfg: &BeamConfig,
) -> Option<Vec<Move>> {
    if *start == GOAL {
        return Some(Vec::new());
    }
    let width = cfg.width.max(1);

    let mut visited: HashSet<State> = HashSet::new();
    visited.insert(*start);

    let mut layer: Vec<BeamNode> =
        vec![BeamNode { state: *start, blank: start.blank_pos(), last: None, path: Vec::new() }];

    let mut expanded: u64 = 0;

    for _depth in 0..cfg.max_depth {
        expanded += layer.len() as u64;
        if cfg.node_budget != 0 && expanded > cfg.node_budget {
            return None;
        }

        // Expand the whole layer into scored children (undo-pruned, deduped vs
        // states already kept in prior layers).
        let mut children: Vec<(u8, BeamNode)> = Vec::new();
        for node in &layer {
            let banned = node.last.map(|m| m.inverse());
            for m in State::legal_moves_at(node.blank).iter() {
                if Some(m) == banned {
                    continue;
                }
                let (ns, nb) = node.state.apply_at(m, node.blank);
                if ns == GOAL {
                    let mut path = node.path.clone();
                    path.push(m);
                    return Some(path);
                }
                if visited.contains(&ns) {
                    continue;
                }
                let hv = h.h(&ns);
                let mut path = node.path.clone();
                path.push(m);
                children.push((hv, BeamNode { state: ns, blank: nb, last: Some(m), path }));
            }
        }

        if children.is_empty() {
            return None;
        }

        // Rank by heuristic ascending; deterministic tie-break on the board bytes.
        children.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.state.0.cmp(&b.1.state.0)));

        // Keep the top `width`, inserting into `visited` on keep (this also dedups
        // duplicates *within* the layer — the second copy is skipped).
        let mut next: Vec<BeamNode> = Vec::with_capacity(width);
        for (_, node) in children {
            if next.len() >= width {
                break;
            }
            if visited.insert(node.state) {
                next.push(node);
            }
        }
        if next.is_empty() {
            return None;
        }
        layer = next;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::ManhattanHeuristic;

    fn replay_reaches_goal(start: &State, moves: &[Move]) -> bool {
        let mut s = *start;
        for &m in moves {
            s = s.apply(m);
        }
        s == GOAL
    }

    #[test]
    fn goal_returns_empty() {
        let cfg = BeamConfig::default();
        assert_eq!(beam_search(&GOAL, &ManhattanHeuristic, &cfg), Some(Vec::new()));
    }

    #[test]
    fn solves_shallow_boards_validly_and_suboptimally() {
        // On depth-<=8 boards a width-200 Manhattan beam should solve every board
        // with a valid path whose length is >= the true optimal (BFS) distance.
        let truth = bfs_distances(8);
        let cfg = BeamConfig { width: 200, max_depth: 60, node_budget: 2_000_000 };
        let mut checked = 0;
        for (i, (raw, &dist)) in truth.iter().enumerate() {
            if checked >= 40 {
                break;
            }
            if i % 5 != 0 {
                continue;
            }
            let start = State(*raw);
            let sol = beam_search(&start, &ManhattanHeuristic, &cfg)
                .unwrap_or_else(|| panic!("beam failed to solve depth-{dist} board {raw:?}"));
            assert!(replay_reaches_goal(&start, &sol), "beam path did not reach GOAL");
            assert!(sol.len() as u8 >= dist, "beam length {} < optimal {}", sol.len(), dist);
            checked += 1;
        }
        assert!(checked > 20, "too few boards checked: {checked}");
    }

    #[test]
    fn width_one_is_greedy_and_terminates() {
        // Width 1 = greedy best-first. It may solve or cleanly give up, but must
        // never loop (the visited set + budget guarantee termination).
        let truth = bfs_distances(6);
        let cfg = BeamConfig { width: 1, max_depth: 200, node_budget: 100_000 };
        for (raw, _) in truth.iter().take(30) {
            let start = State(*raw);
            if let Some(sol) = beam_search(&start, &ManhattanHeuristic, &cfg) {
                assert!(replay_reaches_goal(&start, &sol));
            }
        }
    }

    #[test]
    fn tiny_budget_returns_none() {
        // A scrambled board with a 1-expansion budget must give up, not loop.
        let truth = bfs_distances(8);
        let (raw, _) = truth.iter().find(|(_, &d)| d >= 6).unwrap();
        let cfg = BeamConfig { width: 1000, max_depth: 220, node_budget: 1 };
        assert_eq!(beam_search(&State(*raw), &ManhattanHeuristic, &cfg), None);
    }
}
