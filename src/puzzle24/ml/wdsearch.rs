//! WD-maximizing beam search over walks from GOAL — a **deep-board constructor**.
//!
//! The learned walk-policy generator plateaus at WD~70-90 because greedy
//! single-move Walking-Distance ascent gets stuck at WD local maxima. A beam
//! search keeps a frontier of `width` candidates and can cross those plateaus
//! (go sideways/down, then higher) — the same reason the *solver* uses beam/BWAS
//! rather than a greedy policy. It walks from GOAL, ranks children by
//! **descending WD** (scored incrementally via `WalkingDistanceInc::advance`, one
//! probe per child), keeps the deepest `width` survivors per layer, and returns
//! the final frontier — the deepest boards it found. WD is admissible
//! (`WD ≤ optimal`), so a high-WD board is certifiably deep training material.
//!
//! `WalkingDistanceHeuristic::warm_up()` must run once before use.

use std::collections::HashSet;

use super::scramble::Rng;
use crate::puzzle24::search::{IncHeuristic, SearchStats, WalkingDistanceInc, WdCtx};
use crate::puzzle24::state::{Move, State, GOAL};

#[derive(Clone, Copy, Debug)]
pub struct WdSearchConfig {
    /// Survivors kept per layer (the beam width). Wider = escapes plateaus better.
    pub width: usize,
    /// Layers to expand = walk depth. WD ≤ depth (one move changes WD by ≤1).
    pub target_depth: usize,
    /// Hard cap on node expansions (0 = unlimited); bounds worst-case effort.
    pub node_budget: u64,
    /// How survivors are chosen from the ranked children.
    pub diversity: Diversity,
}

#[derive(Clone, Copy, Debug)]
pub enum Diversity {
    /// Deterministic top-`width` by WD desc (state-bytes tie-break) — repeatable.
    TopK,
    /// Keep `width − random_slots` deterministically, fill the rest by WD-weighted
    /// sampling from the remaining children (frontier diversity for DAVI; helps
    /// escape a single deep basin).
    Stochastic { random_slots: usize, temperature: f32 },
}

#[derive(Clone, Copy)]
struct Node {
    state: State,
    blank: u8,
    last: Option<Move>,
    ctx: WdCtx, // Copy — threads incremental WD along the walk
    wd: u8,     // WD of `state`
}

/// Beam search from GOAL maximizing Walking Distance. Returns up to `n`
/// `(state, wd)` pairs from the final frontier, sorted WD-descending. Pass
/// `n >= width` to get the whole frontier (callers V-rank it down). `warm_up()`
/// must have been called.
/// Run the beam and return the final frontier layer. If `per_layer > 0`, also
/// record up to `per_layer` survivors from *each* layer into `ladder` as
/// `(state, walk_depth)` — the whole GOAL→deep-board path.
fn run_beam(
    cfg: &WdSearchConfig,
    rng: &mut Rng,
    ladder: &mut Vec<(State, u8)>,
    per_layer: usize,
) -> Vec<Node> {
    let width = cfg.width.max(1);
    let wd_inc = WalkingDistanceInc;
    let mut stats = SearchStats::default();

    let (wd0, ctx0) = wd_inc.root(&GOAL, &mut stats);
    let mut layer: Vec<Node> =
        vec![Node { state: GOAL, blank: GOAL.blank_pos(), last: None, ctx: ctx0, wd: wd0 }];
    let mut visited: HashSet<State> = HashSet::new();
    visited.insert(GOAL);
    let mut expanded: u64 = 0;

    for depth0 in 0..cfg.target_depth {
        expanded += layer.len() as u64;
        if cfg.node_budget != 0 && expanded > cfg.node_budget {
            break; // return best frontier so far (not None)
        }

        // Expand the layer into scored children (undo-pruned, deduped vs visited).
        let mut children: Vec<Node> = Vec::new();
        for node in &layer {
            let banned = node.last.map(|m| m.inverse());
            for m in State::legal_moves_at(node.blank).iter() {
                if Some(m) == banned {
                    continue;
                }
                let (ns, nb) = node.state.apply_at(m, node.blank);
                if visited.contains(&ns) {
                    continue;
                }
                let (wd, ctx) = wd_inc.advance(&node.ctx, &ns, m, &mut stats);
                children.push(Node { state: ns, blank: nb, last: Some(m), ctx, wd });
            }
        }
        if children.is_empty() {
            break;
        }

        // Rank DESCENDING by WD; deterministic state-bytes tie-break.
        children.sort_by(|a, b| b.wd.cmp(&a.wd).then(a.state.0.cmp(&b.state.0)));
        let next = select_survivors(&children, width, cfg.diversity, &mut visited, rng);
        if next.is_empty() {
            break;
        }
        // These survivors were reached in depth0+1 moves (their walk length).
        if per_layer > 0 {
            let d = (depth0 + 1).min(u8::MAX as usize) as u8;
            for nd in next.iter().take(per_layer) {
                ladder.push((nd.state, d));
            }
        }
        layer = next;
    }
    layer
}

/// WD-maximizing beam from GOAL; returns up to `n` deepest `(state, wd)` from the
/// final frontier (WD-descending).
pub fn construct_deep_boards(n: usize, cfg: &WdSearchConfig, rng: &mut Rng) -> Vec<(State, u8)> {
    let mut ladder = Vec::new();
    let layer = run_beam(cfg, rng, &mut ladder, 0);
    let mut out: Vec<(State, u8)> = layer.iter().map(|nd| (nd.state, nd.wd)).collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then((a.0).0.cmp(&(b.0).0)));
    out.truncate(n.max(1));
    out
}

/// A **ladder** of `(state, walk_depth)` pairs sampled across the beam's layers —
/// the whole GOAL→deep-board path, for supervised value labels. `walk_depth` is
/// an achievable solution length (the reversed walk), so it upper-bounds optimal
/// (tight where the beam ascends WD efficiently). Returns up to `n` pairs spread
/// over depths `1..=target_depth`.
pub fn construct_deep_ladder(n: usize, cfg: &WdSearchConfig, rng: &mut Rng) -> Vec<(State, u8)> {
    let per_layer = (n / cfg.target_depth.max(1)).max(1);
    let mut ladder: Vec<(State, u8)> = Vec::with_capacity(per_layer * cfg.target_depth + 8);
    let _final = run_beam(cfg, rng, &mut ladder, per_layer);
    // Trim to n while preserving the spread across depths (evenly strided).
    if ladder.len() > n && n > 0 {
        let stride = ladder.len().div_ceil(n);
        ladder = ladder.into_iter().step_by(stride.max(1)).collect();
        ladder.truncate(n);
    }
    ladder
}

/// Pick up to `width` survivors from the WD-desc-sorted `children`, inserting into
/// `visited` on keep (dedups within and across layers).
fn select_survivors(
    children: &[Node],
    width: usize,
    diversity: Diversity,
    visited: &mut HashSet<State>,
    rng: &mut Rng,
) -> Vec<Node> {
    let mut next: Vec<Node> = Vec::with_capacity(width);
    match diversity {
        Diversity::TopK => {
            for node in children {
                if next.len() >= width {
                    break;
                }
                if visited.insert(node.state) {
                    next.push(*node);
                }
            }
        }
        Diversity::Stochastic { random_slots, temperature } => {
            // Deterministic top slots.
            let det = width.saturating_sub(random_slots);
            let mut i = 0;
            while next.len() < det && i < children.len() {
                if visited.insert(children[i].state) {
                    next.push(children[i]);
                }
                i += 1;
            }
            // WD-weighted sampling (with replacement + dedup) for the rest.
            let rest = &children[i..];
            if !rest.is_empty() {
                let temp = temperature.max(1e-3);
                let mx = rest.iter().map(|nd| nd.wd).max().unwrap() as f32;
                let weights: Vec<f32> =
                    rest.iter().map(|nd| ((nd.wd as f32 - mx) / temp).exp()).collect();
                let sum: f32 = weights.iter().sum::<f32>().max(1e-9);
                let mut slots = width.saturating_sub(next.len());
                let mut attempts = 0usize;
                let max_attempts = slots * 8 + 64;
                while slots > 0 && attempts < max_attempts {
                    attempts += 1;
                    let mut acc = rng.gen_f32() * sum;
                    let mut pick = 0;
                    for (k, &w) in weights.iter().enumerate() {
                        pick = k;
                        acc -= w;
                        if acc <= 0.0 {
                            break;
                        }
                    }
                    if visited.insert(rest[pick].state) {
                        next.push(rest[pick]);
                        slots -= 1;
                    }
                }
            }
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};

    fn have_table() -> bool {
        std::path::Path::new("data/wd24.bin").exists()
    }

    fn max_wd(out: &[(State, u8)]) -> u8 {
        out.iter().map(|&(_, w)| w).max().unwrap_or(0)
    }

    #[test]
    fn warm_up_then_reaches_deep() {
        // THE bet: a wide beam must reach WD far past the greedy policy's ~90.
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mut rng = Rng::new(1);
        let cfg = WdSearchConfig {
            width: 2000,
            target_depth: 160,
            node_budget: 0,
            diversity: Diversity::TopK,
        };
        let out = construct_deep_boards(2000, &cfg, &mut rng);
        assert!(max_wd(&out) > 100, "beam only reached WD {} (expected >>90)", max_wd(&out));
    }

    #[test]
    fn wd_non_decreasing_in_width() {
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mx = |width: usize, seed: u64| {
            let mut rng = Rng::new(seed);
            let cfg = WdSearchConfig {
                width,
                target_depth: 120,
                node_budget: 0,
                diversity: Diversity::TopK,
            };
            max_wd(&construct_deep_boards(width, &cfg, &mut rng))
        };
        let (a, b, c) = (mx(50, 1), mx(500, 2), mx(2000, 3));
        assert!(a <= b && b <= c, "WD not non-decreasing in width: {} {} {}", a, b, c);
    }

    #[test]
    fn wd_increases_with_depth() {
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mx = |depth: usize| {
            let mut rng = Rng::new(7);
            let cfg = WdSearchConfig {
                width: 1000,
                target_depth: depth,
                node_budget: 0,
                diversity: Diversity::TopK,
            };
            max_wd(&construct_deep_boards(1000, &cfg, &mut rng))
        };
        let (a, b, c) = (mx(40), mx(80), mx(160));
        assert!(a < b && b <= c, "WD not increasing with depth: {} {} {}", a, b, c);
    }

    #[test]
    fn boards_solvable_and_wd_matches_oracle() {
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mut rng = Rng::new(9);
        let cfg = WdSearchConfig {
            width: 500,
            target_depth: 100,
            node_budget: 0,
            diversity: Diversity::TopK,
        };
        let out = construct_deep_boards(500, &cfg, &mut rng);
        assert!(!out.is_empty());
        for &(s, wd) in &out {
            assert!(s.is_solvable(), "unsolvable board {:?}", s.0);
            assert_eq!(wd, WalkingDistanceHeuristic.h(&s), "incremental WD != oracle on {:?}", s.0);
        }
    }

    #[test]
    fn width_one_and_tiny_budget_terminate() {
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mut rng = Rng::new(3);
        let g = construct_deep_boards(
            1,
            &WdSearchConfig {
                width: 1,
                target_depth: 200,
                node_budget: 0,
                diversity: Diversity::TopK,
            },
            &mut rng,
        );
        assert!(g.len() <= 1);
        let b = construct_deep_boards(
            100,
            &WdSearchConfig {
                width: 1000,
                target_depth: 160,
                node_budget: 1,
                diversity: Diversity::TopK,
            },
            &mut rng,
        );
        assert!(b.len() <= 1000);
    }

    #[test]
    fn ladder_covers_depths_with_upper_bound_labels() {
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mut rng = Rng::new(2);
        let cfg = WdSearchConfig {
            width: 500,
            target_depth: 120,
            node_budget: 0,
            diversity: Diversity::TopK,
        };
        let ladder = construct_deep_ladder(1000, &cfg, &mut rng);
        assert!(!ladder.is_empty());
        let min_d = ladder.iter().map(|&(_, d)| d).min().unwrap();
        let max_d = ladder.iter().map(|&(_, d)| d).max().unwrap();
        assert!(min_d < 40 && max_d > 100, "ladder depth span {}..{}", min_d, max_d);
        // Each label upper-bounds optimal: WD(board) ≤ walk_depth, board solvable.
        for &(s, d) in &ladder {
            assert!(s.is_solvable(), "unsolvable ladder board {:?}", s.0);
            assert!(WalkingDistanceHeuristic.h(&s) <= d, "WD > walk-depth label on {:?}", s.0);
        }
    }

    #[test]
    fn stochastic_diversity_not_collapsed() {
        if !have_table() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let mut rng = Rng::new(11);
        let cfg = WdSearchConfig {
            width: 1000,
            target_depth: 120,
            node_budget: 0,
            diversity: Diversity::Stochastic { random_slots: 300, temperature: 5.0 },
        };
        let out = construct_deep_boards(1000, &cfg, &mut rng);
        let distinct: HashSet<_> = out.iter().map(|&(s, _)| s.0).collect();
        assert!(distinct.len() > 1, "frontier collapsed to one board");
        assert!(max_wd(&out) > 90, "stochastic reached only WD {}", max_wd(&out));
    }
}
