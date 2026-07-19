//! Bidirectional (meet-in-the-middle) beam search for a single hard instance.
//!
//! R is search-bound: an accurate heuristic can't *find* a short 152-deep path in
//! feasible nodes (see the residual-strategy result — accurate V made R *longer*).
//! Meet-in-the-middle attacks the depth directly: run two width-bounded beams —
//! **forward** from R guided toward GOAL, **backward** from GOAL guided toward R —
//! and stitch a solution the instant a state appears in *both* visited sets. Each
//! side only has to reach the ~76-deep middle, not traverse the full 152.
//!
//! The puzzle is reversible, so the backward search is an ordinary forward search
//! from GOAL; a meet at `s` yields `R → s` (forward moves) followed by `s → GOAL`
//! (the backward `GOAL → s` moves inverted and re-ordered).

use std::collections::HashMap;

use crate::puzzle24::state::{Move, State, GOAL};

pub struct BidirConfig {
    /// Beam width kept per layer, per side (ranked by `g + weight·h`).
    pub width: usize,
    /// Max expansion layers before giving up.
    pub max_layers: usize,
    /// Stop once this many child states have been generated (0 = unlimited).
    pub node_budget: u64,
    pub fwd_weight: f32,
    pub bwd_weight: f32,
}

/// Manhattan distance from `s` to `target` (Σ over tiles of |Δrow|+|Δcol| on the
/// 5×5 grid, blank excluded) — an admissible lower bound on `dist(s, target)`,
/// cheap to compute for an arbitrary target board (used as backward guidance).
pub fn manhattan_to(s: &State, target: &State) -> u32 {
    let mut pos = [0usize; 25];
    for (cell, &tile) in target.0.iter().enumerate() {
        pos[tile as usize] = cell;
    }
    let mut total = 0u32;
    for (cell, &tile) in s.0.iter().enumerate() {
        if tile == 0 {
            continue;
        }
        let tgt = pos[tile as usize];
        total += ((cell / 5).abs_diff(tgt / 5) + (cell % 5).abs_diff(tgt % 5)) as u32;
    }
    total
}

#[derive(Clone, Copy)]
struct BNode {
    state: State,
    g: u32,
    h: f32,
    last: Option<Move>,
}

type Visited = HashMap<State, (u32, State, Move)>; // state -> (g, parent, move parent->state)

/// Expand one beam layer: generate undo-pruned children, dedup/relax against this
/// side's `visited`, record any that already sit in `other` as meetings, then
/// keep the best `width` by `g + weight·h`. Grows `expanded` and `meetings`.
#[allow(clippy::too_many_arguments)]
fn expand<H>(
    layer: &[BNode],
    visited: &mut Visited,
    other: &Visited,
    width: usize,
    weight: f32,
    h_batch: &H,
    meetings: &mut Vec<(State, u32, u32)>, // (state, g_this, g_other)
    expanded: &mut u64,
) -> Vec<BNode>
where
    H: Fn(&[State]) -> Vec<f32>,
{
    let mut child_states: Vec<State> = Vec::new();
    let mut child_meta: Vec<(Move, u32)> = Vec::new(); // (move into child, g_child)
    for node in layer {
        let blank = node.state.blank_pos();
        let banned = node.last.map(|m| m.inverse());
        let g_child = node.g + 1;
        for m in State::legal_moves_at(blank).iter() {
            if Some(m) == banned {
                continue;
            }
            let (ns, _) = node.state.apply_at(m, blank);
            match visited.get(&ns) {
                Some(&(g, _, _)) if g <= g_child => continue, // already at least as cheap
                _ => {}
            }
            visited.insert(ns, (g_child, node.state, m));
            if let Some(&(g_other, _, _)) = other.get(&ns) {
                meetings.push((ns, g_child, g_other));
            }
            child_states.push(ns);
            child_meta.push((m, g_child));
        }
    }
    *expanded += child_states.len() as u64;
    if child_states.is_empty() {
        return Vec::new();
    }
    let hs = h_batch(&child_states);
    let mut children: Vec<BNode> = child_states
        .iter()
        .enumerate()
        .map(|(i, &s)| BNode { state: s, g: child_meta[i].1, h: hs[i], last: Some(child_meta[i].0) })
        .collect();
    children.sort_by(|a, b| {
        (a.g as f32 + weight * a.h)
            .total_cmp(&(b.g as f32 + weight * b.h))
            .then(a.state.0.cmp(&b.state.0))
    });
    children.truncate(width.max(1));
    children
}

fn reconstruct(visited: &Visited, from: State, stop: State) -> Vec<Move> {
    let mut moves = Vec::new();
    let mut cur = from;
    while cur != stop {
        let &(_, parent, mv) = visited.get(&cur).expect("broken came_from chain");
        moves.push(mv);
        cur = parent;
    }
    moves // order: `from` back to `stop` (each move is parent->child)
}

/// Meet-in-the-middle beam search from `start` to `GOAL`. `fwd_h(states)` scores
/// distance-to-GOAL (forward guidance, e.g. the learned V); `bwd_h(states)` scores
/// distance-to-`start` (backward guidance, e.g. `manhattan_to(_, start)`). Returns
/// the shortest stitched solution found, or `None` if the beams never meet.
/// Result of a meet-in-the-middle solve: total length, the forward/backward split
/// at the meet (`g_fwd + g_bwd == len`), and the stitched move sequence. A balanced
/// split (both ≈ len/2) means the beams genuinely met in the middle; a lopsided one
/// means one side dominated.
pub struct MitmResult {
    pub len: u32,
    pub g_fwd: u32,
    pub g_bwd: u32,
    pub moves: Vec<Move>,
}

pub fn bidir_search<FH, BH>(
    start: &State,
    fwd_h: FH,
    bwd_h: BH,
    cfg: &BidirConfig,
) -> Option<MitmResult>
where
    FH: Fn(&[State]) -> Vec<f32>,
    BH: Fn(&[State]) -> Vec<f32>,
{
    if *start == GOAL {
        return Some(MitmResult { len: 0, g_fwd: 0, g_bwd: 0, moves: Vec::new() });
    }
    let mut fwd_visited: Visited = HashMap::new();
    let mut bwd_visited: Visited = HashMap::new();
    fwd_visited.insert(*start, (0, *start, Move::Up)); // self-sentinel; reconstruct stops here
    bwd_visited.insert(GOAL, (0, GOAL, Move::Up));
    let mut fwd_layer = vec![BNode { state: *start, g: 0, h: 0.0, last: None }];
    let mut bwd_layer = vec![BNode { state: GOAL, g: 0, h: 0.0, last: None }];

    let mut best: Option<(u32, u32, u32, State)> = None; // (total, g_fwd, g_bwd, meet)
    let mut expanded: u64 = 0;
    let mut meetings: Vec<(State, u32, u32)> = Vec::new();

    for _ in 0..cfg.max_layers {
        meetings.clear();
        fwd_layer = expand(
            &fwd_layer,
            &mut fwd_visited,
            &bwd_visited,
            cfg.width,
            cfg.fwd_weight,
            &fwd_h,
            &mut meetings,
            &mut expanded,
        );
        for &(s, gf, gb) in &meetings {
            if best.is_none_or(|(b, ..)| gf + gb < b) {
                best = Some((gf + gb, gf, gb, s));
            }
        }

        meetings.clear();
        bwd_layer = expand(
            &bwd_layer,
            &mut bwd_visited,
            &fwd_visited,
            cfg.width,
            cfg.bwd_weight,
            &bwd_h,
            &mut meetings,
            &mut expanded,
        );
        for &(s, gb, gf) in &meetings {
            if best.is_none_or(|(b, ..)| gf + gb < b) {
                best = Some((gf + gb, gf, gb, s));
            }
        }

        if cfg.node_budget != 0 && expanded >= cfg.node_budget {
            break;
        }
        if fwd_layer.is_empty() && bwd_layer.is_empty() {
            break;
        }
    }

    let (_total, g_fwd, g_bwd, meet) = best?;
    // R -> meet (forward moves, in order).
    let mut fwd = reconstruct(&fwd_visited, meet, *start);
    fwd.reverse();
    // GOAL -> meet backward moves (meet..GOAL order); meet -> GOAL = invert each in place.
    let back = reconstruct(&bwd_visited, meet, GOAL);
    let mut moves = fwd;
    moves.extend(back.iter().map(|m| m.inverse()));
    Some(MitmResult { len: moves.len() as u32, g_fwd, g_bwd, moves })
}

// ===================== front-to-front meet-in-the-middle =====================

pub struct FfConfig {
    /// Survivors kept per layer, per side.
    pub width: usize,
    pub max_layers: usize,
    pub node_budget: u64,
    /// How many opposite-frontier anchor states each side aims at (front-to-front
    /// proximity = min over anchors of `Manhattan(child, anchor) + g(anchor)`).
    pub n_anchors: usize,
    /// Weight on the base heuristic (strong: V toward GOAL forward, WD-to-R
    /// backward) — keeps paths short. `0` ⇒ pure front-to-front.
    pub w_base: f32,
    /// Weight on the front-to-front proximity term — pulls the frontiers together
    /// so they meet in the middle. `0` ⇒ plain (front-to-end) beam.
    pub w_ff: f32,
}

/// One layer node: `(state, g, last_move)`.
type FfNode = (State, u32, Option<Move>);

/// Evenly-strided sample of up to `k` `(state, g)` anchors from a frontier layer.
fn ff_anchors(layer: &[FfNode], k: usize) -> Vec<(State, u32)> {
    if layer.is_empty() || k == 0 {
        return Vec::new();
    }
    let stride = (layer.len() / k).max(1);
    layer.iter().step_by(stride).take(k).map(|&(s, g, _)| (s, g)).collect()
}

/// Front-to-front cost-to-other-side estimate: the cheapest `Manhattan(child, a) +
/// g(a)` over anchors `a` (est. child→anchor→that-side's-origin). `0` if no anchors
/// yet (the opposite side hasn't expanded), which degenerates to plain BFS.
fn ff_h(child: &State, anchors: &[(State, u32)]) -> f32 {
    let mut best = f32::INFINITY;
    for (a, ag) in anchors {
        let v = manhattan_to(child, a) as f32 + *ag as f32;
        if v < best {
            best = v;
        }
    }
    if best.is_finite() {
        best
    } else {
        0.0
    }
}

/// Expand one side's layer with a front-to-front heuristic aimed at `opp_anchors`;
/// dedup/relax against this side's `survivors` (memory-bounded: only survivors are
/// stored), record survivors that already sit in `opp_survivors` as meetings.
#[allow(clippy::too_many_arguments)]
fn expand_ff<H>(
    layer: &[FfNode],
    survivors: &mut Visited,
    opp_survivors: &Visited,
    opp_anchors: &[(State, u32)],
    width: usize,
    base_batch: &H,
    w_base: f32,
    w_ff: f32,
    meetings: &mut Vec<(State, u32, u32)>, // (state, g_this, g_opp)
    expanded: &mut u64,
) -> Vec<FfNode>
where
    H: Fn(&[State]) -> Vec<f32>,
{
    // Generate undo-pruned children, relaxed against current survivors.
    let mut cs: Vec<State> = Vec::new();
    let mut meta: Vec<(u32, State, Move)> = Vec::new(); // (g_child, parent, move)
    for &(st, g, last) in layer {
        let blank = st.blank_pos();
        let banned = last.map(|m| m.inverse());
        let gc = g + 1;
        for m in State::legal_moves_at(blank).iter() {
            if Some(m) == banned {
                continue;
            }
            let (ns, _) = st.apply_at(m, blank);
            if let Some(&(g0, _, _)) = survivors.get(&ns) {
                if g0 <= gc {
                    continue;
                }
            }
            cs.push(ns);
            meta.push((gc, st, m));
        }
    }
    *expanded += cs.len() as u64;
    if cs.is_empty() {
        return Vec::new();
    }
    // Base heuristic (V / WD-to-R) batched; front-to-front proximity per child.
    let base = if w_base != 0.0 { base_batch(&cs) } else { vec![0.0; cs.len()] };
    let mut scored: Vec<(f32, usize)> = (0..cs.len())
        .map(|i| {
            let f = meta[i].0 as f32 + w_base * base[i] + w_ff * ff_h(&cs[i], opp_anchors);
            (f, i)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(cs[a.1].0.cmp(&cs[b.1].0)));
    scored.truncate(width.max(1));

    let mut next = Vec::with_capacity(scored.len());
    for (_f, i) in scored {
        let ns = cs[i];
        let (gc, parent, mv) = meta[i];
        match survivors.get(&ns) {
            Some(&(g0, _, _)) if g0 <= gc => continue,
            _ => {}
        }
        survivors.insert(ns, (gc, parent, mv));
        if let Some(&(g_opp, _, _)) = opp_survivors.get(&ns) {
            meetings.push((ns, gc, g_opp));
        }
        next.push((ns, gc, Some(mv)));
    }
    next
}

/// **Front-to-front** meet-in-the-middle: each beam is scored by
/// `g + w_base·base(s) + w_ff·min_anchor[Manhattan(s, a) + g(a)]` — the base term
/// (`fwd_base` = V toward GOAL, `bwd_base` = WD-to-R) keeps paths short, the
/// front-to-front term pulls the frontiers together so they meet in the middle.
/// With `w_ff=0` it's a plain (front-to-end) beam; with `w_base=0`, pure
/// front-to-front. Memory-bounded (only survivors stored). Returns the shortest
/// stitched solution found.
pub fn bidir_search_ff<FB, BB>(
    start: &State,
    fwd_base: FB,
    bwd_base: BB,
    cfg: &FfConfig,
) -> Option<MitmResult>
where
    FB: Fn(&[State]) -> Vec<f32>,
    BB: Fn(&[State]) -> Vec<f32>,
{
    if *start == GOAL {
        return Some(MitmResult { len: 0, g_fwd: 0, g_bwd: 0, moves: Vec::new() });
    }
    let mut fwd_s: Visited = HashMap::new();
    let mut bwd_s: Visited = HashMap::new();
    fwd_s.insert(*start, (0, *start, Move::Up));
    bwd_s.insert(GOAL, (0, GOAL, Move::Up));
    let mut fwd_layer: Vec<FfNode> = vec![(*start, 0, None)];
    let mut bwd_layer: Vec<FfNode> = vec![(GOAL, 0, None)];

    let mut best: Option<(u32, u32, u32, State)> = None;
    let mut expanded: u64 = 0;
    let mut meetings: Vec<(State, u32, u32)> = Vec::new();

    for _ in 0..cfg.max_layers {
        let bwd_anchors = ff_anchors(&bwd_layer, cfg.n_anchors);
        meetings.clear();
        fwd_layer = expand_ff(
            &fwd_layer, &mut fwd_s, &bwd_s, &bwd_anchors, cfg.width, &fwd_base, cfg.w_base,
            cfg.w_ff, &mut meetings, &mut expanded,
        );
        for &(s, gf, gb) in &meetings {
            if best.is_none_or(|(b, ..)| gf + gb < b) {
                best = Some((gf + gb, gf, gb, s));
            }
        }

        let fwd_anchors = ff_anchors(&fwd_layer, cfg.n_anchors);
        meetings.clear();
        bwd_layer = expand_ff(
            &bwd_layer, &mut bwd_s, &fwd_s, &fwd_anchors, cfg.width, &bwd_base, cfg.w_base,
            cfg.w_ff, &mut meetings, &mut expanded,
        );
        for &(s, gb, gf) in &meetings {
            if best.is_none_or(|(b, ..)| gf + gb < b) {
                best = Some((gf + gb, gf, gb, s));
            }
        }

        if cfg.node_budget != 0 && expanded >= cfg.node_budget {
            break;
        }
        if fwd_layer.is_empty() && bwd_layer.is_empty() {
            break;
        }
    }

    let (_total, g_fwd, g_bwd, meet) = best?;
    let mut fwd = reconstruct(&fwd_s, meet, *start);
    fwd.reverse();
    let back = reconstruct(&bwd_s, meet, GOAL);
    let mut moves = fwd;
    moves.extend(back.iter().map(|m| m.inverse()));
    Some(MitmResult { len: moves.len() as u32, g_fwd, g_bwd, moves })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::ml::scramble::{scramble, Rng};

    // Manhattan-guided MITM (table-free) must find a VALID solution on a
    // moderate scramble, and no longer than the scramble that produced it.
    #[test]
    fn mitm_finds_valid_solution() {
        let mut rng = Rng::new(5);
        let mut checked = 0;
        for _ in 0..12 {
            let (start, _k) = scramble(&mut rng, 24);
            if start == GOAL {
                continue;
            }
            let cfg = BidirConfig {
                width: 2000,
                max_layers: 60,
                node_budget: 0,
                fwd_weight: 2.0,
                bwd_weight: 2.0,
            };
            let goal = GOAL;
            let out = bidir_search(
                &start,
                |ss: &[State]| ss.iter().map(|s| manhattan_to(s, &goal) as f32).collect(),
                |ss: &[State]| ss.iter().map(|s| manhattan_to(s, &start) as f32).collect(),
                &cfg,
            );
            let res = out.expect("MITM found no solution on a depth-24 board");
            let (len, moves) = (res.len, res.moves);
            assert_eq!(len as usize, moves.len());
            assert_eq!(res.g_fwd + res.g_bwd, len, "meet split must sum to length");
            // Replays to GOAL.
            let mut s = start;
            for &m in &moves {
                s = s.apply(m);
            }
            assert_eq!(s, GOAL, "MITM solution does not reach GOAL");
            assert!(len <= 60, "MITM solution {len} implausibly long for a depth-24 board");
            checked += 1;
        }
        assert!(checked >= 8, "too few boards checked: {checked}");
    }

    // Front-to-front MITM must also find VALID (replay-verified) solutions, with a
    // consistent split, on moderate scrambles.
    #[test]
    fn ff_mitm_finds_valid_solution() {
        let mut rng = Rng::new(9);
        let mut checked = 0;
        for _ in 0..12 {
            let (start, _k) = scramble(&mut rng, 24);
            if start == GOAL {
                continue;
            }
            let cfg = FfConfig {
                width: 3000,
                max_layers: 60,
                node_budget: 0,
                n_anchors: 16,
                w_base: 0.0, // pure front-to-front (table-free)
                w_ff: 2.0,
            };
            let goal = GOAL;
            let res = bidir_search_ff(
                &start,
                |ss: &[State]| ss.iter().map(|s| manhattan_to(s, &goal) as f32).collect(),
                |ss: &[State]| ss.iter().map(|s| manhattan_to(s, &start) as f32).collect(),
                &cfg,
            )
            .expect("FF-MITM found no solution");
            assert_eq!(res.len as usize, res.moves.len());
            assert_eq!(res.g_fwd + res.g_bwd, res.len, "FF meet split must sum to length");
            let mut s = start;
            for &m in &res.moves {
                s = s.apply(m);
            }
            assert_eq!(s, GOAL, "FF-MITM solution does not reach GOAL");
            assert!(res.len <= 60, "FF-MITM solution {} implausibly long", res.len);
            checked += 1;
        }
        assert!(checked >= 8, "too few boards checked: {checked}");
    }
}
