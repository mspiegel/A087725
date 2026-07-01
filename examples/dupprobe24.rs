//! dupprobe24 — Phase 0 of the move-pruning FSM scope.
//!
//! Runs a bounded IDA* exhaust on R (WD heuristic, our existing parent-pruning:
//! skip the inverse of the last move) at a given f-threshold, and counts total
//! expansions vs distinct expanded states. `expansions / distinct` is the
//! duplicate-expansion rate — the *ceiling* on what a move-pruning FSM (or a
//! transposition table) could save. If it's small, the FSM isn't worth building.
//!
//! Usage: dupprobe24 [f-bound]     (default 140 = root h(R); the shallowest
//! meaningful threshold, memory-safe. Deeper thresholds blow up the distinct set.)

use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::state::{Move, State, GOAL};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 25;

fn r_board() -> State {
    let mut c = [0u8; N];
    for i in 1..N {
        c[i] = (25 - i) as u8;
    }
    State(c)
}

/// Pack the 25 cells (each 0..24, 5 bits) into a u128 — an exact, collision-free
/// state key (25·5 = 125 bits).
fn key(s: &State) -> u128 {
    let mut k: u128 = 0;
    for i in 0..N {
        k |= (s.0[i] as u128) << (5 * i);
    }
    k
}

#[inline]
fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// Deterministic 1/(mask+1) sampling of a state by its key (uniform over
/// distinct states, so `sampled_expansions / sampled_distinct` is an unbiased
/// estimate of the overall expansion/distinct ratio).
#[inline]
fn is_sampled(k: u128, mask: u64) -> bool {
    let h = fmix64((k as u64) ^ ((k >> 64) as u64).rotate_left(32));
    (h & mask) == 0
}

struct Counts {
    expansions: u64,     // sampled expansions
    boundary: u64,       // all boundary leaves
    total_exp: u64,      // all expansions (for the true node count)
}

fn dfs(
    s: &State,
    blank: u8,
    g: u8,
    bound: u8,
    last: Option<Move>,
    mask: u64,
    seen: &mut HashSet<u128>,
    c: &mut Counts,
) {
    let h = WalkingDistanceHeuristic.h(s);
    if g.saturating_add(h) > bound {
        c.boundary += 1;
        return;
    }
    if s == &GOAL {
        return;
    }
    c.total_exp += 1;
    let k = key(s);
    if is_sampled(k, mask) {
        c.expansions += 1;
        seen.insert(k);
    }
    for m in State::legal_moves_at(blank).iter() {
        if let Some(p) = last {
            if m == p.inverse() {
                continue;
            }
        }
        let (ns, nb) = s.apply_at(m, blank);
        dfs(&ns, nb, g + 1, bound, Some(m), mask, seen, c);
    }
}

fn main() {
    let bound: u8 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(140);
    // arg 2: sample shift s → track 1/2^s of states (0 = exact/full).
    let shift: u32 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);
    let mask: u64 = (1u64 << shift) - 1;

    WalkingDistanceHeuristic::warm_up_verbose();
    let r = r_board();
    let blank = r.blank_pos();

    let mut seen: HashSet<u128> = HashSet::new();
    let mut c = Counts { expansions: 0, boundary: 0, total_exp: 0 };
    let t = Instant::now();
    dfs(&r, blank, 0, bound, None, mask, &mut seen, &mut c);
    let el = t.elapsed();

    let distinct = seen.len() as u64; // sampled distinct
    let dup = c.expansions.saturating_sub(distinct);
    println!("board R, f-bound    : {}", bound);
    println!("sampling            : 1/{} (shift {})", 1u64 << shift, shift);
    println!("total expansions    : {}", c.total_exp);
    println!("boundary leaves     : {}", c.boundary);
    println!("sampled expansions  : {}", c.expansions);
    println!("sampled distinct    : {}", distinct);
    println!(
        "duplicate expansions: {:.1}% of expansions",
        100.0 * dup as f64 / c.expansions.max(1) as f64
    );
    println!(
        "dup rate (exp/dist) : {:.3}x   <- FSM/TT ceiling",
        c.expansions as f64 / distinct.max(1) as f64
    );
    println!("wall                : {:.2?}", el);
}
