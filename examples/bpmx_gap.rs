//! bpmx_gap — measure cWD *inconsistency* on R's search tree and simulate the
//! pathmax (BPMX) sibling-cut it would enable.
//!
//! Property under test: the puzzle graph is 1-Lipschitz (each move changes the
//! true distance by exactly 1), so for a node `s` and any child `c`,
//! `h*(s) ≥ h*(c) − 1 ≥ h(c) − 1` for any admissible `h`. If cWD were
//! *consistent* (|h(parent) − h(child)| ≤ 1) this is vacuous; but cWD's
//! escape-demand vector jumps under single moves (an H-move can raise a row's
//! demand, worth +2 vertical surcharge, while WD_h itself moves ±1), so jumps
//! of +2/+3 are structurally possible. Whenever a child's h value `H` gives
//! `g + (H − 1) > bound`, the parent's own f is provably above the bound and
//! ALL remaining siblings (and their subtrees) can be skipped — sound for the
//! LB proof, and the returned `g + H − 1` is a valid lower bound for the next
//! IDA* threshold.
//!
//! Modes (both run by default):
//!   1. jump histogram: bounded randomized-order DFS at `thr_hist`, capped
//!      expansions, accumulating Δ = h(child) − h(parent) over all generated
//!      children.
//!   2. A/B exhaust: full DFS exhaust at `thr_ab` (choose small enough to
//!      finish: thr 140 ≈ sub-second, 142 ≈ a minute at ~0.5 M nodes/s) with
//!      and without the sibling-cut, fixed child order both times; reports the
//!      node reduction the cut alone buys (no DFA / neighbor-prune here — this
//!      is a go/no-go signal, production A/B comes later).
//!
//! Usage: cargo run --release --example bpmx_gap [thr_ab] [thr_hist] [hist_cap]
//! Run from repo root so data/wd24.bin + data/cwd_single.bin resolve.

use puzzle8::puzzle24::search::cwd::{Cwd, CwdScratch};
use puzzle8::puzzle24::state::{Move, State};
use std::time::Instant;

fn r_board() -> State {
    let mut a = [0u8; 25];
    for i in 1..25u8 {
        a[i as usize] = 25 - i;
    }
    State(a)
}

struct Hist {
    // Δ = h(child) − h(parent), clamped to [-4, 4] → index 0..=8
    delta: [u64; 9],
    // per-node max Δ over generated children, same clamp
    max_delta: [u64; 9],
    nodes: u64,
    children: u64,
}

impl Hist {
    fn new() -> Self {
        Hist {
            delta: [0; 9],
            max_delta: [0; 9],
            nodes: 0,
            children: 0,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Mode 1: bounded DFS (randomized child order) accumulating the jump histogram.
#[allow(clippy::too_many_arguments)]
fn dfs_hist(
    cwd: &Cwd,
    scratch: &mut CwdScratch,
    s: &State,
    h_parent: u8,
    g: u16,
    thr: u16,
    prev_inv: Option<Move>,
    rng: &mut Rng,
    cap: u64,
    expansions: &mut u64,
    hist: &mut Hist,
) {
    if *expansions >= cap {
        return;
    }
    *expansions += 1;
    hist.nodes += 1;
    let mut kids: Vec<(Move, State, u8)> = Vec::with_capacity(4);
    for m in s.legal_moves().iter() {
        if Some(m) == prev_inv {
            continue;
        }
        let child = s.apply(m);
        let hc = cwd.eval(&child, scratch);
        kids.push((m, child, hc));
    }
    let mut maxd = i16::MIN;
    for &(_, _, hc) in &kids {
        let d = hc as i16 - h_parent as i16;
        hist.delta[(d.clamp(-4, 4) + 4) as usize] += 1;
        hist.children += 1;
        maxd = maxd.max(d);
    }
    if maxd > i16::MIN {
        hist.max_delta[(maxd.clamp(-4, 4) + 4) as usize] += 1;
    }
    // recurse in random order
    for k in (1..kids.len()).rev() {
        let j = (rng.next() as usize) % (k + 1);
        kids.swap(k, j);
    }
    for (m, child, hc) in kids {
        if g + 1 + hc as u16 <= thr {
            dfs_hist(
                cwd,
                scratch,
                &child,
                hc,
                g + 1,
                thr,
                Some(m.inverse()),
                rng,
                cap,
                expansions,
                hist,
            );
            if *expansions >= cap {
                return;
            }
        }
    }
}

/// Mode 2: full exhaust at `thr`, fixed child order, optional pathmax
/// sibling-cut. Returns (expanded nodes, generated children, cut events,
/// siblings skipped by cuts).
#[allow(clippy::too_many_arguments)]
fn dfs_ab(
    cwd: &Cwd,
    scratch: &mut CwdScratch,
    s: &State,
    g: u16,
    thr: u16,
    prev_inv: Option<Move>,
    bpmx: bool,
    stats: &mut (u64, u64, u64, u64),
) {
    stats.0 += 1;
    for (i, m) in s.legal_moves().iter().enumerate() {
        let _ = i;
        if Some(m) == prev_inv {
            continue;
        }
        let child = s.apply(m);
        let hc = cwd.eval(&child, scratch);
        stats.1 += 1;
        // pathmax cut: h*(s) ≥ hc − 1, so if g + hc − 1 > thr no solution ≤ thr
        // passes through s at all — skip every remaining sibling.
        if bpmx && g + (hc as u16).saturating_sub(1) > thr {
            stats.2 += 1;
            // count remaining siblings that would otherwise have been generated
            let mut rest = 0u64;
            let mut seen_self = false;
            for m2 in s.legal_moves().iter() {
                if Some(m2) == prev_inv {
                    continue;
                }
                if seen_self {
                    rest += 1;
                }
                if m2 == m {
                    seen_self = true;
                }
            }
            stats.3 += rest;
            return;
        }
        if g + 1 + hc as u16 <= thr {
            dfs_ab(
                cwd,
                scratch,
                &child,
                g + 1,
                thr,
                Some(m.inverse()),
                bpmx,
                stats,
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // NB: must be ≥ cWD(R) = 144 or the root itself is cut and the A/B is vacuous.
    let thr_ab: u16 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(144);
    let thr_hist: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(144);
    let hist_cap: u64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_000_000);

    let t0 = Instant::now();
    let cwd = Cwd::new();
    assert!(
        cwd.has_overlay(),
        "data/cwd_single.bin missing — production single-line-max path required"
    );
    let mut scratch = CwdScratch::new();
    let r = r_board();
    let h_r = cwd.eval(&r, &mut scratch);
    eprintln!(
        "tables loaded in {:.1}s; cWD(R) = {h_r} (expect 144)",
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(h_r, 144);

    // ---- mode 1: jump histogram ----
    let mut hist = Hist::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut exp = 0u64;
    let t1 = Instant::now();
    dfs_hist(
        &cwd,
        &mut scratch,
        &r,
        h_r,
        0,
        thr_hist,
        None,
        &mut rng,
        hist_cap,
        &mut exp,
        &mut hist,
    );
    eprintln!(
        "hist DFS: {} expansions at thr {} in {:.1}s",
        exp,
        thr_hist,
        t1.elapsed().as_secs_f64()
    );
    println!("=== Δ = h(child) − h(parent), cWD single-line-max, R tree @thr {thr_hist} ===");
    println!("children evaluated: {}", hist.children);
    let labels = ["≤-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", "≥+4"];
    for (i, lab) in labels.iter().enumerate() {
        let pct = hist.delta[i] as f64 / hist.children.max(1) as f64 * 100.0;
        println!("  Δ {lab:>3}: {:>12}  ({pct:.3}%)", hist.delta[i]);
    }
    println!("per-node max Δ over children (n={}):", hist.nodes);
    for (i, lab) in labels.iter().enumerate() {
        let pct = hist.max_delta[i] as f64 / hist.nodes.max(1) as f64 * 100.0;
        println!("  max {lab:>3}: {:>12}  ({pct:.3}%)", hist.max_delta[i]);
    }
    let incon: u64 = hist.delta[6..].iter().sum();
    println!(
        "inconsistency rate (Δ ≥ +2): {:.4}% of children — {}",
        incon as f64 / hist.children.max(1) as f64 * 100.0,
        if incon == 0 {
            "cWD looks CONSISTENT here; pathmax is dead"
        } else {
            "cWD is INCONSISTENT; pathmax can fire"
        }
    );

    // ---- mode 2: A/B exhaust ----
    for bpmx in [false, true] {
        let mut stats = (0u64, 0u64, 0u64, 0u64);
        let t = Instant::now();
        dfs_ab(&cwd, &mut scratch, &r, 0, thr_ab, None, bpmx, &mut stats);
        println!(
            "\nA/B thr {thr_ab} bpmx={bpmx}: expanded={} generated={} cuts={} siblings_skipped={} ({:.1}s)",
            stats.0, stats.1, stats.2, stats.3, t.elapsed().as_secs_f64()
        );
        if bpmx {
            println!(
                "(compare `generated` across the two runs — that is the probe-count the cut saves)"
            );
        }
    }
    println!("\ntotal {:.1}s", t0.elapsed().as_secs_f64());
}
