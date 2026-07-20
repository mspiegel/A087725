//! Deep-board complement profile: where is cWD loose on the R-proof traversal,
//! and how often / how much does the k8 zPDB complement it there?
//!
//! Motivation. What matters for proving R >= 152 is the heuristic across the
//! whole R->goal search tree, not h(R). The coarse A/B (data/cwdzpdb_deepboard_ab.txt)
//! already shows the PDB complement cuts ~40% of nodes on deep boards but at ~2x
//! per-node cost (iso-time, §8k). This is the FINE-GRAINED version: evaluate cWD
//! and k8 on a large sample of the states the R-proof actually visits, and
//! histogram (k8 - cWD) — the complement's hit-rate and magnitude. If k8 wins on
//! a small, structured subset, a cheaper/sparser complement could capture the
//! node-cut without the cost.
//!
//! Samples: random walks FROM R of assorted lengths (the near-R deep region the
//! proof tree lives in) + R's 156-move corridor (where the suffix length is a
//! d* proxy, so we can also measure absolute looseness d* - h).
//!
//! Run from repo root (needs data/wd24.bin, data/cwd_single.bin, and the 30.5 GiB
//! data/pdb24_k8_{a,b,c}.zbin):  cargo run --release --example deepboard_profile [N]

use std::path::Path;

use puzzle8::puzzle24::pdb::{ZPatternDb, ZpdbInc};
use puzzle8::puzzle24::search::{Cwd, Heuristic, IncHeuristicMut};
use puzzle8::puzzle24::state::{Move, State, N_CELLS};

fn r_board() -> State {
    let mut a = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        a[i] = (25 - i) as u8;
    }
    State(a)
}

/// Random walk from `start`, avoiding immediate backtracking.
fn walk(start: State, seed: u64, len: u32) -> State {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = start;
    let mut last: Option<Move> = None;
    for _ in 0..len {
        let opts: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| last.is_none_or(|p: Move| m != p.inverse()))
            .collect();
        let m = opts[(next() % opts.len() as u64) as usize];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

fn parse_moves(s: &str) -> Vec<Move> {
    s.split_whitespace()
        .filter_map(|t| match t {
            "U" => Some(Move::Up),
            "D" => Some(Move::Down),
            "L" => Some(Move::Left),
            "R" => Some(Move::Right),
            _ => None,
        })
        .collect()
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);

    eprintln!("loading cWD…");
    let cwd = Cwd::new();
    eprintln!("mmap'ing k8 zPDB (30.5 GiB, pages fault in on demand)…");
    let k8: Vec<ZPatternDb> = ["pdb24_k8_a.zbin", "pdb24_k8_b.zbin", "pdb24_k8_c.zbin"]
        .iter()
        .map(|f| {
            ZPatternDb::load_mmap(&Path::new("data").join(f)).unwrap_or_else(|e| panic!("{f}: {e}"))
        })
        .collect();
    let z8 = ZpdbInc::new([&k8[0], &k8[1], &k8[2]]);
    let k8_h = |s: &State| -> i32 { IncHeuristicMut::root(&z8, s).0 as i32 };
    let cwd_h = |s: &State| -> i32 { cwd.h(s) as i32 };

    // ---- part A: complement distribution on near-R deep boards ----
    // bucket by cWD range; track how often/much k8 beats cWD.
    let mut buckets: std::collections::BTreeMap<i32, (u64, u64, i64, i32)> =
        std::collections::BTreeMap::new(); // cwd/10 -> (count, k8_wins, sum_gain_when_win, max_gain)
    let (mut tot, mut wins, mut ties, mut sum_gain, mut max_gain) = (0u64, 0u64, 0u64, 0i64, 0i32);
    let mut sum_cwd = 0i64;
    for i in 0..n {
        let seed = (i.wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let len = 1 + (i % 60) as u32; // 1..60 moves from R
        let s = walk(r_board(), seed, len);
        let (c, k) = (cwd_h(&s), k8_h(&s));
        tot += 1;
        sum_cwd += c as i64;
        let gain = k - c;
        let e = buckets.entry(c / 10 * 10).or_insert((0, 0, 0, 0));
        e.0 += 1;
        if gain > 0 {
            wins += 1;
            sum_gain += gain as i64;
            max_gain = max_gain.max(gain);
            e.1 += 1;
            e.2 += gain as i64;
            e.3 = e.3.max(gain);
        } else if gain == 0 {
            ties += 1;
        }
    }

    println!("== deep-board complement profile (k8 vs cWD) ==");
    println!(
        "near-R sample: {tot} states (random walks 1..60 from R); mean cWD {:.1}",
        sum_cwd as f64 / tot.max(1) as f64
    );
    println!(
        "  k8 > cWD (complement fires): {wins}/{tot} = {:.1}%   ties {:.1}%   k8 < cWD {:.1}%",
        100.0 * wins as f64 / tot.max(1) as f64,
        100.0 * ties as f64 / tot.max(1) as f64,
        100.0 * (tot - wins - ties) as f64 / tot.max(1) as f64,
    );
    println!(
        "  when it fires: mean gain {:.2}, max gain {max_gain}",
        sum_gain as f64 / wins.max(1) as f64
    );
    println!("  by cWD bucket:  cWD-range | states | k8-wins (rate) | mean gain | max");
    for (lo, (cnt, w, sg, mx)) in &buckets {
        println!(
            "    {:>3}-{:<3} | {:6} | {:6} ({:4.0}%) | {:6.2} | {}",
            lo,
            lo + 9,
            cnt,
            w,
            100.0 * *w as f64 / (*cnt).max(1) as f64,
            *sg as f64 / (*w).max(1) as f64,
            mx
        );
    }

    // ---- part B: looseness + complement on R's corridor (d* proxy = suffix) ----
    let text = std::fs::read_to_string("data/r156_ours_solution.txt").expect("R path");
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let moves = parse_moves(&body);
    let mut st = r_board();
    let (mut sum_lc, mut sum_lk, mut sum_lm, mut cwins, mut cn) = (0i64, 0i64, 0i64, 0u32, 0u32);
    for i in 0..=moves.len() {
        let suffix = (moves.len() - i) as i32; // d* proxy (156 believed optimal)
        let (c, k) = (cwd_h(&st), k8_h(&st));
        sum_lc += (suffix - c) as i64;
        sum_lk += (suffix - k) as i64;
        sum_lm += (suffix - c.max(k)) as i64;
        cwins += u32::from(k > c);
        cn += 1;
        if i < moves.len() {
            st = st.apply(moves[i]);
        }
    }
    println!();
    println!("R's 156-corridor ({cn} states; d* proxy = suffix length):");
    println!(
        "  mean looseness d*-h:  cWD {:.2}   k8 {:.2}   max(cWD,k8) {:.2}",
        sum_lc as f64 / cn as f64,
        sum_lk as f64 / cn as f64,
        sum_lm as f64 / cn as f64
    );
    println!(
        "  k8 > cWD on {cwins}/{cn} corridor states = {:.1}%",
        100.0 * cwins as f64 / cn as f64
    );
}
