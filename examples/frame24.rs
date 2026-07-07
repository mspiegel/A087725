//! frame24 — Tier-1 test of the m=5 **frame-rule conjecture** (PUZZLE24.md 2B).
//!
//! Conjecture (generalized from the 15-puzzle, where all 17 depth-80 antipodes
//! satisfy it): deep boards are *frame-conformant* — (a) the corner pieces
//! `{_, 1, 5, 21}` sit at their antipodal corners `{0, 24, 20, 4}`, and (b) the
//! 8 corner-neighbor tiles sit within Chebyshev ≤ 1 of their assigned
//! anti-corners.
//!
//! Tier-1 test (search-free): CONSTRUCT frame-conformant boards (fix the frame,
//! randomize the 13 interior tiles), compare their **Walking Distance** — a
//! proven lower bound on depth — against uniform-random solvable controls. If
//! frame boards' proven-LB distribution sits far above control, the rule is a
//! genuine deep-board generator; if the distributions overlap, it dies cheaply.
//!
//!   cargo run --release --example frame24 -- [--n 2000] [--seed 1]

use std::process::ExitCode;

use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::state::{State, N_CELLS};

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// xorshift64* RNG (deterministic).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Corner pieces and their antipodal corner cells: blank→0, 1→24, 5→20, 21→4.
const CORNER_ANTI: [(u8, u8); 4] = [(0, 0), (1, 24), (5, 20), (21, 4)];
/// The 8 corner-neighbor tiles and each one's assigned anti-corner
/// (goal-neighbors of tile 1 = {2,6}→24; of 5 = {4,10}→20; of 21 = {16,22}→4;
/// of the blank = {20,24}→0).
const NBR_ANTI: [(u8, u8); 8] =
    [(2, 24), (6, 24), (4, 20), (10, 20), (16, 4), (22, 4), (20, 0), (24, 0)];

/// Cells within Chebyshev ≤ 1 of `c` on the 5×5 grid.
fn cheb1(c: u8) -> Vec<u8> {
    let (r0, c0) = ((c / 5) as i32, (c % 5) as i32);
    let mut out = Vec::new();
    for dr in -1..=1i32 {
        for dc in -1..=1i32 {
            let (r, cc) = (r0 + dr, c0 + dc);
            if (0..5).contains(&r) && (0..5).contains(&cc) {
                out.push((r * 5 + cc) as u8);
            }
        }
    }
    out
}

/// One frame-conformant board: frame pieces per the rule, interior uniform.
/// Parity is fixed by swapping two interior tiles if needed.
fn construct_frame(rng: &mut Rng) -> Option<State> {
    let mut cells = [u8::MAX; N_CELLS]; // MAX = empty
    let mut placed_tiles = [false; 25];
    // (a) corner pieces at anti-corners.
    for &(t, cell) in &CORNER_ANTI {
        cells[cell as usize] = t;
        placed_tiles[t as usize] = true;
    }
    // (b) each neighbor tile at a random free cell within Chebyshev 1 of its
    // assigned anti-corner (shuffled order so contention resolves randomly).
    let mut order: Vec<usize> = (0..NBR_ANTI.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    for &i in &order {
        let (t, anti) = NBR_ANTI[i];
        let free: Vec<u8> =
            cheb1(anti).into_iter().filter(|&c| cells[c as usize] == u8::MAX).collect();
        if free.is_empty() {
            return None; // contention (rare) — caller retries
        }
        cells[free[rng.below(free.len())] as usize] = t;
        placed_tiles[t as usize] = true;
    }
    // Interior: remaining 13 tiles shuffled into remaining 13 cells.
    let mut rest_tiles: Vec<u8> = (1..25u8).filter(|&t| !placed_tiles[t as usize]).collect();
    for i in (1..rest_tiles.len()).rev() {
        rest_tiles.swap(i, rng.below(i + 1));
    }
    let free_cells: Vec<usize> = (0..N_CELLS).filter(|&c| cells[c] == u8::MAX).collect();
    debug_assert_eq!(free_cells.len(), rest_tiles.len());
    for (c, t) in free_cells.iter().zip(rest_tiles.iter()) {
        cells[*c] = *t;
    }
    let mut s = State(cells);
    if !s.is_solvable() {
        // Swap two interior tiles: one transposition flips permutation parity.
        s.0.swap(free_cells[0], free_cells[1]);
        debug_assert!(s.is_solvable());
    }
    Some(s)
}

/// Uniform-random solvable control: shuffle all 25 pieces, fix parity via an
/// interior-tile swap if needed.
fn construct_control(rng: &mut Rng) -> State {
    let mut cells: Vec<u8> = (0..25u8).collect();
    for i in (1..cells.len()).rev() {
        cells.swap(i, rng.below(i + 1));
    }
    let mut arr = [0u8; N_CELLS];
    arr.copy_from_slice(&cells);
    let mut s = State(arr);
    if !s.is_solvable() {
        // Swap two non-blank pieces.
        let (mut a, mut b) = (0, 1);
        while s.0[a] == 0 {
            a += 2;
        }
        while s.0[b] == 0 || b == a {
            b += 2;
        }
        s.0.swap(a, b);
        debug_assert!(s.is_solvable());
    }
    s
}

fn report(name: &str, wds: &mut Vec<u8>) {
    wds.sort_unstable();
    let n = wds.len();
    let mean = wds.iter().map(|&w| w as f64).sum::<f64>() / n as f64;
    let pct = |p: usize| wds[(n * p / 100).min(n - 1)];
    println!(
        "{:>8}: n={}  mean WD {:.1}  p10 {}  p50 {}  p90 {}  p99 {}  max {}",
        name,
        n,
        mean,
        pct(10),
        pct(50),
        pct(90),
        pct(99),
        wds[n - 1]
    );
    // Coarse histogram, buckets of 10.
    let lo = (wds[0] / 10) * 10;
    let hi = (wds[n - 1] / 10) * 10;
    let mut b = lo;
    while b <= hi {
        let c = wds.iter().filter(|&&w| w >= b && w < b + 10).count();
        if c > 0 {
            println!("    WD {:>3}-{:<3} {:>5}  {}", b, b + 9, c, "#".repeat(c * 60 / n + 1));
        }
        b += 10;
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let n: usize = arg(&argv, "--n", 2000);
    let seed: u64 = arg(&argv, "--seed", 1);

    print!("warming up Walking Distance table... ");
    WalkingDistanceHeuristic::warm_up();
    println!("ready");
    let wd = WalkingDistanceHeuristic;

    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut frame: Vec<(u8, State)> = Vec::with_capacity(n);
    let mut attempts = 0usize;
    while frame.len() < n {
        attempts += 1;
        if let Some(s) = construct_frame(&mut rng) {
            debug_assert!(s.is_solvable());
            frame.push((wd.h(&s), s));
        }
        if attempts > n * 20 {
            eprintln!("constructor contention too high");
            return ExitCode::FAILURE;
        }
    }

    // --out FILE: emit a WD-spread of frame boards for Tier-2 (top 20 by WD +
    // 10 at the median), as 25-token lines (ladder24/solve24 format).
    if let Some(path) = argv.iter().position(|a| a == "--out").and_then(|i| argv.get(i + 1)) {
        let mut sorted = frame.clone();
        sorted.sort_by(|a, b| b.0.cmp(&a.0).then((a.1).0.cmp(&(b.1).0)));
        let mut picks: Vec<&(u8, State)> = sorted.iter().take(20).collect();
        let mid = sorted.len() / 2;
        picks.extend(sorted[mid..].iter().take(10));
        let mut text = String::from("# frame-conformant Tier-2 boards (top-20 WD + 10 median)\n");
        for (w, s) in &picks {
            text.push_str(&format!("# WD = {}\n", w));
            let toks: Vec<String> = s.0.iter().map(|t| t.to_string()).collect();
            text.push_str(&toks.join(" "));
            text.push('\n');
        }
        std::fs::write(path, text).expect("write --out");
        eprintln!("wrote {} boards -> {}", picks.len(), path);
    }
    let mut frame_wd: Vec<u8> = frame.iter().map(|&(w, _)| w).collect();
    let mut control_wd: Vec<u8> = (0..n).map(|_| wd.h(&construct_control(&mut rng))).collect();

    println!();
    println!("== frame-rule Tier-1: proven-LB (WD) distributions, n={} each ==", n);
    report("frame", &mut frame_wd);
    report("control", &mut control_wd);
    println!();
    println!(
        "(reference: WD(R) = 140 is the global max; random-board true depth ~100.8 mean; \
         constructor retries: {})",
        attempts - n
    );
    ExitCode::SUCCESS
}
