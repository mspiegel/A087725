//! corridor15 — the **corridor-conformance gate** for the frame rule.
//!
//! The frame rule (PLAN.md, verified on all 17 known depth-80 antipodes) is a
//! statement about *positions*: deep boards have their corner tiles at the
//! antipodal corners and the 8 corner-neighbor tiles within Chebyshev ≤ 1 of
//! their anti-corners. To use it for *search* (frame-waypoint MITM on the
//! 24-puzzle) we need the **corridor** version: do the intermediate states of
//! an optimal deep-board solution stay frame-structured as depth decreases, or
//! does the structure dissolve immediately?
//!
//! Ground truth exists on the 15-puzzle: solve each of the 17 antipodes
//! optimally (korf-plus, ~9 s each); every prefix of an optimal solution leaves
//! a state whose exact optimal depth is `80 − i`. Measure frame metrics along
//! these corridors, and compare against random states of **matched optimal
//! depth** (random walks from GOAL, each solved optimally to get its true
//! depth).
//!
//!   cargo run --release --features mmap --example corridor15 -- \
//!       [--pdb-dir data] [--antipodes data/pdb15_antipodes.txt] \
//!       [--control-walks 32] [--seed 1]

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::antipodes::load_ranks;
use puzzle8::puzzle15::pdb::{
    AdditivePdbHeuristic, MaxHeuristic, PatternDb, ReflectedHeuristic,
};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    idastar, Heuristic, LinearConflictHeuristic, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::{Move, State, GOAL};

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------- frame rule

/// Chebyshev distance between two cells of the 4×4 grid.
fn cheb(a: u8, b: u8) -> u8 {
    let (ra, ca) = (a / 4, a % 4);
    let (rb, cb) = (b / 4, b % 4);
    (ra.abs_diff(rb)).max(ca.abs_diff(cb))
}

/// The four corner pieces and their **antipodal** corner cells (PLAN.md (a)):
/// blank→0, 1→15, 4→12, 13→3 (each piece's home corner rotated 180°).
const CORNER_ANTI: [(u8, u8); 4] = [(0, 0), (1, 15), (4, 12), (13, 3)];

/// The 8 goal-neighbors of the corner pieces and each one's **assigned
/// anti-corner** (PLAN.md (b)): the anti-corner of the corner piece it borders
/// in GOAL. Neighbors of 1 (cell 0) = {2,5}→15; of 4 (cell 3) = {3,8}→12; of
/// 13 (cell 12) = {9,14}→3; of blank (cell 15) = {12,15}→0.
const NBR_ANTI: [(u8, u8); 8] =
    [(2, 15), (5, 15), (3, 12), (8, 12), (9, 3), (14, 3), (12, 0), (15, 0)];

#[derive(Clone, Copy, Default)]
struct FrameMetrics {
    /// How many of the 4 corner pieces sit exactly at their anti-corner (0–4).
    corners_anti: u8,
    /// Sum over the 8 neighbor tiles of Chebyshev(pos, assigned anti-corner).
    nbr_cheb_sum: u8,
    /// Full frame-conformance per the rule: all 4 corners at anti, all 8
    /// neighbors ≤ Chebyshev 2 with at most one at exactly 2.
    conformant: bool,
}

fn frame_metrics(s: &State) -> FrameMetrics {
    // pos[piece] = cell (piece 0 = blank).
    let mut pos = [0u8; 16];
    for (cell, &t) in s.0.iter().enumerate() {
        pos[t as usize] = cell as u8;
    }
    let corners_anti =
        CORNER_ANTI.iter().filter(|&&(t, anti)| pos[t as usize] == anti).count() as u8;
    let mut nbr_cheb_sum = 0u8;
    let mut over1 = 0; // neighbors with cheb ≥ 2
    let mut over2 = 0; // neighbors with cheb ≥ 3 (hard fail)
    for &(t, anti) in &NBR_ANTI {
        let d = cheb(pos[t as usize], anti);
        nbr_cheb_sum += d;
        if d >= 2 {
            over1 += 1;
        }
        if d >= 3 {
            over2 += 1;
        }
    }
    let conformant = corners_anti == 4 && over1 <= 1 && over2 == 0;
    FrameMetrics { corners_anti, nbr_cheb_sum, conformant }
}

// ------------------------------------------------------------------ sampling

/// xorshift64* — deterministic control-walk RNG.
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
}

/// Random walk from GOAL of length `len` with no immediate undo.
fn walk(rng: &mut Rng, len: usize) -> State {
    let mut s = GOAL;
    let mut last: Option<Move> = None;
    for _ in 0..len {
        let blank = s.blank_pos();
        let banned = last.map(|m| m.inverse());
        let moves: Vec<Move> =
            State::legal_moves_at(blank).iter().filter(|&m| Some(m) != banned).collect();
        let m = moves[(rng.next() % moves.len() as u64) as usize];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

#[derive(Default, Clone)]
struct Bucket {
    n: usize,
    corners: u64,
    cheb: u64,
    conform: usize,
}
impl Bucket {
    fn add(&mut self, m: &FrameMetrics) {
        self.n += 1;
        self.corners += m.corners_anti as u64;
        self.cheb += m.nbr_cheb_sum as u64;
        self.conform += m.conformant as usize;
    }
    fn row(&self) -> String {
        if self.n == 0 {
            return format!("{:>4}  {:>8}  {:>8}  {:>8}", 0, "-", "-", "-");
        }
        format!(
            "{:>4}  {:>8.2}  {:>8.2}  {:>7.0}%",
            self.n,
            self.corners as f64 / self.n as f64,
            self.cheb as f64 / self.n as f64,
            100.0 * self.conform as f64 / self.n as f64
        )
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let pdb_dir = PathBuf::from(arg(&argv, "--pdb-dir", "data".to_string()));
    let anti_path = arg(&argv, "--antipodes", "data/pdb15_antipodes.txt".to_string());
    let control_walks: usize = arg(&argv, "--control-walks", 32);
    let seed: u64 = arg(&argv, "--seed", 1);

    // korf-plus stack (mirrors solve15): max(korf, LC, WD).
    let p7 = match PatternDb::load_mmap(&pdb_dir.join("pdb15_p7_korf.bin")) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error loading p7: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let p8 = match PatternDb::load_mmap(&pdb_dir.join("pdb15_p8_korf.bin")) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error loading p8: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let dbs = [p7, p8];
    let h_add = AdditivePdbHeuristic::new(&dbs);
    let h_refl_inner = AdditivePdbHeuristic::new(&dbs);
    let h_refl = ReflectedHeuristic::new(h_refl_inner);
    let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);
    WalkingDistanceHeuristic::warm_up();
    let h_classical = MaxHeuristic::new(
        &LinearConflictHeuristic as &dyn Heuristic,
        &WalkingDistanceHeuristic as &dyn Heuristic,
    );
    let h_plus = MaxHeuristic::new(&h_korf as &dyn Heuristic, &h_classical as &dyn Heuristic);

    // GOAL sanity: fully anti-conformant is the *far* end; GOAL scores 0/…
    let g = frame_metrics(&GOAL);
    eprintln!(
        "GOAL frame metrics (sanity): corners_anti={} cheb_sum={} conformant={}",
        g.corners_anti, g.nbr_cheb_sum, g.conformant
    );

    // ---- corridors: solve each antipode optimally, measure every prefix state.
    let ranks = match load_ranks(std::path::Path::new(&anti_path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    eprintln!("{} antipodes loaded; solving optimally (korf-plus)...", ranks.len());

    // corridor[d] accumulates metrics of corridor states at remaining depth d.
    let mut corridor: Vec<Bucket> = vec![Bucket::default(); 81];
    for (i, &rk) in ranks.iter().enumerate() {
        let start = unrank(rk);
        let t = Instant::now();
        let moves = match idastar(&start, &h_plus) {
            Some(m) => m,
            None => {
                eprintln!("antipode {} unsolvable?!", i);
                return ExitCode::FAILURE;
            }
        };
        eprintln!(
            "  A{:<2} solved: {} moves in {:.1}s",
            i + 1,
            moves.len(),
            t.elapsed().as_secs_f64()
        );
        assert_eq!(moves.len(), 80, "antipode {} not depth 80", i);
        // Every prefix of an optimal solution leaves remaining depth 80 - k.
        let mut s = start;
        corridor[80].add(&frame_metrics(&s));
        for (k, &m) in moves.iter().enumerate() {
            s = s.apply(m);
            corridor[80 - (k + 1)].add(&frame_metrics(&s));
        }
        assert_eq!(s, GOAL);
    }

    // ---- control: random walks from GOAL, exact-depth via optimal solve.
    eprintln!("control: {} walks × lengths 20..=100 step 10 ...", control_walks);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut control: HashMap<u8, Bucket> = HashMap::new(); // keyed by exact depth
    let t = Instant::now();
    let mut solved = 0usize;
    for len in (20..=100).step_by(10) {
        for _ in 0..control_walks {
            let s = walk(&mut rng, len);
            let d = match idastar(&s, &h_plus) {
                Some(m) => m.len() as u8,
                None => continue,
            };
            control.entry(d).or_default().add(&frame_metrics(&s));
            solved += 1;
        }
    }
    eprintln!("control: {} boards solved in {:.0}s", solved, t.elapsed().as_secs_f64());

    // ---- report: corridor vs depth-matched control, bands of 5.
    println!();
    println!("== corridor vs control (frame metrics by remaining optimal depth) ==");
    println!("corners = mean corner pieces at anti-corner (0-4; GOAL=0, antipodes=4)");
    println!("cheb    = mean Σ Chebyshev(8 corner-neighbor tiles → anti-corner) (0=fully reversed frame)");
    println!("conf%   = fraction fully frame-conformant per the PLAN.md rule");
    println!();
    println!(
        "{:>5} | {:>4} {:>8} {:>8} {:>8} | {:>4} {:>8} {:>8} {:>8}",
        "depth", "N", "corners", "cheb", "conf%", "N", "corners", "cheb", "conf%"
    );
    println!("{:>5} | {:^32} | {:^32}", "", "CORRIDOR (optimal paths)", "CONTROL (random @ depth)");
    for band in (0..=80).rev().step_by(5) {
        let mut c_row = Bucket::default();
        let mut r_row = Bucket::default();
        for d in band..=(band + 4).min(80) {
            let cb = &corridor[d as usize];
            c_row.n += cb.n;
            c_row.corners += cb.corners;
            c_row.cheb += cb.cheb;
            c_row.conform += cb.conform;
            if let Some(kb) = control.get(&(d as u8)) {
                r_row.n += kb.n;
                r_row.corners += kb.corners;
                r_row.cheb += kb.cheb;
                r_row.conform += kb.conform;
            }
        }
        println!("{:>2}-{:<2} | {} | {}", band, (band + 4).min(80), c_row.row(), r_row.row());
    }
    ExitCode::SUCCESS
}
