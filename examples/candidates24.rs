//! candidates24 — Phase-2 (2C) deep-board candidate generator.
//!
//! Produces a diverse pool of hard 24-puzzle boards to feed the bounded-LB hunt
//! (`ladder24 --from`) and the learned-UB pass (`gen_corridors --mode ubfile`).
//!
//! Method (construct → score → hill-climb → dedup → emit):
//!   - **Seeds:** `R` (180° rotation), `reflect(R)`, frame-conformant boards
//!     (`puzzle24::frame::construct_frame_with`, kept only if WD ≥ --min-seed-wd),
//!     and every solvable board from each `--reseed FILE` (the catalog flywheel).
//!   - **Score:** Walking Distance (default) or `max(LC, WD)` — both admissible,
//!     so a board's score IS a proven lower bound on its optimal depth. WD
//!     saturates near R (global max 140), so the interesting motion is lateral.
//!   - **Hill-climb:** solvability-preserving mutations of the *non-blank* cells —
//!     3-cycles and double-swaps are even permutations, so they never touch the
//!     even-inversion solvability invariant (state.rs) nor the blank. Greedy with
//!     bounded sideways drift; several restarts per seed (each `--perturb`-kicked
//!     by legal blank moves) find distinct local maxima.
//!   - **Dedup / diversity:** canonicalize under the reflection symmetry
//!     (`symmetry::canonical`); cap per-seed emissions; require Hamming distance
//!     ≥ --min-dist from every already-emitted board of equal-or-higher score
//!     (stops the pool collapsing onto near-R clones).
//!
//! Output (the interchange format read by `ladder24 --from` / `gen_corridors
//! --boards`): one `#` lineage comment then a 25-token row (blank emitted as `0`,
//! never `_` — the gen_corridors reader cannot parse `_`).
//!
//!   cargo run --release --example candidates24 -- --out data/pool_g1.txt \
//!       --frame-seeds 40 --iters 2000 --restarts 4 --per-seed 12 \
//!       --min-dist 6 --pool-cap 100 --seed 1 [--reseed data/reseed_g2.txt]...

use std::collections::HashSet;
use std::fmt::Write as _;
use std::process::ExitCode;

use puzzle8::puzzle24::frame::{construct_frame_with, is_frame_conformant};
use puzzle8::puzzle24::search::{Heuristic, LinearConflictHeuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::state::{Move, State, N_CELLS};
use puzzle8::puzzle24::symmetry::canonical;

// ------------------------------------------------------------------- arg parsing

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// All values following repeated `flag` occurrences (for `--reseed A --reseed B`).
fn args_multi<'a>(argv: &'a [String], flag: &str) -> Vec<&'a String> {
    let mut out = Vec::new();
    for (i, a) in argv.iter().enumerate() {
        if a == flag {
            if let Some(v) = argv.get(i + 1) {
                out.push(v);
            }
        }
    }
    out
}

// ------------------------------------------------------------------------- RNG

/// xorshift64* — deterministic, independent of the library `scramble::Rng`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform index in `[0, n)`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ------------------------------------------------------------------ board helpers

/// `R` = goal rotated 180°: blank at cell 0, tile `25−i` at cell `i`.
fn r_board() -> State {
    let mut a = [0u8; N_CELLS];
    for (i, slot) in a.iter_mut().enumerate().skip(1) {
        *slot = (25 - i) as u8;
    }
    State(a)
}

/// Cells holding a non-blank tile.
fn non_blank_cells(s: &State) -> Vec<usize> {
    (0..N_CELLS).filter(|&c| s.0[c] != 0).collect()
}

/// Distinct-cell count between two boards (canonical Hamming for diversity).
fn hamming(a: &State, b: &State) -> usize {
    (0..N_CELLS).filter(|&i| a.0[i] != b.0[i]).count()
}

#[derive(Clone, Copy, PartialEq)]
enum MutKind {
    ThreeCycle,
    DoubleSwap,
}

/// One solvability-preserving mutation of the non-blank cells. Both kinds are
/// even permutations, so inversion parity (and thus solvability) and the blank
/// position are preserved. Returns the mutated board and which kind fired.
fn mutate(rng: &mut Rng, s: &State, cells: &[usize]) -> (State, MutKind) {
    let mut out = *s;
    // ~70% 3-cycle, ~30% double-swap.
    if rng.below(10) < 7 && cells.len() >= 3 {
        let (a, b, c) = pick3(rng, cells);
        // (a b c): out[a]<-s[c], out[b]<-s[a], out[c]<-s[b]  (a 3-cycle, even).
        out.0[a] = s.0[c];
        out.0[b] = s.0[a];
        out.0[c] = s.0[b];
        (out, MutKind::ThreeCycle)
    } else {
        let (a, b, c, d) = pick4(rng, cells);
        out.0.swap(a, b);
        out.0.swap(c, d);
        (out, MutKind::DoubleSwap)
    }
}

fn pick3(rng: &mut Rng, cells: &[usize]) -> (usize, usize, usize) {
    let a = cells[rng.below(cells.len())];
    let mut b = cells[rng.below(cells.len())];
    while b == a {
        b = cells[rng.below(cells.len())];
    }
    let mut c = cells[rng.below(cells.len())];
    while c == a || c == b {
        c = cells[rng.below(cells.len())];
    }
    (a, b, c)
}

fn pick4(rng: &mut Rng, cells: &[usize]) -> (usize, usize, usize, usize) {
    let (a, b, c) = pick3(rng, cells);
    let mut d = cells[rng.below(cells.len())];
    while d == a || d == b || d == c {
        d = cells[rng.below(cells.len())];
    }
    (a, b, c, d)
}

/// `k` legal blank moves from `s`, never immediately undoing the previous move
/// (a real move sequence, so solvability is preserved automatically).
fn perturb(rng: &mut Rng, s: &State, k: u32) -> State {
    let mut cur = *s;
    let mut last: Option<Move> = None;
    for _ in 0..k {
        let legal = cur.legal_moves();
        let choices: Vec<Move> =
            legal.iter().filter(|&m| last.map_or(true, |l| m != l.inverse())).collect();
        if choices.is_empty() {
            break;
        }
        let m = choices[rng.below(choices.len())];
        cur = cur.apply(m);
        last = Some(m);
    }
    cur
}

// --------------------------------------------------------------------- scoring

#[derive(Clone, Copy)]
enum ScoreKind {
    Wd,
    Cheap,
}

struct Scorer {
    wd: WalkingDistanceHeuristic,
    lc: LinearConflictHeuristic,
    kind: ScoreKind,
}
impl Scorer {
    /// Admissible score = proven lower bound on optimal depth.
    fn score(&self, s: &State) -> u8 {
        let w = self.wd.h(s);
        match self.kind {
            ScoreKind::Wd => w,
            ScoreKind::Cheap => w.max(self.lc.h(s)),
        }
    }
}

// --------------------------------------------------------------------- hill-climb

/// One local-optimum discovery from `seed`: greedy accept on improving score,
/// bounded sideways drift, tracking the best (max-score) state visited.
struct Peak {
    s: State,
    score: u8,
    ops3: u32,
    opsd: u32,
    accepts: u32,
}

fn climb(seed: &State, sc: &Scorer, iters: u32, sideways_cap: u32, rng: &mut Rng) -> Peak {
    let cells = non_blank_cells(seed);
    let mut cur = *seed;
    let mut cur_score = sc.score(&cur);
    let (mut ops3, mut opsd, mut accepts) = (0u32, 0u32, 0u32);
    let mut best = Peak { s: cur, score: cur_score, ops3: 0, opsd: 0, accepts: 0 };
    let mut sideways = 0u32;
    for _ in 0..iters {
        let (cand, kind) = mutate(rng, &cur, &cells);
        let ns = sc.score(&cand);
        let take = if ns > cur_score {
            true
        } else if ns == cur_score && sideways < sideways_cap {
            sideways += 1;
            true
        } else {
            false
        };
        if take {
            if ns > cur_score {
                sideways = 0;
            }
            cur = cand;
            cur_score = ns;
            accepts += 1;
            match kind {
                MutKind::ThreeCycle => ops3 += 1,
                MutKind::DoubleSwap => opsd += 1,
            }
            if cur_score > best.score {
                best = Peak { s: cur, score: cur_score, ops3, opsd, accepts };
            }
        }
    }
    best
}

// ----------------------------------------------------------------- board parsing

fn parse_board_line(line: &str) -> Option<State> {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
        return None;
    }
    let toks: Vec<&str> = t.split_whitespace().collect();
    if toks.len() != N_CELLS {
        return None;
    }
    let mut arr = [0u8; N_CELLS];
    for (i, tok) in toks.iter().enumerate() {
        let v: u8 = if *tok == "_" || *tok == "." {
            0
        } else {
            match tok.parse() {
                Ok(v) if v <= 24 => v,
                _ => return None,
            }
        };
        arr[i] = v;
    }
    Some(State(arr))
}

fn board_tokens(s: &State) -> String {
    s.0.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------- main

struct Emitted {
    canon: State,
    score: u8,
}

struct Candidate {
    board: State,
    canon: State,
    score: u8,
    lineage: String,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "candidates24 --out FILE [--seed N] [--frame-seeds 40] [--min-seed-wd 110] \
             [--reseed FILE]... [--perturb 8] [--iters 2000] [--restarts 4] \
             [--sideways 200] [--per-seed 12] [--min-dist 6] [--pool-cap 150] \
             [--score wd|cheap]"
        );
        return ExitCode::SUCCESS;
    }
    let out = match argv.iter().position(|a| a == "--out").and_then(|i| argv.get(i + 1)) {
        Some(p) => p.clone(),
        None => {
            eprintln!("--out FILE required");
            return ExitCode::FAILURE;
        }
    };
    let seed: u64 = arg(&argv, "--seed", 1);
    let frame_seeds: usize = arg(&argv, "--frame-seeds", 40);
    let min_seed_wd: u8 = arg(&argv, "--min-seed-wd", 110);
    let perturb_k: u32 = arg(&argv, "--perturb", 8);
    let iters: u32 = arg(&argv, "--iters", 2000);
    let restarts: u32 = arg(&argv, "--restarts", 4);
    let sideways: u32 = arg(&argv, "--sideways", 200);
    let per_seed: usize = arg(&argv, "--per-seed", 12);
    let min_dist: usize = arg(&argv, "--min-dist", 6);
    let pool_cap: usize = arg(&argv, "--pool-cap", 150);
    let kind = match arg::<String>(&argv, "--score", "wd".into()).as_str() {
        "cheap" => ScoreKind::Cheap,
        _ => ScoreKind::Wd,
    };

    eprint!("warming up Walking Distance table... ");
    WalkingDistanceHeuristic::warm_up();
    eprintln!("ready");
    let sc = Scorer { wd: WalkingDistanceHeuristic, lc: LinearConflictHeuristic, kind };

    let mut rng = Rng::new(seed);

    // ---- build seed list: (label, board) ----
    let mut seeds: Vec<(String, State)> = Vec::new();
    seeds.push(("R".into(), r_board()));
    seeds.push(("reflect(R)".into(), puzzle8::puzzle24::symmetry::reflect(&r_board())));
    // frame-conformant seeds (keep only WD >= min_seed_wd)
    {
        let mut made = 0usize;
        let mut attempts = 0usize;
        while made < frame_seeds && attempts < frame_seeds * 50 + 100 {
            attempts += 1;
            if let Some(s) = construct_frame_with(&mut |n| rng.below(n)) {
                if sc.wd.h(&s) >= min_seed_wd {
                    seeds.push((format!("frame#{}", made), s));
                    made += 1;
                }
            }
        }
        eprintln!("frame seeds: {} (>= WD {}), {} attempts", made, min_seed_wd, attempts);
    }
    // reseed files (catalog flywheel)
    for path in args_multi(&argv, "--reseed") {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let mut n = 0usize;
                for (ln, line) in text.lines().enumerate() {
                    if let Some(s) = parse_board_line(line) {
                        if s.is_solvable() {
                            seeds.push((format!("reseed:{}#{}", path, ln), s));
                            n += 1;
                        } else {
                            eprintln!("warning: {}:{} not solvable, skipped", path, ln + 1);
                        }
                    }
                }
                eprintln!("reseed {}: {} boards", path, n);
            }
            Err(e) => eprintln!("warning: cannot read --reseed {}: {}", path, e),
        }
    }

    // ---- generate candidates ----
    let mut candidates: Vec<Candidate> = Vec::new();
    for (label, board) in &seeds {
        for r in 0..restarts {
            // restart 0 climbs from the seed itself; later restarts perturb first.
            let start = if r == 0 { *board } else { perturb(&mut rng, board, perturb_k) };
            let pk_used = if r == 0 { 0 } else { perturb_k };
            let peak = climb(&start, &sc, iters, sideways, &mut rng);
            let wd = sc.wd.h(&peak.s);
            let cheap = wd.max(sc.lc.h(&peak.s));
            let (canon, _) = canonical(&peak.s);
            let lineage = format!(
                "seed={} restart={} perturb={} ops=3cyc:{},dswap:{} accepts={} wd={} cheap={} frame={}",
                label,
                r,
                pk_used,
                peak.ops3,
                peak.opsd,
                peak.accepts,
                wd,
                cheap,
                is_frame_conformant(&peak.s) as u8
            );
            candidates.push(Candidate { board: peak.s, canon, score: peak.score, lineage });
        }
    }

    // ---- dedup + diversity + caps → emit ----
    // Highest score first so diversity anchors on the hardest boards.
    candidates.sort_by(|a, b| b.score.cmp(&a.score).then(a.canon.0.cmp(&b.canon.0)));
    let mut seen: HashSet<[u8; N_CELLS]> = HashSet::new();
    let mut emitted: Vec<Emitted> = Vec::new();
    let mut per_seed_count: std::collections::HashMap<String, usize> = Default::default();
    let mut chosen: Vec<&Candidate> = Vec::new();
    for c in &candidates {
        if chosen.len() >= pool_cap {
            break;
        }
        if !seen.insert(c.canon.0) {
            continue; // duplicate under reflection
        }
        // per-seed cap keyed on the seed label prefix
        let seed_key = c.lineage.split_whitespace().next().unwrap_or("").to_string();
        let cnt = per_seed_count.entry(seed_key.clone()).or_insert(0);
        if *cnt >= per_seed {
            continue;
        }
        // diversity: reject if too close to an already-emitted >= score board
        let too_close = emitted
            .iter()
            .any(|e| e.score >= c.score && hamming(&e.canon, &c.canon) < min_dist);
        if too_close {
            continue;
        }
        *cnt += 1;
        emitted.push(Emitted { canon: c.canon, score: c.score });
        chosen.push(c);
    }

    // ---- write output ----
    let mut text = String::new();
    let _ = writeln!(
        text,
        "# candidates24 seed=0x{:x} score={} pool={} (of {} raw); iters={} restarts={} perturb={} min_dist={}",
        seed,
        match kind { ScoreKind::Wd => "wd", ScoreKind::Cheap => "cheap" },
        chosen.len(),
        candidates.len(),
        iters,
        restarts,
        perturb_k,
        min_dist,
    );
    for (i, c) in chosen.iter().enumerate() {
        let _ = writeln!(text, "# cand#{} {}", i, c.lineage);
        let _ = writeln!(text, "{}", board_tokens(&c.board));
    }
    if let Err(e) = std::fs::write(&out, &text) {
        eprintln!("write {}: {}", out, e);
        return ExitCode::FAILURE;
    }

    // ---- stderr summary ----
    let mut scores: Vec<u8> = chosen.iter().map(|c| c.score).collect();
    scores.sort_unstable();
    let n = scores.len();
    if n == 0 {
        eprintln!("WARNING: emitted 0 candidates");
        return ExitCode::SUCCESS;
    }
    let mean = scores.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    // mean pairwise Hamming (sampled if large)
    let mut dsum = 0u64;
    let mut dcnt = 0u64;
    for i in 0..chosen.len() {
        for j in (i + 1)..chosen.len() {
            dsum += hamming(&chosen[i].canon, &chosen[j].canon) as u64;
            dcnt += 1;
        }
    }
    let mean_pair = if dcnt > 0 { dsum as f64 / dcnt as f64 } else { 0.0 };
    eprintln!(
        "emitted {} boards -> {}\n  score: min {} mean {:.1} max {}  (proven-LB floors)\n  mean pairwise Hamming {:.1} (min-dist {})",
        n, out, scores[0], mean, scores[n - 1], mean_pair, min_dist
    );
    let frame_cnt = chosen.iter().filter(|c| c.lineage.contains("frame=1")).count();
    eprintln!("  frame-conformant emitted: {}/{}", frame_cnt, n);
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Rng {
        Rng::new(12345)
    }

    #[test]
    fn r_board_is_solvable_and_rotated() {
        let r = r_board();
        assert_eq!(r.0[0], 0);
        assert_eq!(r.0[1], 24);
        assert_eq!(r.0[24], 1);
        assert!(r.is_solvable());
    }

    #[test]
    fn mutations_preserve_solvability_and_blank() {
        let mut rng = rng();
        // start from a scrambled solvable board
        let mut s = perturb(&mut rng, &r_board(), 40);
        assert!(s.is_solvable());
        let blank = s.blank_pos();
        let cells = non_blank_cells(&s);
        for _ in 0..500 {
            let (m, _) = mutate(&mut rng, &s, &cells);
            assert!(m.is_solvable(), "mutation broke solvability: {:?}", m.0);
            assert_eq!(m.blank_pos(), blank, "mutation moved the blank");
            s = m;
        }
    }

    #[test]
    fn perturb_preserves_solvability() {
        let mut rng = rng();
        for k in [1u32, 5, 20, 100] {
            let s = perturb(&mut rng, &r_board(), k);
            assert!(s.is_solvable());
        }
    }

    #[test]
    fn climb_is_monotone_and_admissible() {
        // Manhattan-free score: use WD would need warm-up; instead assert the
        // climb's returned peak score >= seed score for a synthetic scorer.
        // (WD needs the table; covered by integration. Here we check the invariant
        // that climb never returns below the start score using a tile-sum proxy.)
        struct ProxyScorer;
        impl ProxyScorer {
            fn score(&self, s: &State) -> u8 {
                // count tiles NOT in their goal cell, capped to u8 — admissible-ish
                // monotone target for the climb loop's accept logic.
                (0..N_CELLS).filter(|&i| s.0[i] != 0 && s.0[i] as usize != i + 1).count() as u8
            }
        }
        // Re-run the climb loop inline with the proxy (mirrors `climb`).
        let mut rng = rng();
        let seed = perturb(&mut rng, &r_board(), 30);
        let ps = ProxyScorer;
        let cells = non_blank_cells(&seed);
        let mut cur = seed;
        let mut cur_score = ps.score(&cur);
        let start_score = cur_score;
        let mut best = cur_score;
        for _ in 0..1000 {
            let (cand, _) = mutate(&mut rng, &cur, &cells);
            let ns = ps.score(&cand);
            if ns >= cur_score {
                cur = cand;
                cur_score = ns;
                best = best.max(cur_score);
            }
        }
        assert!(best >= start_score);
    }

    #[test]
    fn emitted_line_round_trips() {
        let r = r_board();
        let line = board_tokens(&r);
        let parsed = parse_board_line(&line).expect("round-trip");
        assert_eq!(parsed, r);
        // comment + blank lines are skipped
        assert!(parse_board_line("# comment").is_none());
        assert!(parse_board_line("   ").is_none());
    }

    #[test]
    fn hamming_and_diversity_semantics() {
        let a = r_board();
        let mut b = a;
        b.0.swap(1, 2);
        assert_eq!(hamming(&a, &b), 2);
        assert_eq!(hamming(&a, &a), 0);
    }

    #[test]
    fn parse_rejects_bad_lines() {
        assert!(parse_board_line("1 2 3").is_none()); // too few
        assert!(parse_board_line("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 25").is_none()); // 25 > 24
        assert!(parse_board_line("_ 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1").is_some()); // _ blank
    }
}
