//! gen_corridors — build the **geodesic-corridor training dataset** (the
//! "general fix" for deep-board value training).
//!
//! Diagnosis (data/corridor_r_compare.txt): V was trained on WD-maximal
//! *endpoint* boards with loose walk-length labels, so it misprices the states
//! real deep solutions pass through (by +15..+52 on the R corridor). The fix:
//! train on **solution-path states** with tight labels.
//!
//! Two modes:
//! - `--mode exact`   — random-walk boards solved **optimally** (zpdb-k6 IDA*,
//!   per-board deadline; calibration: d≤88 within ~20 s). Every path state gets
//!   an EXACT label `rem = len − k`.
//! - `--mode learned` — WD-search constructed deep boards (WD ~100-126) solved
//!   by the trained BWAS solver; labels are near-tight upper bounds (the solver
//!   beats the WD-beam by ~20 there).
//!
//! Output lines: `rem exact_flag t0 t1 … t24` (tiles row-major, blank = 0),
//! `rem ≥ --min-rem` only (the shallow region belongs to the DAVI bootstrap),
//! deduplicated by state.
//!
//!   cargo run --release --features ml,mmap --bin gen_corridors -- \
//!       --mode exact --pdb-dir data --walk-lengths 120,160,200,240 --reps 50 \
//!       --budget-secs 60 --out data/corridors_exact.txt
//!   cargo run --release --features ml,mmap --bin gen_corridors -- \
//!       --mode learned --checkpoint data/ml24_wds2/value_latest.safetensors \
//!       --depths 100,110,120 --boards-per-depth 40 --out data/corridors_deep.txt

use std::collections::HashSet;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use rayon::prelude::*;

use puzzle8::puzzle24::ml::bwas::{search as bwas_search, BwasConfig, BwasOutcome};
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::scramble::Rng;
use puzzle8::puzzle24::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::ml::wdsearch::{construct_deep_boards, Diversity, WdSearchConfig};
use puzzle8::puzzle24::pdb::{ZPatternDb, ZpdbInc};
use puzzle8::puzzle24::search::idastar_inc_ladder;
use puzzle8::puzzle24::search::LadderOutcome;
use puzzle8::puzzle24::state::{Move, State, GOAL};

const ZPDB_FILES_K6: [&str; 4] = ["pdb24_a.zbin", "pdb24_b.zbin", "pdb24_c.zbin", "pdb24_d.zbin"];

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_list(argv: &[String], flag: &str, default: &[u32]) -> Vec<u32> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

/// No-immediate-undo random walk from GOAL (xorshift, mirrors ladder24).
fn random_walk(seed: u64, len: u32) -> State {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = GOAL;
    let mut last: Option<Move> = None;
    for _ in 0..len {
        let opts: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| last.map_or(true, |p: Move| m != p.inverse()))
            .collect();
        let m = opts[(next() as usize) % opts.len()];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

/// Emit one corridor: every state along `moves` from `start` with remaining
/// depth ≥ `min_rem`, as `(rem, state)` pairs (includes the start itself).
fn corridor(start: &State, moves: &[Move], min_rem: usize) -> Vec<(usize, State)> {
    let len = moves.len();
    let mut out = Vec::with_capacity(len + 1);
    let mut s = *start;
    if len >= min_rem {
        out.push((len, s));
    }
    for (k, &m) in moves.iter().enumerate() {
        s = s.apply(m);
        let rem = len - (k + 1);
        if rem >= min_rem {
            out.push((rem, s));
        }
    }
    debug_assert_eq!(s, GOAL);
    out
}

fn write_lines(
    path: &PathBuf,
    rows: &[(usize, bool, State)],
) -> std::io::Result<(usize, usize)> {
    let mut seen: HashSet<State> = HashSet::new();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut written = 0usize;
    let mut dups = 0usize;
    for (rem, exact, s) in rows {
        if !seen.insert(*s) {
            dups += 1;
            continue;
        }
        let tiles: Vec<String> = s.0.iter().map(|t| t.to_string()).collect();
        writeln!(f, "{} {} {}", rem, *exact as u8, tiles.join(" "))?;
        written += 1;
    }
    Ok((written, dups))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let mode = arg(&argv, "--mode", "exact".to_string());
    let out = PathBuf::from(arg(&argv, "--out", format!("data/corridors_{}.txt", mode)));
    let min_rem: usize = arg(&argv, "--min-rem", 30);

    // ---------------- wdbound mode: certificate audit, no solving needed.
    // For every state, `label − WD(state)` PROVABLY bounds its slack
    // (WD ≤ optimal ≤ label). Slack can hide only where this bound is large —
    // in particular it closes the corridor-HEAD question the exact-solve audit
    // can't reach (suffix-optimality never certifies the head; per the
    // 15-puzzle lesson, excess concentrates in the first moves off deep boards).
    if mode == "wdbound" {
        let input = PathBuf::from(arg(&argv, "--in", "data/corridors_deep.txt".to_string()));
        let rows = match puzzle8::puzzle24::ml::corridor::load_file(&input) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {}", e);
                return ExitCode::FAILURE;
            }
        };
        puzzle8::puzzle24::search::WalkingDistanceHeuristic::warm_up();
        use puzzle8::puzzle24::search::Heuristic as _;
        let wd = puzzle8::puzzle24::search::WalkingDistanceHeuristic;
        // Histogram of the slack bound, split by label band (shallow/mid/head).
        let mut bands: [(u32, Vec<usize>); 3] = [(0, Vec::new()), (0, Vec::new()), (0, Vec::new())];
        for &(s, label) in &rows {
            let bound = (label as i32 - wd.h(&s) as i32).max(0) as usize;
            let b = match label as u32 {
                0..=79 => 0,
                80..=99 => 1,
                _ => 2,
            };
            bands[b].0 += 1;
            if bands[b].1.len() <= bound {
                bands[b].1.resize(bound + 1, 0);
            }
            bands[b].1[bound] += 1;
        }
        for (name, (n, hist)) in ["label<80", "label 80-99", "label>=100"].iter().zip(&bands) {
            let sum: usize = hist.iter().enumerate().map(|(b, c)| b * c).sum();
            println!(
                "{}: {} states, mean slack-bound {:.2}, bound histogram:",
                name,
                n,
                sum as f64 / (*n).max(1) as f64
            );
            for (bound, c) in hist.iter().enumerate() {
                if *c > 0 {
                    println!("  bound {:>2}: {:>5}", bound, c);
                }
            }
        }
        return ExitCode::SUCCESS;
    }

    // ---------------- audit mode: measure learned-tier label looseness.
    // Sample states with labels inside the exactly-solvable band, solve each
    // optimally, and report (label − optimal): the slack suffix-shortening
    // would remove. Zero slack ⇒ shortening is pointless; material slack ⇒ do it.
    if mode == "audit" {
        let input = PathBuf::from(arg(&argv, "--in", "data/corridors_deep.txt".to_string()));
        let pdb_dir = PathBuf::from(arg(&argv, "--pdb-dir", "data".to_string()));
        let band_min: usize = arg(&argv, "--band-min", 40);
        let band_max: usize = arg(&argv, "--band-max", 80);
        let sample_n: usize = arg(&argv, "--sample", 60);
        let budget = Duration::from_secs(arg(&argv, "--budget-secs", 120));

        let rows = match puzzle8::puzzle24::ml::corridor::load_file(&input) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let mut band: Vec<(State, usize)> = rows
            .iter()
            .filter(|(_, l)| (*l as usize) >= band_min && (*l as usize) <= band_max)
            .map(|&(s, l)| (s, l as usize))
            .collect();
        // Deterministic spread over the band (strided sample).
        let stride = (band.len() / sample_n.max(1)).max(1);
        band = band.into_iter().step_by(stride).take(sample_n).collect();
        eprintln!(
            "audit: {} sampled states with label in [{}, {}] from {}",
            band.len(),
            band_min,
            band_max,
            input.display()
        );

        let mut dbs = Vec::with_capacity(4);
        for name in &ZPDB_FILES_K6 {
            match ZPatternDb::load_mmap(&pdb_dir.join(name)) {
                Ok(d) => dbs.push(d),
                Err(e) => {
                    eprintln!("error loading {}: {}", name, e);
                    return ExitCode::FAILURE;
                }
            }
        }
        let t0 = Instant::now();
        let results: Vec<(usize, usize)> = band
            .par_iter()
            .filter_map(|&(s, label)| {
                let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
                let deadline = Instant::now() + budget;
                match idastar_inc_ladder(&s, &inc, u8::MAX, Some(deadline)).0 {
                    LadderOutcome::Solved(moves) => Some((label, moves.len())),
                    _ => None,
                }
            })
            .collect();
        let mut slack_hist: Vec<usize> = Vec::new();
        let mut slack_sum = 0usize;
        for &(label, opt) in &results {
            let slack = label.saturating_sub(opt);
            if slack_hist.len() <= slack {
                slack_hist.resize(slack + 1, 0);
            }
            slack_hist[slack] += 1;
            slack_sum += slack;
            debug_assert!(opt <= label, "label {} below optimal {}?!", label, opt);
        }
        println!(
            "audit: {}/{} solved in {:.0}s; mean slack (label - optimal) = {:.2}",
            results.len(),
            band.len(),
            t0.elapsed().as_secs_f64(),
            slack_sum as f64 / results.len().max(1) as f64
        );
        for (slack, n) in slack_hist.iter().enumerate() {
            if *n > 0 {
                println!("  slack {:>2}: {:>4} states", slack, n);
            }
        }
        return ExitCode::SUCCESS;
    }

    let rows: Vec<(usize, bool, State)> = if mode == "exact" {
        // ---------------- exact mode: optimal solves of random-walk boards.
        let pdb_dir = PathBuf::from(arg(&argv, "--pdb-dir", "data".to_string()));
        let walk_lengths = arg_list(&argv, "--walk-lengths", &[120, 160, 200, 240]);
        let reps: u32 = arg(&argv, "--reps", 50);
        let budget = Duration::from_secs(arg(&argv, "--budget-secs", 60));
        let seed: u64 = arg(&argv, "--seed", 1);

        let mut dbs = Vec::with_capacity(4);
        for name in &ZPDB_FILES_K6 {
            match ZPatternDb::load_mmap(&pdb_dir.join(name)) {
                Ok(d) => dbs.push(d),
                Err(e) => {
                    eprintln!("error loading {}: {}", name, e);
                    return ExitCode::FAILURE;
                }
            }
        }
        let boards: Vec<(u64, u32)> = walk_lengths
            .iter()
            .flat_map(|&len| (0..reps).map(move |r| (r, len)))
            .enumerate()
            .map(|(i, (r, len))| (seed.wrapping_mul(0x9E37_79B9).wrapping_add((i as u64) << 17 | r as u64), len))
            .collect();
        eprintln!(
            "exact mode: {} boards (walks {:?} × {} reps), budget {:?}/board",
            boards.len(),
            walk_lengths,
            reps,
            budget
        );

        let t0 = Instant::now();
        let solved: Vec<(usize, Vec<(usize, State)>)> = boards
            .par_iter()
            .filter_map(|&(bseed, len)| {
                let s = random_walk(bseed, len);
                let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
                let deadline = Instant::now() + budget;
                match idastar_inc_ladder(&s, &inc, u8::MAX, Some(deadline)).0 {
                    LadderOutcome::Solved(moves) => {
                        let c = corridor(&s, &moves, min_rem);
                        Some((moves.len(), c))
                    }
                    _ => None, // timeout / unsolvable-walk (never) — skip
                }
            })
            .collect();
        let n_states: usize = solved.iter().map(|(_, c)| c.len()).sum();
        let mut depth_hist: Vec<usize> = Vec::new();
        for (len, _) in &solved {
            if depth_hist.len() <= *len {
                depth_hist.resize(len + 1, 0);
            }
            depth_hist[*len] += 1;
        }
        eprintln!(
            "exact: {}/{} solved in {:.0}s; {} corridor states; depth range {}..{}",
            solved.len(),
            boards.len(),
            t0.elapsed().as_secs_f64(),
            n_states,
            depth_hist.iter().position(|&c| c > 0).unwrap_or(0),
            depth_hist.len().saturating_sub(1),
        );
        solved
            .into_iter()
            .flat_map(|(_, c)| c.into_iter().map(|(rem, s)| (rem, true, s)))
            .collect()
    } else {
        // ---------------- learned mode: BWAS-solve WD-search deep boards.
        let checkpoint = match argv.iter().position(|a| a == "--checkpoint").and_then(|i| argv.get(i + 1)) {
            Some(p) => PathBuf::from(p),
            None => {
                eprintln!("--checkpoint required for --mode learned");
                return ExitCode::FAILURE;
            }
        };
        let hidden: usize = arg(&argv, "--hidden", DEFAULT_HIDDEN);
        let blocks: usize = arg(&argv, "--blocks", DEFAULT_BLOCKS);
        let depths = arg_list(&argv, "--depths", &[100, 110, 120]);
        let per_depth: usize = arg(&argv, "--boards-per-depth", 40);
        let bwas_budget: u64 = arg(&argv, "--bwas-budget", 500_000);
        let weight: f32 = arg(&argv, "--weight", 2.5);
        let width: usize = arg(&argv, "--search-width", 2000);
        let seed: u64 = arg(&argv, "--seed", 7);

        puzzle8::puzzle24::search::WalkingDistanceHeuristic::warm_up();
        let device = if argv.iter().any(|a| a == "--cpu") {
            candle_core::Device::Cpu
        } else {
            pick_device().unwrap_or(candle_core::Device::Cpu)
        };
        let mut vm = VarMap::new();
        let net = match ValueNet::new(VarBuilder::from_varmap(&vm, DType::F32, &device), hidden, blocks)
        {
            Ok(n) => n,
            Err(e) => {
                eprintln!("error building net: {}", e);
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = vm.load(&checkpoint) {
            eprintln!("error loading {}: {}", checkpoint.display(), e);
            return ExitCode::FAILURE;
        }
        eprintln!(
            "learned mode: depths {:?} × {} boards, bwas {} @ w{}, device {}",
            depths,
            per_depth,
            bwas_budget,
            weight,
            device_kind(&device)
        );

        let mut rows: Vec<(usize, bool, State)> = Vec::new();
        let mut rng = Rng::new(seed);
        let t0 = Instant::now();
        let (mut n_solved, mut n_boards) = (0usize, 0usize);
        for &depth in &depths {
            let cfg = WdSearchConfig {
                width: width.max(per_depth * 4),
                target_depth: depth as usize,
                node_budget: 0,
                diversity: Diversity::Stochastic { random_slots: 64, temperature: 8.0 },
            };
            let boards = construct_deep_boards(per_depth, &cfg, &mut rng);
            for (b, _wd) in boards {
                n_boards += 1;
                let bcfg =
                    BwasConfig { weight, batch_size: 2000, node_budget: bwas_budget };
                let value_of =
                    |states: &[State]| net.values(states, &device).expect("value net forward");
                match bwas_search(&b, &bcfg, value_of) {
                    BwasOutcome::Solved { moves, .. } => {
                        n_solved += 1;
                        for (rem, s) in corridor(&b, &moves, min_rem) {
                            rows.push((rem, false, s));
                        }
                    }
                    BwasOutcome::BudgetExceeded { .. } => {}
                }
            }
            eprintln!(
                "  depth {}: cumulative {}/{} solved, {} states, {:.0}s",
                depth,
                n_solved,
                n_boards,
                rows.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        rows
    };

    match write_lines(&out, &rows) {
        Ok((written, dups)) => {
            eprintln!("wrote {} states ({} dups dropped) -> {}", written, dups, out.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("write {}: {}", out.display(), e);
            ExitCode::FAILURE
        }
    }
}
