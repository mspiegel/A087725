//! corridor_r — replicate the published 156-move solution to `R` in-repo and
//! profile our heuristics along its corridor.
//!
//! Construction (Rokicki/Hannanov, forum.cubeman.org node 238): the 90°-rotated
//! goal `W = ρ(GOAL)` solves optimally in 78 STM; rotations are graph
//! automorphisms, so the same solution with every move rotated 90° solves
//! `R = ρ²(GOAL)` to `W`. Concatenating gives a 156-move solution to `R`,
//! replay-verified here. If `optimal(R) = 156` (Rokicki's belief; proven
//! ∈ [152,156]) this path is a geodesic and state `k` has true remaining
//! distance `156 − k` — the only tight deep-corridor ground truth we have.
//!
//! Profiles along the corridor (and optionally along our own best solution,
//! `--our-moves FILE`): WD(s), V(s), V-error vs remaining, WD-rise counts —
//! diagnosing WHY our search never finds a ≤156 path (misranking vs pruning vs
//! uphill-intolerance).
//!
//!   cargo run --release --features ml --bin corridor_r -- \
//!       --checkpoint data/ml24_wds2/value_latest.safetensors [--hidden 1024] \
//!       [--blocks 6] [--residual] [--our-moves FILE] [--cpu]

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::r_board;
use puzzle8::puzzle24::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::state::{Move, State, GOAL};

/// Optimal 78-STM solution of `W = ρ(GOAL)` (90° clockwise-rotated goal),
/// found by `solve24 --heuristic select --parallel` 2026-07-06 (12,189 nodes,
/// 2.2 s, replay-verified). Note the spiral macro-structure.
const W_SOLUTION: &str = "U U U U R R R D D D D L L L U U U U R R R R D D D D L L L L \
U U U U R R R R D D D L L L U U R R D D L L U U U R R R D D D D L L L L U U U U R R R R D D D D";

/// `W = ρ(GOAL)`, 90° clockwise: top row → right column, blank → cell 20.
fn w_board() -> State {
    let mut c = [0u8; 25];
    for row in 0..5u8 {
        for col in 0..5u8 {
            let cell = (row * 5 + col) as usize;
            if (row, col) == (4, 0) {
                c[cell] = 0; // blank (GOAL's blank at (4,4) rotates here)
            } else {
                c[cell] = 21 + row - 5 * col;
            }
        }
    }
    State(c)
}

fn parse_moves(s: &str) -> Vec<Move> {
    s.split_whitespace()
        .map(|t| match t {
            "U" => Move::Up,
            "D" => Move::Down,
            "L" => Move::Left,
            "R" => Move::Right,
            other => panic!("bad move token {other:?}"),
        })
        .collect()
}

/// Rotate a move direction 90° clockwise / counter-clockwise.
fn cw(m: Move) -> Move {
    match m {
        Move::Up => Move::Right,
        Move::Right => Move::Down,
        Move::Down => Move::Left,
        Move::Left => Move::Up,
    }
}
fn ccw(m: Move) -> Move {
    cw(cw(cw(m)))
}

/// Replay `moves` from `start`; return the end state (panics on illegal move).
fn replay(start: &State, moves: &[Move]) -> State {
    let mut s = *start;
    for &m in moves {
        s = s.apply(m);
    }
    s
}

/// States along a path, including both endpoints.
fn path_states(start: &State, moves: &[Move]) -> Vec<State> {
    let mut out = Vec::with_capacity(moves.len() + 1);
    let mut s = *start;
    out.push(s);
    for &m in moves {
        s = s.apply(m);
        out.push(s);
    }
    out
}

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Profile one path: per-state WD and V, errors vs (len − k), uphill counts.
fn profile(name: &str, states: &[State], values: &[f32]) {
    let len = states.len() - 1;
    println!();
    println!("== {name} (length {len}) ==");
    println!(
        "{:>4} {:>5} {:>4} {:>7} {:>7} {:>7}",
        "k", "rem", "WD", "rem-WD", "V", "V-rem"
    );
    let mut wd_prev = 0u8;
    let mut wd_rises = 0usize;
    let mut max_verr = 0f32;
    let mut max_verr_k = 0usize;
    for (k, s) in states.iter().enumerate() {
        let rem = (len - k) as i32;
        let wd = WalkingDistanceHeuristic.h(s);
        if k > 0 && wd > wd_prev {
            wd_rises += 1;
        }
        wd_prev = wd;
        let v = values[k];
        let verr = v - rem as f32;
        if verr.abs() > max_verr.abs() {
            max_verr = verr;
            max_verr_k = k;
        }
        if k % 4 == 0 || k == len {
            println!(
                "{:>4} {:>5} {:>4} {:>7} {:>7.1} {:>+7.1}",
                k,
                rem,
                wd,
                rem - wd as i32,
                v,
                verr
            );
        }
    }
    println!(
        "-- {name}: WD rises on {wd_rises}/{len} steps; max |V-rem| = {max_verr:+.1} at k={max_verr_k}"
    );
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let checkpoint: Option<PathBuf> = argv
        .iter()
        .position(|a| a == "--checkpoint")
        .and_then(|i| argv.get(i + 1))
        .map(PathBuf::from);
    let hidden: usize = arg(&argv, "--hidden", DEFAULT_HIDDEN);
    let blocks: usize = arg(&argv, "--blocks", DEFAULT_BLOCKS);

    WalkingDistanceHeuristic::warm_up();

    // ---- build + verify the 156-move solution to R.
    let r = r_board();
    let w = w_board();
    let s78 = parse_moves(W_SOLUTION);
    assert_eq!(s78.len(), 78);
    assert_eq!(replay(&w, &s78), GOAL, "W-solution does not solve W");

    // Conjugate: try both rotation directions; keep the one that maps R -> W.
    let conj_cw: Vec<Move> = s78.iter().map(|&m| cw(m)).collect();
    let conj_ccw: Vec<Move> = s78.iter().map(|&m| ccw(m)).collect();
    let first_half = if replay(&r, &conj_cw) == w {
        conj_cw
    } else if replay(&r, &conj_ccw) == w {
        conj_ccw
    } else {
        eprintln!("neither rotation conjugate maps R -> W; conjugation bug");
        return ExitCode::FAILURE;
    };
    let mut r156 = first_half;
    r156.extend_from_slice(&s78);
    assert_eq!(r156.len(), 156);
    assert_eq!(replay(&r, &r156), GOAL, "156 replay failed");
    println!("156-move solution to R constructed and replay-verified (R -> W -> GOAL).");
    let compact: String = r156
        .iter()
        .map(|m| match m {
            Move::Up => "U ",
            Move::Down => "D ",
            Move::Left => "L ",
            Move::Right => "R ",
        })
        .collect();
    println!("moves: {}", compact.trim());

    // ---- value net (optional; without --checkpoint only WD is profiled).
    let device = if argv.iter().any(|a| a == "--cpu") {
        candle_core::Device::Cpu
    } else {
        pick_device().unwrap_or(candle_core::Device::Cpu)
    };
    let states_156 = path_states(&r, &r156);
    let values_156: Vec<f32> = match &checkpoint {
        Some(path) => {
            let mut vm = VarMap::new();
            let mut net = match ValueNet::new(
                VarBuilder::from_varmap(&vm, DType::F32, &device),
                hidden,
                blocks,
            ) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("error building net: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(e) = vm.load(path) {
                eprintln!("error loading {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
            if argv.iter().any(|a| a == "--residual") {
                net.set_residual(true);
            }
            println!(
                "device: {}, checkpoint: {}",
                device_kind(&device),
                path.display()
            );
            net.values(&states_156, &device).expect("value net forward")
        }
        None => vec![0.0; states_156.len()],
    };
    profile("published-156 corridor", &states_156, &values_156);

    // ---- optional: our own solution for comparison.
    if let Some(p) = argv
        .iter()
        .position(|a| a == "--our-moves")
        .and_then(|i| argv.get(i + 1))
    {
        let text = std::fs::read_to_string(p).expect("read --our-moves");
        let ours = parse_moves(&text);
        assert_eq!(replay(&r, &ours), GOAL, "--our-moves does not solve R");
        let states_ours = path_states(&r, &ours);
        let values_ours: Vec<f32> = match &checkpoint {
            Some(path) => {
                let mut vm = VarMap::new();
                let mut net = ValueNet::new(
                    VarBuilder::from_varmap(&vm, DType::F32, &device),
                    hidden,
                    blocks,
                )
                .expect("net");
                vm.load(path).expect("load");
                if argv.iter().any(|a| a == "--residual") {
                    net.set_residual(true);
                }
                net.values(&states_ours, &device)
                    .expect("value net forward")
            }
            None => vec![0.0; states_ours.len()],
        };
        profile(
            &format!("our-{} solution", ours.len()),
            &states_ours,
            &values_ours,
        );

        // Shared states between the two paths (beyond the endpoints).
        let set: std::collections::HashSet<State> = states_156.iter().copied().collect();
        let shared = states_ours.iter().filter(|s| set.contains(s)).count();
        // First divergence index.
        let div = states_156
            .iter()
            .zip(states_ours.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(states_156.len().min(states_ours.len()));
        println!();
        println!(
            "paths share {} states (of {} / {}); first divergence at move {}",
            shared,
            states_156.len(),
            states_ours.len(),
            div
        );
    }
    ExitCode::SUCCESS
}
