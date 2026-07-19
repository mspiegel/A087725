//! solve_ff — solve an ARBITRARY board via front-to-front MITM using the
//! target-conditioned pair-distance net for BOTH directions:
//!   forward (board → GOAL):    V(x | b=24)          = dist(x, GOAL)
//!   backward (GOAL → board):   V(relabel_board(x) | b_board) = dist(x, board)
//! One net, any board pair — the general any-target backward heuristic. With
//! `--bwd-manhattan` the backward base falls back to Manhattan-to-board, for the
//! head-to-head that demonstrates the learned backward's value.
//!
//!   cargo run --release --features ml --bin solve_ff -- \
//!       --checkpoint data/ml24_pair/value_latest.safetensors --hidden 1024 --blocks 6 \
//!       [--position "<25 tokens>"] [--bwd-manhattan] [--width 20000] \
//!       [--ff-base-weight 2.0] [--ff-weight 0.5] [--budget 40000000]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::bidirectional::{bidir_search_ff, manhattan_to, FfConfig};
use puzzle8::puzzle24::ml::bwas::{anytime_search, BwasOutcome};
use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::ml::corridor::relabel_to_ib;
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::state::{State, GOAL, N_CELLS};

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// R board (default target).
fn r_board() -> State {
    let mut c = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        c[i] = (25 - i) as u8;
    }
    State(c)
}

fn parse_position(s: &str) -> Option<State> {
    let mut c = [0u8; N_CELLS];
    let mut n = 0;
    for tok in s.split_whitespace() {
        if n >= N_CELLS {
            return None;
        }
        c[n] = if tok == "_" || tok == "." { 0 } else { tok.parse().ok()? };
        n += 1;
    }
    if n == N_CELLS {
        Some(State(c))
    } else {
        None
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let checkpoint = match argv.iter().position(|a| a == "--checkpoint").and_then(|i| argv.get(i + 1)) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("--checkpoint (conditioned pair net) required");
            return ExitCode::FAILURE;
        }
    };
    let hidden: usize = arg(&argv, "--hidden", DEFAULT_HIDDEN);
    let blocks: usize = arg(&argv, "--blocks", DEFAULT_BLOCKS);
    let start = match argv.iter().position(|a| a == "--position").and_then(|i| argv.get(i + 1)) {
        Some(p) => match parse_position(p) {
            Some(s) => s,
            None => {
                eprintln!("bad --position");
                return ExitCode::FAILURE;
            }
        },
        None => r_board(),
    };
    if !start.is_solvable() {
        eprintln!("start not solvable");
        return ExitCode::FAILURE;
    }
    let bwd_manhattan = argv.iter().any(|a| a == "--bwd-manhattan");
    let cfg = FfConfig {
        width: arg(&argv, "--width", 20_000),
        max_layers: arg(&argv, "--max-layers", 300),
        node_budget: arg(&argv, "--budget", 40_000_000),
        n_anchors: arg(&argv, "--ff-anchors", 16),
        w_base: arg(&argv, "--ff-base-weight", 2.0),
        w_ff: arg(&argv, "--ff-weight", 0.5),
    };

    let device = if argv.iter().any(|a| a == "--cpu") {
        candle_core::Device::Cpu
    } else {
        pick_device().unwrap_or(candle_core::Device::Cpu)
    };
    let mut vm = VarMap::new();
    let net = match ValueNet::new_conditioned(VarBuilder::from_varmap(&vm, DType::F32, &device), hidden, blocks) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("build net: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = vm.load(&checkpoint) {
        eprintln!("load {}: {}", checkpoint.display(), e);
        return ExitCode::FAILURE;
    }
    // Optional SEPARATE forward net (e.g. ml24_frame2) — avoids the confound of
    // using the pair net's (accurate-but-poor-for-search) forward slice. Backward
    // still uses the conditioned pair net.
    let fwd_net = match argv.iter().position(|a| a == "--fwd-checkpoint").and_then(|i| argv.get(i + 1)) {
        Some(fc) => {
            let mut fvm = VarMap::new();
            let fnet = ValueNet::new(VarBuilder::from_varmap(&fvm, DType::F32, &device), hidden, blocks)
                .expect("build fwd net");
            fvm.load(fc).expect("load fwd checkpoint");
            Some(fnet)
        }
        None => None,
    };
    let b_start = start.0.iter().position(|&v| v == 0).unwrap();
    println!(
        "device: {}, target-blank {}, fwd {}, bwd {}, w_base {}, w_ff {}",
        device_kind(&device),
        b_start,
        if fwd_net.is_some() { "separate-net" } else { "pair-net(b=24)" },
        if bwd_manhattan { "manhattan" } else { "pair-net" },
        cfg.w_base,
        cfg.w_ff
    );

    let fwd = |ss: &[State]| match &fwd_net {
        Some(fnet) => fnet.values(ss, &device).expect("fwd"),
        None => net.values_cond(ss, &vec![24usize; ss.len()], &device).expect("fwd"),
    };
    let start_for_bwd = start;
    let bwd = |ss: &[State]| -> Vec<f32> {
        if bwd_manhattan {
            ss.iter().map(|s| manhattan_to(s, &start_for_bwd) as f32).collect()
        } else {
            // dist(x, start) = V(relabel_start(x) | b_start).
            let rel: Vec<State> = ss.iter().map(|s| relabel_to_ib(s, &start_for_bwd).0).collect();
            net.values_cond(&rel, &vec![b_start; ss.len()], &device).expect("bwd")
        }
    };

    // Forward-only baseline: anytime weighted-A* from `start` to GOAL with the
    // SAME forward heuristic (no backward beam) + WD admissible incumbent prune.
    if argv.iter().any(|a| a == "--forward-only") {
        WalkingDistanceHeuristic::warm_up();
        let weights = [2.5f32, 2.0];
        let budget: u64 = arg(&argv, "--fo-budget", 8_000_000);
        println!("forward-only: anytime weights {weights:?}, budget/weight {budget}");
        let t = Instant::now();
        let out = anytime_search(&start, &weights, 2000, budget, fwd, |s: &State| {
            WalkingDistanceHeuristic.h(s)
        });
        let secs = t.elapsed().as_secs_f64();
        match out {
            BwasOutcome::Solved { moves, nodes_expanded } => {
                let mut s = start;
                for &m in &moves {
                    s = s.apply(m);
                }
                println!(
                    "forward-only solved: {} moves, {} nodes, {:.0}s, replay_ok={}",
                    moves.len(),
                    nodes_expanded,
                    secs,
                    s == GOAL
                );
            }
            BwasOutcome::BudgetExceeded { nodes_expanded } => {
                println!("forward-only unsolved ({nodes_expanded} nodes, {secs:.0}s)")
            }
        }
        return ExitCode::SUCCESS;
    }

    let t = Instant::now();
    let res = bidir_search_ff(&start, fwd, bwd, &cfg);
    let secs = t.elapsed().as_secs_f64();
    match res {
        Some(r) => {
            let mut s = start;
            for &m in &r.moves {
                s = s.apply(m);
            }
            let ok = s == GOAL && r.len as usize == r.moves.len();
            println!(
                "solved: {} moves = {} fwd + {} bwd, {:.0}s, replay_ok={}",
                r.len, r.g_fwd, r.g_bwd, secs, ok
            );
        }
        None => println!("unsolved (beams never met) in {secs:.0}s"),
    }
    ExitCode::SUCCESS
}
