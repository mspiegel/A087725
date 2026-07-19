//! verify_antipodes15 — independently verify a trained value net's solutions to
//! the 17 known depth-80 antipodes.
//!
//! For each antipode: run the learned BWAS solver, then **re-apply the returned
//! move sequence to the board from scratch and assert it reaches GOAL** (an
//! independent check of BWAS's path, not just its `Solved` claim), and confirm
//! the length is 80 (optimal, since the antipodes are depth-80 by definition).
//! Runs on the CPU backend for determinism / cross-backend reproduction.
//!
//!   cargo run --release --features ml --bin verify_antipodes15 -- \
//!       --checkpoint data/ml15_big/value_latest.safetensors --hidden 1024 --blocks 4 \
//!       [--budget 500000] [--weight 2.0] [--sbatch 4000]

use std::process::ExitCode;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle15::enumerate::antipodes::load_ranks;
use puzzle8::puzzle15::ml::bwas::{search, BwasConfig, BwasOutcome};
use puzzle8::puzzle15::ml::value_net::ValueNet;
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::state::{State, DIAMETER, GOAL};

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let checkpoint = match argv
        .iter()
        .position(|a| a == "--checkpoint")
        .and_then(|i| argv.get(i + 1))
    {
        Some(p) => p.clone(),
        None => {
            eprintln!("error: --checkpoint required");
            return ExitCode::FAILURE;
        }
    };
    let hidden: usize = arg(&argv, "--hidden", 1024);
    let blocks: usize = arg(&argv, "--blocks", 4);
    let budget: u64 = arg(&argv, "--budget", 500_000);
    let weight: f32 = arg(&argv, "--weight", 2.0);
    let sbatch: usize = arg(&argv, "--sbatch", 4000);
    let path = arg(&argv, "--antipodes", "data/pdb15_antipodes.txt".to_string());

    let device = Device::Cpu; // deterministic, independent of the training backend
    let mut vm = VarMap::new();
    let net = match ValueNet::new(
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
    if let Err(e) = vm.load(&checkpoint) {
        eprintln!("error loading {checkpoint}: {e}");
        return ExitCode::FAILURE;
    }
    println!("loaded {checkpoint} (hidden {hidden}, blocks {blocks}); CPU; weight {weight}, budget {budget}");

    let value_of = |states: &[State]| net.values(states, &device).expect("value net forward");
    let cfg = BwasConfig {
        weight,
        batch_size: sbatch,
        node_budget: budget,
    };

    let antipodes: Vec<State> = match load_ranks(std::path::Path::new(&path)) {
        Ok(ranks) => ranks.into_iter().map(unrank).collect(),
        Err(e) => {
            eprintln!("error loading antipodes {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("loaded {} antipodes", antipodes.len());
    let distinct: std::collections::HashSet<_> = antipodes.iter().collect();
    println!(
        "DISTINCT antipode states after unrank: {}/{}",
        distinct.len(),
        antipodes.len()
    );
    for (i, s) in antipodes.iter().enumerate() {
        println!("  [{i}] blank@{} {:?}", s.blank_pos(), &s.0[..6]);
    }
    if argv.iter().any(|a| a == "--load-only") {
        return ExitCode::SUCCESS;
    }
    println!();

    let mut solved = 0usize;
    let mut verified_optimal = 0usize;
    for (i, s) in antipodes.iter().enumerate() {
        assert_ne!(*s, GOAL, "antipode {i} is GOAL — bad load");
        match search(s, &cfg, value_of) {
            BwasOutcome::Solved {
                moves,
                nodes_expanded,
            } => {
                solved += 1;
                // Independent re-application from the original board.
                let mut b = *s;
                for m in &moves {
                    b = b.apply(*m);
                }
                let reaches_goal = b == GOAL;
                let optimal = moves.len() == DIAMETER as usize;
                if reaches_goal && optimal {
                    verified_optimal += 1;
                }
                let prefix: String = moves
                    .iter()
                    .take(12)
                    .map(|m| format!("{m:?}").chars().next().unwrap())
                    .collect();
                println!(
                    "antipode {:>2}: len {:>3}  nodes {:>7}  reaches_GOAL={}  optimal={}  moves[..12]={}",
                    i,
                    moves.len(),
                    nodes_expanded,
                    reaches_goal,
                    optimal,
                    prefix
                );
            }
            BwasOutcome::BudgetExceeded { nodes_expanded } => {
                println!("antipode {i:>2}: UNSOLVED within budget (expanded {nodes_expanded})");
            }
        }
    }
    println!(
        "\nsolved {}/{}, INDEPENDENTLY VERIFIED optimal (reaches GOAL in {} moves) {}/{}",
        solved,
        antipodes.len(),
        DIAMETER,
        verified_optimal,
        antipodes.len()
    );
    ExitCode::SUCCESS
}
