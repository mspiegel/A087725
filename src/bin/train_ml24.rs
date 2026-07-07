//! train_ml24 — adversarial generator/solver co-training for the 24-puzzle.
//!
//! Trains a DAVI value-net solver and a REINFORCE board generator against each
//! other, with the generator rewarded by regret vs a fast beam-search suboptimal
//! baseline. Validated on mid-depth boards (excess-over-optimal via idastar).
//!
//!   cargo run --release --features ml --bin train_ml24 -- \
//!       [--rounds 40] [--solver-steps 3000] [--gen-steps 400] \
//!       [--batch 4096] [--gen-frac 0.5] [--hidden 1024] [--blocks 6] \
//!       [--solver-k 50] [--gen-k 40] \
//!       [--solver-weight 2.5] [--solver-sbatch 2000] \
//!       [--solver-budget 30000 (generator reward)] [--eval-budget 300000] \
//!       [--baseline wd|manhattan] [--beam-width 1000] [--beam-budget 2000000] \
//!       [--eval-every 2] [--holdout 100] [--depth-min 20] [--depth-max 50] \
//!       [--eval-mode middepth|deep] [--eval-depth-min 60] [--eval-depth-max 120] [--eval-with-r] \
//!       [--gen-reward wd|regret] [--adv-lambda 1.0] [--entropy-beta 0.01] \
//!       [--curriculum --k-start 50 --k-end 160] \
//!       [--gen-source policy|wdsearch] [--search-width 8192] [--search-budget 0] \
//!       [--search-random-slots 0 --search-temp 5.0] [--supervise-walk] [--dry-run] \
//!       [--out data/ml24] [--seed 1] [--metal|--cpu] [--resume] [--fresh-generator] [--quiet]

use std::path::PathBuf;
use std::process::ExitCode;

use puzzle8::puzzle24::ml::alternate::{run, AlternationConfig, Curriculum, EvalSpec};
use puzzle8::puzzle24::ml::beam::BeamConfig;
use puzzle8::puzzle24::ml::bwas::BwasConfig;
use puzzle8::puzzle24::ml::davi::DaviConfig;
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::{DeepEvalConfig, DeepHoldout, EvalConfig, LabelHeuristic};
use puzzle8::puzzle24::ml::generator::{
    BaselineHeuristic, Generator, GeneratorConfig, GeneratorReward, GeneratorSource,
};
use puzzle8::puzzle24::ml::profile;
use puzzle8::puzzle24::ml::scramble::Rng;
use puzzle8::puzzle24::ml::wdsearch::{Diversity, WdSearchConfig};
use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::state::State;

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();

    let rounds: u32 = arg(&argv, "--rounds", 40);
    let solver_steps: u32 = arg(&argv, "--solver-steps", 3000);
    // Defaults reflect the 15-puzzle lessons: batch 1024 (the GPU saturates
    // there — a bigger batch is more compute for the same throughput and fewer
    // updates); gen-steps kept modest (each runs the learned BWAS solve, which is
    // the gen-phase cost).
    let gen_steps: u32 = arg(&argv, "--gen-steps", 100);
    let batch: usize = arg(&argv, "--batch", 1024);
    let gen_frac: f32 = arg(&argv, "--gen-frac", 0.5);
    let hidden: usize = arg(&argv, "--hidden", 1024);
    let blocks: usize = arg(&argv, "--blocks", 6);
    let solver_k: u32 = arg(&argv, "--solver-k", 50);
    let gen_k: u32 = arg(&argv, "--gen-k", 40);
    let solver_weight: f32 = arg(&argv, "--solver-weight", 2.5);
    let solver_sbatch: usize = arg(&argv, "--solver-sbatch", 2000);
    // The generator-reward BWAS wants a LOW budget (cheap, fails fast); the eval
    // BWAS wants a HIGH budget (else the metric is search-limited and reads ~0
    // even when the value function is good — the 15-puzzle lesson). Keep them
    // separate.
    let solver_budget: u64 = arg(&argv, "--solver-budget", 30_000);
    let eval_budget: u64 = arg(&argv, "--eval-budget", 300_000);
    let beam_width: usize = arg(&argv, "--beam-width", 1000);
    let beam_budget: u64 = arg(&argv, "--beam-budget", 2_000_000);
    let eval_every: u32 = arg(&argv, "--eval-every", 2);
    let holdout: usize = arg(&argv, "--holdout", 100);
    let depth_min: u32 = arg(&argv, "--depth-min", 20);
    let depth_max: u32 = arg(&argv, "--depth-max", 50);
    // In-loop eval mode: mid-depth (excess-over-optimal via IDA*, ≤~d50) or deep
    // (excess-over-beam) for the scale-to-R regime, where the mid-depth holdout
    // saturates and stops being informative. Deep reuses the generator beam +
    // baseline; --eval-depth-* set the deep holdout walk-length range.
    let eval_mode = arg(&argv, "--eval-mode", "middepth".to_string());
    let eval_depth_min: u32 = arg(&argv, "--eval-depth-min", 60);
    let eval_depth_max: u32 = arg(&argv, "--eval-depth-max", 120);
    let eval_with_r = argv.iter().any(|a| a == "--eval-with-r");
    // The in-loop deep holdout source. `walk` (default) folds to optimal~60 and is
    // BLIND to the deep regime; `wdsearch` builds genuinely-deep WD-boards so the
    // metric can actually see the depth the training targets.
    let eval_holdout = match arg(&argv, "--eval-holdout", "walk".to_string()).as_str() {
        "wdsearch" => DeepHoldout::WdSearch {
            width: arg(&argv, "--eval-search-width", 4000),
            depth: arg(&argv, "--eval-search-depth", 120),
        },
        _ => DeepHoldout::Walk,
    };
    let out: String = arg(&argv, "--out", "data/ml24".to_string());
    let seed: u64 = arg(&argv, "--seed", 1);
    let baseline = match arg(&argv, "--baseline", "wd".to_string()).as_str() {
        "manhattan" => BaselineHeuristic::Manhattan,
        _ => BaselineHeuristic::Wd,
    };
    // Generator reward: WdDepth (hack-proof depth composite, default) or the
    // legacy Regret. --adv-lambda is the composite's λ; --curriculum ramps the
    // walk length k (depth) from --k-start to --k-end over the run.
    let gen_reward = match arg(&argv, "--gen-reward", "wd".to_string()).as_str() {
        "regret" => GeneratorReward::Regret,
        _ => GeneratorReward::WdDepth,
    };
    let entropy_beta: f32 = arg(&argv, "--entropy-beta", 0.01);
    let adv_lambda: f32 = arg(&argv, "--adv-lambda", 1.0);
    let curriculum = if argv.iter().any(|a| a == "--curriculum") {
        Some(Curriculum { k_start: arg(&argv, "--k-start", 50), k_end: arg(&argv, "--k-end", 160) })
    } else {
        None
    };
    // Generation source: the learned policy (default) or the WD-search deep-board
    // constructor. WdSearch reads its target depth from the curriculum k_max, so
    // use it with --curriculum (--k-start/--k-end).
    let gen_source = arg(&argv, "--gen-source", "policy".to_string());
    let search_width: usize = arg(&argv, "--search-width", 8192);
    let search_budget: u64 = arg(&argv, "--search-budget", 0);
    let search_random_slots: usize = arg(&argv, "--search-random-slots", 0);
    let search_temp: f32 = arg(&argv, "--search-temp", 5.0);
    let source = if gen_source == "wdsearch" {
        let diversity = if search_random_slots > 0 {
            Diversity::Stochastic { random_slots: search_random_slots, temperature: search_temp }
        } else {
            Diversity::TopK
        };
        GeneratorSource::WdSearch(WdSearchConfig {
            width: search_width,
            target_depth: gen_k as usize, // overridden per-round by k_max (curriculum)
            node_budget: search_budget,
            diversity,
        })
    } else {
        GeneratorSource::PolicyRollout
    };
    let verbose = !argv.iter().any(|a| a == "--quiet");
    let resume = argv.iter().any(|a| a == "--resume");
    // Resume the solver but start the generator fresh (e.g. after changing the
    // generator reward — a collapsed old-reward policy is a poor start).
    let reset_generator = argv.iter().any(|a| a == "--fresh-generator");
    // Supervise WdSearch boards with their walk-length (the search's GOAL→board
    // walk reversed is an achievable solution) instead of the Bellman bootstrap.
    let supervise_search_depth = argv.iter().any(|a| a == "--supervise-walk");
    // Strategy #1: WD-residual parameterization (net learns optimal-WD; V=WD+raw).
    let residual = argv.iter().any(|a| a == "--residual");
    // Geodesic-corridor supervision (the "general fix"): comma-separated
    // gen_corridors dataset files + the batch fraction they occupy.
    let corridor_data: Vec<PathBuf> = argv
        .iter()
        .position(|a| a == "--corridor-data")
        .and_then(|i| argv.get(i + 1))
        .map(|s| s.split(',').map(|t| PathBuf::from(t.trim())).collect())
        .unwrap_or_default();
    let corridor_frac: f32 = arg(&argv, "--corridor-frac", 0.25);
    if argv.iter().any(|a| a == "--profile") {
        profile::set_enabled(true);
    }

    // Warm up Walking Distance up front (the eval labels and the WD beam baseline
    // both use it) so the ~590 MB load / BFS is honest, not hidden mid-training.
    print!("warming up Walking Distance table... ");
    let t = std::time::Instant::now();
    WalkingDistanceHeuristic::warm_up();
    println!("ready in {:.1}s", t.elapsed().as_secs_f64());

    // Default to the GPU (Metal-if-available, CPU fallback); `--cpu` forces CPU.
    let device = if argv.iter().any(|a| a == "--cpu") {
        candle_core::Device::Cpu
    } else {
        match pick_device() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not init device: {}", e);
                return ExitCode::FAILURE;
            }
        }
    };
    println!(
        "device: {}, rounds: {}, solver-steps/round: {}, gen-steps/round: {}, batch: {}, hidden: {}, blocks: {}",
        device_kind(&device),
        rounds,
        solver_steps,
        gen_steps,
        batch,
        hidden,
        blocks,
    );

    let solver_bwas =
        BwasConfig { weight: solver_weight, batch_size: solver_sbatch, node_budget: solver_budget };
    let beam = BeamConfig { width: beam_width, max_depth: 220, node_budget: beam_budget };
    // High-budget eval, decoupled from the cheap generator-reward budget so the
    // metric reflects the value function, not the search budget.
    let eval_bwas =
        BwasConfig { weight: solver_weight, batch_size: solver_sbatch, node_budget: eval_budget };
    let eval_spec = if eval_mode == "deep" {
        EvalSpec::Deep(DeepEvalConfig {
            bwas: eval_bwas,
            beam,
            baseline,
            holdout_n: holdout,
            depth_min: eval_depth_min,
            depth_max: eval_depth_max,
            seed: 0x24_5EED,
            include_r: eval_with_r,
            holdout: eval_holdout,
        })
    } else {
        EvalSpec::MidDepth(EvalConfig {
            bwas: eval_bwas,
            holdout_n: holdout,
            depth_min,
            depth_max,
            seed: 0x24_5EED,
            optimal_max_bound: (depth_max as u8).saturating_add(5),
            label_heuristic: LabelHeuristic::LcWd,
        })
    };
    let cfg = AlternationConfig {
        rounds,
        solver_steps_per_round: solver_steps,
        generator_steps_per_round: gen_steps,
        solver_batch: batch,
        generator_frac: gen_frac,
        davi: DaviConfig {
            k_max: solver_k,
            hidden,
            blocks,
            lr: 1e-3,
            target_sync_every: 1000,
            residual,
        },
        generator: GeneratorConfig {
            k_max: gen_k,
            hidden,
            lr: 1e-3,
            baseline_decay: 0.95,
            solver_bwas,
            beam,
            baseline,
            fail_penalty: 400.0,
            reward: gen_reward,
            entropy_beta,
            adv_lambda,
            source,
        },
        eval_every,
        eval: eval_spec,
        curriculum,
        checkpoint_dir: PathBuf::from(out),
        seed,
        verbose,
        resume,
        reset_generator,
        supervise_search_depth,
        corridor_data,
        corridor_frac,
    };

    // --dry-run: build one generator pool and print its WD stats, then exit — a
    // CLI-level smoke of the (WdSearch) board source without any training.
    if argv.iter().any(|a| a == "--dry-run") {
        let mut g = match Generator::new(&cfg.generator, device.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("dry-run: generator init failed: {}", e);
                return ExitCode::FAILURE;
            }
        };
        if let Some(c) = &cfg.curriculum {
            g.set_k_max(c.k_end); // deepest curriculum level
        }
        let t = std::time::Instant::now();
        let pool = g
            .sample_pool(2000, |s: &[State]| vec![0.0f32; s.len()], &mut Rng::new(1))
            .expect("dry-run sample_pool");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let wds: Vec<u8> = pool.iter().map(|s| WalkingDistanceHeuristic.h(s)).collect();
        let max_wd = wds.iter().copied().max().unwrap_or(0);
        let mean_wd = if wds.is_empty() {
            0.0
        } else {
            wds.iter().map(|&w| w as f64).sum::<f64>() / wds.len() as f64
        };
        println!(
            "dry-run: {} boards, max WD {}, mean WD {:.1}, in {:.0} ms",
            pool.len(),
            max_wd,
            mean_wd,
            ms
        );
        return ExitCode::SUCCESS;
    }

    match run(&cfg, device) {
        Ok(()) => {
            println!("training complete; checkpoints + metrics in {}", cfg.checkpoint_dir.display());
            if profile::is_enabled() {
                print!("\n{}", profile::report());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("training error: {}", e);
            ExitCode::FAILURE
        }
    }
}
