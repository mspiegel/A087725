//! solve_r — solve the canonical hard `R` board with a trained value net via
//! **anytime weighted-A\*** (a descending weight ladder with incumbent pruning),
//! to get the shortest solution the current solver can find without retraining.
//!
//!   cargo run --release --features ml --bin solve_r -- \
//!       --checkpoint data/ml24_wds3/value_latest.safetensors [--hidden 1024] \
//!       [--blocks 6] [--weights 2.5,2.0,1.5,1.2] [--budget 5000000] [--batch 2000] [--cpu]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::bidirectional::{
    bidir_search, bidir_search_ff, manhattan_to, BidirConfig, FfConfig,
};
use puzzle8::puzzle24::ml::bwas::{anytime_search, BwasOutcome};
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::r_board;
use puzzle8::puzzle24::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic, WalkingDistanceTo};
use puzzle8::puzzle24::state::{State, GOAL};

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
    let budget: u64 = arg(&argv, "--budget", 5_000_000);
    let batch: usize = arg(&argv, "--batch", 2000);
    let weights: Vec<f32> = arg(&argv, "--weights", "2.5,2.0,1.5,1.2".to_string())
        .split(',')
        .filter_map(|w| w.trim().parse().ok())
        .collect();

    print!("warming up Walking Distance table... ");
    let t = Instant::now();
    WalkingDistanceHeuristic::warm_up();
    println!("ready in {:.1}s", t.elapsed().as_secs_f64());

    let device = if argv.iter().any(|a| a == "--cpu") {
        candle_core::Device::Cpu
    } else {
        match pick_device() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not init device: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    let mut varmap = VarMap::new();
    let mut net =
        match ValueNet::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), hidden, blocks) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("error building net: {e}");
                return ExitCode::FAILURE;
            }
        };
    match &checkpoint {
        Some(path) => match varmap.load(path) {
            Ok(()) => println!("loaded checkpoint: {}", path.display()),
            Err(e) => {
                eprintln!("error loading checkpoint {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
        },
        None => {
            eprintln!("--checkpoint required");
            return ExitCode::FAILURE;
        }
    }
    if argv.iter().any(|a| a == "--residual") {
        net.set_residual(true); // V = WD + raw (must match how the net was trained)
        println!("residual mode: V = WD + raw");
    }
    println!(
        "device: {}, weights: {:?}, budget/weight: {}",
        device_kind(&device),
        weights,
        budget
    );

    let r = r_board();
    // --wd-floor: clamp V to the admissible WD floor at inference —
    // V' = max(V, WD(s)). Corrects label-ceiling saturation (a net trained on
    // labels <= ~126 predicts below the provable floor on deeper states, which
    // flattens deep ranking); where V saturates, ranking falls back to WD.
    let wd_floor = argv.iter().any(|a| a == "--wd-floor");
    if wd_floor {
        println!("wd-floor: V = max(V, WD)");
    }
    let value_of = |states: &[State]| {
        let mut vs = net.values(states, &device).expect("value net forward");
        if wd_floor {
            for (v, s) in vs.iter_mut().zip(states) {
                *v = v.max(WalkingDistanceHeuristic.h(s) as f32);
            }
        }
        vs
    };
    // Direct extrapolation check: V(R) vs WD(R)=140 (residual mode) / vs optimal
    // 152. A tight residual net should read ~152; a loose one reads much higher.
    println!(
        "V(R) = {:.1}  (WD(R)=140, opt>=152; residual={})",
        value_of(&[r])[0],
        net.is_residual()
    );
    let t = Instant::now();

    // Front-to-front MITM: each beam aims at the OTHER frontier (pure Manhattan
    // front-to-front, no net), so the two converge in the middle. Memory-bounded.
    if argv.iter().any(|a| a == "--bidir-ff") {
        let cfg = FfConfig {
            width: arg(&argv, "--width", 20_000),
            max_layers: arg(&argv, "--max-layers", 300),
            node_budget: budget,
            n_anchors: arg(&argv, "--ff-anchors", 16),
            w_base: arg(&argv, "--ff-base-weight", 2.0),
            w_ff: arg(&argv, "--ff-weight", 1.0),
        };
        // Backward base. Default WD-to-R. `--bwd-rot180` (Phase-0.5 GATE, R-only):
        // use the STRONG learned heuristic V(rot180(x)) = dist(x, R) via the exact
        // rot180(GOAL)=R automorphism — as strong as the forward net, for free.
        let bwd_rot180 = argv.iter().any(|a| a == "--bwd-rot180");
        let wd_to_r = if cfg.w_base != 0.0 && !bwd_rot180 {
            print!("building WD-to-R backward table... ");
            let tb = Instant::now();
            let wd = WalkingDistanceTo::new(&r);
            println!("ready in {:.1}s", tb.elapsed().as_secs_f64());
            Some(wd)
        } else {
            None
        };
        let rot180 = |s: &State| {
            let mut c = [0u8; 25];
            for i in 0..25 {
                c[i] = s.0[24 - i];
            }
            State(c)
        };
        println!(
            "FF-MITM: width {}, max_layers {}, budget {}, anchors {}, w_base {}, w_ff {}, bwd {}",
            cfg.width,
            cfg.max_layers,
            cfg.node_budget,
            cfg.n_anchors,
            cfg.w_base,
            cfg.w_ff,
            if bwd_rot180 { "V(rot180)" } else { "WD-to-R" }
        );
        let result = bidir_search_ff(
            &r,
            |ss: &[State]| value_of(ss), // fwd base = V (wd-floor-clamped if set)
            |ss: &[State]| {
                if bwd_rot180 {
                    // dist(x, R) = V(rot180(x)); strong learned backward guidance.
                    let rot: Vec<State> = ss.iter().map(rot180).collect();
                    value_of(&rot)
                } else {
                    match &wd_to_r {
                        Some(wd) => ss.iter().map(|s| wd.h(s) as f32).collect(),
                        None => vec![0.0; ss.len()],
                    }
                }
            },
            &cfg,
        );
        let secs = t.elapsed().as_secs_f64();
        match result {
            Some(res) => {
                let mut s = r;
                for &m in &res.moves {
                    s = s.apply(m);
                }
                let ok = s == GOAL && (res.len as usize == res.moves.len());
                println!(
                    "R solved (FF-MITM): {} moves = {} fwd + {} bwd (WD LB 140, LB 152), {:.0}s, replay_ok={}",
                    res.len, res.g_fwd, res.g_bwd, secs, ok
                );
                if argv.iter().any(|a| a == "--print-moves") {
                    let s: Vec<&str> = res
                        .moves
                        .iter()
                        .map(|m| match m {
                            puzzle8::puzzle24::state::Move::Up => "U",
                            puzzle8::puzzle24::state::Move::Down => "D",
                            puzzle8::puzzle24::state::Move::Left => "L",
                            puzzle8::puzzle24::state::Move::Right => "R",
                        })
                        .collect();
                    println!("moves: {}", s.join(" "));
                }
            }
            None => println!("R unsolved (FF-MITM: beams never met) in {secs:.0}s"),
        }
        return ExitCode::SUCCESS;
    }

    // Bidirectional / meet-in-the-middle: forward beam from R guided to GOAL by the
    // learned V; backward beam from GOAL guided to R by Manhattan-to-R; stitch on
    // the first shared state. Halves the effective search depth (~76 vs 152).
    if argv.iter().any(|a| a == "--bidir") {
        let cfg = BidirConfig {
            width: arg(&argv, "--width", 20_000),
            max_layers: arg(&argv, "--max-layers", 300),
            node_budget: budget, // reuse --budget as total children generated
            fwd_weight: arg(&argv, "--fwd-weight", 2.0),
            bwd_weight: arg(&argv, "--bwd-weight", 2.0),
        };
        // Backward heuristic: strong WD-to-R by default (retargetable Walking
        // Distance, ~16s one-time build), or Manhattan-to-R with --bwd-manhattan.
        let wd_to_r = if argv.iter().any(|a| a == "--bwd-manhattan") {
            None
        } else {
            print!("building WD-to-R backward table... ");
            let tb = Instant::now();
            let wd = WalkingDistanceTo::new(&r);
            println!("ready in {:.1}s", tb.elapsed().as_secs_f64());
            Some(wd)
        };
        println!(
            "MITM: width {}, max_layers {}, budget {}, fwd_w {}, bwd_w {}, bwd_heur {}",
            cfg.width,
            cfg.max_layers,
            cfg.node_budget,
            cfg.fwd_weight,
            cfg.bwd_weight,
            if wd_to_r.is_some() { "WD-to-R" } else { "manhattan" }
        );
        let result = bidir_search(
            &r,
            |ss: &[State]| value_of(ss),
            |ss: &[State]| {
                ss.iter()
                    .map(|s| match &wd_to_r {
                        Some(wd) => wd.h(s) as f32,
                        None => manhattan_to(s, &r) as f32,
                    })
                    .collect()
            },
            &cfg,
        );
        let secs = t.elapsed().as_secs_f64();
        match result {
            Some(res) => {
                let mut s = r;
                for &m in &res.moves {
                    s = s.apply(m);
                }
                let ok = s == GOAL && (res.len as usize == res.moves.len());
                println!(
                    "R solved (MITM): {} moves = {} fwd + {} bwd (WD LB 140, LB 152), {:.0}s, replay_ok={}",
                    res.len, res.g_fwd, res.g_bwd, secs, ok
                );
            }
            None => println!("R unsolved (MITM: beams never met) in {secs:.0}s"),
        }
        return ExitCode::SUCCESS;
    }
    let outcome = anytime_search(&r, &weights, batch, budget, value_of, |s: &State| {
        WalkingDistanceHeuristic.h(s)
    });
    let secs = t.elapsed().as_secs_f64();

    match outcome {
        BwasOutcome::Solved { moves, nodes_expanded } => {
            // Self-certify: replay reaches GOAL.
            let mut s = r;
            for &m in &moves {
                s = s.apply(m);
            }
            let ok = s == puzzle8::puzzle24::state::GOAL;
            println!(
                "R solved: {} moves (WD LB 140, known-best 156, LB 152), {} nodes, {:.0}s, replay_ok={}",
                moves.len(),
                nodes_expanded,
                secs,
                ok
            );
        }
        BwasOutcome::BudgetExceeded { nodes_expanded } => {
            println!("R unsolved ({nodes_expanded} nodes, {secs:.0}s)");
        }
    }
    ExitCode::SUCCESS
}
