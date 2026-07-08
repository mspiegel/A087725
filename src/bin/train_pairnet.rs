//! train_pairnet — train the target-conditioned pair-distance value net.
//!
//! `V(x | b) ≈ dist(x, I_b)` for the canonical target with blank-class `b`. This
//! turns the solver into an ANY-BOARD-PAIR distance net: backward search toward
//! any target S becomes forward search on `relabel_S(state)` conditioned on S's
//! blank-class. Supervised regression on exact-ish pair triples
//! (`gen_corridors --pairs-out`), warm-started from a forward checkpoint (the
//! zero-init b_embed reproduces the forward net until it trains).
//!
//!   cargo run --release --features ml --bin train_pairnet -- \
//!       --pairs data/pairs_frame.txt --warm data/ml24_frame2/value_latest.safetensors \
//!       --out data/ml24_pair --hidden 1024 --blocks 6 --steps 30000 --batch 1024

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use puzzle8::puzzle24::ml::corridor::r156_verr;
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::encoding::encode_batch;
use puzzle8::puzzle24::ml::scramble::Rng;
use puzzle8::puzzle24::ml::value_net::{onehot, ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::state::{State, N_CELLS};

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Load pair triples `b label t0..t24`.
fn load_pairs(path: &Path) -> Result<Vec<(usize, f32, State)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tok: Vec<&str> = line.split_whitespace().collect();
        if tok.len() != 2 + N_CELLS {
            return Err(format!("{}:{}: {} tokens", path.display(), ln + 1, tok.len()));
        }
        let b: usize = tok[0].parse().map_err(|e| format!("{}:{} b: {}", path.display(), ln + 1, e))?;
        let label: f32 = tok[1].parse().map_err(|e| format!("{}:{} label: {}", path.display(), ln + 1, e))?;
        let mut c = [0u8; N_CELLS];
        for (i, t) in tok[2..].iter().enumerate() {
            c[i] = t.parse().map_err(|e| format!("{}:{} tile: {}", path.display(), ln + 1, e))?;
        }
        out.push((b, label, State(c)));
    }
    Ok(out)
}

/// Warm-start: copy the trunk tensors from `from` into `vm` by name, leaving vars
/// absent from the file (the zero-init `b_embed`) untouched. (candle's
/// `VarMap::load` errors on such vars, so we can't use it directly.)
fn warm_start(vm: &VarMap, from: &Path, device: &Device) -> candle_core::Result<usize> {
    let loaded = candle_core::safetensors::load(from, device)?;
    let mut copied = 0;
    for (name, var) in vm.data().lock().unwrap().iter() {
        if let Some(t) = loaded.get(name) {
            var.set(t)?;
            copied += 1;
        }
    }
    Ok(copied)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let pair_files: Vec<PathBuf> = argv
        .iter()
        .position(|a| a == "--pairs")
        .and_then(|i| argv.get(i + 1))
        .map(|s| s.split(',').map(|t| PathBuf::from(t.trim())).collect())
        .unwrap_or_default();
    if pair_files.is_empty() {
        eprintln!("--pairs required");
        return ExitCode::FAILURE;
    }
    let warm: Option<PathBuf> =
        argv.iter().position(|a| a == "--warm").and_then(|i| argv.get(i + 1)).map(PathBuf::from);
    let out = PathBuf::from(arg(&argv, "--out", "data/ml24_pair".to_string()));
    let hidden: usize = arg(&argv, "--hidden", DEFAULT_HIDDEN);
    let blocks: usize = arg(&argv, "--blocks", DEFAULT_BLOCKS);
    let steps: usize = arg(&argv, "--steps", 30_000);
    let batch: usize = arg(&argv, "--batch", 1024);
    let lr: f64 = arg(&argv, "--lr", 5e-4);
    let eval_every: usize = arg(&argv, "--eval-every", 2000);
    let seed: u64 = arg(&argv, "--seed", 1);

    // WD warm-up for the R156-OOD forward probe (values_cond is raw, but the
    // r156 states use no WD; keep for parity if any consumer needs it).
    let device = if argv.iter().any(|a| a == "--cpu") {
        Device::Cpu
    } else {
        pick_device().unwrap_or(Device::Cpu)
    };

    // ---- data
    let mut all: Vec<(usize, f32, State)> = Vec::new();
    for p in &pair_files {
        match load_pairs(p) {
            Ok(mut v) => {
                eprintln!("loaded {} pairs from {}", v.len(), p.display());
                all.append(&mut v);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                return ExitCode::FAILURE;
            }
        }
    }
    if all.len() < 100 {
        eprintln!("too few pairs: {}", all.len());
        return ExitCode::FAILURE;
    }
    // Deterministic shuffle + 5% held-out.
    let mut rng = Rng::new(seed);
    for i in (1..all.len()).rev() {
        all.swap(i, rng.gen_range(0, i as u32) as usize);
    }
    let n_hold = (all.len() / 20).max(200).min(all.len() / 2);
    let (hold, train) = all.split_at(n_hold);
    eprintln!("train {} / held-out {} pairs; device {}", train.len(), hold.len(), device_kind(&device));

    // ---- net (conditioned) + warm start
    let vm = VarMap::new();
    let net = match ValueNet::new_conditioned(VarBuilder::from_varmap(&vm, DType::F32, &device), hidden, blocks) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("build net: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Some(w) = &warm {
        match warm_start(&vm, w, &device) {
            Ok(c) => eprintln!("warm-started {} trunk tensors from {} (b_embed fresh)", c, w.display()),
            Err(e) => {
                eprintln!("warm-start {}: {}", w.display(), e);
                return ExitCode::FAILURE;
            }
        }
    }
    let mut opt = match AdamW::new(vm.all_vars(), ParamsAdamW { lr, ..Default::default() }) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("optimizer: {}", e);
            return ExitCode::FAILURE;
        }
    };
    std::fs::create_dir_all(&out).ok();

    // held-out eval by label band
    let eval_hold = |net: &ValueNet| -> (f32, f32, f32) {
        let bands = [(0u32, 79u32), (80, 119), (120, 200)];
        let mut sse = [0f64; 3];
        let mut n = [0usize; 3];
        for chunk in hold.chunks(4096) {
            let states: Vec<State> = chunk.iter().map(|&(_, _, s)| s).collect();
            let bs: Vec<usize> = chunk.iter().map(|&(b, _, _)| b).collect();
            let preds = net.values_cond(&states, &bs, &device).expect("cond forward");
            for ((_, label, _), p) in chunk.iter().zip(preds.iter()) {
                let bi = bands.iter().position(|&(lo, hi)| *label as u32 >= lo && *label as u32 <= hi).unwrap_or(2);
                sse[bi] += ((p - label) as f64).powi(2);
                n[bi] += 1;
            }
        }
        let rmse = |i: usize| (sse[i] / n[i].max(1) as f64).sqrt() as f32;
        (rmse(0), rmse(1), rmse(2))
    };

    let t0 = Instant::now();
    let mut loss_acc = 0.0f32;
    for step in 1..=steps {
        let mut states = Vec::with_capacity(batch);
        let mut bs = Vec::with_capacity(batch);
        let mut labels = Vec::with_capacity(batch);
        for _ in 0..batch {
            let idx = rng.gen_range(0, (train.len() - 1) as u32) as usize;
            let (b, label, s) = train[idx];
            states.push(s);
            bs.push(b);
            labels.push(label);
        }
        let x = match encode_batch(&states, &device) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("encode: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let oh = onehot(&bs, &device).expect("onehot");
        let preds = net.forward_cond_tensor(&x, &oh).expect("forward");
        let target = Tensor::from_vec(labels, (batch, 1), &device).expect("target");
        let l = loss::mse(&preds, &target).expect("mse");
        opt.backward_step(&l).expect("step");
        loss_acc += l.to_scalar::<f32>().expect("scalar");

        if step % eval_every == 0 || step == steps {
            let (r0, r1, r2) = eval_hold(&net);
            let (od, om, ol) = r156_verr(&|s: &[State]| {
                net.values_cond(s, &vec![24usize; s.len()], &device).expect("cond")
            });
            eprintln!(
                "step {:>6}  train_mse {:.3}  heldout_rmse [<80 {:.1} | 80-119 {:.1} | 120+ {:.1}]  R156-OOD(b24) [{:+.1} {:+.1} {:+.1}]  {:.0}s",
                step,
                loss_acc / eval_every as f32,
                r0, r1, r2, od, om, ol,
                t0.elapsed().as_secs_f64()
            );
            loss_acc = 0.0;
            if let Err(e) = vm.save(out.join("value_latest.safetensors")) {
                eprintln!("save: {}", e);
            }
        }
    }
    eprintln!("done; checkpoint in {}", out.display());
    ExitCode::SUCCESS
}
