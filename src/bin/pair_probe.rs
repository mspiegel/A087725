//! pair_probe — Phase-0 gate for the target-conditioned pair-distance net.
//!
//! The reduction: tile-relabeling is a graph automorphism, so
//!   dist(x, t) = dist(relabel_t(x), I_b),   b = t's blank cell,
//! where I_b is the canonical identity board with blank at b. The current net is
//! the b=24 (GOAL) slice. This probe checks two things with NO training:
//!
//!   A. dist(x, R) = V(rot180(x))  — R = rot180(GOAL), so the b=0-corner target
//!      reduces to the existing net on rotated boards. If this tracks the true
//!      backward distance along R's corridor, the reduction plumbing is correct
//!      and the free (symmetry-only) backward heuristic for R is strong.
//!   B. V(I_b) for canonical targets across blank-classes — the "self-distance
//!      offset" a b=24 net wrongly reports (should be 0 for a perfect pair-net).
//!      Large, class-dependent offsets prove a conditioned net is needed for the
//!      GENERAL (non-symmetric) case.
//!
//!   cargo run --release --features ml --bin pair_probe -- \
//!       --checkpoint data/ml24_frame2/value_latest.safetensors --hidden 1024 --blocks 6

use std::process::ExitCode;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::corridor::r156_states;
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

/// 180° board rotation (pure position permutation, cell c <-> 24-c). A graph
/// automorphism that maps GOAL <-> R.
fn rot180(s: &State) -> State {
    let mut c = [0u8; N_CELLS];
    for i in 0..N_CELLS {
        c[i] = s.0[N_CELLS - 1 - i];
    }
    State(c)
}

/// Canonical identity board with the blank at cell `b`: tiles 1..24 fill the
/// other cells in row-major order. `identity_b(24) == GOAL`.
fn identity_b(b: usize) -> State {
    let mut c = [0u8; N_CELLS];
    let mut t = 1u8;
    for (cell, slot) in c.iter_mut().enumerate() {
        if cell == b {
            *slot = 0;
        } else {
            *slot = t;
            t += 1;
        }
    }
    State(c)
}

/// Fix parity to solvable by swapping two non-blank tiles (blank stays put).
fn make_solvable(mut s: State) -> State {
    if !s.is_solvable() {
        let (mut a, mut b) = (usize::MAX, usize::MAX);
        for (i, &v) in s.0.iter().enumerate() {
            if v != 0 {
                if a == usize::MAX {
                    a = i;
                } else {
                    b = i;
                    break;
                }
            }
        }
        s.0.swap(a, b);
    }
    debug_assert!(s.is_solvable());
    s
}

fn class_name(b: usize) -> &'static str {
    let (r, c) = (b / 5, b % 5);
    let corner = (r == 0 || r == 4) && (c == 0 || c == 4);
    let edge = r == 0 || r == 4 || c == 0 || c == 4;
    if corner {
        "corner"
    } else if edge {
        "edge"
    } else if b == 12 {
        "center"
    } else {
        "interior"
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let checkpoint = match argv.iter().position(|a| a == "--checkpoint").and_then(|i| argv.get(i + 1)) {
        Some(p) => p.clone(),
        None => {
            eprintln!("--checkpoint required");
            return ExitCode::FAILURE;
        }
    };
    let hidden: usize = arg(&argv, "--hidden", DEFAULT_HIDDEN);
    let blocks: usize = arg(&argv, "--blocks", DEFAULT_BLOCKS);

    assert_eq!(identity_b(24), GOAL, "identity_b(24) must equal GOAL (convention check)");
    assert_eq!(rot180(&GOAL), r(), "rot180(GOAL) must equal R");

    let device = if argv.iter().any(|a| a == "--cpu") {
        candle_core::Device::Cpu
    } else {
        pick_device().unwrap_or(candle_core::Device::Cpu)
    };
    let mut vm = VarMap::new();
    let net = match ValueNet::new(VarBuilder::from_varmap(&vm, DType::F32, &device), hidden, blocks) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("build net: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = vm.load(&checkpoint) {
        eprintln!("load {checkpoint}: {e}");
        return ExitCode::FAILURE;
    }
    let value_of = |s: &[State]| net.values(s, &device).expect("forward");
    println!("device: {}, checkpoint: {}", device_kind(&device), checkpoint);

    // ---- Measurement A: rotation reduction along R's corridor.
    // corridor[i] = (s_i, rem-to-GOAL = 156-i); dist(s_i, R) = i; the reduction
    // says dist(s_i, R) = V(rot180(s_i)).
    let corridor = r156_states();
    let rotated: Vec<State> = corridor.iter().map(|&(s, _)| rot180(&s)).collect();
    let vrot = value_of(&rotated);
    println!("\n== A. dist(s_i, R) vs V(rot180(s_i))  (should track i) ==");
    println!("{:>4} {:>10} {:>8}", "i", "V(rot180)", "err");
    let mut abserr = 0.0f32;
    for (i, v) in vrot.iter().enumerate() {
        let true_d = i as f32;
        abserr += (v - true_d).abs();
        if i % 12 == 0 || i == corridor.len() - 1 {
            println!("{:>4} {:>10.1} {:>+8.1}", i, v, v - true_d);
        }
    }
    println!("mean |V(rot180(s_i)) - i| = {:.1}  (small ⇒ reduction valid + net calibrated on R-frame)", abserr / vrot.len() as f32);

    // ---- Measurement B: per-class self-distance offset V(I_b).
    // For a perfect pair-net conditioned on b, dist(I_b, I_b)=0. The b=24 net
    // instead reports V(I_b)=dist(I_b, GOAL) — the offset it must unlearn.
    println!("\n== B. V(I_b) self-distance offset per blank-class (0 for a perfect pair-net) ==");
    println!("{:>4} {:>10} {:>10}", "b", "class", "V(I_b)");
    for &b in &[24usize, 0, 4, 20, 2, 22, 7, 17, 12] {
        let ib = make_solvable(identity_b(b));
        let v = value_of(&[ib])[0];
        println!("{:>4} {:>10} {:>10.1}", b, class_name(b), v);
    }
    println!("(b=24 ≈ 0 confirms the net; large offsets elsewhere ⇒ the general case needs a b-conditioned net)");
    ExitCode::SUCCESS
}

/// R board, inline (avoid pulling eval just for this).
fn r() -> State {
    let mut c = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        c[i] = (25 - i) as u8;
    }
    State(c)
}
