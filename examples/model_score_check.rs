//! Correctness gate: score the validation sample with the ported Rust evaluator
//! and compare to sklearn's predict_proba (in val_scores.txt). Must match within
//! tight tolerance before the model is trusted at scale.
//!
//! Run: cargo run --release --example model_score_check -- MODEL.txt VAL_SCORES.txt

#[path = "common/tree_model.rs"]
mod tree_model;

use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path: PathBuf = std::env::args().nth(1).ok_or("usage: model_score_check MODEL VAL")?.into();
    let val_path: PathBuf = std::env::args().nth(2).ok_or("need VAL")?.into();

    let m = tree_model::TreeModel::load(&model_path)?;
    eprintln!("loaded model: nfeat={}", m.nfeat);

    let text = fs::read_to_string(&val_path)?;
    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    let mut n = 0u64;
    for line in text.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() != m.nfeat + 1 { continue; }
        let x: Vec<f64> = toks[..m.nfeat].iter().map(|t| t.parse().unwrap()).collect();
        let expected: f64 = toks[m.nfeat].parse().unwrap();
        let got = m.score(&x);
        let d = (got - expected).abs();
        if d > max_diff { max_diff = d; }
        sum_diff += d;
        n += 1;
    }
    eprintln!("compared {n} rows");
    eprintln!("max abs diff : {max_diff:.3e}");
    eprintln!("mean abs diff: {:.3e}", sum_diff / n as f64);
    if max_diff < 1e-5 {
        println!("PASS — Rust scorer matches sklearn (max diff {max_diff:.2e})");
    } else {
        println!("FAIL — max diff {max_diff:.2e} exceeds 1e-5");
        std::process::exit(1);
    }
    Ok(())
}
