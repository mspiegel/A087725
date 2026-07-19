//! ml_bench — GPU-tuning micro-benchmark for the value-net hot path.
//!
//! Answers two questions before committing to a long training run:
//!   1. Roofline: what fraction of the device's peak FLOP/s does a bare matmul
//!      of our shape reach? (Isolates candle/Metal's matmul ceiling from our
//!      code structure.)
//!   2. Norm A/B: how much do the per-block LayerNorm / RmsNorm ops cost vs a
//!      no-norm block? (candle 0.9 has no fused Metal layer-norm, so ours is a
//!      pile of primitive-op kernels — this quantifies that tax.)
//!
//!   cargo run --release --features ml --bin ml_bench -- \
//!       [--metal] [--hidden 1024] [--batch 1024] [--iters 100] [--peak 6.0]

use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{layer_norm_no_bias, linear, VarBuilder, VarMap};

use puzzle8::puzzle15::ml::device::{device_kind, pick_device};

/// Hand-written RmsNorm from primitive ops (Metal-compatible, unlike candle's
/// fused `rms_norm`): `x / sqrt(mean(x^2) + eps) * weight`.
fn manual_rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let d = x.dim(D::Minus1)? as f64;
    let ms = (x.sqr()?.sum_keepdim(D::Minus1)? / d)?;
    let xn = x.broadcast_div(&(ms + eps)?.sqrt()?)?;
    xn.broadcast_mul(weight)
}

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Warm up once, sync, then time `iters` calls and sync at the end. Returns
/// milliseconds per iteration. `f` must return a tensor so its work is enqueued.
fn bench<F: FnMut() -> Result<Tensor>>(dev: &Device, iters: usize, mut f: F) -> Result<f64> {
    let _ = f()?;
    dev.synchronize()?;
    let t = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(f()?);
    }
    // Force materialization of the final result, then wait for the whole queue.
    if let Some(t) = last {
        let _ = t.sum_all()?.to_scalar::<f32>()?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let hidden: usize = arg(&argv, "--hidden", 1024);
    let batch: usize = arg(&argv, "--batch", 1024);
    let iters: usize = arg(&argv, "--iters", 100);
    let peak_tflops: f64 = arg(&argv, "--peak", 6.0);

    let device = if argv.iter().any(|a| a == "--metal") {
        match pick_device() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not init Metal device: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        Device::Cpu
    };

    match run(&device, hidden, batch, iters, peak_tflops) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(dev: &Device, h: usize, b: usize, iters: usize, peak_tflops: f64) -> Result<()> {
    println!(
        "device: {}, hidden: {}, batch: {}, iters: {}, assumed peak: {:.1} TFLOP/s (fp32)",
        device_kind(dev),
        h,
        b,
        iters,
        peak_tflops
    );

    // ---- 1. Matmul roofline: [b,h] x [h,h] -> [b,h] (2*b*h*h FLOP) ----
    println!("\n── matmul roofline (achieved GFLOP/s vs peak) ──");
    for &m in &[b, 4 * b] {
        let x = Tensor::randn(0f32, 1f32, (m, h), dev)?;
        let w = Tensor::randn(0f32, 1f32, (h, h), dev)?;
        let ms = bench(dev, iters, || x.matmul(&w))?;
        let gflop = 2.0 * m as f64 * h as f64 * h as f64 / 1e9;
        let gflops = gflop / (ms / 1000.0);
        let pct = 100.0 * gflops / (peak_tflops * 1000.0);
        println!(
            "  [{m:>5},{h}] x [{h},{h}]  {ms:>7.3} ms/iter  {gflops:>7.0} GFLOP/s  ({pct:>4.1}% of peak)"
        );
    }

    // ---- 2. Per-op cost on [b,h] ----
    println!("\n── per-op forward on [{b},{h}] (ms/iter) ──");
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let lin = linear(h, h, vb.pp("lin"))?;
    let ln = layer_norm_no_bias(h, 1e-5, vb.pp("ln"))?;
    let rms_w = Tensor::ones(h, DType::F32, dev)?;
    let x = Tensor::randn(0f32, 1f32, (b, h), dev)?;

    let t_lin = bench(dev, iters, || lin.forward(&x))?;
    let t_ln = bench(dev, iters, || ln.forward(&x))?;
    let t_rn = bench(dev, iters, || manual_rmsnorm(&x, &rms_w, 1e-5))?;
    let t_relu = bench(dev, iters, || x.relu())?;
    println!("  linear(h->h)         {t_lin:>7.3}");
    println!("  layer_norm_no_bias   {:>7.3}   ({:.2}x a linear)", t_ln, t_ln / t_lin);
    println!("  manual_rmsnorm       {:>7.3}   ({:.2}x a linear)", t_rn, t_rn / t_lin);
    println!("  relu                 {t_relu:>7.3}");

    // ---- 3. Residual-block A/B (the real block shape: 2 linears + 2 norms) ----
    println!("\n── residual block forward on [{b},{h}] (ms/iter) ──");
    let l1 = linear(h, h, vb.pp("b_l1"))?;
    let l2 = linear(h, h, vb.pp("b_l2"))?;
    let n1 = layer_norm_no_bias(h, 1e-5, vb.pp("b_n1"))?;
    let n2 = layer_norm_no_bias(h, 1e-5, vb.pp("b_n2"))?;

    let block_ln = bench(dev, iters, || {
        let hh = n1.forward(&l1.forward(&x)?)?.relu()?;
        let hh = n2.forward(&l2.forward(&hh)?)?;
        (&x + hh)?.relu()
    })?;
    let block_rn = bench(dev, iters, || {
        let hh = manual_rmsnorm(&l1.forward(&x)?, &rms_w, 1e-5)?.relu()?;
        let hh = manual_rmsnorm(&l2.forward(&hh)?, &rms_w, 1e-5)?;
        (&x + hh)?.relu()
    })?;
    let block_none = bench(dev, iters, || {
        let hh = l1.forward(&x)?.relu()?;
        let hh = l2.forward(&hh)?;
        (&x + hh)?.relu()
    })?;
    println!("  block w/ LayerNorm (current)  {block_ln:>7.3}");
    println!("  block w/ manual RmsNorm        {block_rn:>7.3}");
    println!("  block, no norm                 {block_none:>7.3}");
    let ln_tax = 100.0 * (block_ln - block_none) / block_ln;
    let block_speedup = block_ln / block_rn;
    println!(
        "  => LayerNorm is {ln_tax:.0}% of the block; switching to manual RmsNorm makes the block {block_speedup:.2}x faster"
    );

    Ok(())
}
