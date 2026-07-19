//! ml_probe — cheap sanity check that the candle `ml` feature links and runs.
//!
//! Validates the whole Metal-linking premise (Step 1 of TRAINING.md) before any
//! puzzle logic is written: pick a device, report which backend, allocate and
//! do one trivial tensor op. Run:
//!   cargo run --release --features ml --bin ml_probe

use puzzle8::puzzle15::ml::device::{device_kind, pick_device};

use candle_core::{DType, Device, Tensor};

fn main() -> candle_core::Result<()> {
    let dev: Device = pick_device()?;
    println!("candle device backend: {}", device_kind(&dev));

    // Trivial tensor op to confirm the backend actually executes kernels.
    let a = Tensor::ones((4, 4), DType::F32, &dev)?;
    let b = (&a + &a)?;
    let sum = b.sum_all()?.to_scalar::<f32>()?;
    println!("2 * ones(4,4) summed = {sum} (expected 32)");
    assert_eq!(sum, 32.0);

    println!("ml_probe OK");
    Ok(())
}
