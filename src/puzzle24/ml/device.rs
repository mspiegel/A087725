//! Device selection for the candle-based ML tools.
//!
//! Prefers the Metal GPU backend on Apple Silicon (compiled in only on macOS
//! via the `ml` feature's target-specific `candle-core` dependency), falling
//! back to CPU everywhere else — or on macOS if Metal initialization fails.

use candle_core::Device;

/// Pick the best available compute device: Metal if present, else CPU.
///
/// `Device::metal_if_available` already does the try-Metal-else-CPU dance, so
/// this is a thin, named wrapper that also reports which backend was chosen.
pub fn pick_device() -> candle_core::Result<Device> {
    let dev = Device::metal_if_available(0)?;
    Ok(dev)
}

/// Human-readable backend name for logging (`"metal"` / `"cpu"` / `"cuda"`).
pub fn device_kind(dev: &Device) -> &'static str {
    if dev.is_metal() {
        "metal"
    } else if dev.is_cuda() {
        "cuda"
    } else {
        "cpu"
    }
}
