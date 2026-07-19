//! Per-depth histogram of the solve cache: how many boards are stored at each
//! depth. Used to see which layers are complete.
//!
//! Run: `cargo run --release --features "mmap parallel" --example cache_depth_hist -- [CACHE]`

use puzzle8::puzzle15::enumerate::cache;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    let c = cache::load(&cache_path)?;
    let mut hist = [0u64; 256];
    for &d in c.values() {
        hist[d as usize] += 1;
    }
    let lo = (0..256).find(|&i| hist[i] > 0).unwrap();
    let hi = (0..256).rev().find(|&i| hist[i] > 0).unwrap();
    let total: u64 = hist.iter().sum();
    println!("total cache entries: {total}");
    println!("depth |       count");
    for d in lo..=hi {
        println!("  {d:>3} | {:>11}", hist[d]);
    }
    Ok(())
}
