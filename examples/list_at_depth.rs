//! List ranks of all cache entries at a given depth, optionally filtered by
//! blank cell. One rank per line — designed for piping into other tools.
//!
//! Run: `cargo run --release --features "mmap parallel" --example list_at_depth -- DEPTH [BLANK] [CACHE]`

use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::rank::unrank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let depth: u8 = std::env::args().nth(1)
        .ok_or("usage: list_at_depth DEPTH [BLANK] [CACHE]")?
        .parse()?;
    let blank_filter: Option<u8> = std::env::args().nth(2)
        .and_then(|s| s.parse().ok());
    let cache_path: PathBuf = std::env::args().nth(3)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();

    let c = cache::load(&cache_path)?;
    for (&r, &d) in &c {
        if d != depth { continue; }
        if let Some(bf) = blank_filter {
            if unrank(r).blank_pos() != bf { continue; }
        }
        println!("{}", r);
    }
    Ok(())
}
