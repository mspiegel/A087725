//! Dump board grids (16 space-separated tile values per line) for all cache
//! entries at a given depth. Cell index 0..15 row-major, value = tile (0=blank).
//!
//! Run: `cargo run --release --features "mmap parallel" --example dump_grids -- DEPTH [CACHE]`

use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::rank::unrank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let depth: u8 = std::env::args().nth(1)
        .ok_or("usage: dump_grids DEPTH [CACHE]")?
        .parse()?;
    let cache_path: PathBuf = std::env::args().nth(2)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();

    let c = cache::load(&cache_path)?;
    let mut out = String::new();
    for (&r, &d) in &c {
        if d != depth { continue; }
        let s = unrank(r);
        for (i, &v) in s.0.iter().enumerate() {
            if i > 0 { out.push(' '); }
            out.push_str(&v.to_string());
        }
        out.push('\n');
    }
    print!("{}", out);
    Ok(())
}
