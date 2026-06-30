//! List ranks of all cached boards at DEPTH and blank@cell with h >= MIN_H.
//! BLANK == 255 (or 'any') means no blank filter — emit ranks for every cached
//! board at DEPTH with h >= MIN_H.
//!
//! Run: `cargo run --release --features "mmap parallel" --example list_high_h -- BLANK MIN_H [CACHE] [DEPTH]`

use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let blank_arg = std::env::args().nth(1).ok_or("need BLANK (or 'any'/255)")?;
    let blank: Option<u8> = if blank_arg == "any" || blank_arg == "255" {
        None
    } else {
        Some(blank_arg.parse()?)
    };
    let min_h: u8 = std::env::args().nth(2).ok_or("need MIN_H")?.parse()?;
    let cache_path: PathBuf = std::env::args().nth(3)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    let depth: u8 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(78);

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let c = cache::load(&cache_path)?;
    for (&r, &d) in &c {
        if d != depth { continue; }
        let s = unrank(r);
        if let Some(b) = blank {
            if s.blank_pos() != b { continue; }
        }
        let mut stats = SearchStats::default();
        let (hv, _) = h.root(&s, &mut stats);
        if hv >= min_h {
            println!("{}", r);
        }
    }
    Ok(())
}
