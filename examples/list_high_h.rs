//! List ranks of all cached d=78 boards at blank@cell with h >= MIN_H.
//!
//! Run: `cargo run --release --features "mmap parallel" --example list_high_h -- BLANK MIN_H [CACHE]`

use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let blank: u8 = std::env::args().nth(1).ok_or("need BLANK")?.parse()?;
    let min_h: u8 = std::env::args().nth(2).ok_or("need MIN_H")?.parse()?;
    let cache_path: PathBuf = std::env::args().nth(3)
        .unwrap_or_else(|| "data/enum/solve_cache.bin".into())
        .into();

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let c = cache::load(&cache_path)?;
    for (&r, &d) in &c {
        if d != 78 { continue; }
        let s = unrank(r);
        if s.blank_pos() != blank { continue; }
        let mut stats = SearchStats::default();
        let (hv, _) = h.root(&s, &mut stats);
        if hv >= min_h {
            println!("{}", r);
        }
    }
    Ok(())
}
