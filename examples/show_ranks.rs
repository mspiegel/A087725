//! Display decimal ranks (one per line; #-comments ok) as 4x4 grids with
//! depth (from cache, if given), h, blank cell, and reflection partner.
//!
//! Run: cargo run --release --features "mmap parallel" --example show_ranks -- RANKS_FILE [CACHE]

use std::fs;
use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::symmetry::reflect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ranks_path: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: show_ranks RANKS_FILE [CACHE]")?
        .into();
    let cache_path: Option<PathBuf> = std::env::args().nth(2).map(Into::into);

    let ranks: Vec<u64> = fs::read_to_string(&ranks_path)?
        .lines()
        .filter_map(|l| {
            let s = l.trim();
            if s.is_empty() || s.starts_with('#') {
                None
            } else {
                s.parse::<u64>().ok()
            }
        })
        .collect();

    let cache = match &cache_path {
        Some(p) => Some(cache::load(p)?),
        None => None,
    };

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    println!("{} ranks from {}\n", ranks.len(), ranks_path.display());
    for r in &ranks {
        let s = unrank(*r);
        let mut st = SearchStats::default();
        let (hv, _) = h.root(&s, &mut st);
        let d = cache.as_ref().and_then(|c| c.get(r).copied());
        let rr = rank(&reflect(&unrank(*r)));
        let refl_cached = cache.as_ref().map(|c| c.contains_key(&rr)).unwrap_or(false);
        println!(
            "rank={r:>14}  d={}  h={hv}  blank@{}  refl={rr}{}",
            d.map(|x| x.to_string()).unwrap_or("?".into()),
            s.blank_pos(),
            if cache.is_some() {
                if refl_cached {
                    " (refl cached)"
                } else {
                    " (refl MISSING)"
                }
            } else {
                ""
            }
        );
        for row in 0..4 {
            print!("    ");
            for col in 0..4 {
                let t = s.0[row * 4 + col];
                if t == 0 {
                    print!(" __");
                } else {
                    print!(" {t:>2}");
                }
            }
            println!();
        }
        println!();
    }
    Ok(())
}
