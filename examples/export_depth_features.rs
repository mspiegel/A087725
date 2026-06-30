//! Export per-board features + labels for a depth>=75 classifier. Positives are
//! cache boards at depth>=75 (all d76..80, sampled d75); negatives are HARD
//! negatives at depth 70..74 (deep but not deep enough). Features are cheap:
//! the combined heuristic h, blank cell, Manhattan distance, min-neighbor-h and
//! pit flag (the heuristic-landscape signal), and the 16 raw tile-at-cell values
//! (lets a tree learn the corner/quadrant structure directly).
//!
//! Output: TSV with header. Columns: label depth h blank manhattan min_nb_h pit c0..c15
//!
//! Run: cargo run --release --features "mmap parallel" --example export_depth_features -- OUT.tsv [CACHE]

use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::State;

// deterministic per-rank hash for sampling (splitmix64)
fn h64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path: PathBuf = std::env::args().nth(1).ok_or("usage: export_depth_features OUT.tsv [CACHE]")?.into();
    let cache_path: PathBuf = std::env::args().nth(2).unwrap_or_else(|| "data/enum/solve_cache.bin".into()).into();

    let c = cache::load(&cache_path)?;
    eprintln!("cache: {} entries", c.len());

    // choose sampled (rank, depth) by class
    let mut kept: Vec<(u64, u8)> = Vec::new();
    for (&r, &d) in &c {
        let keep = match d {
            75 => h64(r) % 10 == 0,        // ~150K of 1.5M
            76..=80 => true,               // all (~302K)
            70..=74 => h64(r) % 170 == 0,  // ~480K hard negatives
            _ => false,
        };
        if keep { kept.push((r, d)); }
    }
    let npos = kept.iter().filter(|(_, d)| *d >= 75).count();
    eprintln!("sampled {} rows ({} pos / {} neg)", kept.len(), npos, kept.len() - npos);

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let rows: Vec<String> = kept.par_iter().map(|&(r, d)| {
        let s = unrank(r);
        let blank = s.blank_pos();
        let mut st = SearchStats::default();
        let bh = h.root(&s, &mut st).0;
        // manhattan
        let mut man = 0u32;
        for (i, &v) in s.0.iter().enumerate() {
            if v == 0 { continue; }
            let (gr, gc) = ((v as usize) / 4, (v as usize) % 4);
            let (cr, cc) = (i / 4, i % 4);
            man += (gr as i32 - cr as i32).unsigned_abs() + (gc as i32 - cc as i32).unsigned_abs();
        }
        // min neighbor h + pit
        let mut min_nb = u8::MAX;
        for m in State::legal_moves_at(blank).iter() {
            let (ns, _) = s.apply_at(m, blank);
            let mut st2 = SearchStats::default();
            let nh = h.root(&ns, &mut st2).0;
            if nh < min_nb { min_nb = nh; }
        }
        let pit = if min_nb > bh { 1 } else { 0 };
        let label = if d >= 75 { 1 } else { 0 };
        let mut line = format!("{label}\t{d}\t{bh}\t{blank}\t{man}\t{min_nb}\t{pit}");
        for &v in &s.0 { line.push('\t'); line.push_str(&v.to_string()); }
        line
    }).collect();

    let mut w = BufWriter::new(std::fs::File::create(&out_path)?);
    write!(w, "label\tdepth\th\tblank\tmanhattan\tmin_nb_h\tpit")?;
    for i in 0..16 { write!(w, "\tc{i}")?; }
    writeln!(w)?;
    for l in &rows { writeln!(w, "{l}")?; }
    w.flush()?;
    eprintln!("wrote {} rows -> {}", rows.len(), out_path.display());
    Ok(())
}
