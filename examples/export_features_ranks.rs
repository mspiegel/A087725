//! Export the SAME features as export_depth_features, but for an explicit list
//! of `rank depth` pairs (e.g. solved quadrant-generated candidates). Used to
//! build a fair test set whose negatives match the inference distribution.
//!
//! Run: cargo run --release --features "mmap parallel" --example export_features_ranks -- PAIRS.tsv OUT.tsv

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use rayon::prelude::*;

use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pairs_path: PathBuf = std::env::args().nth(1).ok_or("usage: export_features_ranks PAIRS.tsv OUT.tsv")?.into();
    let out_path: PathBuf = std::env::args().nth(2).ok_or("need OUT.tsv")?.into();

    let f = std::fs::File::open(&pairs_path)?;
    let pairs: Vec<(u64, u8)> = std::io::BufReader::new(f).lines().filter_map(|l| {
        let l = l.ok()?; let mut it = l.split_whitespace();
        let r = it.next()?.parse::<u64>().ok()?;
        let d = it.next()?.parse::<u8>().ok()?;
        Some((r, d))
    }).collect();
    eprintln!("{} pairs", pairs.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let rows: Vec<String> = pairs.par_iter().map(|&(r, d)| {
        let s = unrank(r);
        let blank = s.blank_pos();
        let mut st = SearchStats::default();
        let bh = h.root(&s, &mut st).0;
        let mut man = 0u32;
        for (i, &v) in s.0.iter().enumerate() {
            if v == 0 { continue; }
            let (gr, gc) = ((v as usize) / 4, (v as usize) % 4);
            let (cr, cc) = (i / 4, i % 4);
            man += (gr as i32 - cr as i32).unsigned_abs() + (gc as i32 - cc as i32).unsigned_abs();
        }
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
