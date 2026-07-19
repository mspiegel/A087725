//! Read the solve cache and compute the (true_depth, h) joint distribution,
//! where h is the root-bound from ZpdbPlusInc (= max of ZPDB-sum, LC, WD).
//!
//! Tells us how tight the heuristic is at high depth, which bounds how
//! effective a PDB-shell enumeration ("enumerate states with h >= T, verify
//! each") would be at finding the missing d77/d78 boards.
//!
//! Run: `cargo run --release --features "mmap parallel" --example pdb_shell_analysis`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    let pdb_dir: PathBuf = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data".into())
        .into();

    let t0 = Instant::now();
    println!("loading cache {} ...", cache_path.display());
    let cache = cache::load(&cache_path)?;
    println!("  {} entries in {:.1?}", cache.len(), t0.elapsed());

    let t1 = Instant::now();
    println!("loading ZPDBs from {} ...", pdb_dir.display());
    let p7 = ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p8.zbin"))?;
    println!("  loaded in {:.1?}", t1.elapsed());

    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    // Materialize cache entries so rayon can chunk them.
    let entries: Vec<(u64, u8)> = cache.iter().map(|(&r, &d)| (r, d)).collect();
    drop(cache);

    let t2 = Instant::now();
    println!("evaluating h(s) on {} boards ...", entries.len());

    // Per-thread (depth, h) -> count, merged at the end.
    let hist: BTreeMap<(u8, u8), u64> = entries
        .par_iter()
        .fold(
            BTreeMap::<(u8, u8), u64>::new,
            |mut acc, &(r, d)| {
                let s = unrank(r);
                let mut stats = SearchStats::default();
                let (hv, _ctx) = h.root(&s, &mut stats);
                *acc.entry((d, hv)).or_insert(0) += 1;
                acc
            },
        )
        .reduce(BTreeMap::<(u8, u8), u64>::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        });

    println!("  histogram built in {:.1?}\n", t2.elapsed());

    // Per-depth summary: count, min/max/mean h, mean slack.
    let mut depths: Vec<u8> = hist.keys().map(|&(d, _)| d).collect();
    depths.sort();
    depths.dedup();

    println!("{:>5}  {:>12}  {:>5} {:>5} {:>7}  {:>8}  {:>8}",
             "depth", "count", "min_h", "max_h", "mean_h", "mean_slk", "max_slk");
    for &d in &depths {
        let row: Vec<(u8, u64)> = hist
            .iter()
            .filter(|((dd, _), _)| *dd == d)
            .map(|((_, h), &c)| (*h, c))
            .collect();
        let n: u64 = row.iter().map(|(_, c)| c).sum();
        let min_h = row.iter().map(|(h, _)| *h).min().unwrap_or(0);
        let max_h = row.iter().map(|(h, _)| *h).max().unwrap_or(0);
        let mean_h = row.iter().map(|(h, c)| (*h as u64) * c).sum::<u64>() as f64 / n as f64;
        let mean_slack = (d as f64) - mean_h;
        let max_slack = d.saturating_sub(min_h);
        println!("{d:>5}  {n:>12}  {min_h:>5} {max_h:>5}  {mean_h:>6.2}  {mean_slack:>8.2}  {max_slack:>8}");
    }

    // Focused dump: for high-depth rows (≥75), show the full h-histogram so we
    // can read PDB-shell threshold + count tradeoff directly.
    println!("\nfull h-distribution for depths 75..=80 (h: count):");
    for &d in depths.iter().filter(|&&d| d >= 75) {
        let row: Vec<(u8, u64)> = hist
            .iter()
            .filter(|((dd, _), _)| *dd == d)
            .map(|((_, h), &c)| (*h, c))
            .collect();
        let n: u64 = row.iter().map(|(_, c)| c).sum();
        print!("  d={d:>2} (n={n:>6}):");
        for (hv, c) in &row {
            print!(" {hv}:{c}");
        }
        println!();
    }

    // Per-threshold recall sweep: if we PDB-shell at h >= T, how many cached
    // boards pass the filter at each true-depth tier?
    println!("\nfilter sweep (rows = threshold T on h, cols = true depth bucket):");
    type DepthTier = (&'static str, Box<dyn Fn(u8) -> bool>);
    let depth_tiers: Vec<DepthTier> = vec![
        ("d>=80", Box::new(|d| d >= 80)),
        ("d=79",  Box::new(|d| d == 79)),
        ("d=78",  Box::new(|d| d == 78)),
        ("d=77",  Box::new(|d| d == 77)),
        ("d=76",  Box::new(|d| d == 76)),
        ("d=75",  Box::new(|d| d == 75)),
        ("d<75",  Box::new(|d| d < 75)),
    ];
    print!("{:>5}", "T");
    for (label, _) in &depth_tiers {
        print!("  {label:>10}");
    }
    println!();
    for t in (60u8..=80).rev() {
        print!("  >={t:>2}");
        for (_, pred) in &depth_tiers {
            let count: u64 = hist
                .iter()
                .filter(|((d, hv), _)| *hv >= t && pred(*d))
                .map(|(_, &c)| c)
                .sum();
            print!("  {count:>10}");
        }
        println!();
    }

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}
