//! Surface the most-rare known d=78 boards along two outlier axes — lowest
//! heuristic value (`h(s)`) and rarest blank cell. Print each outlier's tile
//! grid and compute pairwise Hamming distance on tile positions to see
//! whether outliers cluster structurally.
//!
//! Hypothesis: the 4 missing d=78 boards are extreme cases of these existing
//! outlier patterns. If outliers cluster tightly in tile-arrangement space,
//! the residue probably clusters with them — giving us a generate-and-verify
//! target region.
//!
//! Run: `cargo run --release --features "mmap parallel" --example d78_outliers`

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let cache_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    let cache = cache::load(&cache_path)?;
    println!("loaded {} cache entries from {}", cache.len(), cache_path.display());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    // Pull all d=78 boards.
    let d78: Vec<u64> = cache.iter()
        .filter(|(_, &d)| d == 78)
        .map(|(&r, _)| r)
        .collect();
    println!("{} d=78 boards total\n", d78.len());

    // Compute h + blank for each.
    let entries: Vec<Entry> = d78.par_iter()
        .map(|&r| {
            let s = unrank(r);
            let mut stats = SearchStats::default();
            let (hv, _) = h.root(&s, &mut stats);
            Entry { rank: r, state: s, h: hv, blank: s.blank_pos() }
        })
        .collect();

    // === Outlier set #1: lowest h-bucket (h=62, expected 6 boards) ===
    println!("=== Outlier set 1: lowest h ===");
    let min_h = entries.iter().map(|e| e.h).min().unwrap();
    let low_h: Vec<&Entry> = entries.iter().filter(|e| e.h == min_h).collect();
    println!("{} d=78 boards with h={}", low_h.len(), min_h);
    print_grids(&low_h);
    let hamming = pairwise_hamming(&low_h);
    println!("Pairwise tile-position Hamming distances (low_h):");
    print_matrix(&hamming);

    // Mean intra-group Hamming for low-h:
    let intra_low = mean_off_diagonal(&hamming);
    println!("Mean intra-group Hamming distance: {intra_low:.2}");

    // === Outlier set #2: rarest blank cell (cell 10, expected 26 boards) ===
    // First find the rarest blank cell among d=78 boards.
    let mut by_blank: HashMap<u8, u32> = HashMap::new();
    for e in &entries {
        *by_blank.entry(e.blank).or_insert(0) += 1;
    }
    let mut blank_counts: Vec<(u8, u32)> = by_blank.into_iter().collect();
    blank_counts.sort_by_key(|&(_, c)| c);
    let rarest_blank = blank_counts[0].0;
    println!("\n=== Outlier set 2: rarest blank cell ===");
    println!("blank distribution (sorted ascending):");
    for (cell, n) in &blank_counts {
        println!("  cell {cell:>2}: {n} boards");
    }
    let rare_blank: Vec<&Entry> = entries.iter().filter(|e| e.blank == rarest_blank).collect();
    println!("\n{} d=78 boards with blank at rarest cell {}:", rare_blank.len(), rarest_blank);
    let rare_show: Vec<&Entry> = rare_blank.iter().take(10).copied().collect();
    print_grids(&rare_show);

    let hamming_rare = pairwise_hamming(&rare_blank);
    let intra_rare = mean_off_diagonal(&hamming_rare);
    println!("Mean intra-group Hamming distance (blank={rarest_blank}): {intra_rare:.2}");

    // === Cross-group: do low_h boards overlap with rare_blank boards? ===
    let mut overlap = 0;
    for e in &low_h {
        if e.blank == rarest_blank { overlap += 1; }
    }
    println!("\nOverlap (h={min_h} AND blank={rarest_blank}): {overlap} boards");

    // === Baseline for context: pick a random sample of "typical" d=78 boards ===
    // (Use first 20 boards in cache iteration order — biased but cheap.)
    let baseline: Vec<&Entry> = entries.iter().take(20).collect();
    let hamming_base = pairwise_hamming(&baseline);
    let intra_base = mean_off_diagonal(&hamming_base);
    println!("\nBaseline mean intra-group Hamming (20 arbitrary d=78 boards): {intra_base:.2}");
    println!("\n→ Low intra-group Hamming relative to baseline means outliers cluster structurally.");

    // === Cross-group distances ===
    let cross_low_rare = mean_cross(&low_h, &rare_blank);
    println!("Mean Hamming distance from low-h group to rare-blank group: {cross_low_rare:.2}");

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}

#[derive(Clone, Copy)]
struct Entry {
    rank: u64,
    state: State,
    h: u8,
    blank: u8,
}

fn print_grids(group: &[&Entry]) {
    for (i, e) in group.iter().enumerate() {
        println!("#{:>2} rank {:>14} h={} blank@{}:", i + 1, e.rank, e.h, e.blank);
        let s = &e.state;
        for row in 0..4 {
            print!("    ");
            for col in 0..4 {
                let t = s.0[row * 4 + col];
                if t == 0 { print!(" __"); } else { print!(" {t:>2}"); }
            }
            println!();
        }
    }
}

fn hamming(a: &State, b: &State) -> u8 {
    let mut d = 0u8;
    for i in 0..16 {
        if a.0[i] != b.0[i] { d += 1; }
    }
    d
}

fn pairwise_hamming(group: &[&Entry]) -> Vec<Vec<u8>> {
    let n = group.len();
    let mut m = vec![vec![0u8; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = hamming(&group[i].state, &group[j].state);
        }
    }
    m
}

fn print_matrix(m: &[Vec<u8>]) {
    let n = m.len();
    print!("       ");
    for j in 0..n { print!(" {:>3}", j + 1); }
    println!();
    for i in 0..n {
        print!("  #{:>2}: ", i + 1);
        for j in 0..n { print!(" {:>3}", m[i][j]); }
        println!();
    }
}

fn mean_off_diagonal(m: &[Vec<u8>]) -> f64 {
    let n = m.len();
    let mut sum = 0u64;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += m[i][j] as u64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum as f64 / count as f64 }
}

fn mean_cross(a: &[&Entry], b: &[&Entry]) -> f64 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for x in a {
        for y in b {
            if x.rank == y.rank { continue; }
            sum += hamming(&x.state, &y.state) as u64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum as f64 / count as f64 }
}
