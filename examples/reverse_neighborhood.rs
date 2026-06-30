//! Enumerate every solvable state within Hamming distance K of the perfect
//! tile-reverse arrangement (singleton #17 from d78_clusters):
//!
//!     __ 15 14 13
//!     12 11 10  9
//!      8  7  6  5
//!      4  3  2  1
//!
//! IDA*-verify each cache-miss candidate that passes a heuristic floor.
//! Report any new d=78 / d=77 boards — these are the residue candidates if
//! the missing maxima cluster near this near-antipodal pattern.
//!
//! Hamming neighborhood: each candidate differs from REFERENCE in exactly
//! `k` cells for some k in 2..=K, with the tiles in those k cells permuted
//! as a derangement (no fixed points). Total enumeration size:
//!   K=2: 120 states
//!   K=4: ~17,620 states
//!   K=6: ~2.3M states
//!   K=8: ~210M states
//!
//! Run: `cargo run --release --features "mmap parallel" --example reverse_neighborhood -- [K] [MIN_H] [CACHE] [REFERENCE_RANK]`
//!
//! REFERENCE_RANK (optional): rank of an alternate reference state to
//! enumerate around. Default is the perfect-reverse arrangement
//! (singleton #17, rank 653837183999). Pass another singleton's rank to
//! pivot to a different geometric outlier sub-pocket.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, IncHeuristic, LinearConflictInc, SearchStats,
    WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;

const DEFAULT_REFERENCE: [u8; 16] = [0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let min_h: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let cache_path: PathBuf = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    let reference_rank: Option<u64> = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok());
    let reference: [u8; 16] = match reference_rank {
        Some(r) => unrank(r).0,
        None => DEFAULT_REFERENCE,
    };

    println!("config: K={}, min_h={}, cache={}", max_k, min_h, cache_path.display());
    match reference_rank {
        Some(r) => println!("reference (rank {}):", r),
        None => println!("reference (default: singleton #17, perfect reverse, h=70):"),
    };
    for row in 0..4 {
        print!("  ");
        for col in 0..4 {
            let t = reference[row * 4 + col];
            if t == 0 { print!(" __"); } else { print!(" {:>2}", t); }
        }
        println!();
    }

    let t0 = Instant::now();
    let cache = cache::load(&cache_path)?;
    println!("loaded cache: {} entries", cache.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    // Generate the Hamming-K neighborhood. For each k in 2..=K and each
    // k-subset of cells, enumerate every derangement on those k positions
    // and apply it to REFERENCE.
    let t_gen = Instant::now();
    let raw_states: Vec<[u8; 16]> = enumerate_neighborhood(max_k, &reference);
    println!("\nenumerated {} raw states in {:.1?}", raw_states.len(), t_gen.elapsed());

    let t_filter = Instant::now();
    // Parallel solvability + h filter + cache-miss filter.
    let candidates: Vec<u64> = raw_states
        .par_iter()
        .filter_map(|&arr| {
            let s = State(arr);
            if !s.is_solvable() { return None; }
            let mut stats = SearchStats::default();
            let (hv, _) = h.root(&s, &mut stats);
            if hv < min_h { return None; }
            let r = rank(&s);
            if cache.contains_key(&r) { return None; }
            Some(r)
        })
        .collect();
    println!("filter (solvable + h>={} + cache-miss): {} candidates in {:.1?}",
             min_h, candidates.len(), t_filter.elapsed());

    if candidates.is_empty() {
        println!("\nNo candidates to verify.");
        return Ok(());
    }

    // Parallel IDA*-verify.
    let t_verify = Instant::now();
    let results: Vec<(u64, u8)> = candidates
        .par_iter()
        .map(|&r| {
            let (sol, _) = idastar_inc_with_stats(&unrank(r), &h);
            (r, sol.map(|v| v.len() as u8).unwrap_or(u8::MAX))
        })
        .collect();
    println!("verified {} cands in {:.1?} ({:.1}/s)",
             results.len(), t_verify.elapsed(),
             results.len() as f64 / t_verify.elapsed().as_secs_f64());

    // Histogram.
    let mut hist: BTreeMap<u8, u32> = BTreeMap::new();
    for &(_, d) in &results { *hist.entry(d).or_insert(0) += 1; }
    println!("\nverified depth histogram:");
    for (d, c) in &hist {
        let mark = if *d >= 76 { " ***" } else { "" };
        println!("  d={:>3}: {:>8}{}", d, c, mark);
    }

    for target_d in (76..=80).rev() {
        let finds: Vec<u64> = results.iter()
            .filter(|&&(_, d)| d == target_d)
            .map(|&(r, _)| r)
            .collect();
        if finds.is_empty() { continue; }
        println!("\n=== NEW d={} BOARDS: {} ===", target_d, finds.len());
        for (i, &r) in finds.iter().enumerate() {
            let s = unrank(r);
            print!("  #{:>3} rank {:>14} blank@{:>2}: ", i + 1, r, s.blank_pos());
            for (j, &t) in s.0.iter().enumerate() {
                if t == 0 { print!("__"); } else { print!("{:>2}", t); }
                if j % 4 == 3 { print!("  "); } else { print!(","); }
            }
            println!();
        }
    }

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}

/// Enumerate every distinct state differing from REFERENCE in exactly k cells
/// (with the tiles in those cells permuted as a derangement) for each
/// k in 2..=max_k. Returns the raw state arrays.
fn enumerate_neighborhood(max_k: usize, reference: &[u8; 16]) -> Vec<[u8; 16]> {
    let mut out = Vec::new();
    for k in 2..=max_k {
        // Iterate all k-combinations of cells from [0, 16).
        let mut combo: Vec<usize> = (0..k).collect();
        loop {
            let original_tiles: Vec<u8> = combo.iter().map(|&c| reference[c]).collect();
            // Iterate all k! permutations of (0..k), keeping only derangements.
            let mut perm: Vec<usize> = (0..k).collect();
            loop {
                if is_derangement(&perm) {
                    let mut arr = *reference;
                    for (slot, &cell) in combo.iter().enumerate() {
                        arr[cell] = original_tiles[perm[slot]];
                    }
                    out.push(arr);
                }
                if !next_permutation(&mut perm) { break; }
            }
            if !next_combination(&mut combo, 16) { break; }
        }
    }
    out
}

fn is_derangement(p: &[usize]) -> bool {
    p.iter().enumerate().all(|(i, &v)| i != v)
}

/// Lexicographically next permutation of `arr`. Returns false when at the
/// last permutation (descending).
fn next_permutation(arr: &mut [usize]) -> bool {
    let n = arr.len();
    if n < 2 { return false; }
    let mut i = n - 1;
    while i > 0 && arr[i - 1] >= arr[i] { i -= 1; }
    if i == 0 { return false; }
    let mut j = n - 1;
    while arr[j] <= arr[i - 1] { j -= 1; }
    arr.swap(i - 1, j);
    arr[i..].reverse();
    true
}

/// Lex-next k-combination of [0, n). `combo` is sorted ascending; returns
/// false when at the last combination.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    if k == 0 { return false; }
    let mut i = k - 1;
    loop {
        if combo[i] < n - k + i {
            combo[i] += 1;
            for j in i + 1..k { combo[j] = combo[j - 1] + 1; }
            return true;
        }
        if i == 0 { return false; }
        i -= 1;
    }
}

// Quiet `unused`.
#[allow(dead_code)]
fn _unused() -> Option<(HashMap<u64, u8>, HashSet<u64>)> { None }
