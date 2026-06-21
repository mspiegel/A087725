//! Batch K=4 sweep of d=77 sub-singletons: read a file of reference ranks,
//! run a Hamming-K neighborhood enumeration around each, IDA*-verify
//! cache-miss high-h candidates, accumulate any depth>=76 finds across all
//! references, deduplicated. Reflection-aware: each find is mirrored.
//!
//! Cache + PDBs are loaded once; refs are processed sequentially but each
//! ref's verify phase is parallel (rayon).
//!
//! Run: `cargo run --release --features "mmap parallel" --example d77_sweep \
//!     -- REFS_FILE [K] [MIN_H] [CACHE]`
//!
//! REFS_FILE: one decimal rank per line, blank lines and #-prefixed comments
//! are ignored.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
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
use puzzle8::puzzle15::symmetry::reflect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let refs_path: PathBuf = std::env::args().nth(1)
        .ok_or("usage: d77_sweep REFS_FILE [K] [MIN_H] [CACHE]")?
        .into();
    let max_k: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let min_h: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(63);
    let cache_path: PathBuf = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "data/enum/solve_cache.bin".into())
        .into();

    let t0 = Instant::now();

    let refs: Vec<u64> = fs::read_to_string(&refs_path)?
        .lines()
        .filter_map(|l| {
            let s = l.trim();
            if s.is_empty() || s.starts_with('#') { return None; }
            s.parse::<u64>().ok()
        })
        .collect();
    println!("loaded {} references from {}", refs.len(), refs_path.display());

    let cache = cache::load(&cache_path)?;
    println!("loaded cache: {} entries", cache.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);
    println!("config: K={}, min_h={}", max_k, min_h);

    let mut all_finds: HashMap<u64, u8> = HashMap::new();
    let mut new_d77_total: usize = 0;
    let mut new_d76_total: usize = 0;
    let mut hist_total: BTreeMap<u8, u32> = BTreeMap::new();

    for (idx, &ref_rank) in refs.iter().enumerate() {
        let t_ref = Instant::now();
        let reference: [u8; 16] = unrank(ref_rank).0;

        let raw_states: Vec<[u8; 16]> = enumerate_neighborhood(max_k, &reference);

        // Solvable + high-h + cache-miss + not-already-found filter.
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

        // Drop already-found candidates from prior refs.
        let candidates: Vec<u64> = candidates
            .into_iter()
            .filter(|r| !all_finds.contains_key(r))
            .collect();

        if candidates.is_empty() {
            println!("ref {:>3}/{:<3} rank={:<14} cand=0 in {:.1?}",
                     idx + 1, refs.len(), ref_rank, t_ref.elapsed());
            continue;
        }

        let results: Vec<(u64, u8)> = candidates
            .par_iter()
            .map(|&r| {
                let (sol, _) = idastar_inc_with_stats(&unrank(r), &h);
                (r, sol.map(|v| v.len() as u8).unwrap_or(u8::MAX))
            })
            .collect();

        let mut ref_new_d77 = 0;
        let mut ref_new_d76 = 0;
        for &(r, d) in &results {
            *hist_total.entry(d).or_insert(0) += 1;
            if d >= 76 {
                if all_finds.insert(r, d).is_none() {
                    // Also insert reflection partner.
                    let refl_r = rank(&reflect(&unrank(r)));
                    if refl_r != r && !cache.contains_key(&refl_r) {
                        all_finds.insert(refl_r, d);
                    }
                    if d == 77 { ref_new_d77 += 1; }
                    if d == 76 { ref_new_d76 += 1; }
                }
            }
        }
        new_d77_total += ref_new_d77;
        new_d76_total += ref_new_d76;

        println!("ref {:>3}/{:<3} rank={:<14} cand={} new_d77={} new_d76={} in {:.1?} (total d77={} d76={})",
                 idx + 1, refs.len(), ref_rank,
                 candidates.len(), ref_new_d77, ref_new_d76,
                 t_ref.elapsed(),
                 new_d77_total, new_d76_total);
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    println!("\n=== sweep complete in {:.1?} ===", t0.elapsed());
    println!("total verified-depth histogram (across all refs, cache-miss only):");
    for (d, c) in &hist_total {
        let mark = if *d >= 76 { " ***" } else { "" };
        println!("  d={:>3}: {:>8}{}", d, c, mark);
    }

    // Dump all unique d>=76 finds (including reflection partners we auto-added).
    let mut finds_sorted: Vec<(u64, u8)> = all_finds.into_iter().collect();
    finds_sorted.sort_by_key(|&(r, d)| (std::cmp::Reverse(d), r));
    println!("\n=== ALL DEPTH>=76 FINDS (incl. reflection partners): {} ===", finds_sorted.len());
    let mut by_depth: BTreeMap<u8, Vec<u64>> = BTreeMap::new();
    for (r, d) in &finds_sorted { by_depth.entry(*d).or_default().push(*r); }
    for (d, ranks) in by_depth.iter().rev() {
        println!("\n-- d={} : {} boards --", d, ranks.len());
        for &r in ranks {
            let s = unrank(r);
            print!("FIND d={} rank={:>14} blank@{:>2}: ", d, r, s.blank_pos());
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
        let mut combo: Vec<usize> = (0..k).collect();
        loop {
            let original_tiles: Vec<u8> = combo.iter().map(|&c| reference[c]).collect();
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

#[allow(dead_code)]
fn _unused() -> Option<(HashMap<u64, u8>, HashSet<u64>)> { None }
