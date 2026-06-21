//! Frame-completion enumeration for d=77 maxima.
//!
//! Observation: cached d=77 maxima keep the high tiles {9..15}+blank in the top
//! 8 cells and the low tiles {1..8} in the bottom 8, clustering tightly and
//! differing only by permutations of the bottom cells. So: for each reference
//! board, FIX cells 0..7 (the "frame") and enumerate EVERY permutation of the
//! tiles currently in cells 8..15. Solvable + h>=min_h + cache-miss completions
//! are IDA*-verified; any that solve to d=77 are missing maxima in that frame.
//!
//! Frames are deduped by their top-8 signature. Reflection-aware on output.
//!
//! Run: cargo run --release --features "mmap parallel" --example d77_frame_complete \
//!          -- REFS_FILE [MIN_H] [CACHE]

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{idastar_inc_with_stats, IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::State;
use puzzle8::puzzle15::symmetry::reflect;

fn next_permutation(a: &mut [u8]) -> bool {
    let n = a.len();
    if n < 2 { return false; }
    let mut i = n - 1;
    while i > 0 && a[i-1] >= a[i] { i -= 1; }
    if i == 0 { return false; }
    let mut j = n - 1;
    while a[j] <= a[i-1] { j -= 1; }
    a.swap(i-1, j);
    a[i..].reverse();
    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let refs_path: PathBuf = std::env::args().nth(1).ok_or("usage: d77_frame_complete REFS_FILE [MIN_H] [CACHE]")?.into();
    let min_h: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(66);
    let cache_path: PathBuf = std::env::args().nth(3).unwrap_or_else(|| "data/enum/solve_cache.bin".into()).into();

    let t0 = Instant::now();
    let refs: Vec<u64> = fs::read_to_string(&refs_path)?.lines()
        .filter_map(|l| { let s=l.trim(); if s.is_empty()||s.starts_with('#') {None} else {s.parse::<u64>().ok()} }).collect();
    let cache = cache::load(&cache_path)?;
    println!("loaded {} refs, cache {} entries", refs.len(), cache.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);
    println!("min_h={min_h}");

    // dedup frames by top-8 signature
    let mut frames: HashMap<[u8;8], [u8;16]> = HashMap::new();
    for &r in &refs {
        let a = unrank(r).0;
        let mut top = [0u8;8]; top.copy_from_slice(&a[0..8]);
        frames.entry(top).or_insert(a);
    }
    println!("distinct frames: {}", frames.len());

    let mut all_finds: HashMap<u64,u8> = HashMap::new();
    let mut hist: BTreeMap<u8,u32> = BTreeMap::new();
    let mut total_cand = 0u64;

    for (idx, (top, arr)) in frames.iter().enumerate() {
        let mut bottom: Vec<u8> = arr[8..16].to_vec();
        bottom.sort_unstable();
        // enumerate all permutations of the bottom tiles over cells 8..15
        let mut cands: Vec<u64> = Vec::new();
        loop {
            let mut b = [0u8;16];
            b[0..8].copy_from_slice(top);
            b[8..16].copy_from_slice(&bottom);
            let s = State(b);
            if s.is_solvable() {
                let mut st = SearchStats::default();
                if h.root(&s,&mut st).0 >= min_h {
                    let r = rank(&s);
                    if !cache.contains_key(&r) { cands.push(r); }
                }
            }
            if !next_permutation(&mut bottom) { break; }
        }
        total_cand += cands.len() as u64;
        // verify
        let results: Vec<(u64,u8)> = cands.par_iter().map(|&r| {
            let (sol,_) = idastar_inc_with_stats(&unrank(r), &h);
            (r, sol.map(|v| v.len() as u8).unwrap_or(u8::MAX))
        }).collect();
        let mut fnew77 = 0;
        for (r,d) in results {
            *hist.entry(d).or_insert(0)+=1;
            if d>=76 && all_finds.insert(r,d).is_none() {
                let rr = rank(&reflect(&unrank(r)));
                if rr!=r { all_finds.entry(rr).or_insert(d); }
                if d==77 { fnew77+=1; }
            }
        }
        if fnew77>0 || (idx % 20 == 0) {
            println!("frame {:>3}/{:<3} cand={} new_d77={} (total d77={})",
                     idx+1, frames.len(), cands.len(), fnew77,
                     all_finds.values().filter(|&&d| d==77).count());
        }
        if fnew77>0 {
            for (&r,&d) in &all_finds {
                if d!=77 { continue; }
                let s=unrank(r);
                print!("*** FIND d=77 rank={r} blank@{}: ", s.blank_pos());
                for (j,&t) in s.0.iter().enumerate(){ if t==0{print!("__");}else{print!("{t:>2}");} if j%4==3{print!("  ");}else{print!(",");} }
                println!();
            }
        }
    }

    println!("\n=== done in {:.1?} : {} frames, {} candidates verified ===", t0.elapsed(), frames.len(), total_cand);
    for (d,c) in &hist { let m = if *d>=76 {" ***"} else {""}; println!("  d={d:>3}: {c:>7}{m}"); }
    let d77: Vec<u64> = all_finds.iter().filter(|(_,&d)| d==77).map(|(&r,_)| r).collect();
    println!("\n*** NEW d=77: {} ***", d77.len());
    for r in &d77 { println!("{r} 77"); }
    Ok(())
}
