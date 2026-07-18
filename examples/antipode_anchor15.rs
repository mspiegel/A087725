//! Antipode-anchored heuristic study (ENUMERATION.md follow-up).
//!
//! Idea: depth(s) ≈ 80 − dist(s, nearest antipode). We approximate
//! dist(s, antipode) by Manhattan distance toward that antipode's arrangement
//! (cheap, no PDB build), so `anchor_est(s) = 80 − min_i MtA(s, antipode_i)`.
//!
//! Test: enumerate the known-depth deep boards (depths 78/79/80 via the band
//! BFS), then compare, per true depth, the antipode-anchored estimate against
//! the goal-side `korf-plus` admissible lower bound. The decisive question is
//! whether the estimate is sharp at the deep end and, in particular, whether it
//! separates depth-79 from depth-78 (needed if it is to guide generation of the
//! depth-79 local maxima).
//!
//! Run: `cargo run --release --example antipode_anchor15 --features "mmap parallel" -- data`

use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::{antipodes, frontier, histogram, Store};
use puzzle8::puzzle15::pdb::{
    AdditivePdbHeuristic, MaxHeuristic, PatternDb, ReflectedHeuristic,
};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    idastar, Heuristic, LinearConflictHeuristic, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;

/// `pos_of[tile]` = cell index of `tile` in this antipode.
fn position_table(a: &State) -> [u8; 16] {
    let mut pos = [0u8; 16];
    for (cell, &t) in a.0.iter().enumerate() {
        pos[t as usize] = cell as u8;
    }
    pos
}

/// Manhattan distance from `s` to the arrangement described by `pos_of`
/// (treating that antipode as the goal), summed over the 15 tiles.
fn manhattan_to(s: &State, pos_of: &[u8; 16]) -> u32 {
    let mut sum = 0u32;
    for (cell, &t) in s.0.iter().enumerate() {
        if t == 0 {
            continue;
        }
        let q = pos_of[t as usize] as usize;
        let dr = (cell / 4).abs_diff(q / 4);
        let dc = (cell % 4).abs_diff(q % 4);
        sum += (dr + dc) as u32;
    }
    sum
}

fn main() {
    let dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("data"));

    let hist = histogram::load(&dir.join("pdb15_depth_histogram.txt")).expect("histogram");
    let antipode_ranks = antipodes::load_ranks(&dir.join("pdb15_antipodes.txt")).expect("antipodes");
    let antipode_pos: Vec<[u8; 16]> =
        antipode_ranks.iter().map(|&r| position_table(&unrank(r))).collect();
    let anchor_est = |s: &State| -> i32 {
        let best = antipode_pos.iter().map(|p| manhattan_to(s, p)).min().unwrap();
        80i32 - best as i32
    };

    // Goal-side korf-plus (admissible lower bound) for comparison + verification.
    let dbs = [
        PatternDb::load_mmap(&dir.join("pdb15_p7_korf.bin")).expect("p7"),
        PatternDb::load_mmap(&dir.join("pdb15_p8_korf.bin")).expect("p8"),
    ];
    WalkingDistanceHeuristic::warm_up();
    let h = MaxHeuristic::new(
        MaxHeuristic::new(
            AdditivePdbHeuristic::new(&dbs),
            ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs)),
        ),
        MaxHeuristic::new(LinearConflictHeuristic, WalkingDistanceHeuristic),
    );
    let verify = |r: u64| -> u8 { idastar(&unrank(r), &h).map(|v| v.len() as u8).unwrap_or(u8::MAX) };

    // Collect known-depth deep boards via the band BFS over [78,80].
    let mut store = Store::new();
    for r in &antipode_ranks {
        store.insert(*r, 80);
    }
    let out = dir.join("enum");
    println!("filling store with depths 78..80 via band BFS (this takes a couple minutes)...");
    let mut cache = puzzle8::puzzle15::enumerate::cache::Cache::new();
    let _ = frontier::run_band(&mut store, &hist, 78, 78, 0, &mut cache, None, &out, verify, |_l| {});

    // Per-true-depth stats for both estimators.
    println!("\n true | n     | korf-plus (goal-side LB)        | anchor 80-MtA (inadmissible)");
    println!(" depth|       | mean   min  max   mean gap      | mean   min  max   mean err");
    println!("------+-------+--------------------------------+-----------------------------");
    for d in (78u8..=80).rev() {
        let layer = store.layer(d);
        if layer.is_empty() {
            continue;
        }
        let (mut ks, mut kmin, mut kmax) = (0i64, i32::MAX, i32::MIN);
        let (mut as_, mut amin, mut amax) = (0i64, i32::MAX, i32::MIN);
        for &r in layer {
            let s = unrank(r);
            let k = h.h(&s) as i32;
            let a = anchor_est(&s);
            ks += k as i64; kmin = kmin.min(k); kmax = kmax.max(k);
            as_ += a as i64; amin = amin.min(a); amax = amax.max(a);
        }
        let n = layer.len() as i64;
        let kmean = ks as f64 / n as f64;
        let amean = as_ as f64 / n as f64;
        println!(
            " {:>4} | {:>5} | {:>5.1}  {:>3}  {:>3}   {:>5.1} short  | {:>5.1}  {:>3}  {:>3}   {:>+5.1}",
            d, n, kmean, kmin, kmax, d as f64 - kmean, amean, amin, amax, amean - d as f64,
        );
    }

    // Separation test: does any anchor threshold cleanly split depth-79 from 78?
    if !store.layer(79).is_empty() && !store.layer(78).is_empty() {
        let a79: Vec<i32> = store.layer(79).iter().map(|&r| anchor_est(&unrank(r))).collect();
        let a78: Vec<i32> = store.layer(78).iter().map(|&r| anchor_est(&unrank(r))).collect();
        let min79 = *a79.iter().min().unwrap();
        let max78 = *a78.iter().max().unwrap();
        println!(
            "\nseparation (anchor): min over depth-79 = {}, max over depth-78 = {} → {}",
            min79, max78,
            if min79 > max78 { "SEPARABLE (a threshold splits 79 from 78)" }
            else { "OVERLAP (no anchor threshold separates 79 from 78)" },
        );
        // How sharp at the deep end? Spearman-ish: fraction of (79,78) pairs ranked correctly.
        let (mut correct, mut total) = (0u64, 0u64);
        for &x in &a79 { for &y in &a78 { total += 1; if x > y { correct += 1; } } }
        println!(
            "pairwise: anchor ranks depth-79 above depth-78 in {}/{} = {:.1}% of pairs",
            correct, total, 100.0 * correct as f64 / total as f64,
        );
    }
}
