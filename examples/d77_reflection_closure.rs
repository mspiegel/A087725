//! Reflection-closure diff of a cached depth layer.
//!
//! The solve cache is supposed to be symmetry-closed: for every board `r` at
//! depth `d`, its horizontal reflection `reflect(r)` is also cached at the same
//! depth (reflection preserves optimal distance). If that closure was ever
//! broken (e.g. an insert overwritten by a concurrent enumerate15 save), then
//! some cached board's reflection is *absent* — and that absent reflection is a
//! missing residue board, constructible for free.
//!
//! This tool reports, for a given depth, every cached board whose reflection is
//! NOT in the cache. Each such reflection is provably at the same depth.
//!
//! Read-only. Safe to run while enumerate15 holds the live cache (point it at a
//! backup snapshot to avoid a torn read).
//!
//! Run: cargo run --release --features "mmap parallel" --example d77_reflection_closure -- [cache_path] [depth]

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, LinearConflictInc, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;
use puzzle8::puzzle15::symmetry::reflect;

fn print_grid(prefix: &str, r: u64) {
    let s: State = unrank(r);
    print!("{prefix} rank={r:>14} blank@{:>2}: ", s.blank_pos());
    for (j, &t) in s.0.iter().enumerate() {
        if t == 0 {
            print!("__");
        } else {
            print!("{t:>2}");
        }
        if j % 4 == 3 {
            print!("  ");
        } else {
            print!(",");
        }
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin.bak.preenum73".into())
        .into();
    let depth: u8 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(77);

    let t0 = Instant::now();
    let cache = cache::load(&cache_path)?;
    println!(
        "loaded cache: {} entries from {}",
        cache.len(),
        cache_path.display()
    );

    // Collect the target layer.
    let layer: Vec<u64> = cache
        .iter()
        .filter(|(_, &d)| d == depth)
        .map(|(&r, _)| r)
        .collect();
    println!("depth {depth}: {} cached boards", layer.len());

    // Count fixed points (self-symmetric boards) for parity insight.
    let mut fixed = 0usize;
    let mut orphans: Vec<(u64, u64)> = Vec::new(); // (cached board, its absent reflection)
    for &r in &layer {
        let rr = rank(&reflect(&unrank(r)));
        if rr == r {
            fixed += 1;
            continue;
        }
        if !cache.contains_key(&rr) {
            orphans.push((r, rr));
        }
    }
    println!("self-symmetric (fixed-point) boards: {fixed}");
    println!(
        "cached boards whose reflection is ABSENT: {}",
        orphans.len()
    );

    if orphans.is_empty() {
        println!("\n==> Layer is reflection-closed.");
        println!("    The missing boards form a closed set under reflection:");
        println!("    a reflection PAIR (find one => get both for free) or two fixed points.");
        println!("    No free board recoverable here.\n");
        println!("total elapsed {:.1?}", t0.elapsed());
        return Ok(());
    }

    // Dedup absent reflections (each missing board may appear once).
    let missing: HashSet<u64> = orphans.iter().map(|&(_, rr)| rr).collect();
    println!(
        "\n==> {} distinct missing board(s) recovered as reflections!\n",
        missing.len()
    );

    // Verify each with admissible IDA* (cheap; should all come back == depth).
    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let mut verified: Vec<u64> = Vec::new();
    for &r in &missing {
        let (sol, _) = idastar_inc_with_stats(&unrank(r), &h);
        let d = sol.map(|v| v.len() as u8).unwrap_or(u8::MAX);
        print_grid(&format!("[d={d}]"), r);
        if d == depth {
            verified.push(r);
        }
    }

    println!("\n=== RANKS (one per line, for cache_insert) ===");
    let mut out: Vec<u64> = verified.clone();
    out.sort();
    for r in &out {
        println!("{r}");
    }

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}
