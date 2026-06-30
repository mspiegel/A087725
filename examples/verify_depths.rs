//! IDA*-verify the optimal depth of each rank in a file. Reports the depth
//! histogram and flags any board at depth >= 76 (a missing d76 find, since
//! layers 77..80 are complete in the cache). Optional LIMIT solves only the
//! first N (for per-board rate estimation).
//!
//! Run: cargo run --release --features "mmap parallel" --example verify_depths \
//!          -- RANKS_FILE [OUT_PAIRS] [LIMIT]
//!
//! If OUT_PAIRS is given, writes a `rank depth` line for every solved board
//! (feed to `cache_insert --from-pairs`). LIMIT solves only the first N.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{idastar_inc_with_stats, LinearConflictInc, WalkingDistanceHeuristic};
use puzzle8::puzzle15::symmetry::reflect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ranks_path: PathBuf = std::env::args().nth(1).ok_or("usage: verify_depths RANKS_FILE [OUT_PAIRS] [LIMIT]")?.into();
    let out_pairs: Option<PathBuf> = std::env::args().nth(2).filter(|s| s.parse::<usize>().is_err()).map(PathBuf::from);
    let limit: Option<usize> = std::env::args().nth(2).and_then(|s| s.parse().ok())
        .or_else(|| std::env::args().nth(3).and_then(|s| s.parse().ok()));

    let mut ranks: Vec<u64> = fs::read_to_string(&ranks_path)?.lines()
        .filter_map(|l| l.trim().parse::<u64>().ok()).collect();
    if let Some(n) = limit { ranks.truncate(n); }
    eprintln!("verifying {} ranks", ranks.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let done = AtomicU64::new(0);
    let found = AtomicU64::new(0);
    let t0 = Instant::now();
    let n = ranks.len();
    // incremental checkpoint: write each solved pair as it's found (crash-safe;
    // lets us `grep` finds live mid-run).
    let ckpt = out_pairs.as_ref().map(|p| Mutex::new(BufWriter::new(fs::File::create(p).unwrap())));
    let results: Vec<(u64, u8)> = ranks.par_iter().map(|&r| {
        let (sol, _) = idastar_inc_with_stats(&unrank(r), &h);
        let d = sol.map(|v| v.len() as u8).unwrap_or(u8::MAX);
        if d != u8::MAX {
            if let Some(m) = &ckpt {
                let mut w = m.lock().unwrap();
                let _ = writeln!(w, "{r} {d}");
                if d >= 75 { let _ = w.flush(); }
            }
        }
        if d >= 75 {
            let nf = found.fetch_add(1, Ordering::Relaxed) + 1;
            let s = unrank(r);
            let mut g = String::new();
            for (j, &t) in s.0.iter().enumerate() {
                if t == 0 { g.push_str("__"); } else { g.push_str(&format!("{t:>2}")); }
                g.push_str(if j % 4 == 3 { "  " } else { "," });
            }
            eprintln!("*** FIND #{nf} d={d} rank={r} blank@{}: {g}", s.blank_pos());
        }
        let c = done.fetch_add(1, Ordering::Relaxed) + 1;
        if c % 5000 == 0 {
            let el = t0.elapsed().as_secs_f64();
            eprintln!("  {c}/{n}  {:.1} solves/s  elapsed {:.0}s  (d>=75 finds: {})",
                      c as f64 / el, el, found.load(Ordering::Relaxed));
        }
        (r, d)
    }).collect();
    if let Some(m) = &ckpt { let _ = m.lock().unwrap().flush(); }

    let mut hist: BTreeMap<u8, u64> = BTreeMap::new();
    let mut finds: Vec<(u64, u8)> = Vec::new();
    for &(r, d) in &results {
        *hist.entry(d).or_insert(0) += 1;
        if d >= 75 { finds.push((r, d)); }
    }
    let el = t0.elapsed().as_secs_f64();
    eprintln!("\n=== verified {} in {:.0}s ({:.1} solves/s) ===", n, el, n as f64 / el);
    eprintln!("depth histogram:");
    for (d, c) in &hist { eprintln!("  d={d:>3}: {c}"); }

    if let Some(p) = &out_pairs {
        let nw = results.iter().filter(|(_, d)| *d != u8::MAX).count();
        eprintln!("checkpointed {nw} (rank depth) pairs -> {}", p.display());
    }

    println!("\n*** d>=75 finds: {} ***", finds.len());
    for &(r, d) in &finds {
        let s = unrank(r);
        let rr = rank(&reflect(&s));
        print!("FIND d={d} rank={r} (reflect={rr}) blank@{}: ", s.blank_pos());
        for (j, &t) in s.0.iter().enumerate() {
            if t == 0 { print!("__"); } else { print!("{t:>2}"); }
            if j % 4 == 3 { print!("  "); } else { print!(","); }
        }
        println!();
    }
    Ok(())
}
