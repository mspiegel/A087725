//! Bellman / value-iteration closure on the solve cache. A cache-miss board B
//! whose EVERY puzzle-move neighbor is cached has exact optimal depth
//! d(B) = 1 + min(neighbor depths) (unit-cost shortest path) — no IDA* needed.
//! Seeds from cached boards at depth >= DMIN, expands to cache-miss neighbors,
//! and iterates to a fixpoint (each insert re-enables its neighbors). Mirrors
//! every insert under reflection. Backs up the cache before writing.
//!
//! Run: cargo run --release --features "mmap parallel" --example bellman_closure -- [DMIN] [CACHE]

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::state::State;
use puzzle8::puzzle15::symmetry::reflect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dmin: u8 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(75);
    let cache_path: PathBuf = std::env::args().nth(2).unwrap_or_else(|| "data/enum15/solve_cache.bin".into()).into();

    let t0 = Instant::now();
    let mut c = cache::load(&cache_path)?;
    let n_before = c.len();
    eprintln!("cache: {n_before} entries; seeding from depth >= {dmin}");

    // backup first
    let backup = cache_path.with_extension("bin.bak");
    cache::save(&backup, &c)?;
    eprintln!("backed up to {}", backup.display());

    // collect cache-miss neighbors of deep cached boards = initial frontier
    let seeds: Vec<u64> = c.iter().filter(|(_, &d)| d >= dmin).map(|(&r, _)| r).collect();
    eprintln!("seeds (depth>={dmin}): {}", seeds.len());

    let neighbors = |r: u64| -> Vec<u64> {
        let s = unrank(r);
        let b = s.blank_pos();
        State::legal_moves_at(b).iter().map(|m| rank(&s.apply_at(m, b).0)).collect()
    };

    let mut inwork: HashSet<u64> = HashSet::new();
    let mut stack: Vec<u64> = Vec::new();
    for &r in &seeds {
        for nr in neighbors(r) {
            if !c.contains_key(&nr) && inwork.insert(nr) { stack.push(nr); }
        }
    }
    eprintln!("initial frontier: {} candidates", stack.len());

    let mut new_by_depth = [0u64; 128];
    let mut processed = 0u64;
    while let Some(r) = stack.pop() {
        inwork.remove(&r);
        if c.contains_key(&r) { continue; }
        processed += 1;
        let nbrs = neighbors(r);
        let mut mind = u8::MAX;
        let mut all_cached = true;
        for &nr in &nbrs {
            match c.get(&nr) {
                Some(&d) => { if d < mind { mind = d; } }
                None => { all_cached = false; break; }
            }
        }
        if !all_cached { continue; }
        let d = mind + 1;
        c.insert(r, d);
        new_by_depth[d as usize] += 1;
        // mirror
        let rr = rank(&reflect(&unrank(r)));
        if rr != r && !c.contains_key(&rr) {
            c.insert(rr, d);
            new_by_depth[d as usize] += 1;
            for nr in neighbors(rr) { if !c.contains_key(&nr) && inwork.insert(nr) { stack.push(nr); } }
        }
        // re-enable neighbors (they now have one more cached neighbor)
        for nr in nbrs { if !c.contains_key(&nr) && inwork.insert(nr) { stack.push(nr); } }
        if processed % 2_000_000 == 0 {
            eprintln!("  processed {processed}, stack {}, new so far {}", stack.len(),
                      new_by_depth.iter().sum::<u64>());
        }
    }

    let total_new: u64 = new_by_depth.iter().sum();
    eprintln!("\n=== bellman closure done in {:.1?} ===", t0.elapsed());
    eprintln!("processed {processed} candidates; inserted {total_new} new (incl mirrors)");
    eprintln!("new boards by depth:");
    for d in 0..128 { if new_by_depth[d] > 0 { eprintln!("  d={d:>3}: {}", new_by_depth[d]); } }

    if total_new > 0 {
        eprintln!("saving cache ({n_before} -> {}) ...", c.len());
        cache::save(&cache_path, &c)?;
        eprintln!("done");
    } else {
        eprintln!("no new boards; cache unchanged");
    }
    Ok(())
}
