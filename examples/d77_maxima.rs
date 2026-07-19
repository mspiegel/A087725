//! Profile the cached d=77 *local maxima* (boards with no d=78 puzzle-move
//! neighbor), and emit high-h maxima ranks as targeted sweep seeds.
//!
//! The 2 missing d=77 boards are strict maxima (every neighbor is d=76; no d=78
//! neighbor — confirmed by d78_expand_d77 finding 0). They therefore share the
//! structural profile of the *known* d=77 maxima. This tool tallies the known
//! maxima by (blank cell, h bucket) so the next sweep can seed from the buckets
//! the missing pair most likely inhabit, and dumps maxima ranks at/above a
//! given h as seeds.
//!
//! Run: cargo run --release --features "mmap parallel" --example d77_maxima \
//!          -- [MIN_H_SEED] [CACHE]   (MIN_H_SEED default 68; ranks -> stdout)

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let min_h_seed: u8 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(68);
    let cache_path: PathBuf = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let c = cache::load(&cache_path)?;
    eprintln!("loaded cache: {} entries", c.len());

    let d78: HashSet<u64> = c
        .iter()
        .filter(|(_, &d)| d == 78)
        .map(|(&r, _)| r)
        .collect();
    let d77: Vec<u64> = c
        .iter()
        .filter(|(_, &d)| d == 77)
        .map(|(&r, _)| r)
        .collect();
    eprintln!("d77 cached: {}, d78 cached: {}", d77.len(), d78.len());

    // (blank, h) -> count for maxima; also count non-maxima for contrast.
    let mut max_by_blank: BTreeMap<u8, u32> = BTreeMap::new();
    let mut max_by_h: BTreeMap<u8, u32> = BTreeMap::new();
    let mut max_by_blank_h: BTreeMap<(u8, u8), u32> = BTreeMap::new();
    let mut n_max = 0u32;
    let mut n_nonmax = 0u32;
    let mut seeds: Vec<(u64, u8, u8)> = Vec::new(); // (rank, blank, h)

    for &r in &d77 {
        let s = unrank(r);
        let blank = s.blank_pos();
        // maximum iff no puzzle-move neighbor is a cached d78 board
        let is_max = !State::legal_moves_at(blank).iter().any(|m| {
            let (ns, _) = s.apply_at(m, blank);
            d78.contains(&rank(&ns))
        });
        if !is_max {
            n_nonmax += 1;
            continue;
        }
        n_max += 1;
        let mut stats = SearchStats::default();
        let (hv, _) = h.root(&s, &mut stats);
        *max_by_blank.entry(blank).or_insert(0) += 1;
        *max_by_h.entry(hv).or_insert(0) += 1;
        *max_by_blank_h.entry((blank, hv)).or_insert(0) += 1;
        if hv >= min_h_seed {
            seeds.push((r, blank, hv));
        }
    }

    eprintln!("\nd77 maxima: {n_max}   non-maxima: {n_nonmax}");
    eprintln!("\nmaxima by blank cell:");
    for (b, n) in &max_by_blank {
        eprintln!("  blank@{b:>2}: {n:>6}");
    }
    eprintln!("\nmaxima by h:");
    for (hv, n) in &max_by_h {
        eprintln!("  h={hv:>2}: {n:>6}");
    }
    eprintln!("\nmaxima by (blank, h) — sparsest buckets are where rare maxima live:");
    let mut bh: Vec<((u8, u8), u32)> = max_by_blank_h.into_iter().collect();
    bh.sort_by_key(|&(_, n)| n);
    for ((b, hv), n) in bh.iter().take(30) {
        eprintln!("  blank@{b:>2} h={hv:>2}: {n:>5}");
    }

    eprintln!(
        "\nemitting {} maxima seeds with h >= {} to stdout",
        seeds.len(),
        min_h_seed
    );
    seeds.sort_by_key(|&(_, b, hv)| (b, std::cmp::Reverse(hv)));
    for (r, _, _) in &seeds {
        println!("{r}");
    }
    Ok(())
}
