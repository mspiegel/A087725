//! K-step expand from an explicit list of known residue board ranks, verify
//! each cache-miss with IDA*, report the resulting depth histogram. Used to
//! grow the residue pocket starting from concrete examples we've already
//! found.
//!
//! Run: `cargo run --release --features "mmap parallel" --example residue_pocket -- CACHE K RANK [RANK...]`

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, LinearConflictInc, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::{Move, State};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_path: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: residue_pocket CACHE K RANK [RANK...]")?
        .into();
    let k: u8 = std::env::args()
        .nth(2)
        .ok_or("usage: residue_pocket CACHE K RANK [RANK...]")?
        .parse()?;
    let seeds: Vec<u64> = std::env::args()
        .skip(3)
        .filter_map(|a| a.parse::<u64>().ok())
        .collect();
    if seeds.is_empty() {
        eprintln!("need at least one seed rank");
        std::process::exit(2);
    }
    println!(
        "config: K={}, {} seed ranks, cache={}",
        k,
        seeds.len(),
        cache_path.display()
    );

    let t0 = Instant::now();
    let cache = cache::load(&cache_path)?;
    println!("loaded cache: {} entries", cache.len());

    let antipodes: HashSet<u64> = load_u48_ranks(Path::new("data/enum15/depth80.ranks"))?;
    println!("loaded {} antipodes", antipodes.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    // K-step expansion from each seed, union into a global candidate set.
    let mut candidates: HashSet<u64> = HashSet::new();
    for &r in &seeds {
        k_step_expand(r, k, &cache, &antipodes, &seeds, &mut candidates);
    }
    println!(
        "collected {} cache-miss candidates from K={} expansion",
        candidates.len(),
        k
    );

    let cands: Vec<u64> = candidates.into_iter().collect();
    if cands.is_empty() {
        println!("no candidates to verify");
        return Ok(());
    }

    // Parallel IDA*-verify.
    let t_verify = Instant::now();
    let results: Vec<(u64, u8)> = cands
        .par_iter()
        .map(|&r| {
            let (sol, _) = idastar_inc_with_stats(&unrank(r), &h);
            (r, sol.map(|v| v.len() as u8).unwrap_or(u8::MAX))
        })
        .collect();
    println!(
        "verified {} cands in {:.1?} ({:.1}/s)",
        cands.len(),
        t_verify.elapsed(),
        cands.len() as f64 / t_verify.elapsed().as_secs_f64()
    );

    // Depth histogram.
    let mut hist: BTreeMap<u8, u32> = BTreeMap::new();
    for &(_, d) in &results {
        *hist.entry(d).or_insert(0) += 1;
    }
    println!("\nverified depth histogram:");
    for (d, c) in &hist {
        let mark = if *d >= 76 { " ***" } else { "" };
        println!("  d={d:>3}: {c:>6}{mark}");
    }

    // Dump deep finds with cell grids.
    for target_d in (76..=80).rev() {
        let finds: Vec<u64> = results
            .iter()
            .filter(|&&(_, d)| d == target_d)
            .map(|&(r, _)| r)
            .collect();
        if finds.is_empty() {
            continue;
        }
        println!("\n=== d={} finds: {} ===", target_d, finds.len());
        for (i, &r) in finds.iter().enumerate() {
            let s = unrank(r);
            print!(
                "  #{:>3} rank {:>14} blank@{:>2}: ",
                i + 1,
                r,
                s.blank_pos()
            );
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
        if finds.len() > 30 {
            println!("  ... ({} more)", finds.len() - 30);
        }
    }

    // Optional: write all solved (rank, depth) pairs for cache_insert --from-pairs.
    if let Ok(out) = std::env::var("POCKET_OUT") {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&out)?);
        let mut n = 0u64;
        for &(r, d) in &results {
            if d != u8::MAX {
                writeln!(w, "{r} {d}")?;
                n += 1;
            }
        }
        w.flush()?;
        println!("wrote {n} (rank depth) pairs -> {out}");
    }

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}

fn k_step_expand(
    r: u64,
    k: u8,
    cache: &HashMap<u64, u8>,
    antipodes: &HashSet<u64>,
    seed_set: &[u64],
    out: &mut HashSet<u64>,
) {
    if k == 0 {
        return;
    }
    let s = unrank(r);
    let mut seen: HashSet<u64> = HashSet::from([r]);
    for &s_other in seed_set {
        seen.insert(s_other);
    }
    let mut frontier: Vec<(State, Option<Move>)> = vec![(s, None)];
    for _ in 0..k {
        let mut next = Vec::with_capacity(frontier.len() * 3);
        for &(ref x, last) in &frontier {
            let blank = x.blank_pos();
            for m in State::legal_moves_at(blank).iter() {
                if last.is_some_and(|lm| m == lm.inverse()) {
                    continue;
                }
                let (ns, _) = x.apply_at(m, blank);
                let nr = rank(&ns);
                if seen.insert(nr) {
                    next.push((ns, Some(m)));
                    if !cache.contains_key(&nr) && !antipodes.contains(&nr) {
                        out.insert(nr);
                    }
                }
            }
        }
        frontier = next;
    }
}

fn load_u48_ranks(path: &Path) -> Result<HashSet<u64>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut bytes)?;
    let mut out = HashSet::with_capacity(bytes.len() / 6);
    for chunk in bytes.chunks_exact(6) {
        let mut le = [0u8; 8];
        le[..6].copy_from_slice(chunk);
        out.insert(u64::from_le_bytes(le));
    }
    Ok(out)
}
