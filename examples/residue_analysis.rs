//! Structural analysis of the deepest cached boards. Goal: find features that
//! distinguish "easy" (descent-reachable) d=77/78 boards from the residue we
//! can't surface with band-BFS at floor=72.
//!
//! For each board, we compute:
//!   - blank cell (0..15)
//!   - degree = number of legal blank moves (2 corner / 3 edge / 4 interior)
//!   - h(s) = ZpdbPlusInc root bound
//!   - above_deg = how many of its neighbors are stored at depth d+1
//!   - below_deg = how many of its neighbors are stored at depth d-1
//!
//! Boards with above_deg == 0 are Bellman-max candidates — descent from above
//! could not have reached them. Comparing their distributions to the residue
//! sample (e.g. d=79 boards not adjacent to any antipode) lets us see if a
//! generation signature exists for the 124 missing d=77 + 4 missing d=78.
//!
//! Run: `cargo run --release --features "mmap parallel" --example residue_analysis`

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    let pdb_dir: PathBuf = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data".into())
        .into();

    let t0 = Instant::now();
    println!("loading cache {} ...", cache_path.display());
    let cache = cache::load(&cache_path)?;
    println!("  {} entries in {:.1?}", cache.len(), t0.elapsed());

    let t1 = Instant::now();
    let p7 = ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);
    println!("loaded PDBs in {:.1?}", t1.elapsed());

    // Antipodes (d=80) are pre-seeded into the BFS Store but never go through
    // verify(), so they're absent from the cache. Load depth80.ranks so the
    // above-neighbor check at d=79 sees them.
    let antipode_path = Path::new("data/enum15/depth80.ranks");
    let antipodes: HashSet<u64> = load_u48_ranks(antipode_path)?;
    println!("loaded {} antipodes from {}", antipodes.len(), antipode_path.display());

    // Restrict analysis to deep boards: d >= 75. The rest is noise.
    let deep: Vec<(u64, u8)> = cache
        .iter()
        .filter(|(_, &d)| d >= 75)
        .map(|(&r, &d)| (r, d))
        .collect();
    println!("{} boards at d>=75 in cache\n", deep.len());

    // Compute per-board features in parallel. We need cache lookups for
    // above/below-degree, so keep the cache borrowed.
    let t2 = Instant::now();
    let features: Vec<Feature> = deep
        .par_iter()
        .map(|&(r, d)| {
            let s = unrank(r);
            let mut stats = SearchStats::default();
            let (hv, _) = h.root(&s, &mut stats);
            let blank = s.blank_pos();
            let lm = State::legal_moves_at(blank);
            let degree = lm.len() as u8;
            let mut above = 0u8;
            let mut below = 0u8;
            for m in lm.iter() {
                let (ns, _) = s.apply_at(m, blank);
                let nr = rank(&ns);
                let nd = cache.get(&nr).copied()
                    .or_else(|| if antipodes.contains(&nr) { Some(80) } else { None });
                if let Some(nd) = nd {
                    if nd == d + 1 { above += 1; }
                    else if d > 0 && nd == d - 1 { below += 1; }
                }
            }
            Feature { rank: r, depth: d, blank, degree, h: hv, above, below }
        })
        .collect();
    println!("computed features in {:.1?}\n", t2.elapsed());

    // Per-depth summary.
    println!("Per-depth summary:");
    println!("{:>5}  {:>9}  {:>5}  {:>5}  {:>6}  {:>10}  {:>10}",
             "depth", "count", "min_h", "max_h", "mean_h", "above=0", "above>=1");
    let by_depth: HashMap<u8, Vec<&Feature>> = group_by(&features, |f| f.depth);
    let mut depths: Vec<u8> = by_depth.keys().copied().collect();
    depths.sort();
    for d in &depths {
        let fs = &by_depth[d];
        let n = fs.len();
        let min_h = fs.iter().map(|f| f.h).min().unwrap_or(0);
        let max_h = fs.iter().map(|f| f.h).max().unwrap_or(0);
        let mean_h = fs.iter().map(|f| f.h as f64).sum::<f64>() / n as f64;
        let bmax = fs.iter().filter(|f| f.above == 0).count();
        let nonmax = n - bmax;
        println!("{d:>5}  {n:>9}  {min_h:>5}  {max_h:>5}  {mean_h:>6.2}  {bmax:>10}  {nonmax:>10}");
    }

    // Blank-cell distribution per depth, separately for Bellman-max candidates
    // (above==0) and the rest. If maxima cluster in specific cells, that's a
    // generation signature.
    println!("\nBlank-cell distribution (Bellman-max candidates | other):");
    println!("  cell layout: 0..3 top row, 4..7, 8..11, 12..15 bottom row");
    for d in &depths {
        let fs = &by_depth[d];
        let mut bmax = [0u32; 16];
        let mut other = [0u32; 16];
        for f in fs {
            if f.above == 0 { bmax[f.blank as usize] += 1; }
            else { other[f.blank as usize] += 1; }
        }
        let nb: u32 = bmax.iter().sum();
        let no: u32 = other.iter().sum();
        println!("  d={d:>2} (max={nb:>6}, oth={no:>6}):");
        for row in 0..4 {
            print!("    bmax:");
            for col in 0..4 {
                print!(" {:>5}", bmax[row * 4 + col]);
            }
            print!("   |   other:");
            for col in 0..4 {
                print!(" {:>5}", other[row * 4 + col]);
            }
            println!();
        }
    }

    // Degree (2/3/4) × above-degree breakdown per depth.
    println!("\nDegree (legal moves) × above-degree breakdown per depth:");
    println!("  Each cell = # boards with that combo; rows = legal-move count");
    for d in &depths {
        let fs = &by_depth[d];
        let mut tab = [[0u32; 5]; 5]; // tab[degree][above]
        for f in fs {
            tab[f.degree as usize][f.above as usize] += 1;
        }
        println!("  d={d:>2}:");
        println!("           above=0 above=1 above=2 above=3 above=4");
        for deg in 2..=4 {
            let row = &tab[deg];
            println!("    deg={}  {:>6} {:>6} {:>6} {:>6} {:>6}",
                     deg, row[0], row[1], row[2], row[3], row[4]);
        }
    }

    // h-distribution of Bellman-max candidates per depth.
    println!("\nh-distribution of Bellman-max candidates (above==0) per depth:");
    for d in &depths {
        let bmaxes: Vec<&&Feature> = by_depth[d].iter().filter(|f| f.above == 0).collect();
        if bmaxes.is_empty() { continue; }
        let mut h_hist: HashMap<u8, u32> = HashMap::new();
        for f in &bmaxes {
            *h_hist.entry(f.h).or_insert(0) += 1;
        }
        let mut hs: Vec<(u8, u32)> = h_hist.into_iter().collect();
        hs.sort();
        print!("  d={:>2} (n={}):", d, bmaxes.len());
        for (hv, c) in hs {
            print!(" h{hv}={c}");
        }
        println!();
    }

    // Targeted: list a handful of d=79 Bellman maxima as concrete examples to
    // eyeball later (we have all 70 d=79 boards in the cache).
    println!("\nFirst 20 d=79 Bellman-max candidates (cell layout):");
    let d79_bmax: Vec<&Feature> = features.iter().filter(|f| f.depth == 79 && f.above == 0).collect();
    for f in d79_bmax.iter().take(20) {
        let s = unrank(f.rank);
        print!("  rank {} blank@{} deg={} h={} above={} below={}: ",
               f.rank, f.blank, f.degree, f.h, f.above, f.below);
        for (i, &t) in s.0.iter().enumerate() {
            if t == 0 { print!("__"); } else { print!("{t:>2}"); }
            if i % 4 == 3 { print!(" "); } else { print!(","); }
        }
        println!();
    }
    println!("(total d=79 Bellman-max candidates: {})", d79_bmax.len());

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}

#[derive(Clone, Copy)]
struct Feature {
    rank: u64,
    depth: u8,
    blank: u8,
    degree: u8,
    h: u8,
    above: u8,
    below: u8,
}

/// Read a u48-packed ranks file (6 bytes little-endian per rank) into a set.
fn load_u48_ranks(path: &Path) -> Result<HashSet<u64>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut bytes)?;
    if bytes.len() % 6 != 0 {
        return Err(format!("{}: not a u48-packed ranks file ({} bytes)",
                           path.display(), bytes.len()).into());
    }
    let mut out = HashSet::with_capacity(bytes.len() / 6);
    for chunk in bytes.chunks_exact(6) {
        let mut le = [0u8; 8];
        le[..6].copy_from_slice(chunk);
        out.insert(u64::from_le_bytes(le));
    }
    Ok(out)
}

fn group_by<T, K, F>(items: &[T], key: F) -> HashMap<K, Vec<&T>>
where
    K: std::hash::Hash + Eq,
    F: Fn(&T) -> K,
{
    let mut out: HashMap<K, Vec<&T>> = HashMap::new();
    for x in items {
        out.entry(key(x)).or_default().push(x);
    }
    out
}
