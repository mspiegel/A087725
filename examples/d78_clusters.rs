//! Tile-position clustering on the d=78 boards in the cache. Uses connected
//! components on the Hamming-distance graph at threshold `eps` (default 6).
//! For each cluster reports: size, universal tile signature (cells where
//! every member agrees on a tile), dominant signature (≥80% agree),
//! reflection-pair structure, h-distribution, blank-cell distribution, and
//! mean intra-cluster Hamming distance.
//!
//! Goal: surface tile-arrangement clusters that the univariate (h, blank)
//! outlier analysis missed. The smallest dense clusters are the most
//! interesting candidates for residue hunting — generate-and-verify each
//! cluster's signature space and look for missing d=78 boards.
//!
//! Run: `cargo run --release --features "mmap parallel" --example d78_clusters [EPS] [DEPTH] [CACHE]`

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;
use puzzle8::puzzle15::symmetry::reflect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eps: u8 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let depth: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(78);
    let cache_path: PathBuf = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "data/enum15/solve_cache.bin".into())
        .into();
    // Optional: restrict the clustering population to boards whose blank
    // sits in this cell. Passing this filter lets us find "sub-singletons"
    // within a blank-cell sub-population.
    let blank_filter: Option<u8> = std::env::args().nth(4).and_then(|s| s.parse().ok());

    let t0 = Instant::now();
    let cache = cache::load(&cache_path)?;
    println!("loaded {} cache entries", cache.len());

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let mut boards: Vec<u64> = cache.iter()
        .filter(|(_, &d)| d == depth)
        .map(|(&r, _)| r)
        .collect();

    if let Some(bf) = blank_filter {
        boards.retain(|&r| unrank(r).blank_pos() == bf);
        println!("filtered to blank@{}: {} boards remain", bf, boards.len());
    }
    let n = boards.len();
    println!("clustering {} d={} boards at eps={}\n", n, depth, eps);

    let states: Vec<State> = boards.iter().map(|&r| unrank(r)).collect();

    let h_values: Vec<u8> = states.par_iter()
        .map(|s| {
            let mut stats = SearchStats::default();
            let (hv, _) = h.root(s, &mut stats);
            hv
        })
        .collect();

    // Build connected components via union-find at Hamming distance ≤ eps.
    let t_cluster = Instant::now();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if hamming(&states[i], &states[j]) <= eps {
                uf.union(i, j);
            }
        }
    }
    let labels: Vec<usize> = (0..n).map(|i| uf.find(i)).collect();

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &lbl) in labels.iter().enumerate() {
        clusters.entry(lbl).or_default().push(i);
    }
    let n_clusters = clusters.len();
    let n_singletons = clusters.values().filter(|v| v.len() == 1).count();
    println!("clustering done in {:.1?}", t_cluster.elapsed());
    println!("found {} clusters ({} singletons)", n_clusters, n_singletons);

    // Cluster size histogram.
    let mut size_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for v in clusters.values() {
        *size_hist.entry(v.len()).or_insert(0) += 1;
    }
    println!("\ncluster size distribution:");
    for (size, count) in &size_hist {
        println!("  size {:>4}: {} clusters", size, count);
    }

    // Sort clusters by size descending.
    let mut cluster_list: Vec<(usize, Vec<usize>)> = clusters.into_iter().collect();
    cluster_list.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));

    // Print top N largest clusters.
    println!("\n=== Top 10 largest clusters ===");
    for (i, (_, members)) in cluster_list.iter().take(10).enumerate() {
        if members.len() < 2 { break; }
        print_cluster(i + 1, members, &states, &h_values, &boards);
    }

    // Print smallest non-singleton clusters (potential extreme outliers).
    let small: Vec<&(usize, Vec<usize>)> = cluster_list.iter()
        .filter(|(_, v)| v.len() >= 2 && v.len() <= 4)
        .collect();
    println!("\n=== Smallest non-singleton clusters (size 2..=4): {} total ===",
             small.len());
    for (i, (_, members)) in small.iter().enumerate().take(20) {
        print_cluster(i + 1, members, &states, &h_values, &boards);
    }

    // Singleton dump: list every board with no neighbor within Hamming eps.
    // These are the geometric outliers of d=78 — the cleanest "extreme" set
    // for residue hunting.
    let singletons: Vec<usize> = cluster_list.iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(_, v)| v[0])
        .collect();
    if !singletons.is_empty() {
        println!("\n=== All {} singletons (no d={} neighbor within Hamming {}) ===",
                 singletons.len(), depth, eps);
        // For each singleton, find its nearest d=78 neighbor and Hamming to it.
        for (k, &i) in singletons.iter().enumerate() {
            let s = &states[i];
            let mut min_ham = u8::MAX;
            let mut nearest_idx = i;
            for j in 0..n {
                if j == i { continue; }
                let d = hamming(s, &states[j]);
                if d < min_ham { min_ham = d; nearest_idx = j; }
            }
            let refl_r = rank(&reflect(s));
            let is_self_sym = refl_r == boards[i];
            println!("\nSingleton #{:>2} (rank {}, h={}, blank@{}, nearest d=78 at Hamming {}):",
                     k + 1, boards[i], h_values[i], s.blank_pos(), min_ham);
            println!("  grid:");
            for row in 0..4 {
                print!("    ");
                for col in 0..4 {
                    let t = s.0[row * 4 + col];
                    if t == 0 { print!(" __"); } else { print!(" {:>2}", t); }
                }
                println!();
            }
            println!("  reflection: rank {} {}",
                     refl_r,
                     if is_self_sym { "(self-symmetric)" } else { "(distinct pair)" });
            // Show the nearest neighbor's grid for context.
            let nb = &states[nearest_idx];
            println!("  nearest d=78 (rank {}, h={}, blank@{}):",
                     boards[nearest_idx], h_values[nearest_idx], nb.blank_pos());
            for row in 0..4 {
                print!("    ");
                for col in 0..4 {
                    let t = nb.0[row * 4 + col];
                    if t == 0 { print!(" __"); } else { print!(" {:>2}", t); }
                }
                println!();
            }
        }
    }

    // Quick scan: clusters that pass a "compact + extreme-h" filter.
    println!("\n=== Compact + extreme-h clusters (size 2..=20, fixed_cells>=8, min_h<=64 or max_h>=68) ===");
    let mut interesting: Vec<&(usize, Vec<usize>)> = cluster_list.iter()
        .filter(|(_, v)| v.len() >= 2 && v.len() <= 20)
        .filter(|(_, v)| {
            let n_fixed = signature(v, &states).iter().filter(|s| s.is_some()).count();
            n_fixed >= 8
        })
        .filter(|(_, v)| {
            let h_min = v.iter().map(|&i| h_values[i]).min().unwrap();
            let h_max = v.iter().map(|&i| h_values[i]).max().unwrap();
            h_min <= 64 || h_max >= 68
        })
        .collect();
    interesting.sort_by_key(|(_, v)| v.len());
    println!("found {} interesting clusters", interesting.len());
    for (i, (_, members)) in interesting.iter().enumerate().take(15) {
        print_cluster(i + 1, members, &states, &h_values, &boards);
    }

    println!("\ntotal elapsed {:.1?}", t0.elapsed());
    Ok(())
}

fn hamming(a: &State, b: &State) -> u8 {
    let mut d = 0;
    for i in 0..16 { if a.0[i] != b.0[i] { d += 1; } }
    d
}

/// Universal signature: cells where every cluster member has the same tile.
fn signature(members: &[usize], states: &[State]) -> [Option<u8>; 16] {
    let mut sig = [None; 16];
    for cell in 0..16 {
        let first = states[members[0]].0[cell];
        if members.iter().all(|&i| states[i].0[cell] == first) {
            sig[cell] = Some(first);
        }
    }
    sig
}

/// Dominant signature: cells where ≥80% of cluster members agree on a tile,
/// returning that majority tile.
fn dominant_signature(members: &[usize], states: &[State]) -> [Option<u8>; 16] {
    let mut sig = [None; 16];
    for cell in 0..16 {
        let mut freqs: HashMap<u8, u32> = HashMap::new();
        for &i in members {
            *freqs.entry(states[i].0[cell]).or_insert(0) += 1;
        }
        if let Some((&tile, &count)) = freqs.iter().max_by_key(|(_, c)| **c) {
            if count as f64 / members.len() as f64 >= 0.8 {
                sig[cell] = Some(tile);
            }
        }
    }
    sig
}

fn reflection_structure(members: &[usize], states: &[State], boards: &[u64]) -> (usize, usize, usize) {
    let member_ranks: HashSet<u64> = members.iter().map(|&i| boards[i]).collect();
    let mut self_sym = 0;
    let mut paired = 0;
    let mut singletons = 0;
    for &i in members {
        let refl_r = rank(&reflect(&states[i]));
        if refl_r == boards[i] {
            self_sym += 1;
        } else if member_ranks.contains(&refl_r) {
            paired += 1;
        } else {
            singletons += 1;
        }
    }
    (self_sym, paired / 2, singletons)
}

fn mean_intra_hamming(members: &[usize], states: &[State]) -> f64 {
    let n = members.len();
    if n < 2 { return 0.0; }
    let mut sum = 0u64;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += hamming(&states[members[i]], &states[members[j]]) as u64;
            count += 1;
        }
    }
    sum as f64 / count as f64
}

fn print_cluster(idx: usize, members: &[usize], states: &[State], h_values: &[u8], boards: &[u64]) {
    let n_members = members.len();
    // For very small clusters (2..=4), dump every member's rank with a tag
    // so callers can grep them out.
    if n_members >= 2 && n_members <= 4 {
        for &i in members {
            println!("MEMBER cluster#{} rank={} h={} blank@{}",
                     idx, boards[i], h_values[i], states[i].blank_pos());
        }
    }
    let sig_u = signature(members, states);
    let sig_d = dominant_signature(members, states);
    let n_fixed_u = sig_u.iter().filter(|s| s.is_some()).count();
    let n_fixed_d = sig_d.iter().filter(|s| s.is_some()).count();
    let mean_ham = mean_intra_hamming(members, states);
    let (self_sym, pairs, single) = reflection_structure(members, states, boards);

    let mut h_dist: BTreeMap<u8, u32> = BTreeMap::new();
    for &i in members { *h_dist.entry(h_values[i]).or_insert(0) += 1; }
    let mut blank_dist: BTreeMap<u8, u32> = BTreeMap::new();
    for &i in members { *blank_dist.entry(states[i].blank_pos()).or_insert(0) += 1; }

    println!("\nCluster #{}: {} boards, mean_ham={:.2}, fixed(universal)={}, fixed(dominant_80%)={}",
             idx, n_members, mean_ham, n_fixed_u, n_fixed_d);
    println!("  reflection: {} self-sym, {} pair(s), {} singletons",
             self_sym, pairs, single);
    println!("  universal signature ('?' = varies):");
    for row in 0..4 {
        print!("    ");
        for col in 0..4 {
            match sig_u[row * 4 + col] {
                Some(0) => print!(" __"),
                Some(t) => print!(" {:>2}", t),
                None => print!("  ?"),
            }
        }
        println!();
    }
    if n_fixed_d > n_fixed_u {
        println!("  dominant signature (80% threshold):");
        for row in 0..4 {
            print!("    ");
            for col in 0..4 {
                match sig_d[row * 4 + col] {
                    Some(0) => print!(" __"),
                    Some(t) => print!(" {:>2}", t),
                    None => print!("  ?"),
                }
            }
            println!();
        }
    }
    print!("  h: ");
    for (hv, c) in &h_dist { print!("h{}={} ", hv, c); }
    println!();
    print!("  blanks: ");
    for (b, c) in &blank_dist { print!("@{}={} ", b, c); }
    println!();
}

struct UnionFind {
    parents: Vec<usize>,
    ranks: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parents: (0..n).collect(), ranks: vec![0; n] }
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.parents[i] != i {
            self.parents[i] = self.parents[self.parents[i]];
            i = self.parents[i];
        }
        i
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        match self.ranks[ra].cmp(&self.ranks[rb]) {
            std::cmp::Ordering::Less => self.parents[ra] = rb,
            std::cmp::Ordering::Greater => self.parents[rb] = ra,
            std::cmp::Ordering::Equal => { self.parents[rb] = ra; self.ranks[ra] += 1; }
        }
    }
}
