//! Simultaneous d75 + d76 hunt. For each quadrant tile-set partition, enumerate
//! within-quadrant arrangements ONCE, keep solvable + cache-miss + h>=59, score
//! with the depth>=75 structure model, and route by PARITY:
//!   even-h  -> d76 candidate   (h>=60)
//!   odd-h   -> d75 candidate   (h>=59)
//! (board depth always has h's parity, so even-h boards are d76 and odd-h are
//! d75; the model predicts depth>=75 so it ranks both.) Keeps per-partition
//! top-K of each, writes OUT.d76 and OUT.d75 (ranks, score-descending).
//!
//! Run: cargo run --release --features "mmap parallel" --example d75d76_hunt \
//!          -- PART_GRIDS MODEL.txt K OUT_PREFIX [CACHE]

#[path = "common/tree_model.rs"]
mod tree_model;

use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::rank;
use puzzle8::puzzle15::search::{IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::State;

const QUAD: [[usize; 4]; 4] = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]];

fn perm24() -> Vec<[usize; 4]> {
    let mut out = Vec::with_capacity(24);
    for i in 0..4 { for j in 0..4 { if j==i {continue;} for k in 0..4 { if k==i||k==j {continue;}
        out.push([i, j, k, 6-i-j-k]); } } }
    out
}

/// A scored candidate board: the model score and the board's rank.
#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    rank: u64,
}

fn topk(mut v: Vec<Cand>, k: usize) -> Vec<Cand> {
    if v.len() > k {
        v.select_nth_unstable_by(k, |a, b| b.score.partial_cmp(&a.score).unwrap());
        v.truncate(k);
    }
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grids_path: PathBuf = std::env::args().nth(1).ok_or("usage: d75d76_hunt PART_GRIDS MODEL K OUT_PREFIX [CACHE]")?.into();
    let model_path: PathBuf = std::env::args().nth(2).ok_or("need MODEL")?.into();
    let k: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(500);
    let out_prefix: String = std::env::args().nth(4).ok_or("need OUT_PREFIX")?;
    let cache_path: PathBuf = std::env::args().nth(5).unwrap_or_else(|| "data/enum15/solve_cache.bin".into()).into();

    let t0 = Instant::now();
    let model = tree_model::TreeModel::load(&model_path)?;
    eprintln!("model nfeat={}", model.nfeat);

    let text = fs::read_to_string(&grids_path)?;
    let grids: Vec<[u8;16]> = text.lines().filter_map(|l| {
        let v: Vec<u8> = l.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len()==16 { let mut a=[0u8;16]; a.copy_from_slice(&v); Some(a) } else { None }
    }).collect();

    let mut seen: HashSet<[[u8;4];4]> = HashSet::new();
    let mut partitions: Vec<[[u8;4];4]> = Vec::new();
    for b in &grids {
        let mut key=[[0u8;4];4];
        for (q,cells) in QUAD.iter().enumerate() { for (i,&c) in cells.iter().enumerate(){ key[q][i]=b[c]; } key[q].sort_unstable(); }
        if seen.insert(key) { partitions.push(key); }
    }
    eprintln!("partitions: {}  K={k}", partitions.len());

    let c = cache::load(&cache_path)?;
    eprintln!("cache: {} entries", c.len());
    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7,&p8]);
    let perms = perm24();

    // per partition: (even/d76 top-K, odd/d75 top-K)
    let per: Vec<(Vec<Cand>, Vec<Cand>)> = partitions.par_iter().map(|key| {
        let q: [Vec<[u8;4]>;4] = [
            perms.iter().map(|p| [key[0][p[0]],key[0][p[1]],key[0][p[2]],key[0][p[3]]]).collect(),
            perms.iter().map(|p| [key[1][p[0]],key[1][p[1]],key[1][p[2]],key[1][p[3]]]).collect(),
            perms.iter().map(|p| [key[2][p[0]],key[2][p[1]],key[2][p[2]],key[2][p[3]]]).collect(),
            perms.iter().map(|p| [key[3][p[0]],key[3][p[1]],key[3][p[2]],key[3][p[3]]]).collect(),
        ];
        let mut st = SearchStats::default();
        let mut feats = [0f64; 17];
        let mut even: Vec<Cand> = Vec::new();
        let mut odd: Vec<Cand> = Vec::new();
        for a in &q[0] { for b in &q[1] { for cc in &q[2] { for d in &q[3] {
            let mut g=[0u8;16];
            for t in 0..4 { g[QUAD[0][t]]=a[t]; g[QUAD[1][t]]=b[t]; g[QUAD[2][t]]=cc[t]; g[QUAD[3][t]]=d[t]; }
            let s = State(g);
            if !s.is_solvable() { continue; }
            let r = rank(&s);
            if c.contains_key(&r) { continue; }
            let hv = h.root(&s, &mut st).0;
            if hv < 59 { continue; }   // d76 even>=60, d75 odd>=59
            feats[0] = s.blank_pos() as f64;
            for i in 0..16 { feats[i+1] = g[i] as f64; }
            let sc = model.score(&feats) as f32;
            if hv % 2 == 0 { even.push(Cand { score: sc, rank: r }); } else { odd.push(Cand { score: sc, rank: r }); }
        }}}}
        (topk(even, k), topk(odd, k))
    }).collect();

    let mut d76: Vec<Cand> = Vec::new();
    let mut d75: Vec<Cand> = Vec::new();
    for (e, o) in per { d76.extend(e); d75.extend(o); }
    d76.sort_unstable_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    d75.sort_unstable_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    eprintln!("kept d76(even): {}  d75(odd): {}  in {:.1?}", d76.len(), d75.len(), t0.elapsed());

    // per-layer files: "rank score layer" (score-descending)
    for (lbl, v) in [("d76", &d76), ("d75", &d75)] {
        let p = format!("{out_prefix}.{lbl}");
        let mut w = BufWriter::new(fs::File::create(&p)?);
        for c in v.iter() { writeln!(w, "{} {} {lbl}", c.rank, c.score)?; }
        w.flush()?;
        let hi = v.iter().filter(|c| c.score >= 0.9).count();
        eprintln!("wrote {} -> {p}  (score>=0.9: {hi})", v.len());
    }
    // INTERLEAVED merge: both layers sorted together by score, ranks only,
    // for top-down verify_depths (finds whichever layer's residue scores highest).
    let mut merged: Vec<Cand> = d76.iter().copied().chain(d75.iter().copied()).collect();
    merged.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let pm = format!("{out_prefix}.merged");
    let mut w = BufWriter::new(fs::File::create(&pm)?);
    for c in merged.iter() { writeln!(w, "{}", c.rank)?; }
    w.flush()?;
    eprintln!("wrote {} merged ranks (score-interleaved) -> {pm}", merged.len());
    Ok(())
}
