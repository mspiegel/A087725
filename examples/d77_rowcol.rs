//! Combined row + column confinement of cached d=77 maxima.
//!
//! For each tile, report which rows and which columns it occupies across all
//! d=77 maxima (and the high-h subset), plus the number of distinct CELLS it
//! occupies (= the row x column intersection). Tight per-tile cell-sets mean
//! the maxima live in a small combinatorial space that can be enumerated
//! exhaustively — a seed-independent route to the missing pair.
//!
//! Run: cargo run --release --features "mmap parallel" --example d77_rowcol -- [MIN_H] [CACHE]

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::PathBuf;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let min_h: u8 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cache_path: PathBuf = std::env::args().nth(2).unwrap_or_else(|| "data/enum15/solve_cache.bin".into()).into();

    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    let c = cache::load(&cache_path)?;
    let d78: HashSet<u64> = c.iter().filter(|(_,&d)| d==78).map(|(&r,_)| r).collect();

    // tile -> cell -> count, over d=77 maxima with h>=min_h
    let mut cell_count = [[0u32;16];16]; // [tile][cell]
    let mut n = 0u32;
    for (&r,&d) in &c {
        if d!=77 { continue; }
        let s=unrank(r); let b=s.blank_pos();
        let is_max = !State::legal_moves_at(b).iter().any(|m|{let(ns,_)=s.apply_at(m,b); d78.contains(&rank(&ns))});
        if !is_max { continue; }
        if min_h>0 { let mut st=SearchStats::default(); if h.root(&s,&mut st).0 < min_h { continue; } }
        n+=1;
        for cell in 0..16 { cell_count[s.0[cell] as usize][cell]+=1; }
    }
    println!("d=77 maxima with h>={min_h}: {n}\n");

    println!("tile : #cells | rows{{r:pct}} | cols{{c:pct}} | cells(count)");
    let mut prod_log = 0f64;
    for t in 1..=15u8 {
        let mut cells: Vec<(usize,u32)> = (0..16).filter(|&cl| cell_count[t as usize][cl]>0)
            .map(|cl|(cl,cell_count[t as usize][cl])).collect();
        cells.sort_by_key(|&(_,c)| std::cmp::Reverse(c));
        let ncells = cells.len();
        prod_log += (ncells as f64).ln();
        // row/col marginals
        let mut rows: BTreeMap<usize,u32>=BTreeMap::new();
        let mut cols: BTreeMap<usize,u32>=BTreeMap::new();
        for &(cl,cnt) in &cells { *rows.entry(cl/4).or_insert(0)+=cnt; *cols.entry(cl%4).or_insert(0)+=cnt; }
        let rstr: String = rows.iter().map(|(r,c)| format!("{r}:{}%",100*c/n)).collect::<Vec<_>>().join(",");
        let cstr: String = cols.iter().map(|(cl,c)| format!("{cl}:{}%",100*c/n)).collect::<Vec<_>>().join(",");
        let cellstr: String = cells.iter().take(8).map(|&(cl,cnt)| format!("{cl}({cnt})")).collect::<Vec<_>>().join(" ");
        println!("T{t:>2} : {ncells:>2}     | {rstr:<28} | {cstr:<28} | {cellstr}");
    }
    // blank cell distribution
    let mut blanks: BTreeSet<usize> = BTreeSet::new();
    let mut blank_count=[0u32;16];
    for (&r,&d) in &c {
        if d!=77 {continue;}
        let s=unrank(r); let b=s.blank_pos();
        let is_max = !State::legal_moves_at(b).iter().any(|m|{let(ns,_)=s.apply_at(m,b); d78.contains(&rank(&ns))});
        if !is_max {continue;}
        if min_h>0 { let mut st=SearchStats::default(); if h.root(&s,&mut st).0 < min_h { continue; } }
        blanks.insert(b as usize); blank_count[b as usize]+=1;
    }
    println!("\nblank cells: {:?}", blanks.iter().map(|&b| format!("{b}({})",blank_count[b])).collect::<Vec<_>>());
    println!("\nrough product of per-tile distinct-cell counts: e^{:.1} = {:.2e}", prod_log, prod_log.exp());
    Ok(())
}
