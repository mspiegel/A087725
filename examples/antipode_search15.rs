//! Anchor-guided completion test for layer 79 (ENUMERATION.md follow-up).
//!
//! Best-first search from the 17 antipodes, ordered by the plain antipode-anchored
//! estimate `anchor_est(s) = 80 − min_i Manhattan(s → antipode_i)`. Each popped
//! board's exact depth is verified by idastar (parallel batches); boards at depth
//! ≥ `floor` are expanded (floor < 78 lets the search dip below the disconnected
//! [78,80] components to reach the depth-79 local maxima). Collect depth-79 boards
//! until all N(79)=70 are found or the solve budget is spent.
//!
//! Instrumented to answer: (a) does anchor-guidance reach the 34 far maxima, (b)
//! how many idastar solves vs the brute band, (c) is the estimate still accurate on
//! the far maxima (the open crux).
//!
//! Run: `cargo run --release --example antipode_search15 --features "mmap parallel" -- data [budget] [floor]`

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use puzzle8::puzzle15::enumerate::{antipodes, histogram};
use puzzle8::puzzle15::pdb::{AdditivePdbHeuristic, MaxHeuristic, PatternDb, ReflectedHeuristic};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{idastar, LinearConflictHeuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::State;

fn position_table(a: &State) -> [u8; 16] {
    let mut pos = [0u8; 16];
    for (cell, &t) in a.0.iter().enumerate() {
        pos[t as usize] = cell as u8;
    }
    pos
}

fn manhattan_to(s: &State, pos_of: &[u8; 16]) -> u32 {
    let mut sum = 0u32;
    for (cell, &t) in s.0.iter().enumerate() {
        if t == 0 {
            continue;
        }
        let q = pos_of[t as usize] as usize;
        sum += ((cell / 4).abs_diff(q / 4) + (cell % 4).abs_diff(q % 4)) as u32;
    }
    sum
}

fn neighbors(r: u64) -> Vec<u64> {
    let s = unrank(r);
    let blank = s.blank_pos();
    let mut out = Vec::with_capacity(4);
    for m in State::legal_moves_at(blank).iter() {
        out.push(rank(&s.apply_at(m, blank).0));
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"));
    let budget: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8000);
    let floor: u8 = args.next().and_then(|s| s.parse().ok()).unwrap_or(77);

    let hist = histogram::load(&dir.join("pdb15_depth_histogram.txt")).expect("histogram");
    let antipode_ranks =
        antipodes::load_ranks(&dir.join("pdb15_antipodes.txt")).expect("antipodes");
    let antipode_set: HashSet<u64> = antipode_ranks.iter().copied().collect();
    let antipode_pos: Vec<[u8; 16]> = antipode_ranks
        .iter()
        .map(|&r| position_table(&unrank(r)))
        .collect();
    let anchor = |r: u64| -> i32 {
        let s = unrank(r);
        80 - antipode_pos
            .iter()
            .map(|p| manhattan_to(&s, p))
            .min()
            .unwrap() as i32
    };

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
    let verify = |r: u64| -> u8 {
        idastar(&unrank(r), &h)
            .map(|v| v.len() as u8)
            .unwrap_or(u8::MAX)
    };

    println!("anchor-guided search: budget {budget} solves, expand floor {floor}");
    let t0 = Instant::now();

    let mut store: HashMap<u64, u8> = HashMap::new();
    let mut enqueued: HashSet<u64> = HashSet::new();
    let mut heap: BinaryHeap<(i32, u64)> = BinaryHeap::new();
    let mut found79: Vec<u64> = Vec::new();
    let mut pophist = [0u64; 82];
    let mut solves = 0u64;

    for &r in &antipode_ranks {
        store.insert(r, 80);
        for nb in neighbors(r) {
            if enqueued.insert(nb) {
                heap.push((anchor(nb), nb));
            }
        }
    }

    const BATCH: usize = 256;
    let want79 = hist[79];
    while !heap.is_empty() && solves < budget && (found79.len() as u64) < want79 {
        let mut batch = Vec::with_capacity(BATCH);
        while batch.len() < BATCH {
            match heap.pop() {
                Some((_, r)) if !store.contains_key(&r) => batch.push(r),
                Some(_) => {}
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let results: Vec<(u64, u8)> = batch.par_iter().map(|&r| (r, verify(r))).collect();
        solves += batch.len() as u64;
        for (r, d) in results {
            if store.insert(r, d).is_some() {
                continue;
            }
            pophist[d.min(81) as usize] += 1;
            if d == 79 {
                found79.push(r);
            }
            if d >= floor {
                for nb in neighbors(r) {
                    if !store.contains_key(&nb) && enqueued.insert(nb) {
                        let a = anchor(nb);
                        if a >= floor as i32 - 2 {
                            heap.push((a, nb));
                        }
                    }
                }
            }
        }
    }

    // ---- report ----
    println!(
        "\nfound {}/{} depth-79 boards in {} solves, {:.1?}",
        found79.len(),
        want79,
        solves,
        t0.elapsed()
    );
    println!("\npop-depth histogram (boards verified at each depth):");
    for d in (72u8..=80).rev() {
        if pophist[d as usize] > 0 {
            println!("  depth {:>2}: {:>7}", d, pophist[d as usize]);
        }
    }

    // anchor accuracy across everything verified.
    let mut err_sum = [0i64; 82];
    let mut err_cnt = [0u64; 82];
    for (&r, &d) in &store {
        let e = anchor(r) - d as i32;
        err_sum[d as usize] += e as i64;
        err_cnt[d as usize] += 1;
    }
    println!("\nanchor accuracy (mean signed error = anchor − true):");
    for d in (76u8..=80).rev() {
        if err_cnt[d as usize] > 0 {
            println!(
                "  depth {:>2}: mean err {:+.2}  (n={})",
                d,
                err_sum[d as usize] as f64 / err_cnt[d as usize] as f64,
                err_cnt[d as usize]
            );
        }
    }

    // The crux: split depth-79 finds into antipode-neighbors vs local maxima.
    let mut neigh = Vec::new();
    let mut maxima = Vec::new();
    for &r in &found79 {
        if neighbors(r).iter().any(|nb| antipode_set.contains(nb)) {
            neigh.push(r);
        } else {
            maxima.push(r);
        }
    }
    let stats = |v: &[u64]| -> (f64, i32, i32) {
        if v.is_empty() {
            return (0.0, 0, 0);
        }
        let errs: Vec<i32> = v.iter().map(|&r| anchor(r) - 79).collect();
        (
            errs.iter().sum::<i32>() as f64 / v.len() as f64,
            *errs.iter().min().unwrap(),
            *errs.iter().max().unwrap(),
        )
    };
    let (nm, nmin, nmax) = stats(&neigh);
    let (mm, mmin, mmax) = stats(&maxima);
    println!("\ndepth-79 split (the crux — accuracy on the FAR maxima):");
    println!(
        "  antipode-neighbors: {:>3}   anchor err mean {:+.2}  range [{:+},{:+}]",
        neigh.len(),
        nm,
        nmin,
        nmax
    );
    println!(
        "  local maxima      : {:>3}   anchor err mean {:+.2}  range [{:+},{:+}]",
        maxima.len(),
        mm,
        mmin,
        mmax
    );
}
