//! `rceiling24` — Phase 1C / Option 1: the **R-ceiling** analysis.
//!
//! Question: can *any* k=6 partition (and hence a collection, which takes the MAX
//! over partitions) reach a higher admissible heuristic on the deep board
//! `R` (the 180° rotation) than plain Walking Distance does? WD scores 140 on `R`
//! and the bounded-LB frontier there is 144 (`data/ladder24_calibration.txt`); a
//! 6-tile additive/zero-aware PDB scores only ~126 because the additive relaxation
//! collapses on a fully-scrambled board. A collection cannot exceed its
//! best single partition's value (max, not sum), so this measures that ceiling
//! over a *curated, diverse* set of k=6 partitions.
//!
//! Method: for each candidate 6-6-6-6 partition, build its four zero-aware PDBs in
//! RAM (`build_zpdb_parallel` → `ZPatternDb::from_dist`, nothing written to disk),
//! then evaluate the production Korf-max `ZpdbInc::root` h on a set of constructed
//! deep boards, alongside Manhattan / Linear-Conflict / Walking-Distance for
//! reference. Reports per-partition h, the MAX across partitions, and the verdict
//! on `R`.
//!
//! Run: `cargo run --release --features parallel,mmap --example rceiling24`
//! (~15 min: ~8 partitions × 4 PDBs × ~24 s, in-RAM; WD table warm-up ~25 s).

use std::time::Instant;

use puzzle8::puzzle24::pdb::pattern::Pattern;
use puzzle8::puzzle24::pdb::zbuild::build_zpdb_parallel;
use puzzle8::puzzle24::pdb::zdb::ZPatternDb;
use puzzle8::puzzle24::pdb::ZpdbInc;
use puzzle8::puzzle24::search::{
    Heuristic, IncHeuristic, LinearConflictHeuristic, ManhattanHeuristic, SearchStats,
    WalkingDistanceHeuristic,
};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};
use puzzle8::puzzle24::symmetry::reflect;

/// `R` = 180° board rotation: blank fixed at cell 0, tile `25-i` at cell `i`.
fn r_board() -> State {
    let mut c = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        c[i] = (25 - i) as u8;
    }
    State(c)
}

/// No-immediate-undo random walk of `len` steps from GOAL (deep general boards).
fn random_walk(seed: u64, len: u32) -> State {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = GOAL;
    let mut last: Option<Move> = None;
    for _ in 0..len {
        let opts: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| last.map_or(true, |p: Move| m != p.inverse()))
            .collect();
        let m = opts[(next() as usize) % opts.len()];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

/// A seeded random disjoint 6-6-6-6 partition of tiles 1..=24.
fn random_partition(seed: u64) -> [[u8; 6]; 4] {
    let mut tiles: Vec<u8> = (1..=24u8).collect();
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // Fisher-Yates.
    for i in (1..tiles.len()).rev() {
        let j = (next() as usize) % (i + 1);
        tiles.swap(i, j);
    }
    let mut p = [[0u8; 6]; 4];
    for (b, block) in p.iter_mut().enumerate() {
        block.copy_from_slice(&tiles[b * 6..b * 6 + 6]);
    }
    p
}

/// Assert a partition is pairwise-disjoint and covers tiles 1..=24 exactly.
fn check_partition(name: &str, p: &[[u8; 6]; 4]) {
    let mut seen = [false; 25];
    for block in p {
        for &t in block {
            assert!((1..=24).contains(&t), "{}: tile {} out of range", name, t);
            assert!(!seen[t as usize], "{}: tile {} repeats", name, t);
            seen[t as usize] = true;
        }
    }
    for (t, &s) in seen.iter().enumerate().take(25).skip(1) {
        assert!(s, "{}: tile {} missing", name, t);
    }
}

/// Build a partition's four zero-aware PDBs in RAM (no disk).
fn build_partition(p: &[[u8; 6]; 4]) -> Vec<ZPatternDb> {
    p.iter()
        .map(|block| {
            let pattern = Pattern::new(block);
            let (dist, _) = build_zpdb_parallel(pattern);
            ZPatternDb::from_dist(pattern, &dist)
        })
        .collect()
}

/// Production Korf-max zero-aware h (`max(Σ zpdb, Σ refl-zpdb)`) of a partition.
fn zpdb_root_h(dbs: &[ZPatternDb], s: &State) -> u8 {
    let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
    let mut stats = SearchStats::default();
    inc.root(s, &mut stats).0
}

fn main() {
    println!("rceiling24 — Phase 1C Option-1 R-ceiling analysis\n");

    // Constructed deep boards. R is the canonical deep board (Rokicki LB 152; the
    // true antipode is unknown). reflect(R) is its diagonal image. Deep random
    // walks give deep *general* boards for context.
    let mut boards: Vec<(String, State)> = vec![
        ("R".into(), r_board()),
        ("reflect(R)".into(), reflect(&r_board())),
    ];
    for len in [200u32, 300, 400] {
        let s = random_walk(0x24_C0FFEE_0000 ^ (len as u64), len);
        boards.push((format!("walk{}", len), s));
    }

    // Reference (non-PDB) heuristics. WD on R is the number to beat (≈140).
    WalkingDistanceHeuristic::warm_up_verbose();

    // Candidate partitions (our convention; tiles 1..=24, blank goal cell 24).
    let mut partitions: Vec<(String, [[u8; 6]; 4])> = vec![
        // Canonical Korf 6-6-6-6 (our existing built set).
        ("korf".into(), [[1, 2, 3, 6, 7, 8], [4, 5, 9, 10, 14, 15], [11, 12, 16, 17, 21, 22], [13, 18, 19, 20, 23, 24]]),
        // Value-contiguous strata.
        ("strata".into(), [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]),
        // Alternate spatial blocks (left 2 cols / next block / middle row / bottom).
        ("spatial".into(), [[1, 2, 6, 7, 11, 12], [3, 4, 5, 8, 9, 10], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]),
    ];
    for seed in 0..5u64 {
        partitions.push((format!("rnd{}", seed), random_partition(0x9E3779B97F4A7C15u64.wrapping_mul(seed + 1))));
    }

    // h[partition][board].
    let n_boards = boards.len();
    let mut hmat: Vec<Vec<u8>> = Vec::with_capacity(partitions.len());
    let mut names: Vec<String> = Vec::with_capacity(partitions.len());
    for (name, p) in &partitions {
        check_partition(name, p);
        let t = Instant::now();
        let dbs = build_partition(p);
        let row: Vec<u8> = boards.iter().map(|(_, s)| zpdb_root_h(&dbs, s)).collect();
        eprintln!("  built+scored partition {:<8} in {:?}: {:?}", name, t.elapsed(), row);
        names.push(name.clone());
        hmat.push(row);
        // dbs dropped here (frees ~88 MiB before the next partition).
    }

    // Reference heuristics per board.
    let md: Vec<u8> = boards.iter().map(|(_, s)| ManhattanHeuristic.h(s)).collect();
    let lc: Vec<u8> = boards.iter().map(|(_, s)| LinearConflictHeuristic.h(s)).collect();
    let wd: Vec<u8> = boards.iter().map(|(_, s)| WalkingDistanceHeuristic.h(s)).collect();

    // Sanity: WD on R reproduces the documented 140.
    assert_eq!(wd[0], 140, "WD(R) expected 140, got {}", wd[0]);

    // ---- Report ----
    println!("\n=== per-board heuristic values ===");
    println!(
        "{:<12} {:>4} {:>4} {:>4} | {:>8} {:<10} | per-partition zpdb",
        "board", "MD", "LC", "WD", "zpdb-max", "best"
    );
    let mut r_zpdb_max = 0u8;
    let mut r_best = String::new();
    for b in 0..n_boards {
        let mut bestv = 0u8;
        let mut besti = 0usize;
        for (pi, row) in hmat.iter().enumerate() {
            if row[b] > bestv {
                bestv = row[b];
                besti = pi;
            }
        }
        let per: Vec<String> = names.iter().zip(&hmat).map(|(n, row)| format!("{}={}", n, row[b])).collect();
        println!(
            "{:<12} {:>4} {:>4} {:>4} | {:>8} {:<10} | {}",
            boards[b].0, md[b], lc[b], wd[b], bestv, names[besti], per.join(" ")
        );
        if boards[b].0 == "R" {
            r_zpdb_max = bestv;
            r_best = names[besti].clone();
        }
    }

    println!("\n=== VERDICT (R) ===");
    println!("  WD(R)               = {}", wd[0]);
    println!("  best k6 zpdb(R)     = {}  (partition: {})", r_zpdb_max, r_best);
    println!("  bounded-LB frontier = 144 (WD, 120s; data/ladder24_calibration.txt)");
    if r_zpdb_max > wd[0] {
        println!(
            "  => A k6 collection CAN beat WD on R (best {} > WD {}). Build the collection and \n     measure bounded-LB on R.",
            r_zpdb_max, wd[0]
        );
    } else {
        println!(
            "  => No k6 partition beats WD on R (best {} <= WD {}). A collection's value is the \n     GENERAL regime, not deep boards like R — keep WD/the selector for R.",
            r_zpdb_max, wd[0]
        );
    }
}
