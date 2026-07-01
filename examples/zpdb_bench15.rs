//! Benchmark: zero-aware Korf 7-8 vs blank-agnostic korf-plus on deep boards.
//!
//! Measures, on the 17 depth-80 antipodes (the worst case): the heuristic value
//! (gap below the true depth 80) and the idastar node count + wall time for each
//! heuristic. This quantifies the zero-aware tightening and the solve speedup.
//!
//! Run: cargo run --release --example zpdb_bench15 --features "mmap parallel" -- data [n_solves]

use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::antipodes;
use puzzle8::puzzle15::pdb::{
    AdditivePdbHeuristic, MaxHeuristic, PatternDb, ReflectedHeuristic, ZPatternDb, ZpdbInc,
};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, idastar_with_stats, Heuristic, IncHeuristic,
    LinearConflictHeuristic, SearchStats, WalkingDistanceHeuristic,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("data"));
    let n_solves: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let antipode_ranks = antipodes::load_ranks(&dir.join("pdb15_antipodes.txt")).expect("antipodes");

    // Blank-agnostic korf-plus (current heuristic).
    let dbs = [
        PatternDb::load_mmap(&dir.join("pdb15_p7_korf.bin")).expect("p7"),
        PatternDb::load_mmap(&dir.join("pdb15_p8_korf.bin")).expect("p8"),
    ];
    WalkingDistanceHeuristic::warm_up();
    let korf_plus = MaxHeuristic::new(
        MaxHeuristic::new(
            AdditivePdbHeuristic::new(&dbs),
            ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs)),
        ),
        MaxHeuristic::new(LinearConflictHeuristic, WalkingDistanceHeuristic),
    );

    // Zero-aware Korf 7-8 (incremental, max of normal + reflected).
    let zdbs = [
        ZPatternDb::load_mmap(&dir.join("zpdb15_p7.zbin")).expect("zp7"),
        ZPatternDb::load_mmap(&dir.join("zpdb15_p8.zbin")).expect("zp8"),
    ];
    let zpdb = ZpdbInc::new([&zdbs[0], &zdbs[1]]);

    // Raw additive sums (no reflection, no LC/WD) to isolate the zero-aware tightening,
    // plus LC and WD alone, plus a fair "zpdb-plus" = max(zpdb, reflected-zpdb, LC, WD).
    let add_korf = AdditivePdbHeuristic::new(&dbs);
    let add_zpdb = puzzle8::puzzle15::pdb::AdditiveZpdbHeuristic::new(&zdbs);
    let zpdb_plus = MaxHeuristic::new(
        MaxHeuristic::new(
            puzzle8::puzzle15::pdb::AdditiveZpdbHeuristic::new(&zdbs),
            ReflectedHeuristic::new(puzzle8::puzzle15::pdb::AdditiveZpdbHeuristic::new(&zdbs)),
        ),
        MaxHeuristic::new(LinearConflictHeuristic, WalkingDistanceHeuristic),
    );

    // Per-pattern domination check: zpdb_i(s) must be >= korf_i(s) for each pattern.
    println!("Per-pattern domination check (z_i must be >= k_i):");
    println!("  #   k7  z7   k8  z8   | sum_k sum_z  VIOLATION?");
    for (i, &r) in antipode_ranks.iter().enumerate() {
        let s = unrank(r);
        let k7 = dbs[0].h(&s);
        let z7 = zdbs[0].cold_lookup(&s);
        let k8 = dbs[1].h(&s);
        let z8 = zdbs[1].cold_lookup(&s);
        let viol = z7 < k7 || z8 < k8;
        println!("  {:>2}  {:>3} {:>3}  {:>3} {:>3}  | {:>4} {:>4}    {}",
                 i + 1, k7, z7, k8, z8, k7 as u32 + k8 as u32, z7 as u32 + z8 as u32,
                 if viol { format!("YES  p7:{} p8:{}", if z7<k7{"<"}else{"ok"}, if z8<k8{"<"}else{"ok"}) } else { "no".into() });
    }
    println!();

    println!("Heuristic value on the 17 depth-80 antipodes (true depth = 80):");
    println!("  #   add_korf add_zpdb   LC   WD | korf+  zpdb+ | maxKorf maxZpdb");
    let (mut kp_sum, mut za_sum, mut zp_sum) = (0u32, 0u32, 0u32);
    for (i, &r) in antipode_ranks.iter().enumerate() {
        let s = unrank(r);
        let ak = add_korf.h(&s);
        let az = add_zpdb.h(&s);
        let lc = LinearConflictHeuristic.h(&s);
        let wd = WalkingDistanceHeuristic.h(&s);
        let kp = korf_plus.h(&s);
        let zp = zpdb_plus.h(&s);
        let mz = IncHeuristic::root(&zpdb, &s, &mut SearchStats::default()).0;
        kp_sum += kp as u32;
        za_sum += mz as u32;
        zp_sum += zp as u32;
        println!("  {:>2}     {:>3}    {:>3}    {:>3}  {:>3} |  {:>3}    {:>3} |   {:>3}    {:>3}",
                 i + 1, ak, az, lc, wd, kp, zp, kp, mz);
    }
    println!("  zpdb-plus mean = {:.1} (gap {:.1}) vs korf-plus mean = {:.1} (gap {:.1})",
             zp_sum as f64 / antipode_ranks.len() as f64, 80.0 - zp_sum as f64 / antipode_ranks.len() as f64,
             kp_sum as f64 / antipode_ranks.len() as f64, 80.0 - kp_sum as f64 / antipode_ranks.len() as f64);
    let n = antipode_ranks.len() as f64;
    println!(
        "  mean  {:>5.1}      {:>5.1}      gap {:.1} → {:.1}",
        kp_sum as f64 / n, za_sum as f64 / n,
        80.0 - kp_sum as f64 / n, 80.0 - za_sum as f64 / n,
    );

    let k = n_solves.min(antipode_ranks.len());
    println!("\nidastar solve on {} antipodes (nodes / wall):", k);
    println!("  #        korf-plus                 zero-aware            speedup");
    let (mut kp_nodes, mut za_nodes, mut kp_t, mut za_t) = (0u64, 0u64, 0.0f64, 0.0f64);
    for &r in antipode_ranks.iter().take(k) {
        let s = unrank(r);
        let t0 = Instant::now();
        let (sol_kp, st_kp) = idastar_with_stats(&s, &korf_plus);
        let dt_kp = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let (sol_za, st_za) = idastar_inc_with_stats(&s, &zpdb);
        let dt_za = t1.elapsed().as_secs_f64();
        assert_eq!(sol_kp.as_ref().map(|v| v.len()), Some(80));
        assert_eq!(sol_za.as_ref().map(|v| v.len()), Some(80), "zero-aware returned non-optimal!");
        kp_nodes += st_kp.nodes; za_nodes += st_za.nodes; kp_t += dt_kp; za_t += dt_za;
        println!(
            "        {:>13} {:>7.2}s   {:>13} {:>7.2}s    {:>4.1}x nodes, {:>4.1}x time",
            st_kp.nodes, dt_kp, st_za.nodes, dt_za,
            st_kp.nodes as f64 / st_za.nodes.max(1) as f64,
            dt_kp / dt_za.max(1e-9),
        );
    }
    println!(
        "  total  {:>12} {:>7.2}s   {:>13} {:>7.2}s    {:>4.1}x nodes, {:>4.1}x time",
        kp_nodes, kp_t, za_nodes, za_t,
        kp_nodes as f64 / za_nodes.max(1) as f64, kp_t / za_t.max(1e-9),
    );
}
