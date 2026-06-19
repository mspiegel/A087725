//! Enumerate every 15-puzzle board at optimal depth ≥ T, walking T down from 80.
//!
//! Two modes:
//! - `descent` (default): solve-free top-down frontier + Bellman maxima recovery.
//!   Fast, but completes a layer only when its local maxima are reachable from
//!   the antipode shell (true deeper down, not at the very top).
//! - `band`: BFS over the connected band `[floor..80]`, exact depth per board by
//!   idastar (parallel). Reaches the top-layer maxima; cost scales with
//!   `N(≥floor)`, so it is only practical near the top.
//!
//! ```text
//! enumerate15 --pdb-dir data --data-dir data --mode band --down-to 79 --floor 78 --out data/enum
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::{antipodes, cache, frontier, histogram, Store};
use puzzle8::puzzle15::state::DIAMETER;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, LinearConflictInc, WalkingDistanceHeuristic,
};
#[cfg(feature = "verifier-stats")]
use puzzle8::puzzle15::search::SearchStats;
#[cfg(feature = "verifier-stats")]
use std::sync::Mutex;

struct Args {
    pdb_dir: PathBuf,
    data_dir: PathBuf,
    mode_band: bool,
    down_to: u8,
    floor: Option<u8>,
    budget: u64,
    cache: Option<PathBuf>,
    no_cache: bool,
    out: PathBuf,
    no_verify: bool,
    seed_ranks: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = PathBuf::from("data");
    let mut data_dir = PathBuf::from("data");
    let mut mode_band = false;
    let mut down_to: u8 = 79;
    let mut floor: Option<u8> = None;
    let mut budget: u64 = 0;
    let mut cache: Option<PathBuf> = None;
    let mut no_cache = false;
    let mut out = PathBuf::from("data/enum");
    let mut no_verify = false;
    let mut seed_ranks: Option<PathBuf> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => { i += 1; pdb_dir = PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?); }
            "--data-dir" => { i += 1; data_dir = PathBuf::from(argv.get(i).ok_or("--data-dir needs a value")?); }
            "--mode" => {
                i += 1;
                mode_band = match argv.get(i).ok_or("--mode needs a value")?.as_str() {
                    "band" => true,
                    "descent" => false,
                    other => return Err(format!("unknown mode {:?}", other)),
                };
            }
            "--down-to" => { i += 1; down_to = argv.get(i).ok_or("--down-to needs a value")?.parse().map_err(|e| format!("--down-to: {}", e))?; }
            "--floor" => { i += 1; floor = Some(argv.get(i).ok_or("--floor needs a value")?.parse().map_err(|e| format!("--floor: {}", e))?); }
            "--budget" => { i += 1; budget = argv.get(i).ok_or("--budget needs a value")?.parse().map_err(|e| format!("--budget: {}", e))?; }
            "--cache" => { i += 1; cache = Some(PathBuf::from(argv.get(i).ok_or("--cache needs a value")?)); }
            "--no-cache" => { no_cache = true; }
            "--out" => { i += 1; out = PathBuf::from(argv.get(i).ok_or("--out needs a value")?); }
            "--no-verify" => { no_verify = true; }
            "--seed-ranks" => { i += 1; seed_ranks = Some(PathBuf::from(argv.get(i).ok_or("--seed-ranks needs a value")?)); }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    if !(1..=80).contains(&down_to) {
        return Err(format!("--down-to {} out of range 1..=80", down_to));
    }
    if seed_ranks.is_some() && !mode_band {
        return Err("--seed-ranks requires --mode band".into());
    }
    Ok(Args { pdb_dir, data_dir, mode_band, down_to, floor, budget, cache, no_cache, out, no_verify, seed_ranks })
}

fn load_zpdbs(dir: &Path) -> Result<[ZPatternDb; 2], String> {
    let p7 = ZPatternDb::load_mmap(&dir.join("zpdb15_p7.zbin")).map_err(|e| format!("zp7: {}", e))?;
    let p8 = ZPatternDb::load_mmap(&dir.join("zpdb15_p8.zbin")).map_err(|e| format!("zp8: {}", e))?;
    Ok([p7, p8])
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let t0 = Instant::now();

    let hist = histogram::load(&args.data_dir.join("pdb15_depth_histogram.txt"))?;
    let antipode_ranks = antipodes::load_ranks(&args.data_dir.join("pdb15_antipodes.txt"))?;

    let mut store = Store::new();
    let lo = args.floor.unwrap_or(args.down_to.saturating_sub(1)) as usize;
    let reserve: u64 = (lo..=80).map(|d| hist[d]).sum();
    store.reserve(reserve.min(2_000_000_000) as usize);
    for r in &antipode_ranks {
        store.insert(*r, 80);
    }

    // Zero-aware "zpdb-plus" verifier — pointwise dominates the additive
    // korf-plus it replaces (each zero-aware ZPDB ≥ its additive PDB), so it
    // never expands more nodes, while the ZPDB component is advanced
    // incrementally per node. Sync, so it drives the parallel band solves.
    let zdbs = load_zpdbs(&args.pdb_dir)?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);
    // Under `--features verifier-stats`: cross-thread accumulator for the
    // per-solve SearchStats. Lock is held briefly once per solve — solves
    // take milliseconds, so contention is negligible and the resulting totals
    // let us compute ns-per-call per component. Off by default.
    #[cfg(feature = "verifier-stats")]
    let stats_total: Mutex<SearchStats> = Mutex::new(SearchStats::default());
    let verify = |r: u64| -> u8 {
        let (sol, _stats) = idastar_inc_with_stats(&unrank(r), &h);
        #[cfg(feature = "verifier-stats")]
        stats_total.lock().unwrap().add(&_stats);
        sol.map(|v| v.len() as u8).unwrap_or(u8::MAX)
    };

    let reports = if args.mode_band {
        let floor = args.floor.unwrap_or(args.down_to.saturating_sub(1));
        // Resolve cache path (default: on, under the output dir) and load it.
        let cache_path: Option<PathBuf> = if args.no_cache {
            None
        } else {
            Some(args.cache.clone().unwrap_or_else(|| args.out.join("solve_cache.bin")))
        };
        let mut cache = match &cache_path {
            Some(p) => cache::load(p).map_err(|e| format!("cache load {}: {}", p.display(), e))?,
            None => cache::Cache::new(),
        };
        println!(
            "band mode: floor {}, target down-to {}, budget {}, cache {} ({} solved boards loaded)",
            floor, args.down_to,
            if args.budget == 0 { "unlimited".to_string() } else { args.budget.to_string() },
            cache_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "disabled".into()),
            cache.len(),
        );
        // --seed-ranks: ensure the listed ranks are in the cache with their
        // exact depths. Solve any misses with parallel IDA* and persist the
        // cache. Pure cache enrichment — the Store gets populated downstream by
        // auto-seed-from-cache, so this block doesn't touch the Store directly.
        if let Some(seed_path) = args.seed_ranks.as_deref() {
            use rayon::prelude::*;
            let ranks = Store::read_ranks_file(seed_path)
                .map_err(|e| format!("seed-ranks {}: {}", seed_path.display(), e))?;
            let mut cache_hits = 0u64;
            let mut misses: Vec<u64> = Vec::new();
            for r in &ranks {
                if cache.contains_key(r) {
                    cache_hits += 1;
                } else {
                    misses.push(*r);
                }
            }
            let solved: Vec<(u64, u8)> = misses.par_iter().map(|&r| (r, verify(r))).collect();
            for &(r, d) in &solved {
                if d == u8::MAX {
                    return Err(format!("seed rank {} not solvable", r));
                }
                cache.insert(r, d);
            }
            let solved_n = solved.len() as u64;
            if solved_n > 0 {
                if let Some(p) = cache_path.as_deref() {
                    cache::save(p, &cache).map_err(|e| format!("cache save {}: {}", p.display(), e))?;
                }
            }
            println!(
                "seed-ranks {}: {} loaded, {} cache hits, {} solved",
                seed_path.display(), ranks.len(), cache_hits, solved_n,
            );
        }
        // Auto-seed-from-cache: pre-populate the Store with every cached board
        // in the band. Without this, the BFS would re-derive these by walking
        // from the antipode shell — correct but wasteful, and fragile when the
        // cached set has multiple connected components (BFS only reaches the
        // one containing the antipodes). Runs after --seed-ranks so it sees
        // the freshest cache.
        let mut cache_seeded = 0u64;
        for (&r, &d) in &cache {
            if (floor..=DIAMETER).contains(&d) && store.insert(r, d) {
                cache_seeded += 1;
            }
        }
        if cache_seeded > 0 {
            println!("seeded {} boards from cache at depths {}..={}", cache_seeded, floor, DIAMETER);
        }
        frontier::run_band(&mut store, &hist, args.down_to, floor, args.budget,
                           &mut cache, cache_path.as_deref(), &args.out, &verify, |l| println!("{}", l))?
    } else {
        let verifier_opt: Option<&dyn Fn(u64) -> u8> = if args.no_verify { None } else { Some(&verify) };
        frontier::run(&mut store, &hist, args.down_to, &args.out, verifier_opt, |l| println!("{}", l))?
    };

    let all_ok = reports.iter().filter(|r| r.depth >= args.down_to).all(|r| r.complete);
    println!(
        "\n{} boards stored, {:.1?} elapsed → {}",
        store.total(),
        t0.elapsed(),
        if all_ok { "target layers complete" } else { "stopped: a target layer incomplete" },
    );

    // Per-component call totals across every verify solve in this run. The
    // wall-time profile (samply) tells us *where* the CPU went as a % of
    // self-time; these counts tell us *how often* each function ran. Combine
    // them: `ns/call = (profile_pct/100) × total_cpu_ns / calls`, immune to
    // inlining attribution because the counter sits inside the function body.
    // Gated behind `--features verifier-stats` — the counter bumps cost
    // ~5-10% wall time, off by default.
    #[cfg(feature = "verifier-stats")]
    {
        let st = stats_total.into_inner().unwrap();
        let wall_ns = t0.elapsed().as_nanos() as f64;
        let threads: f64 = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let total_cpu_ns = wall_ns * threads;
        let row = |label: &str, calls: u64, hint: &str| {
            let per_call_at_100pct = if calls == 0 {
                "—".to_string()
            } else {
                format!("{:.1} ns/call @100% CPU", total_cpu_ns / calls as f64)
            };
            println!("  {:<18} {:>16}  {:<28} {}", label, calls, per_call_at_100pct, hint);
        };
        println!("\nverifier call totals  (wall {:.2}s × {} threads = {:.1}s CPU):",
                 wall_ns / 1e9, threads as u32, total_cpu_ns / 1e9);
        row("nodes",           st.nodes,           "(one search_inc per call)");
        row("zpdb_advances",   st.zpdb_advances,   "");
        row("lc_advances",     st.lc_advances,     "");
        row("wd_advances",     st.wd_advances,     "");
        row("proj_applies",    st.proj_applies,    "(= 2*N*zpdb_advances)");
        row("zpdb_rank_calls", st.zpdb_rank_calls, "(only cost-1 projected edges)");
        println!("  {:<18} {:>16}", "iterations", st.iterations);
        println!("  (multiply ns/call by the component's samply self-time %% to get the real per-call cost)");
    }

    if !all_ok {
        return Err("an enumerated layer did not match its A087725 count".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => {
            eprintln!("usage: enumerate15 --pdb-dir DIR --data-dir DIR [--mode descent|band] --down-to T [--floor F] --out DIR [--no-verify] [--seed-ranks FILE]");
            ExitCode::SUCCESS
        }
        Err(e) => { eprintln!("error: {}", e); ExitCode::FAILURE }
    }
}
