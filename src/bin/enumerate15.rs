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
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{idastar_inc_with_stats, WalkingDistanceHeuristic};

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
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    if !(1..=80).contains(&down_to) {
        return Err(format!("--down-to {} out of range 1..=80", down_to));
    }
    Ok(Args { pdb_dir, data_dir, mode_band, down_to, floor, budget, cache, no_cache, out, no_verify })
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
    let h = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);
    let verify = |r: u64| -> u8 {
        idastar_inc_with_stats(&unrank(r), &h).0.map(|v| v.len() as u8).unwrap_or(u8::MAX)
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
    if !all_ok {
        return Err("an enumerated layer did not match its A087725 count".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => {
            eprintln!("usage: enumerate15 --pdb-dir DIR --data-dir DIR [--mode descent|band] --down-to T [--floor F] --out DIR [--no-verify]");
            ExitCode::SUCCESS
        }
        Err(e) => { eprintln!("error: {}", e); ExitCode::FAILURE }
    }
}
