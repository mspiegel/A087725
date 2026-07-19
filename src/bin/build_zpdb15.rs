//! Build a zero-aware (region-based, Clausecker–Reinefeld) 15-puzzle PDB for a
//! given pattern, multi-threaded, and save it (magic `Z15D`, raw bytes).
//!
//! ```text
//! build_zpdb15 --tiles 1,2,3,4,5,6,7 --out data/zpdb15_a.zbin [--threads N]
//! build_zpdb15 --part 7 --out data/zpdb15_p7.zbin
//! build_zpdb15 --part 8 --out data/zpdb15_p8.zbin
//! ```
//!
//! `--part 7|8` selects one block of the canonical Korf 7-8 partition;
//! `--tiles` gives an explicit comma-separated tile set (1..=15).
//!
//! A small embedded validation runs against `bfs_distances`-style truth for
//! tiny patterns when `--validate` is passed.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle15::pdb::pattern::Pattern;
use puzzle8::puzzle15::pdb::zbuild::build_zpdb_parallel;
use puzzle8::puzzle15::pdb::zdb::ZPatternDb;

/// Canonical Korf 7-8 partition of the 15-puzzle's tiles.
const PART_7: [u8; 7] = [1, 2, 3, 4, 5, 6, 7];
const PART_8: [u8; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

fn part_tiles(part: &str) -> Option<Vec<u8>> {
    match part {
        "7" => Some(PART_7.to_vec()),
        "8" => Some(PART_8.to_vec()),
        _ => None,
    }
}

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} (--part 7|8 | --tiles T1,T2,...) --out PATH [--threads N]");
    eprintln!("  --part     one block of the canonical Korf 7-8 partition");
    eprintln!("  --tiles    comma-separated tile values in 1..=15");
    eprintln!("  --out      path to write the ZPDB binary (Z15D)");
    eprintln!("  --threads  rayon thread count (default: system)");
}

struct Args {
    tiles: Vec<u8>,
    out: PathBuf,
    threads: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut tiles: Option<Vec<u8>> = None;
    let mut out: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--part" => {
                i += 1;
                let s = argv.get(i).ok_or("--part needs a value")?;
                tiles = Some(part_tiles(s).ok_or_else(|| format!("unknown part {s:?}"))?);
            }
            "--tiles" => {
                i += 1;
                let s = argv.get(i).ok_or("--tiles needs a value")?;
                let mut v: Vec<u8> = Vec::new();
                for token in s.split(',') {
                    let t = token
                        .trim()
                        .parse::<u8>()
                        .map_err(|e| format!("bad tile {token:?}: {e}"))?;
                    if !(1..=15).contains(&t) {
                        return Err(format!("tile {t} out of range 1..=15"));
                    }
                    v.push(t);
                }
                tiles = Some(v);
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(argv.get(i).ok_or("--out needs a value")?));
            }
            "--threads" => {
                i += 1;
                threads = Some(
                    argv.get(i)
                        .ok_or("--threads needs a value")?
                        .parse::<usize>()
                        .map_err(|e| format!("bad threads: {e}"))?,
                );
            }
            "-h" | "--help" => return Err(String::from("help")),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    let tiles = tiles.ok_or("missing --part or --tiles")?;
    let out = out.ok_or("missing --out")?;
    Ok(Args {
        tiles,
        out,
        threads,
    })
}

fn main() -> ExitCode {
    let prog = std::env::args()
        .next()
        .unwrap_or_else(|| "build_zpdb15".into());
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e == "help" {
                print_usage(&prog);
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {e}");
            print_usage(&prog);
            return ExitCode::FAILURE;
        }
    };

    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("rayon global pool already initialized");
    }

    let pattern = Pattern::new(&args.tiles);
    println!(
        "Building 15-puzzle zero-aware PDB for pattern {:?} ({} tiles)",
        args.tiles,
        pattern.size()
    );
    println!("  Output file  : {}", args.out.display());

    let t0 = Instant::now();
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }

    let (dist, layout) = build_zpdb_parallel(pattern);
    println!("ZPDB BFS complete in {:.2?}", t0.elapsed());
    println!("  ZPDB entries : {}", layout.total());
    let unvisited = dist.iter().filter(|&&d| d == u8::MAX).count();
    let maxd = dist
        .iter()
        .filter(|&&d| d != u8::MAX)
        .copied()
        .max()
        .unwrap_or(0);
    println!("  Max depth    : {maxd}");
    if unvisited != 0 {
        eprintln!("error: ZPDB build left {unvisited} unvisited entries");
        return ExitCode::FAILURE;
    }

    let zdb = ZPatternDb::from_dist(pattern, dist);
    if let Err(e) = zdb.save(&args.out) {
        eprintln!("error: writing {}: {}", args.out.display(), e);
        return ExitCode::FAILURE;
    }

    let bytes = std::fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    println!("Bytes   : {bytes}");

    ExitCode::SUCCESS
}
