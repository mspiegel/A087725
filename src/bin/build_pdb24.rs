//! Build a 24-puzzle additive or zero-aware PDB for a given pattern, multi-threaded.
//!
//! ```text
//! build_pdb24 --part a|b|c|d --out data/pdb24_a.bin [--threads N] [--write-sha PATH]
//! build_pdb24 --tiles 1,2,3,6,7,8 --out data/pdb24_a.bin [--threads N] [--verify-sha PATH]
//! build_pdb24 --zero-aware --part a --out data/pdb24_a.bin
//! ```
//!
//! `--part` selects one block of the canonical Korf 6-6-6-6 partition; `--tiles`
//! gives an explicit comma-separated tile set (1..=24).
//!
//! Without `--zero-aware`: standard additive PDB ([`PatternDb::build_parallel`]).
//! With `--zero-aware`: zero-aware 1-bit PDB ([`ZPatternDb::build`] over the
//! region-aware BFS — `zbuild_parallel`). The latter produces a ~21.6 MiB
//! `Z24D` file for a 6-tile pattern; the additive form produces a 127 MB
//! `P24D` file. See `docs/zpdb-codec-spec.md`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::pdb::pattern::Pattern;
use puzzle8::puzzle24::pdb::zbuild::{build_zpdb_2bit_packed, build_zpdb_parallel};
use puzzle8::puzzle24::pdb::zdb::ZPatternDb;
use puzzle8::puzzle24::pdb::PatternDb;

use sha2::{Digest, Sha256};

/// Canonical Korf 6-6-6-6 partition of the 24-puzzle's tiles.
///
/// Goal board (tile at goal cell):
/// ```text
///  1  2  3  4  5
///  6  7  8  9 10
/// 11 12 13 14 15
/// 16 17 18 19 20
/// 21 22 23 24  _
/// ```
const PART_A: [u8; 6] = [1, 2, 3, 6, 7, 8];
const PART_B: [u8; 6] = [4, 5, 9, 10, 14, 15];
const PART_C: [u8; 6] = [11, 12, 16, 17, 21, 22];
const PART_D: [u8; 6] = [13, 18, 19, 20, 23, 24];

/// The 7-7-7-3 partition (three 7-tile PDBs + one 3-tile), geometrically
/// contiguous blocks covering all 24 tiles disjointly. Bigger root `h` than the
/// 6-6-6-6 set (≈485 MiB / 7-tile PDB at 1 bit). The exact tile grouping only
/// tunes h̄; this is a reasonable spatial decomposition.
const PART7_A: [u8; 7] = [1, 2, 3, 6, 7, 8, 11]; // top-left block
const PART7_B: [u8; 7] = [4, 5, 9, 10, 14, 15, 20]; // top-right + right edge
const PART7_C: [u8; 7] = [12, 13, 16, 17, 18, 21, 22]; // centre / bottom-left
const PART7_D: [u8; 3] = [19, 23, 24]; // bottom-right corner

fn part_tiles(partition: Partition, part: char) -> Option<Vec<u8>> {
    match (partition, part) {
        (Partition::K6, 'a') => Some(PART_A.to_vec()),
        (Partition::K6, 'b') => Some(PART_B.to_vec()),
        (Partition::K6, 'c') => Some(PART_C.to_vec()),
        (Partition::K6, 'd') => Some(PART_D.to_vec()),
        (Partition::K7, 'a') => Some(PART7_A.to_vec()),
        (Partition::K7, 'b') => Some(PART7_B.to_vec()),
        (Partition::K7, 'c') => Some(PART7_C.to_vec()),
        (Partition::K7, 'd') => Some(PART7_D.to_vec()),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum Partition {
    K6,
    K7,
}

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} (--part a|b|c|d [--partition k6|k7] | --tiles T1,T2,...) --out PATH [--zero-aware] [--threads N] [--verify-sha PATH] [--write-sha PATH]");
    eprintln!("  --part       one block of the selected partition (default k6)");
    eprintln!("  --partition  k6 (6-6-6-6, default) or k7 (7-7-7-3)");
    eprintln!("  --tiles      comma-separated tile values in 1..=24");
    eprintln!("  --out        path to write the PDB binary");
    eprintln!("  --zero-aware build a zero-aware 1-bit PDB (Z24D) instead of the additive (P24D)");
    eprintln!("  --threads    rayon thread count (default: system)");
    eprintln!("  --verify-sha read this file as expected SHA-256 hex; fail on mismatch");
    eprintln!("  --write-sha  write the computed SHA-256 hex to this file");
}

struct Args {
    tiles: Vec<u8>,
    out: PathBuf,
    threads: Option<usize>,
    verify_sha: Option<PathBuf>,
    write_sha: Option<PathBuf>,
    zero_aware: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut explicit_tiles: Option<Vec<u8>> = None;
    let mut part: Option<char> = None;
    let mut partition = Partition::K6;
    let mut out: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    let mut verify_sha: Option<PathBuf> = None;
    let mut write_sha: Option<PathBuf> = None;
    let mut zero_aware = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--part" => {
                i += 1;
                let s = argv.get(i).ok_or("--part needs a value")?;
                part = Some(s.chars().next().ok_or("--part empty")?);
            }
            "--partition" => {
                i += 1;
                partition = match argv.get(i).ok_or("--partition needs a value")?.as_str() {
                    "k6" => Partition::K6,
                    "k7" => Partition::K7,
                    other => return Err(format!("unknown partition {other:?} (k6|k7)")),
                };
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
                    if !(1..=24).contains(&t) {
                        return Err(format!("tile {t} out of range 1..=24"));
                    }
                    v.push(t);
                }
                explicit_tiles = Some(v);
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
            "--verify-sha" => {
                i += 1;
                verify_sha = Some(PathBuf::from(
                    argv.get(i).ok_or("--verify-sha needs a value")?,
                ));
            }
            "--write-sha" => {
                i += 1;
                write_sha = Some(PathBuf::from(
                    argv.get(i).ok_or("--write-sha needs a value")?,
                ));
            }
            "--zero-aware" => {
                zero_aware = true;
            }
            "-h" | "--help" => return Err(String::from("help")),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    let tiles = match (explicit_tiles, part) {
        (Some(_), Some(_)) => return Err("give either --tiles or --part, not both".into()),
        (Some(v), None) => v,
        (None, Some(c)) => part_tiles(partition, c)
            .ok_or_else(|| format!("unknown part {c:?} for the selected partition"))?,
        (None, None) => return Err("missing --part or --tiles".into()),
    };
    let out = out.ok_or("missing --out")?;
    Ok(Args {
        tiles,
        out,
        threads,
        verify_sha,
        write_sha,
        zero_aware,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        use std::fmt::Write;
        write!(&mut s, "{b:02x}").unwrap();
    }
    s
}

fn main() -> ExitCode {
    let prog = std::env::args()
        .next()
        .unwrap_or_else(|| "build_pdb24".into());
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
    if args.zero_aware {
        println!(
            "Building 24-puzzle zero-aware (1-bit) PDB for pattern {:?} ({} tiles)",
            args.tiles,
            pattern.size()
        );
    } else {
        println!(
            "Building 24-puzzle additive PDB for pattern {:?} ({} tiles)",
            args.tiles,
            pattern.size()
        );
        println!("  PDB entries  : {}", pattern.num_projected_states());
        println!("  BFS visited  : {}", pattern.num_bfs_states());
    }
    println!("  Output file  : {}", args.out.display());

    let t0 = Instant::now();
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }
    if args.zero_aware {
        // k>=8: the frontier build's byte `dist` (~80 GiB at k=8) and u32 frontier
        // are infeasible; use the frontier-free 2-bit builder (~20 GiB, packs
        // directly). k<=7: the fast, SHA-pinned frontier build.
        let zdb = if pattern.size() >= 8 {
            let (packed, layout, maxd) = build_zpdb_2bit_packed(pattern);
            println!(
                "ZPDB (frontier-free 2-bit) BFS complete in {:.2?}",
                t0.elapsed()
            );
            println!("  ZPDB entries : {}", layout.total());
            println!("  Packed bytes : {}", packed.len());
            println!("  Max depth    : {maxd}  (eccentricity — sum across a partition for the Σ-ecc bound)");
            ZPatternDb::from_packed(pattern, packed)
        } else {
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
            ZPatternDb::from_dist(pattern, &dist)
        };
        if let Err(e) = zdb.save(&args.out) {
            eprintln!("error: writing {}: {}", args.out.display(), e);
            return ExitCode::FAILURE;
        }
    } else {
        let pdb = PatternDb::build_parallel(pattern);
        println!("Build complete in {:.2?}", t0.elapsed());
        if let Err(e) = pdb.save(&args.out) {
            eprintln!("error: writing {}: {}", args.out.display(), e);
            return ExitCode::FAILURE;
        }
    }

    let file_bytes = std::fs::read(&args.out).expect("read just-written file");
    let sha = sha256_hex(&file_bytes);
    println!("SHA-256 : {sha}");
    println!("Bytes   : {}", file_bytes.len());

    if let Some(verify_path) = &args.verify_sha {
        let expected = std::fs::read_to_string(verify_path)
            .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
            .unwrap_or_else(|e| {
                eprintln!("error reading {}: {}", verify_path.display(), e);
                std::process::exit(1);
            });
        if expected != sha {
            eprintln!("error: SHA-256 mismatch\n  expected: {expected}\n  computed: {sha}");
            return ExitCode::FAILURE;
        }
        println!("SHA-256 matches pinned {}", verify_path.display());
    }
    if let Some(write_path) = &args.write_sha {
        std::fs::write(write_path, format!("{sha}\n")).expect("writing SHA file");
        println!("Wrote SHA-256 → {}", write_path.display());
    }

    ExitCode::SUCCESS
}
