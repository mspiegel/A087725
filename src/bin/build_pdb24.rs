//! Build a 24-puzzle additive PDB for a given pattern, multi-threaded.
//!
//! ```text
//! build_pdb24 --part a|b|c|d --out data/pdb24_a.bin [--threads N] [--write-sha PATH]
//! build_pdb24 --tiles 1,2,3,6,7,8 --out data/pdb24_a.bin [--threads N] [--verify-sha PATH]
//! ```
//!
//! `--part` selects one block of the canonical Korf 6-6-6-6 partition; `--tiles`
//! gives an explicit comma-separated tile set (1..=24). Builds via
//! [`PatternDb::build_parallel`] (rayon, layer-synchronous BFS); output is
//! byte-identical to the sequential build and across runs.
//!
//! A 6-tile build is P(25,7) = 2,422,728,000 BFS-visited states (~2.4 GB
//! transient as a byte array) and P(25,6) = 127,512,000 output entries
//! (~127 MB). See `docs/zpdb-codec-spec.md` for the zero-aware 1-bit successor.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::pdb::pattern::Pattern;
use puzzle8::puzzle24::pdb::PatternDb;

use sha2::{Digest, Sha256};

/// Canonical Korf 6-6-6-6 partition of the 24-puzzle's tiles.
const PART_A: [u8; 6] = [1, 2, 3, 6, 7, 8];
const PART_B: [u8; 6] = [4, 5, 9, 10, 14, 15];
const PART_C: [u8; 6] = [11, 12, 16, 17, 21, 22];
const PART_D: [u8; 6] = [13, 18, 19, 20, 23, 24];

fn part_tiles(part: char) -> Option<Vec<u8>> {
    match part {
        'a' => Some(PART_A.to_vec()),
        'b' => Some(PART_B.to_vec()),
        'c' => Some(PART_C.to_vec()),
        'd' => Some(PART_D.to_vec()),
        _ => None,
    }
}

fn print_usage(prog: &str) {
    eprintln!("usage: {} (--part a|b|c|d | --tiles T1,T2,...) --out PATH [--threads N] [--verify-sha PATH] [--write-sha PATH]", prog);
    eprintln!("  --part       one block of the canonical Korf 6-6-6-6 partition");
    eprintln!("  --tiles      comma-separated tile values in 1..=24");
    eprintln!("  --out        path to write the PDB binary");
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
}

fn parse_args() -> Result<Args, String> {
    let mut tiles: Option<Vec<u8>> = None;
    let mut out: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    let mut verify_sha: Option<PathBuf> = None;
    let mut write_sha: Option<PathBuf> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--part" => {
                i += 1;
                let s = argv.get(i).ok_or("--part needs a value")?;
                let c = s.chars().next().ok_or("--part empty")?;
                tiles = Some(part_tiles(c).ok_or_else(|| format!("unknown part {:?}", s))?);
            }
            "--tiles" => {
                i += 1;
                let s = argv.get(i).ok_or("--tiles needs a value")?;
                let mut v: Vec<u8> = Vec::new();
                for token in s.split(',') {
                    let t = token.trim().parse::<u8>().map_err(|e| format!("bad tile {:?}: {}", token, e))?;
                    if !(1..=24).contains(&t) {
                        return Err(format!("tile {} out of range 1..=24", t));
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
                    argv.get(i).ok_or("--threads needs a value")?
                        .parse::<usize>().map_err(|e| format!("bad threads: {}", e))?,
                );
            }
            "--verify-sha" => {
                i += 1;
                verify_sha = Some(PathBuf::from(argv.get(i).ok_or("--verify-sha needs a value")?));
            }
            "--write-sha" => {
                i += 1;
                write_sha = Some(PathBuf::from(argv.get(i).ok_or("--write-sha needs a value")?));
            }
            "-h" | "--help" => return Err(String::from("help")),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }

    let tiles = tiles.ok_or("missing --part or --tiles")?;
    let out = out.ok_or("missing --out")?;
    Ok(Args { tiles, out, threads, verify_sha, write_sha })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

fn main() -> ExitCode {
    let prog = std::env::args().next().unwrap_or_else(|| "build_pdb24".into());
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e == "help" {
                print_usage(&prog);
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {}", e);
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
    println!("Building 24-puzzle PDB for pattern {:?} ({} tiles)", args.tiles, pattern.size());
    println!("  PDB entries  : {}", pattern.num_projected_states());
    println!("  BFS visited  : {}", pattern.num_bfs_states());
    println!("  Output file  : {}", args.out.display());

    let t0 = Instant::now();
    let pdb = PatternDb::build_parallel(pattern);
    println!("Build complete in {:.2?}", t0.elapsed());

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }
    if let Err(e) = pdb.save(&args.out) {
        eprintln!("error: writing {}: {}", args.out.display(), e);
        return ExitCode::FAILURE;
    }

    let file_bytes = std::fs::read(&args.out).expect("read just-written file");
    let sha = sha256_hex(&file_bytes);
    println!("SHA-256 : {}", sha);
    println!("Bytes   : {}", file_bytes.len());

    if let Some(verify_path) = &args.verify_sha {
        let expected = std::fs::read_to_string(verify_path)
            .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
            .unwrap_or_else(|e| {
                eprintln!("error reading {}: {}", verify_path.display(), e);
                std::process::exit(1);
            });
        if expected != sha {
            eprintln!("error: SHA-256 mismatch\n  expected: {}\n  computed: {}", expected, sha);
            return ExitCode::FAILURE;
        }
        println!("SHA-256 matches pinned {}", verify_path.display());
    }
    if let Some(write_path) = &args.write_sha {
        std::fs::write(write_path, format!("{}\n", sha)).expect("writing SHA file");
        println!("Wrote SHA-256 → {}", write_path.display());
    }

    ExitCode::SUCCESS
}
