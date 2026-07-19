//! Build a 15-puzzle PDB for a given pattern, multi-threaded.
//!
//! Usage:
//!
//! ```text
//! build_pdb15 --tiles 1,2,3,4,5,6,7 --out data/pdb15_p7_korf.bin [--threads N]
//!             [--verify-sha data/pdb15_p7_korf.sha256]
//!             [--write-sha data/pdb15_p7_korf.sha256]
//! ```
//!
//! Builds via [`PatternDb::build_parallel`] (rayon, layer-synchronous BFS).
//! The output is byte-identical to the sequential build and to a previously
//! built PDB for the same pattern.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle15::pdb::pattern::Pattern;
use puzzle8::puzzle15::pdb::PatternDb;

use sha2::{Digest, Sha256};

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} --tiles <T1,T2,...> --out <PATH> [--threads N] [--verify-sha PATH] [--write-sha PATH]");
    eprintln!("  --tiles      comma-separated tile values in 1..=15");
    eprintln!("  --out        path to write the PDB binary");
    eprintln!("  --threads    rayon thread count (default: system)");
    eprintln!("  --verify-sha read this file as the expected SHA-256 hex; fail if mismatch");
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
                let n = argv
                    .get(i)
                    .ok_or("--threads needs a value")?
                    .parse::<usize>()
                    .map_err(|e| format!("bad threads: {e}"))?;
                threads = Some(n);
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
            "-h" | "--help" => {
                return Err(String::from("help"));
            }
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    let tiles = tiles.ok_or("missing --tiles")?;
    let out = out.ok_or("missing --out")?;
    Ok(Args {
        tiles,
        out,
        threads,
        verify_sha,
        write_sha,
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
        .unwrap_or_else(|| "build_pdb15".into());
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
        "Building PDB for pattern {:?} ({} tiles)",
        args.tiles,
        pattern.size()
    );
    println!("  PDB entries  : {}", pattern.num_projected_states());
    println!("  BFS visited  : {}", pattern.num_bfs_states());
    println!("  Output file  : {}", args.out.display());

    let t0 = Instant::now();
    let pdb = PatternDb::build_parallel(pattern);
    let elapsed = t0.elapsed();
    println!("Build complete in {elapsed:.2?}");

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }

    if let Err(e) = pdb.save(&args.out) {
        eprintln!("error: writing {}: {}", args.out.display(), e);
        return ExitCode::FAILURE;
    }

    // Compute SHA-256 of the on-disk file for reporting and pin verification.
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
            eprintln!("error: SHA-256 mismatch");
            eprintln!("  expected: {expected}");
            eprintln!("  computed: {sha}");
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
