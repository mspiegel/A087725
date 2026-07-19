//! Build the 8-puzzle optimal-policy table from a distance table.
//!
//! Usage:
//!
//! ```text
//! build_pol8 --dist8 data/dist8.bin --out data/pol8.bin
//!            [--verify-sha data/pol8.sha256]
//!            [--write-sha data/pol8.sha256]
//! ```
//!
//! For each of the 181,440 8-puzzle ranks, computes the [`MoveSet`] bitmask of
//! optimal first moves — moves `m` such that `dist(s.apply(m)) == dist(s) - 1`.
//! The output is byte-identical across runs and machines.
//!
//! File format:
//!
//! ```text
//!   offset  bytes   description
//!   ----------------------------
//!   0       4       magic "P8PL"
//!   4       4       version (u32, little-endian) = 1
//!   8       181440  MoveSet payload (one byte per rank)
//! ```
//!
//! Total file size: 181_448 bytes.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle8::io;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::state::N_STATES;

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"P8PL";
const VERSION: u32 = 1;
const FILE_SIZE: usize = 8 + N_STATES as usize;

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} --dist8 <PATH> --out <PATH> [--verify-sha PATH] [--write-sha PATH]");
    eprintln!("  --dist8      path to the 8-puzzle distance table (dist8.bin)");
    eprintln!("  --out        path to write the policy binary");
    eprintln!("  --verify-sha read this file as the expected SHA-256 hex; fail if mismatch");
    eprintln!("  --write-sha  write the computed SHA-256 hex to this file");
}

struct Args {
    dist8: PathBuf,
    out: PathBuf,
    verify_sha: Option<PathBuf>,
    write_sha: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut dist8: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut verify_sha: Option<PathBuf> = None;
    let mut write_sha: Option<PathBuf> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--dist8" => {
                i += 1;
                dist8 = Some(PathBuf::from(argv.get(i).ok_or("--dist8 needs a value")?));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(argv.get(i).ok_or("--out needs a value")?));
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
            "-h" | "--help" => return Err(String::from("help")),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    let dist8 = dist8.ok_or("missing --dist8")?;
    let out = out.ok_or("missing --out")?;
    Ok(Args {
        dist8,
        out,
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
        .unwrap_or_else(|| "build_pol8".into());
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

    eprintln!("Loading distance table from {}", args.dist8.display());
    let table = match io::load(&args.dist8) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: loading {}: {}", args.dist8.display(), e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Computing optimal-move bitmasks for {N_STATES} states");
    let t0 = Instant::now();
    let mut payload = vec![0u8; N_STATES as usize];
    for r in 0..N_STATES {
        let s = unrank(r);
        payload[r as usize] = table.optimal_moves(&s).0;
    }
    let elapsed = t0.elapsed();
    eprintln!("Derivation complete in {elapsed:.2?}");

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }

    let mut file_bytes = Vec::with_capacity(FILE_SIZE);
    file_bytes.extend_from_slice(MAGIC);
    file_bytes.extend_from_slice(&VERSION.to_le_bytes());
    file_bytes.extend_from_slice(&payload);
    debug_assert_eq!(file_bytes.len(), FILE_SIZE);

    if let Err(e) = std::fs::write(&args.out, &file_bytes) {
        eprintln!("error: writing {}: {}", args.out.display(), e);
        return ExitCode::FAILURE;
    }

    let sha = sha256_hex(&file_bytes);
    println!("SHA-256 : {sha}");
    println!("Bytes   : {}", file_bytes.len());

    if let Some(verify_path) = &args.verify_sha {
        let expected = match std::fs::read_to_string(verify_path) {
            Ok(s) => s.split_whitespace().next().unwrap_or("").to_string(),
            Err(e) => {
                eprintln!("error reading {}: {}", verify_path.display(), e);
                return ExitCode::FAILURE;
            }
        };
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
