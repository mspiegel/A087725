//! Build the 8-puzzle adjacency table.
//!
//! Usage:
//!
//! ```text
//! build_adj8 --out data/adj8.bin
//!            [--verify-sha data/adj8.sha256]
//!            [--write-sha data/adj8.sha256]
//! ```
//!
//! For each of the 181,440 8-puzzle ranks, emits the ranks of its up-to-4
//! neighbors keyed by [`Move`] slot. `u32::MAX` is the sentinel for "move
//! illegal from this state". Replaces the `unrank → apply → rank` cascade with
//! a single array read for every neighbor query — useful for graph walks,
//! bidirectional BFS, GNN-style mining, and rejection sampling.
//!
//! This binary needs no input file: adjacency depends only on the puzzle's
//! combinatorial structure (`unrank`, `apply`, `rank`), not BFS results.
//!
//! File format:
//!
//! ```text
//!   offset  bytes   description
//!   ----------------------------
//!   0       4       magic "P8AJ"
//!   4       4       version (u32, LE) = 1
//!   8+r*16  16      record for rank r: four u32 LE neighbors, slots U/D/L/R
//! ```
//!
//! Total file size: 8 + 16 * 181_440 = 2,903,048 bytes.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle8::rank::{rank, unrank};
use puzzle8::puzzle8::state::{Move, N_STATES};

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"P8AJ";
const VERSION: u32 = 1;
const RECORD_SIZE: usize = 16;
const HEADER_BYTES: usize = 8;
const FILE_SIZE: usize = HEADER_BYTES + RECORD_SIZE * (N_STATES as usize);

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} --out <PATH> [--verify-sha PATH] [--write-sha PATH]");
    eprintln!("  --out        path to write the adjacency binary");
    eprintln!("  --verify-sha read this file as the expected SHA-256 hex; fail if mismatch");
    eprintln!("  --write-sha  write the computed SHA-256 hex to this file");
}

struct Args {
    out: PathBuf,
    verify_sha: Option<PathBuf>,
    write_sha: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut out: Option<PathBuf> = None;
    let mut verify_sha: Option<PathBuf> = None;
    let mut write_sha: Option<PathBuf> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(argv.get(i).ok_or("--out needs a value")?));
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
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    let out = out.ok_or("missing --out")?;
    Ok(Args { out, verify_sha, write_sha })
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
    let prog = std::env::args().next().unwrap_or_else(|| "build_adj8".into());
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

    eprintln!("Computing neighbor ranks for {N_STATES} states");
    let t0 = Instant::now();
    let mut file_bytes = Vec::with_capacity(FILE_SIZE);
    file_bytes.extend_from_slice(MAGIC);
    file_bytes.extend_from_slice(&VERSION.to_le_bytes());

    for r in 0..N_STATES {
        let s = unrank(r);
        let legal = s.legal_moves();
        let mut rec = [0u8; RECORD_SIZE];
        for (slot, &m) in Move::ALL.iter().enumerate() {
            let neighbor = if legal.contains(m) {
                rank(&s.apply(m))
            } else {
                u32::MAX
            };
            rec[slot * 4..slot * 4 + 4].copy_from_slice(&neighbor.to_le_bytes());
        }
        file_bytes.extend_from_slice(&rec);
    }
    debug_assert_eq!(file_bytes.len(), FILE_SIZE);
    let elapsed = t0.elapsed();
    eprintln!("Derivation complete in {elapsed:.2?}");

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }

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
        std::fs::write(write_path, format!("{sha}\n"))
            .expect("writing SHA file");
        println!("Wrote SHA-256 → {}", write_path.display());
    }

    ExitCode::SUCCESS
}
