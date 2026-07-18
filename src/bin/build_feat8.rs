//! Build the 8-puzzle per-state feature table from a distance table.
//!
//! Usage:
//!
//! ```text
//! build_feat8 --dist8 data/dist8.bin --out data/feat8.bin
//!             [--verify-sha data/feat8.sha256]
//!             [--write-sha data/feat8.sha256]
//! ```
//!
//! For each of the 181,440 8-puzzle ranks, emits a fixed-width 32-byte record
//! holding the raw board, derived scalars (blank position, Manhattan distance,
//! inversions, correct-tile count/mask, optimal-move count, self-symmetry), per
//! -tile Manhattan terms, and the reflected canonical rank. Together with
//! [`build_pol8`] and [`build_adj8`], this turns `dist8.bin` from a verification
//! oracle into a mining corpus.
//!
//! File format:
//!
//! ```text
//!   offset  bytes   description
//!   ----------------------------
//!   0       4       magic "P8FT"
//!   4       4       version (u32, LE) = 1
//!   8       4       record_size (u32, LE) = 32
//!   12      4       reserved (u32, LE) = 0
//!   16+r*32 32      record for rank r (see RECORD_SIZE below)
//! ```
//!
//! Per record (32 bytes):
//!
//! ```text
//!   0       9   board: [u8;9] = unrank(r).0  (row-major; blank = 0)
//!   9       1   blank_pos                     (0..=8)
//!   10      1   distance                      (0..=31)
//!   11      1   manhattan_sum                 (0..=24)
//!   12      1   inversions                    (0..=28)
//!   13      1   correct_tile_count            (0..=8)
//!   14      1   correct_tile_mask             bit (t-1) set iff tile t at goal
//!   15      1   num_optimal_moves             (1..=4 non-goal; 0 for GOAL)
//!   16      1   self_symmetric_flag           (0 or 1)
//!   17      3   padding (must be 0)
//!   20      4   reflected_canonical_rank      (u32, LE)
//!   24      8   per_tile_manhattan[1..=8]     (1 byte per tile, 0..=4)
//! ```
//!
//! Total file size: 16 + 32 * 181_440 = 5,806,096 bytes.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle8::io;
use puzzle8::puzzle8::rank::{rank, unrank};
use puzzle8::puzzle8::state::N_STATES;
use puzzle8::puzzle8::symmetry::reflect;

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"P8FT";
const VERSION: u32 = 1;
const RECORD_SIZE: u32 = 32;
const RESERVED: u32 = 0;
const HEADER_BYTES: usize = 16;
const FILE_SIZE: usize = HEADER_BYTES + (RECORD_SIZE as usize) * (N_STATES as usize);

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} --dist8 <PATH> --out <PATH> [--verify-sha PATH] [--write-sha PATH]");
    eprintln!("  --dist8      path to the 8-puzzle distance table (dist8.bin)");
    eprintln!("  --out        path to write the feature binary");
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

    let dist8 = dist8.ok_or("missing --dist8")?;
    let out = out.ok_or("missing --out")?;
    Ok(Args { dist8, out, verify_sha, write_sha })
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

/// Manhattan distance of tile `t` (1..=8) from its goal cell (`t - 1`), given
/// the tile's current cell `p` (0..=8). Mirrors `ManhattanHeuristic::h` in
/// `src/puzzle8/search/heuristic.rs`.
#[inline]
fn tile_manhattan(t: u8, p: usize) -> u8 {
    let goal_pos = (t - 1) as usize;
    let cur_row = (p / 3) as i32;
    let cur_col = (p % 3) as i32;
    let goal_row = (goal_pos / 3) as i32;
    let goal_col = (goal_pos % 3) as i32;
    ((cur_row - goal_row).unsigned_abs() + (cur_col - goal_col).unsigned_abs()) as u8
}

fn main() -> ExitCode {
    let prog = std::env::args().next().unwrap_or_else(|| "build_feat8".into());
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

    eprintln!("Computing 32-byte feature records for {N_STATES} states");
    let t0 = Instant::now();
    let mut file_bytes = Vec::with_capacity(FILE_SIZE);
    file_bytes.extend_from_slice(MAGIC);
    file_bytes.extend_from_slice(&VERSION.to_le_bytes());
    file_bytes.extend_from_slice(&RECORD_SIZE.to_le_bytes());
    file_bytes.extend_from_slice(&RESERVED.to_le_bytes());

    for r in 0..N_STATES {
        let s = unrank(r);
        let mut rec = [0u8; RECORD_SIZE as usize];

        // 0..9: board
        rec[0..9].copy_from_slice(&s.0);

        // 9: blank position
        rec[9] = s.blank_pos();

        // 10: distance
        rec[10] = table.dist_of_rank(r);

        // 11, 13, 14, 24..32: tile-positional features (manhattan_sum, correct_*)
        let mut manhattan_sum: u32 = 0;
        let mut correct_count: u8 = 0;
        let mut correct_mask: u8 = 0;
        for t in 1u8..=8u8 {
            // Find tile t's current position.
            let mut pos = 9;
            for (i, &v) in s.0.iter().enumerate() {
                if v == t {
                    pos = i;
                    break;
                }
            }
            debug_assert!(pos < 9, "tile {} not found in state {:?}", t, s.0);
            let m = tile_manhattan(t, pos);
            rec[24 + (t as usize - 1)] = m;
            manhattan_sum += m as u32;
            if pos == (t as usize - 1) {
                correct_count += 1;
                correct_mask |= 1 << (t - 1);
            }
        }
        rec[11] = manhattan_sum as u8;
        rec[13] = correct_count;
        rec[14] = correct_mask;

        // 12: inversions
        rec[12] = s.inversions() as u8;

        // 15: number of optimal moves
        rec[15] = table.optimal_moves(&s).len() as u8;

        // 16: self-symmetric flag
        let s_refl = reflect(&s);
        rec[16] = (s_refl == s) as u8;

        // 17..20: padding (already zero from initialization)

        // 20..24: reflected canonical rank
        let r_refl = rank(&s_refl);
        let canonical = r.min(r_refl);
        rec[20..24].copy_from_slice(&canonical.to_le_bytes());

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
