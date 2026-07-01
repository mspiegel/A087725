//! Precompute the 24-puzzle Walking-Distance table once and persist it, so
//! solving harnesses load it (~seconds) instead of rebuilding it via BFS
//! (~17–25 s) on every process start.
//!
//! ```text
//! build_wd24 --out data/wd24.bin [--verify-sha data/wd24.bin.sha256]
//! build_wd24 --out data/wd24.bin [--write-sha  data/wd24.bin.sha256]
//! ```
//!
//! The BFS is deterministic and the artifact is written in a byte-deterministic
//! (key-sorted) layout, so the file — and hence its SHA-256 — is reproducible.
//! Only the `.sha256` pin is committed (the `.bin` is `.gitignore`d, like the
//! ZPDB artifacts); regenerate the `.bin` locally with this tool.
//!
//! Once `data/wd24.bin` exists, every 24-puzzle harness picks it up
//! automatically (or point `WD24_TABLE` at an explicit path). This also
//! establishes the persistence path for the future wildcard-complement `hB`
//! heuristic, which reuses the same `save_dist_table`/`load_dist_table` format.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::search::{
    build_full_table, load_dist_table, save_dist_table, FULL_WD_ENTRIES, WD_KIND_FULL,
};

use sha2::{Digest, Sha256};

struct Args {
    out: PathBuf,
    verify_sha: Option<PathBuf>,
    write_sha: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut out = None;
    let mut verify_sha = None;
    let mut write_sha = None;

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
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    Ok(Args {
        out: out.ok_or("--out is required")?,
        verify_sha,
        write_sha,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if msg != "help" {
                eprintln!("error: {}", msg);
            }
            eprintln!(
                "usage: build_wd24 --out data/wd24.bin \
                 [--verify-sha PATH] [--write-sha PATH]"
            );
            return if msg == "help" { ExitCode::SUCCESS } else { ExitCode::FAILURE };
        }
    };

    let t0 = Instant::now();
    let table = build_full_table();
    println!("WD BFS complete in {:.2?}", t0.elapsed());
    println!("  WD entries : {}", table.len());
    if table.len() as u64 != FULL_WD_ENTRIES {
        eprintln!(
            "error: built {} entries, expected {} — BFS transition rule drifted",
            table.len(),
            FULL_WD_ENTRIES
        );
        return ExitCode::FAILURE;
    }

    if let Err(e) = save_dist_table(&args.out, WD_KIND_FULL, &table) {
        eprintln!("error: writing {}: {}", args.out.display(), e);
        return ExitCode::FAILURE;
    }

    // Round-trip guard: reload the just-written artifact and confirm it decodes to
    // exactly the built table. Cheap relative to the BFS, and catches any codec or
    // header regression before the artifact is pinned or shipped.
    match load_dist_table(&args.out, WD_KIND_FULL, Some(FULL_WD_ENTRIES)) {
        Ok(reloaded) => {
            if reloaded != table {
                eprintln!("error: reloaded table != built table (serialization bug)");
                return ExitCode::FAILURE;
            }
            println!("  Round-trip : OK ({} entries reloaded)", reloaded.len());
        }
        Err(e) => {
            eprintln!("error: reloading {}: {}", args.out.display(), e);
            return ExitCode::FAILURE;
        }
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
