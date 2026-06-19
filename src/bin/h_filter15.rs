//! Filter a `.ranks` file by a Korf-max combined heuristic
//! (ZPDB + Walking-Distance + Linear-Conflict, max'd against the reflection).
//!
//! Reads 6-byte LE ranks from stdin (or `--in PATH`), computes
//! `max(h_zpdb(s), h_wd(s), h_lc(s), h_zpdb(reflect(s)), h_wd(reflect(s)),
//! h_lc(reflect(s)))` for each, and emits to stdout only those with `h >=
//! --min-h`. Each component is admissible, so the max is admissible too.
//! Surviving ranks have `depth >= min_h`.
//!
//! ```text
//! h_filter15 --pdb-dir data --min-h 78 < /tmp/cand.ranks > /tmp/filtered.ranks
//! ```

use std::io::{self, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use puzzle8::puzzle15::pdb::{AdditiveZpdbHeuristic, ZPatternDb};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    Heuristic, LinearConflictHeuristic, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::symmetry::reflect;

struct Args {
    pdb_dir: PathBuf,
    min_h: u8,
    in_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = PathBuf::from("data");
    let mut min_h: u8 = 78;
    let mut in_path: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => { i += 1; pdb_dir = PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?); }
            "--min-h" => { i += 1; min_h = argv.get(i).ok_or("--min-h needs a value")?.parse().map_err(|e: std::num::ParseIntError| format!("--min-h: {}", e))?; }
            "--in" => { i += 1; in_path = Some(PathBuf::from(argv.get(i).ok_or("--in needs a value")?)); }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    Ok(Args { pdb_dir, min_h, in_path })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let p7 = ZPatternDb::load_mmap(&args.pdb_dir.join("zpdb15_p7.zbin"))
        .map_err(|e| format!("zpdb15_p7: {}", e))?;
    let p8 = ZPatternDb::load_mmap(&args.pdb_dir.join("zpdb15_p8.zbin"))
        .map_err(|e| format!("zpdb15_p8: {}", e))?;
    let zdbs = [p7, p8];
    let h_zpdb = AdditiveZpdbHeuristic::new(&zdbs);
    WalkingDistanceHeuristic::warm_up();
    let h_wd = WalkingDistanceHeuristic;
    let h_lc = LinearConflictHeuristic;

    // Read all input into memory.
    let mut input = Vec::new();
    match args.in_path.as_deref() {
        Some(p) => {
            std::fs::File::open(p).map_err(|e| format!("{}: {}", p.display(), e))?
                .read_to_end(&mut input).map_err(|e| format!("{}: {}", p.display(), e))?;
        }
        None => {
            io::stdin().read_to_end(&mut input).map_err(|e| format!("stdin: {}", e))?;
        }
    }
    if input.len() % 6 != 0 {
        return Err(format!("input length {} not a multiple of 6", input.len()));
    }
    let n = input.len() / 6;
    eprintln!("h_filter15: {} input ranks, min_h = {}", n, args.min_h);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut kept = 0u64;
    let mut tick = 0u64;
    for chunk in input.chunks_exact(6) {
        let mut buf = [0u8; 8];
        buf[..6].copy_from_slice(chunk);
        let r = u64::from_le_bytes(buf);
        let s = unrank(r);
        let sr = reflect(&s);
        // Korf-max over 3 admissible heuristics × 2 reflections.
        let h_direct = h_zpdb.h(&s).max(h_wd.h(&s)).max(h_lc.h(&s));
        let h_refl = h_zpdb.h(&sr).max(h_wd.h(&sr)).max(h_lc.h(&sr));
        let h_korf = h_direct.max(h_refl);
        if h_korf >= args.min_h {
            out.write_all(chunk).map_err(|e| format!("stdout: {}", e))?;
            kept += 1;
        }
        tick += 1;
        if tick % 1_000_000 == 0 {
            eprintln!("  processed {}/{} ({} kept)", tick, n, kept);
        }
    }
    out.flush().map_err(|e| format!("stdout flush: {}", e))?;
    eprintln!("h_filter15: kept {} / {}", kept, n);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => {
            eprintln!("usage: h_filter15 --pdb-dir DIR [--min-h N] [--in PATH] (stdin/stdout)");
            ExitCode::SUCCESS
        }
        Err(e) => { eprintln!("error: {}", e); ExitCode::FAILURE }
    }
}
