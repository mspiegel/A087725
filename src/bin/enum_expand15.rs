//! Inspect / re-verify a `depthNN.ranks` enumeration file.
//!
//! Usage:
//! ```text
//! enum_expand15 --file data/enum15/depth79.ranks [--limit N] [--print]
//!               [--verify --depth 79 --pdb-dir data [--sample K]]
//! ```
//! Each board is a 6-byte little-endian rank. `--print` renders boards as
//! 16-token rows; `--verify` re-solves a sample with `korf-plus` and asserts the
//! optimal length equals `--depth` (and that every rank round-trips).

use std::path::PathBuf;
use std::process::ExitCode;

use puzzle8::puzzle15::enumerate::Store;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{idastar_inc_with_stats, WalkingDistanceHeuristic};

fn board_tokens(r: u64) -> String {
    let s = unrank(r);
    s.0.iter()
        .map(|&v| {
            if v == 0 {
                "_".to_string()
            } else {
                v.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut file: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut print = false;
    let mut verify = false;
    let mut depth: Option<u8> = None;
    let mut pdb_dir = PathBuf::from("data");
    let mut sample: usize = 64;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--file" => {
                i += 1;
                file = Some(PathBuf::from(argv.get(i).ok_or("--file needs a value")?));
            }
            "--limit" => {
                i += 1;
                limit = Some(
                    argv.get(i)
                        .ok_or("--limit needs a value")?
                        .parse()
                        .map_err(|e| format!("--limit: {e}"))?,
                );
            }
            "--print" => print = true,
            "--verify" => verify = true,
            "--depth" => {
                i += 1;
                depth = Some(
                    argv.get(i)
                        .ok_or("--depth needs a value")?
                        .parse()
                        .map_err(|e| format!("--depth: {e}"))?,
                );
            }
            "--pdb-dir" => {
                i += 1;
                pdb_dir = PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?);
            }
            "--sample" => {
                i += 1;
                sample = argv
                    .get(i)
                    .ok_or("--sample needs a value")?
                    .parse()
                    .map_err(|e| format!("--sample: {e}"))?;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }
    let path = file.ok_or("--file is required")?;
    let ranks = Store::read_ranks_file(&path).map_err(|e| e.to_string())?;
    println!("{}: {} boards", path.display(), ranks.len());

    // Round-trip check on every rank (cheap): rank(unrank(r)) == r and solvable.
    for &r in &ranks {
        let s = unrank(r);
        if !s.is_solvable() || rank(&s) != r {
            return Err(format!("rank {r} fails round-trip / solvability"));
        }
    }
    println!("round-trip + solvability: OK ({} ranks)", ranks.len());

    if print {
        let n = limit.unwrap_or(ranks.len()).min(ranks.len());
        for &r in &ranks[..n] {
            println!("{}", board_tokens(r));
        }
    }

    if verify {
        let d = depth.ok_or("--verify needs --depth")?;
        // Zero-aware "zpdb-plus" verifier — pointwise dominates the additive
        // korf-plus it replaces, advanced incrementally per node.
        let zdbs = [
            ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p7.zbin"))
                .map_err(|e| format!("zp7: {e}"))?,
            ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p8.zbin"))
                .map_err(|e| format!("zp8: {e}"))?,
        ];
        WalkingDistanceHeuristic::warm_up();
        let h_plus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);

        // Evenly-spaced sample so we cover the whole file.
        let step = (ranks.len() / sample.max(1)).max(1);
        let mut checked = 0usize;
        for &r in ranks.iter().step_by(step) {
            let got = idastar_inc_with_stats(&unrank(r), &h_plus)
                .0
                .map(|v| v.len() as u8)
                .unwrap_or(u8::MAX);
            if got != d {
                return Err(format!("rank {r} has optimal depth {got} != claimed {d}"));
            }
            checked += 1;
        }
        println!("verify: OK — {checked} sampled boards all solve to depth {d}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
