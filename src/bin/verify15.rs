//! Verify the 15-puzzle solver: end-to-end correctness check.
//!
//! Primary mode — pass `--antipodes data/pdb15_antipodes.txt` to load the 17
//! Korf–Schultze (AAAI 2005) depth-80 antipodes and assert IDA\*+PDB returns
//! depth exactly 80 on each, with a path that reaches `GOAL`. The positions
//! in that file were decoded from stannic's 2017 Domain-of-the-Cube post
//! (`forum.cubeman.org/?q=node/view/555`, comment "Nodecounts") via
//! `scripts/decode_antipodes.py`; the Korf–Schultze paper publishes only the
//! depth histogram, not the antipode positions themselves.
//!
//! Fallback mode — without `--antipodes`, the binary generates `--samples N`
//! random scrambles via depth-`D` reverse-move walks and verifies each
//! solution length is `≤ D`. Pair with `--cross-check` to additionally
//! require agreement with IDA\*+Manhattan (feasible only at shallow depths).
//! Use this when iterating without re-running the full antipode set.
//!
//! Usage:
//!
//! ```text
//! verify15 --pdb-dir DIR [--antipodes FILE]
//!          [--samples N] [--depth D] [--seed S] [--cross-check]
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle15::pdb::{AdditivePdbHeuristic, MaxHeuristic, PatternDb, ReflectedHeuristic};
use puzzle8::puzzle15::search::{idastar, Heuristic, ManhattanHeuristic};
use puzzle8::puzzle15::state::{Move, State, DIAMETER, GOAL, N_CELLS};

const P7_PATH: &str = "pdb15_p7_korf.bin";
const P8_PATH: &str = "pdb15_p8_korf.bin";

struct Args {
    pdb_dir: PathBuf,
    antipodes: Option<PathBuf>,
    samples: u32,
    depth: u32,
    seed: u64,
    cross_check: bool,
}

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} --pdb-dir DIR [--antipodes FILE] [--samples N] [--depth D] [--seed S] [--cross-check]");
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir: Option<PathBuf> = None;
    let mut antipodes: Option<PathBuf> = None;
    let mut samples: u32 = 50;
    let mut depth: u32 = 25;
    let mut seed: u64 = 0xDEAD_BEEF_F00D_BABE;
    let mut cross_check = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => {
                i += 1;
                pdb_dir = Some(PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?));
            }
            "--antipodes" => {
                i += 1;
                antipodes = Some(PathBuf::from(
                    argv.get(i).ok_or("--antipodes needs a value")?,
                ));
            }
            "--samples" => {
                i += 1;
                samples = argv
                    .get(i)
                    .ok_or("--samples needs a value")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
            }
            "--depth" => {
                i += 1;
                depth = argv
                    .get(i)
                    .ok_or("--depth needs a value")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
            }
            "--seed" => {
                i += 1;
                seed = argv
                    .get(i)
                    .ok_or("--seed needs a value")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
            }
            "--cross-check" => {
                cross_check = true;
            }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }
    let pdb_dir = pdb_dir.ok_or("missing --pdb-dir")?;
    Ok(Args {
        pdb_dir,
        antipodes,
        samples,
        depth,
        seed,
        cross_check,
    })
}

fn parse_position(s: &str) -> Result<State, String> {
    let mut cells = [0u8; N_CELLS];
    let mut count = 0usize;
    for tok in s.split_whitespace() {
        if count >= N_CELLS {
            return Err(format!("more than {N_CELLS} tokens"));
        }
        let v = if tok == "_" || tok == "." {
            0
        } else {
            tok.parse::<u8>()
                .map_err(|e| format!("token {tok:?}: {e}"))?
        };
        if v > 15 {
            return Err(format!("value {v} out of range"));
        }
        cells[count] = v;
        count += 1;
    }
    if count != N_CELLS {
        return Err(format!("expected {N_CELLS} tokens, got {count}"));
    }
    let mut seen = [false; N_CELLS];
    for &v in &cells {
        if seen[v as usize] {
            return Err(format!("duplicate {v}"));
        }
        seen[v as usize] = true;
    }
    Ok(State(cells))
}

fn load_pdbs(dir: &Path) -> Result<(PatternDb, PatternDb), String> {
    let p7 = PatternDb::load_mmap(&dir.join(P7_PATH))
        .map_err(|e| format!("loading {}: {}", dir.join(P7_PATH).display(), e))?;
    let p8 = PatternDb::load_mmap(&dir.join(P8_PATH))
        .map_err(|e| format!("loading {}: {}", dir.join(P8_PATH).display(), e))?;
    Ok((p7, p8))
}

/// Splitmix64-style PRNG for deterministic sampling.
fn next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn random_scramble(seed: u64, depth: u32) -> State {
    let mut s = GOAL;
    let mut rng = seed;
    let mut last: Option<Move> = None;
    let mut steps = 0u32;
    while steps < depth {
        let r = next(&mut rng);
        let candidate = Move::ALL[(r as usize) & 3];
        if !s.legal_moves().contains(candidate) {
            continue;
        }
        if let Some(p) = last {
            if candidate == p.inverse() {
                continue;
            }
        }
        s = s.apply(candidate);
        last = Some(candidate);
        steps += 1;
    }
    s
}

fn verify_antipodes(pdb_dir: &Path, antipodes_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(antipodes_path)
        .map_err(|e| format!("reading {}: {}", antipodes_path.display(), e))?;

    let mut positions: Vec<State> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let s = parse_position(trimmed).map_err(|e| format!("line {}: {}", lineno + 1, e))?;
        if !s.is_solvable() {
            return Err(format!("line {}: unsolvable position", lineno + 1));
        }
        positions.push(s);
    }

    if positions.is_empty() {
        return Err(
            "antipodes file contains no positions — fill it in from a verified source".into(),
        );
    }

    if positions.len() != 17 {
        eprintln!("note: expected 17 antipodes, got {}", positions.len());
    }

    let (p7_db, p8_db) = load_pdbs(pdb_dir)?;
    let dbs = [p7_db, p8_db];
    let h_add = AdditivePdbHeuristic::new(&dbs);
    let h_refl_inner = AdditivePdbHeuristic::new(&dbs);
    let h_refl = ReflectedHeuristic::new(h_refl_inner);
    let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);

    let mut all_pass = true;
    for (i, s) in positions.iter().enumerate() {
        let t0 = Instant::now();
        let sol = idastar(s, &h_korf).ok_or_else(|| format!("antipode {} unsolved", i + 1))?;
        let elapsed = t0.elapsed();
        let len = sol.len() as u8;
        let ok = len == DIAMETER;
        if !ok {
            all_pass = false;
        }
        println!(
            "antipode {:>2}/{}: depth={} (expected {}) in {:.2?} -- {}",
            i + 1,
            positions.len(),
            len,
            DIAMETER,
            elapsed,
            if ok { "OK" } else { "FAIL" }
        );
        // Sanity: applying solution must reach goal.
        let mut cur = *s;
        for m in &sol {
            cur = cur.apply(*m);
        }
        if cur != GOAL {
            println!("  WARNING: solution does not reach GOAL from this antipode");
            all_pass = false;
        }
    }
    if !all_pass {
        return Err("one or more antipodes failed verification".into());
    }
    Ok(())
}

fn verify_random_samples(args: &Args) -> Result<(), String> {
    let (p7_db, p8_db) = load_pdbs(&args.pdb_dir)?;
    let dbs = [p7_db, p8_db];
    let h_add = AdditivePdbHeuristic::new(&dbs);
    let h_refl_inner = AdditivePdbHeuristic::new(&dbs);
    let h_refl = ReflectedHeuristic::new(h_refl_inner);
    let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);

    println!(
        "Verifying {} random scrambles of depth {} (seed={}, cross_check={})",
        args.samples, args.depth, args.seed, args.cross_check
    );

    let mut all_pass = true;
    let mut max_len = 0u8;
    for k in 0..args.samples {
        let scramble_seed = args.seed.wrapping_add(k as u64);
        let s = random_scramble(scramble_seed, args.depth);
        debug_assert!(s.is_solvable());

        let t0 = Instant::now();
        let sol = idastar(&s, &h_korf).ok_or_else(|| format!("sample {k} unsolved"))?;
        let elapsed = t0.elapsed();
        let len = sol.len() as u8;
        if (len as u32) > args.depth {
            println!(
                "sample {:>4}: depth={} > walk depth {} — FAIL",
                k, len, args.depth
            );
            all_pass = false;
            continue;
        }
        max_len = max_len.max(len);

        // Apply solution.
        let mut cur = s;
        for m in &sol {
            cur = cur.apply(*m);
        }
        if cur != GOAL {
            println!("sample {k:>4}: solution does not reach GOAL — FAIL");
            all_pass = false;
            continue;
        }

        if args.cross_check {
            // Cross-check against IDA*+Manhattan when feasible (depth ≤ 25 typically OK).
            let m_sol = idastar(&s, &ManhattanHeuristic)
                .ok_or_else(|| format!("sample {k} unsolved by Manhattan"))?;
            if (m_sol.len() as u8) != len {
                println!(
                    "sample {:>4}: PDB={} != Manhattan={} — FAIL",
                    k,
                    len,
                    m_sol.len()
                );
                all_pass = false;
                continue;
            }
        }

        if k % 10 == 0 || k == args.samples - 1 {
            println!(
                "sample {:>4}/{}: depth={} in {:.2?}",
                k, args.samples, len, elapsed
            );
        }
    }

    if all_pass {
        println!(
            "\nAll {} samples passed. Max observed optimal depth: {}",
            args.samples, max_len
        );
        Ok(())
    } else {
        Err("one or more samples failed".into())
    }
}

fn main() -> ExitCode {
    let prog = std::env::args().next().unwrap_or_else(|| "verify15".into());
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

    let result = if let Some(path) = &args.antipodes {
        verify_antipodes(&args.pdb_dir, path)
    } else {
        verify_random_samples(&args)
    };

    match result {
        Ok(()) => {
            println!("\nverification: PASS");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\nverification: FAIL — {e}");
            ExitCode::FAILURE
        }
    }
}
