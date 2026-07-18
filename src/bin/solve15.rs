//! Solve a 15-puzzle position optimally via IDA* with a PDB-based heuristic.
//!
//! Usage:
//!
//! ```text
//! solve15 --pdb-dir data/ [--position "<16 tokens>"] [--from FILE]
//!         [--heuristic korf|additive|manhattan]
//! ```
//!
//! Position format: 16 whitespace-separated tokens in row-major order, with
//! `_` or `0` for the blank and decimal values `1..=15` for tiles.
//!
//! Heuristic options (default `korf`):
//! - `manhattan` : sum of Manhattan distances (no PDB needed)
//! - `additive`  : additive of `pdb15_p7_korf` + `pdb15_p8_korf` (Korf 7-8)
//! - `korf`      : `max(additive, additive(reflect(s)))` — Korf's tightest
//!                 admissible heuristic at half the storage of separate
//!                 reflected PDB files. Default.
//! - `korf-plus` : `max(korf, manhattan + linear_conflict, walking_distance)`.
//!                 Free upside on top of Korf: ~25 KB of WD tables plus
//!                 cheap arithmetic, occasionally tightens beyond the PDB
//!                 where the PDB's subset-blindness loses signal.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle15::pdb::{
    AdditivePdbHeuristic, MaxHeuristic, PatternDb, ReflectedHeuristic,
};
use puzzle8::puzzle15::search::{
    idastar, Heuristic, LinearConflictHeuristic, ManhattanHeuristic,
    WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::{Move, State, GOAL, N_CELLS};

const P7_PATH: &str = "pdb15_p7_korf.bin";
const P8_PATH: &str = "pdb15_p8_korf.bin";

#[derive(PartialEq)]
enum HeuristicChoice {
    Manhattan,
    Additive,
    Korf,
    KorfPlus,
}

struct Args {
    pdb_dir: Option<PathBuf>,
    position: Option<String>,
    from: Option<PathBuf>,
    heuristic: HeuristicChoice,
}

fn print_usage(prog: &str) {
    eprintln!("usage: {prog} --pdb-dir DIR [--position \"...\"] [--from FILE] [--heuristic korf-plus|korf|additive|manhattan]");
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir: Option<PathBuf> = None;
    let mut position: Option<String> = None;
    let mut from: Option<PathBuf> = None;
    let mut heuristic = HeuristicChoice::Korf;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => {
                i += 1;
                pdb_dir = Some(PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?));
            }
            "--position" => {
                i += 1;
                position = Some(argv.get(i).ok_or("--position needs a value")?.clone());
            }
            "--from" => {
                i += 1;
                from = Some(PathBuf::from(argv.get(i).ok_or("--from needs a value")?));
            }
            "--heuristic" => {
                i += 1;
                heuristic = match argv.get(i).ok_or("--heuristic needs a value")?.as_str() {
                    "manhattan" => HeuristicChoice::Manhattan,
                    "additive" => HeuristicChoice::Additive,
                    "korf" => HeuristicChoice::Korf,
                    "korf-plus" => HeuristicChoice::KorfPlus,
                    other => return Err(format!("unknown heuristic {other:?}")),
                };
            }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    Ok(Args { pdb_dir, position, from, heuristic })
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
            tok.parse::<u8>().map_err(|e| format!("token {tok:?}: {e}"))?
        };
        if v > 15 {
            return Err(format!("value {v} out of range 0..=15"));
        }
        cells[count] = v;
        count += 1;
    }
    if count != N_CELLS {
        return Err(format!("expected {N_CELLS} tokens, got {count}"));
    }
    // Sanity: each value 0..=15 appears exactly once.
    let mut seen = [false; N_CELLS];
    for &v in &cells {
        if seen[v as usize] {
            return Err(format!("value {v} appears more than once"));
        }
        seen[v as usize] = true;
    }
    Ok(State(cells))
}

fn load_pdbs(dir: &Path) -> Result<(PatternDb, PatternDb), String> {
    let p7 = dir.join(P7_PATH);
    let p8 = dir.join(P8_PATH);
    let p7_db = PatternDb::load_mmap(&p7)
        .map_err(|e| format!("loading {}: {}", p7.display(), e))?;
    let p8_db = PatternDb::load_mmap(&p8)
        .map_err(|e| format!("loading {}: {}", p8.display(), e))?;
    Ok((p7_db, p8_db))
}

fn print_solution(start: &State, sol: &[Move], elapsed: std::time::Duration) {
    println!("Solution length: {}", sol.len());
    let moves: Vec<&str> = sol.iter().map(|m| match m {
        Move::Up => "U",
        Move::Down => "D",
        Move::Left => "L",
        Move::Right => "R",
    }).collect();
    println!("Moves          : {}", moves.join(" "));
    println!("Wall-clock     : {elapsed:.2?}");

    // Sanity-check: applying the solution must reach GOAL.
    let mut cur = *start;
    for m in sol {
        cur = cur.apply(*m);
    }
    if cur == GOAL {
        println!("Verified       : reaches GOAL");
    } else {
        println!("WARNING        : solution does NOT reach GOAL — state: {:?}", cur.0);
    }
}

fn main() -> ExitCode {
    let prog = std::env::args().next().unwrap_or_else(|| "solve15".into());
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e == "help" { print_usage(&prog); return ExitCode::SUCCESS; }
            eprintln!("error: {e}");
            print_usage(&prog);
            return ExitCode::FAILURE;
        }
    };

    // Resolve position.
    let position_str = match (args.position, args.from) {
        (Some(p), _) => p,
        (None, Some(f)) => match std::fs::read_to_string(&f) {
            Ok(s) => s,
            Err(e) => { eprintln!("error reading {}: {}", f.display(), e); return ExitCode::FAILURE; }
        },
        _ => { eprintln!("error: provide --position or --from"); return ExitCode::FAILURE; }
    };
    let start = match parse_position(&position_str) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: position parse: {e}"); return ExitCode::FAILURE; }
    };

    if !start.is_solvable() {
        eprintln!("error: position is not solvable");
        return ExitCode::FAILURE;
    }

    // Build heuristic and run.
    let t0 = Instant::now();
    let sol = match args.heuristic {
        HeuristicChoice::Manhattan => {
            idastar(&start, &ManhattanHeuristic)
        }
        _ => {
            let dir = match &args.pdb_dir {
                Some(d) => d.clone(),
                None => { eprintln!("error: --pdb-dir required for PDB heuristic"); return ExitCode::FAILURE; }
            };
            let (p7_db, p8_db) = match load_pdbs(&dir) {
                Ok(x) => x,
                Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            };
            let dbs = [p7_db, p8_db];
            let h_add = AdditivePdbHeuristic::new(&dbs);
            match args.heuristic {
                HeuristicChoice::Additive => idastar(&start, &h_add),
                HeuristicChoice::Korf => {
                    // Reflected variant uses its own AdditivePdbHeuristic (also borrows dbs).
                    let h_refl_inner = AdditivePdbHeuristic::new(&dbs);
                    let h_refl = ReflectedHeuristic::new(h_refl_inner);
                    let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);
                    idastar(&start, &h_korf)
                }
                HeuristicChoice::KorfPlus => {
                    let h_refl_inner = AdditivePdbHeuristic::new(&dbs);
                    let h_refl = ReflectedHeuristic::new(h_refl_inner);
                    let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);
                    WalkingDistanceHeuristic::warm_up();
                    let h_classical = MaxHeuristic::new(
                        &LinearConflictHeuristic as &dyn Heuristic,
                        &WalkingDistanceHeuristic as &dyn Heuristic,
                    );
                    let h_plus = MaxHeuristic::new(
                        &h_korf as &dyn Heuristic,
                        &h_classical as &dyn Heuristic,
                    );
                    idastar(&start, &h_plus)
                }
                HeuristicChoice::Manhattan => unreachable!(),
            }
        }
    };

    let elapsed = t0.elapsed();
    match sol {
        Some(s) => { print_solution(&start, &s, elapsed); ExitCode::SUCCESS }
        None => { eprintln!("no solution found"); ExitCode::FAILURE }
    }
}
