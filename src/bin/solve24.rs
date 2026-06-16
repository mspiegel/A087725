//! Solve a 24-puzzle position optimally via IDA* with a PDB-based heuristic.
//!
//! ```text
//! solve24 --pdb-dir data/ [--position "<25 tokens>"] [--from FILE]
//!         [--heuristic korf|manhattan|zpdb]
//! ```
//!
//! Position format: 25 whitespace-separated tokens in row-major order, `_`/`0`
//! for the blank and `1..=24` for tiles.
//!
//! Heuristics:
//! - `manhattan` : sum of Manhattan distances (no PDB needed).
//! - `korf`      : incremental `max(additive, additive(reflect(s)))` over the
//!                 four canonical 6-6-6-6 additive PDBs (P24D files). Default.
//! - `zpdb`      : same composition, but on **zero-aware 1-bit PDBs** (Z24D
//!                 files). Strictly dominates `korf` per Clausecker &
//!                 Reinefeld 2019 (≈1.6× on a single 6-tile PDB).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::pdb::{KorfPdbInc, PatternDb, ZPatternDb, ZpdbInc};
use puzzle8::puzzle24::search::{idastar, idastar_inc_with_stats, ManhattanHeuristic};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

const PDB_FILES: [&str; 4] = ["pdb24_a.bin", "pdb24_b.bin", "pdb24_c.bin", "pdb24_d.bin"];
const ZPDB_FILES: [&str; 4] = ["pdb24_a.zbin", "pdb24_b.zbin", "pdb24_c.zbin", "pdb24_d.zbin"];

#[derive(PartialEq)]
enum HeuristicChoice {
    Manhattan,
    Korf,
    Zpdb,
}

struct Args {
    pdb_dir: Option<PathBuf>,
    position: Option<String>,
    from: Option<PathBuf>,
    heuristic: HeuristicChoice,
}

fn print_usage(prog: &str) {
    eprintln!("usage: {} --pdb-dir DIR [--position \"...\"] [--from FILE] [--heuristic korf|manhattan|zpdb]", prog);
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = None;
    let mut position = None;
    let mut from = None;
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
                    "korf" => HeuristicChoice::Korf,
                    "zpdb" => HeuristicChoice::Zpdb,
                    other => return Err(format!("unknown heuristic {:?}", other)),
                };
            }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
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
            return Err(format!("more than {} tokens", N_CELLS));
        }
        let v = if tok == "_" || tok == "." {
            0
        } else {
            tok.parse::<u8>().map_err(|e| format!("token {:?}: {}", tok, e))?
        };
        if v > 24 {
            return Err(format!("value {} out of range 0..=24", v));
        }
        cells[count] = v;
        count += 1;
    }
    if count != N_CELLS {
        return Err(format!("expected {} tokens, got {}", N_CELLS, count));
    }
    let mut seen = [false; N_CELLS];
    for &v in &cells {
        if seen[v as usize] {
            return Err(format!("value {} appears more than once", v));
        }
        seen[v as usize] = true;
    }
    Ok(State(cells))
}

fn load_pdbs(dir: &Path) -> Result<Vec<PatternDb>, String> {
    let mut dbs = Vec::with_capacity(4);
    for name in PDB_FILES {
        let path = dir.join(name);
        let db = PatternDb::load_mmap(&path)
            .map_err(|e| format!("loading {}: {}", path.display(), e))?;
        dbs.push(db);
    }
    Ok(dbs)
}

fn load_zpdbs(dir: &Path) -> Result<Vec<ZPatternDb>, String> {
    let mut dbs = Vec::with_capacity(4);
    for name in ZPDB_FILES {
        let path = dir.join(name);
        let db = ZPatternDb::load_mmap(&path)
            .map_err(|e| format!("loading {}: {}", path.display(), e))?;
        dbs.push(db);
    }
    Ok(dbs)
}

fn print_solution(start: &State, sol: &[Move], elapsed: std::time::Duration) {
    println!("Solution length: {}", sol.len());
    let moves: Vec<&str> = sol
        .iter()
        .map(|m| match m {
            Move::Up => "U",
            Move::Down => "D",
            Move::Left => "L",
            Move::Right => "R",
        })
        .collect();
    println!("Moves          : {}", moves.join(" "));
    println!("Wall-clock     : {:.2?}", elapsed);

    let mut cur = *start;
    for m in sol {
        cur = cur.apply(*m);
    }
    if cur == GOAL {
        println!("Verified       : reaches GOAL");
    } else {
        println!("WARNING        : solution does NOT reach GOAL — {:?}", cur.0);
    }
}

fn main() -> ExitCode {
    let prog = std::env::args().next().unwrap_or_else(|| "solve24".into());
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

    let position_str = match (args.position, args.from) {
        (Some(p), _) => p,
        (None, Some(f)) => match std::fs::read_to_string(&f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error reading {}: {}", f.display(), e);
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("error: provide --position or --from");
            return ExitCode::FAILURE;
        }
    };
    let start = match parse_position(&position_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: position parse: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if !start.is_solvable() {
        eprintln!("error: position is not solvable");
        return ExitCode::FAILURE;
    }

    let t0 = Instant::now();
    let (sol, stats) = match args.heuristic {
        HeuristicChoice::Manhattan => {
            let s = idastar(&start, &ManhattanHeuristic);
            (s, None)
        }
        HeuristicChoice::Korf => {
            let dir = match &args.pdb_dir {
                Some(d) => d.clone(),
                None => {
                    eprintln!("error: --pdb-dir required for korf heuristic");
                    return ExitCode::FAILURE;
                }
            };
            let dbs = match load_pdbs(&dir) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let inc = KorfPdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            let (s, st) = idastar_inc_with_stats(&start, &inc);
            (s, Some(st))
        }
        HeuristicChoice::Zpdb => {
            let dir = match &args.pdb_dir {
                Some(d) => d.clone(),
                None => {
                    eprintln!("error: --pdb-dir required for zpdb heuristic");
                    return ExitCode::FAILURE;
                }
            };
            let dbs = match load_zpdbs(&dir) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            let (s, st) = idastar_inc_with_stats(&start, &inc);
            (s, Some(st))
        }
    };

    let elapsed = t0.elapsed();
    match sol {
        Some(s) => {
            print_solution(&start, &s, elapsed);
            if let Some(st) = stats {
                println!("Nodes          : {}", st.nodes);
                println!("Iterations     : {}", st.iterations);
            }
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("no solution found");
            ExitCode::FAILURE
        }
    }
}
