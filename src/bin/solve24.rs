//! Solve a 24-puzzle position optimally via IDA* with a PDB-based heuristic, or
//! prove a lower bound on its optimal depth.
//!
//! ```text
//! solve24 --pdb-dir data/ [--position "<25 tokens>"] [--from FILE]
//!         [--heuristic korf|manhattan|zpdb|zpdb-plus]
//!         [--pdb-set k6|k7]
//!         [--max-bound T | --prove-at-least T]
//! ```
//!
//! Position format: 25 whitespace-separated tokens in row-major order, `_`/`0`
//! for the blank and `1..=24` for tiles.
//!
//! Heuristics:
//! - `manhattan` : sum of Manhattan distances (no PDB needed).
//! - `korf`      : incremental `max(additive, additive(reflect(s)))` over the
//!                 four canonical 6-6-6-6 additive PDBs (`pdb24_*.bin`).
//! - `zpdb`      : same composition on **zero-aware 1-bit PDBs** (`*.zbin`).
//!                 Strictly dominates `korf`. Default.
//! - `zpdb-plus` : `max(zpdb, refl-zpdb, MD+LinearConflict, WalkingDistance)` —
//!                 the strongest admissible heuristic here, all advanced
//!                 incrementally per node.
//! - `lc` / `wd` : standalone Linear-Conflict / Walking-Distance (WD is the
//!                 strongest term on deep boards like the 180° rotation `R`).
//! - `select`    : per-board 3-way auto-pick — cheap `max(LC,WD)` / pure `zpdb` /
//!                 `zpdb-plus` from the root heuristics (Phase 1C policy: WD on
//!                 deep boards, zpdb on general). `--combine-slack` tunes it.
//!
//! PDB set (for `zpdb`/`zpdb-plus`):
//! - `k6` : the 6-6-6-6 partition `pdb24_{a,b,c,d}.zbin` (default).
//! - `k7` : the 7-7-7-3 partition `pdb24_k7_{a,b,c,d}.zbin` (larger, tighter).
//!
//! Modes:
//! - default        : optimal solve (first solution found is optimal).
//! - `--max-bound T` : bounded / lower-bound search capped at threshold `T`.
//!                     Prints `Lower bound: depth >= K` if it exhausts `T`, or the
//!                     optimal solution if found at/below `T`.
//! - `--prove-at-least T` : sugar for `--max-bound (T-1)` — succeeds in proving
//!                     `depth >= T` iff the search exhausts every threshold < T.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::pdb::{KorfPdbInc, PatternDb, ZPatternDb, ZpdbInc, ZpdbPlusInc};
use puzzle8::puzzle24::search::{
    idastar_inc_bounded_parallel_mut, idastar_inc_bounded_parallel_mut_pruned,
    idastar_inc_mut_bounded_with_stats, idastar_inc_mut_with_stats, BoundedOutcome, Cwd,
    IncHeuristic, IncHeuristicMut, IncManhattan, LadderOutcome, LinearConflictInc, MaxInc, MoveDfa,
    SearchStats, WalkingDistanceHeuristic, WalkingDistanceInc,
};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

const PDB_FILES: [&str; 4] = ["pdb24_a.bin", "pdb24_b.bin", "pdb24_c.bin", "pdb24_d.bin"];
const ZPDB_FILES_K6: [&str; 4] = ["pdb24_a.zbin", "pdb24_b.zbin", "pdb24_c.zbin", "pdb24_d.zbin"];
const ZPDB_FILES_K7: [&str; 4] = [
    "pdb24_k7_a.zbin",
    "pdb24_k7_b.zbin",
    "pdb24_k7_c.zbin",
    "pdb24_k7_d.zbin",
];

#[derive(PartialEq, Clone, Copy)]
enum HeuristicChoice {
    Manhattan,
    Lc,
    Wd,
    /// Escape-constrained Walking Distance (`≥ WD`, table-free per-node A*).
    Cwd,
    Korf,
    Zpdb,
    ZpdbPlus,
    /// Per-board 3-way auto-selector: cheap `max(LC,WD)` / pure `zpdb` / `zpdb-plus`
    /// picked from the root heuristics (the settled Phase 1C policy — WD on
    /// deep boards, zpdb on general boards). Uses the `--pdb-set` PDBs.
    Select,
}

#[derive(PartialEq, Clone, Copy)]
enum PdbSet {
    K6,
    K7,
}

struct Args {
    pdb_dir: Option<PathBuf>,
    position: Option<String>,
    from: Option<PathBuf>,
    heuristic: HeuristicChoice,
    pdb_set: PdbSet,
    /// Cap the IDA* threshold here; `None` = unbounded optimal solve.
    max_bound: Option<u8>,
    /// Use the shared-memory parallel IDA* driver (2P).
    parallel: bool,
    /// Stack the move-pruning DFA (Taylor–Korf duplicate elimination) on the
    /// heuristic. Applies to the parallel driver; sound for lower-bound proofs.
    move_dfa: bool,
    /// For `--heuristic select`: pick zpdb-plus (over pure zpdb) when the classical
    /// terms are within this many of the PDB root. `0` = never combine (default).
    combine_slack: u8,
}

/// The `select` heuristic's 3-way choice (mirrors `examples/ladder24.rs`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    Cheap,
    Zpdb,
    ZpdbPlus,
}

/// Pick from the root heuristics: cheap `max(LC,WD)` unless the PDB root wins, then
/// pure zpdb (weak classical terms) or zpdb-plus (classical terms within `slack`).
fn pick_heuristic(cheap_root: u8, zpdb_root: u8, slack: u8) -> Pick {
    if zpdb_root <= cheap_root {
        Pick::Cheap
    } else if cheap_root + slack >= zpdb_root {
        Pick::ZpdbPlus
    } else {
        Pick::Zpdb
    }
}

fn print_usage(prog: &str) {
    eprintln!(
        "usage: {} --pdb-dir DIR [--position \"...\"] [--from FILE]\n         \
         [--heuristic manhattan|lc|wd|cwd|korf|zpdb|zpdb-plus|select] [--pdb-set k6|k7]\n         \
         [--max-bound T | --prove-at-least T] [--parallel] [--move-dfa] [--combine-slack S]",
        prog
    );
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = None;
    let mut position = None;
    let mut from = None;
    let mut heuristic = HeuristicChoice::Zpdb;
    let mut pdb_set = PdbSet::K6;
    let mut max_bound = None;
    let mut parallel = false;
    let mut move_dfa = false;
    let mut combine_slack = 0u8;

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
                    "lc" => HeuristicChoice::Lc,
                    "wd" => HeuristicChoice::Wd,
                    "cwd" => HeuristicChoice::Cwd,
                    "korf" => HeuristicChoice::Korf,
                    "zpdb" => HeuristicChoice::Zpdb,
                    "zpdb-plus" => HeuristicChoice::ZpdbPlus,
                    "select" => HeuristicChoice::Select,
                    other => return Err(format!("unknown heuristic {:?}", other)),
                };
            }
            "--pdb-set" => {
                i += 1;
                pdb_set = match argv.get(i).ok_or("--pdb-set needs a value")?.as_str() {
                    "k6" => PdbSet::K6,
                    "k7" => PdbSet::K7,
                    other => return Err(format!("unknown pdb-set {:?}", other)),
                };
            }
            "--max-bound" => {
                i += 1;
                let t: u8 = argv
                    .get(i)
                    .ok_or("--max-bound needs a value")?
                    .parse()
                    .map_err(|e| format!("--max-bound: {}", e))?;
                max_bound = Some(t);
            }
            "--prove-at-least" => {
                i += 1;
                let t: u8 = argv
                    .get(i)
                    .ok_or("--prove-at-least needs a value")?
                    .parse()
                    .map_err(|e| format!("--prove-at-least: {}", e))?;
                if t == 0 {
                    return Err("--prove-at-least must be >= 1".into());
                }
                // Exhausting every threshold < T proves depth >= T.
                max_bound = Some(t - 1);
            }
            "--parallel" => parallel = true,
            "--move-dfa" => move_dfa = true,
            "--combine-slack" => {
                i += 1;
                combine_slack = argv
                    .get(i)
                    .ok_or("--combine-slack needs a value")?
                    .parse()
                    .map_err(|e| format!("--combine-slack: {}", e))?;
            }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    Ok(Args { pdb_dir, position, from, heuristic, pdb_set, max_bound, parallel, move_dfa, combine_slack })
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

fn load_zpdbs(dir: &Path, set: PdbSet) -> Result<Vec<ZPatternDb>, String> {
    let files = match set {
        PdbSet::K6 => ZPDB_FILES_K6,
        PdbSet::K7 => ZPDB_FILES_K7,
    };
    let mut dbs = Vec::with_capacity(4);
    for name in files {
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

fn print_stats(st: &SearchStats, elapsed: std::time::Duration) {
    println!("Nodes          : {}", st.nodes);
    println!("Iterations     : {}", st.iterations);
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        println!("Throughput     : {:.2} Mnodes/s", st.nodes as f64 / secs / 1e6);
    }
}

/// Drive an incremental heuristic, dispatching on bounded vs optimal mode.
// Every heuristic is driven through the make/unmake ([`IncHeuristicMut`]) path —
// strictly cheaper per node than the old `Copy` context path and the default
// everywhere (there is no toggle). `IncHeuristic` is still required because the
// parallel driver's shallow split reuses it to grow the work frontier.
fn run_inc<E: IncHeuristic + IncHeuristicMut + Sync>(
    start: &State,
    e: &E,
    max_bound: Option<u8>,
    parallel: bool,
    move_dfa: bool,
    t0: Instant,
) -> ExitCode
where
    <E as IncHeuristic>::Ctx: Send + Sync,
{
    if parallel {
        // Shared-memory parallel IDA* (2P). max_bound = None -> unbounded solve.
        let mb = max_bound.unwrap_or(u8::MAX);
        let (outcome, st) = if move_dfa {
            // Stack the move-pruning DFA (Taylor–Korf duplicate elimination) on top
            // of the heuristic. Sound for lower-bound proofs; build() self-verifies.
            let dfa = MoveDfa::build_default();
            eprintln!(
                "move-pruning DFA: {} states, {} KiB (L2-resident)",
                dfa.states(),
                dfa.table_bytes() / 1024
            );
            idastar_inc_bounded_parallel_mut_pruned(start, e, &dfa, mb, None)
        } else {
            idastar_inc_bounded_parallel_mut(start, e, mb, None)
        };
        let elapsed = t0.elapsed();
        return match outcome {
            LadderOutcome::Solved(s) => {
                if let Some(mb) = max_bound {
                    println!("Found within bound {} (optimal):", mb);
                }
                print_solution(start, &s, elapsed);
                print_stats(&st, elapsed);
                ExitCode::SUCCESS
            }
            LadderOutcome::ProvedAtLeast(k) => {
                println!("Lower bound: depth >= {}", k);
                println!("Wall-clock     : {:.2?}", elapsed);
                print_stats(&st, elapsed);
                ExitCode::SUCCESS
            }
            LadderOutcome::TimedOut(k) => {
                // No deadline is passed here, so this is not expected.
                println!("Lower bound: depth >= {} (timed out)", k);
                print_stats(&st, elapsed);
                ExitCode::SUCCESS
            }
            LadderOutcome::Unsolvable => {
                eprintln!("position is unsolvable");
                ExitCode::FAILURE
            }
        };
    }
    match max_bound {
        None => {
            let (sol, st) = idastar_inc_mut_with_stats(start, e);
            let elapsed = t0.elapsed();
            match sol {
                Some(s) => {
                    print_solution(start, &s, elapsed);
                    print_stats(&st, elapsed);
                    ExitCode::SUCCESS
                }
                None => {
                    eprintln!("no solution found");
                    ExitCode::FAILURE
                }
            }
        }
        Some(mb) => {
            let (outcome, st) = idastar_inc_mut_bounded_with_stats(start, e, mb);
            let elapsed = t0.elapsed();
            match outcome {
                BoundedOutcome::Solved(s) => {
                    println!("Found within bound {} (optimal):", mb);
                    print_solution(start, &s, elapsed);
                    print_stats(&st, elapsed);
                    ExitCode::SUCCESS
                }
                BoundedOutcome::ProvedAtLeast(k) => {
                    println!("Lower bound: depth >= {}", k);
                    println!("Wall-clock     : {:.2?}", elapsed);
                    print_stats(&st, elapsed);
                    ExitCode::SUCCESS
                }
                BoundedOutcome::Unsolvable => {
                    eprintln!("position is unsolvable");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

fn require_dir(args: &Args, label: &str) -> Result<PathBuf, ExitCode> {
    match &args.pdb_dir {
        Some(d) => Ok(d.clone()),
        None => {
            eprintln!("error: --pdb-dir required for {} heuristic", label);
            Err(ExitCode::FAILURE)
        }
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

    let position_str = match (&args.position, &args.from) {
        (Some(p), _) => p.clone(),
        (None, Some(f)) => match std::fs::read_to_string(f) {
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
    match args.heuristic {
        HeuristicChoice::Manhattan => run_inc(&start, &IncManhattan, args.max_bound, args.parallel, args.move_dfa, t0),
        HeuristicChoice::Lc => run_inc(&start, &LinearConflictInc, args.max_bound, args.parallel, args.move_dfa, t0),
        HeuristicChoice::Wd => {
            WalkingDistanceHeuristic::warm_up_verbose();
            run_inc(&start, &WalkingDistanceInc, args.max_bound, args.parallel, args.move_dfa, t0)
        }
        HeuristicChoice::Cwd => {
            eprintln!("cWD: loading tables…");
            let cwd = Cwd::new();
            eprintln!(
                "cWD ready: {} path",
                if cwd.has_overlay() { "fast single-line-max table" } else { "reference per-node A*" }
            );
            run_inc(&start, &cwd, args.max_bound, args.parallel, args.move_dfa, t0)
        }
        HeuristicChoice::Korf => {
            let dir = match require_dir(&args, "korf") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_pdbs(&dir) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let inc = KorfPdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, t0)
        }
        HeuristicChoice::Zpdb => {
            let dir = match require_dir(&args, "zpdb") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, t0)
        }
        HeuristicChoice::ZpdbPlus => {
            let dir = match require_dir(&args, "zpdb-plus") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            // Walking Distance needs its (heavy) table built before the search.
            WalkingDistanceHeuristic::warm_up_verbose();
            let inc = ZpdbPlusInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, t0)
        }
        HeuristicChoice::Select => {
            let dir = match require_dir(&args, "select") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            WalkingDistanceHeuristic::warm_up_verbose();
            // Auto-pick per board from the root heuristics (Phase 1C policy).
            let cheap = MaxInc::new(LinearConflictInc, WalkingDistanceInc);
            let zpdb = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            let mut st = SearchStats::default();
            // UFCS: both heuristics implement IncHeuristic and IncHeuristicMut,
            // whose `root`s collide under method syntax.
            let cheap_root = IncHeuristic::root(&cheap, &start, &mut st).0;
            let zpdb_root = IncHeuristic::root(&zpdb, &start, &mut st).0;
            match pick_heuristic(cheap_root, zpdb_root, args.combine_slack) {
                Pick::Cheap => {
                    println!("Selected       : max(LC,WD)  (cheap_h {} >= zpdb_h {})", cheap_root, zpdb_root);
                    run_inc(&start, &cheap, args.max_bound, args.parallel, args.move_dfa, t0)
                }
                Pick::Zpdb => {
                    println!("Selected       : zpdb  (zpdb_h {} > cheap_h {})", zpdb_root, cheap_root);
                    run_inc(&start, &zpdb, args.max_bound, args.parallel, args.move_dfa, t0)
                }
                Pick::ZpdbPlus => {
                    println!("Selected       : zpdb-plus  (classical terms within slack of zpdb_h {})", zpdb_root);
                    let zplus = ZpdbPlusInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
                    run_inc(&start, &zplus, args.max_bound, args.parallel, args.move_dfa, t0)
                }
            }
        }
    }
}
