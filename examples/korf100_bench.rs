//! Korf-100 benchmark: solve Korf's 100 standard 15-puzzle instances optimally
//! and report search effort (nodes) and wall-clock per instance plus aggregates.
//!
//! This is the regression/measurement harness referenced in `OPTIMIZATION.md`:
//! node count and wall-clock are tracked separately so each optimization can be
//! attributed correctly (incremental-heuristic and codegen changes move
//! wall-clock at fixed nodes; move-ordering and duplicate-pruning move nodes).
//!
//! Run with:
//!
//! ```text
//! cargo run --release --features mmap --example korf100_bench -- --pdb-dir data
//! ```
//!
//! Flags:
//! - `--pdb-dir DIR`     directory holding `pdb15_p7_korf.bin` / `pdb15_p8_korf.bin` (default `data`)
//! - `--instances FILE`  instance file (default `data/korf100.txt`)
//! - `--heuristic NAME`  `korf` (default), `korf-plus`, `additive`, or `manhattan`
//! - `--limit N`         solve only the first `N` instances (smoke testing)
//! - `--quiet`           suppress the per-instance table, print only the summary
//! - `--scratch`         force the from-scratch `korf` heuristic instead of the
//!                       incremental evaluator (for A/B comparison)
//!
//! Correctness: every instance is solved, the solution is replayed to confirm it
//! reaches `GOAL`, and its length is asserted equal to Korf's published optimal
//! (the `optimal` column of `data/korf100.txt`). Any mismatch is collected and
//! reported, and the process exits non-zero.
//!
//! Goal convention: Korf's instances use a blank-top-left goal; this repo uses
//! blank-bottom-right. [`to_repo_goal`] applies the distance-preserving 180°
//! rotation + value-complement transform on load (see `data/korf100.txt`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use puzzle8::puzzle15::pdb::{
    AdditivePdbHeuristic, KorfPdbInc, MaxHeuristic, PatternDb, ReflectedHeuristic,
};
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, idastar_with_stats, Heuristic, LinearConflictHeuristic,
    ManhattanHeuristic, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::{Move, State, GOAL, N_CELLS};

const P7_PATH: &str = "pdb15_p7_korf.bin";
const P8_PATH: &str = "pdb15_p8_korf.bin";

#[derive(PartialEq, Clone, Copy)]
enum HeuristicChoice {
    Manhattan,
    Additive,
    Korf,
    KorfPlus,
}

struct Args {
    pdb_dir: PathBuf,
    instances: PathBuf,
    heuristic: HeuristicChoice,
    limit: Option<usize>,
    quiet: bool,
    /// Force the from-scratch heuristic path even for `korf` (which otherwise
    /// uses the incremental [`KorfPdbInc`] evaluator). For A/B measurement.
    scratch: bool,
}

struct Instance {
    index: u32,
    optimal: u8,
    state: State,
}

/// One solved instance's measured effort.
struct Row {
    index: u32,
    optimal: u8,
    nodes: u64,
    iterations: u32,
    elapsed: Duration,
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = PathBuf::from("data");
    let mut instances = PathBuf::from("data/korf100.txt");
    let mut heuristic = HeuristicChoice::Korf;
    let mut limit: Option<usize> = None;
    let mut quiet = false;
    let mut scratch = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => {
                i += 1;
                pdb_dir = PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?);
            }
            "--instances" => {
                i += 1;
                instances = PathBuf::from(argv.get(i).ok_or("--instances needs a value")?);
            }
            "--heuristic" => {
                i += 1;
                heuristic = match argv.get(i).ok_or("--heuristic needs a value")?.as_str() {
                    "manhattan" => HeuristicChoice::Manhattan,
                    "additive" => HeuristicChoice::Additive,
                    "korf" => HeuristicChoice::Korf,
                    "korf-plus" => HeuristicChoice::KorfPlus,
                    other => return Err(format!("unknown heuristic {:?}", other)),
                };
            }
            "--limit" => {
                i += 1;
                limit = Some(
                    argv.get(i)
                        .ok_or("--limit needs a value")?
                        .parse::<usize>()
                        .map_err(|e| format!("--limit: {}", e))?,
                );
            }
            "--quiet" => quiet = true,
            "--scratch" => scratch = true,
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }

    Ok(Args { pdb_dir, instances, heuristic, limit, quiet, scratch })
}

/// Convert a Korf-convention board (blank top-left, value == cell index in the
/// goal) into this repo's blank-bottom-right `State`.
///
/// The map is a 180° board rotation (cell `i` -> cell `15 - i`) composed with
/// the value complement `v -> 16 - v` (blank fixed). It sends Korf's goal to
/// this repo's goal and preserves move adjacency, so optimal distance is
/// invariant.
fn to_repo_goal(korf: &[u8; N_CELLS]) -> State {
    let mut cells = [0u8; N_CELLS];
    for i in 0..N_CELLS {
        let v = korf[i];
        cells[N_CELLS - 1 - i] = if v == 0 { 0 } else { 16 - v };
    }
    State(cells)
}

fn parse_instances(path: &Path) -> Result<Vec<Instance>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading {}: {}", path.display(), e))?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (head, tiles_str) = line
            .split_once(':')
            .ok_or_else(|| format!("line {}: missing ':' separator", lineno + 1))?;
        let head: Vec<&str> = head.split_whitespace().collect();
        if head.len() != 2 {
            return Err(format!(
                "line {}: expected '<index> <optimal>' before ':', got {:?}",
                lineno + 1,
                head
            ));
        }
        let index: u32 = head[0]
            .parse()
            .map_err(|e| format!("line {}: index: {}", lineno + 1, e))?;
        let optimal: u8 = head[1]
            .parse()
            .map_err(|e| format!("line {}: optimal: {}", lineno + 1, e))?;

        let mut cells = [0u8; N_CELLS];
        let mut count = 0usize;
        for tok in tiles_str.split_whitespace() {
            if count >= N_CELLS {
                return Err(format!("line {}: more than {} tiles", lineno + 1, N_CELLS));
            }
            let v: u8 = tok
                .parse()
                .map_err(|e| format!("line {}: tile {:?}: {}", lineno + 1, tok, e))?;
            if v > 15 {
                return Err(format!("line {}: tile {} out of range 0..=15", lineno + 1, v));
            }
            cells[count] = v;
            count += 1;
        }
        if count != N_CELLS {
            return Err(format!(
                "line {}: expected {} tiles, got {}",
                lineno + 1,
                N_CELLS,
                count
            ));
        }
        // Sanity: a permutation of 0..=15.
        let mut seen = [false; N_CELLS];
        for &v in &cells {
            if seen[v as usize] {
                return Err(format!("line {}: value {} repeats", lineno + 1, v));
            }
            seen[v as usize] = true;
        }

        let state = to_repo_goal(&cells);
        if !state.is_solvable() {
            return Err(format!(
                "line {}: instance {} not solvable after goal-convention transform",
                lineno + 1,
                index
            ));
        }
        out.push(Instance { index, optimal, state });
    }
    Ok(out)
}

fn load_pdbs(dir: &Path) -> Result<(PatternDb, PatternDb), String> {
    let p7 = dir.join(P7_PATH);
    let p8 = dir.join(P8_PATH);
    let p7_db = PatternDb::load_mmap(&p7).map_err(|e| format!("loading {}: {}", p7.display(), e))?;
    let p8_db = PatternDb::load_mmap(&p8).map_err(|e| format!("loading {}: {}", p8.display(), e))?;
    Ok((p7_db, p8_db))
}

/// Solve every instance with `solve`, measuring nodes and wall-clock. Verifies
/// each solution reaches `GOAL` and matches the published optimal. Returns the
/// per-instance rows and the list of optimality mismatches `(index, found, want)`.
///
/// `solve` is a closure (rather than a `&Heuristic`) so callers can route
/// through either [`idastar_with_stats`] (scratch heuristics) or
/// [`idastar_inc_with_stats`] (the incremental Korf evaluator).
fn run<F>(solve: F, instances: &[Instance], quiet: bool) -> (Vec<Row>, Vec<(u32, u8, u8)>)
where
    F: Fn(&State) -> (Option<Vec<Move>>, SearchStats),
{
    let mut rows = Vec::with_capacity(instances.len());
    let mut mismatches = Vec::new();

    if !quiet {
        println!(
            "{:>4}  {:>3}  {:>5}  {:>15}  {:>5}  {:>10}",
            "inst", "opt", "found", "nodes", "iters", "time"
        );
    }

    for inst in instances {
        let t = Instant::now();
        let (sol, stats) = solve(&inst.state);
        let elapsed = t.elapsed();
        let sol = sol.expect("Korf instances are solvable");

        // Replay to confirm validity.
        let mut cur = inst.state;
        for m in &sol {
            cur = cur.apply(*m);
        }
        assert_eq!(cur, GOAL, "instance {}: solution does not reach GOAL", inst.index);

        let found = sol.len() as u8;
        if found != inst.optimal {
            mismatches.push((inst.index, found, inst.optimal));
        }

        if !quiet {
            let flag = if found == inst.optimal { "" } else { "  <-- MISMATCH" };
            println!(
                "{:>4}  {:>3}  {:>5}  {:>15}  {:>5}  {:>10}{}",
                inst.index,
                inst.optimal,
                found,
                stats.nodes,
                stats.iterations,
                format!("{:.2?}", elapsed),
                flag
            );
        }

        rows.push(Row {
            index: inst.index,
            optimal: inst.optimal,
            nodes: stats.nodes,
            iterations: stats.iterations,
            elapsed,
        });
    }

    (rows, mismatches)
}

fn print_summary(name: &str, rows: &[Row], mismatches: &[(u32, u8, u8)]) {
    let n = rows.len() as u64;
    if n == 0 {
        println!("\nNo instances solved.");
        return;
    }
    let total_nodes: u64 = rows.iter().map(|r| r.nodes).sum();
    let total_iters: u64 = rows.iter().map(|r| r.iterations as u64).sum();
    let total_time: Duration = rows.iter().map(|r| r.elapsed).sum();
    let total_optimal: u64 = rows.iter().map(|r| r.optimal as u64).sum();
    let secs = total_time.as_secs_f64();
    let nps = if secs > 0.0 { total_nodes as f64 / secs } else { 0.0 };

    let slowest = rows.iter().max_by_key(|r| r.elapsed).unwrap();
    let heaviest = rows.iter().max_by_key(|r| r.nodes).unwrap();

    println!("\n=== Korf-100 summary ({}, n={}) ===", name, n);
    println!("total optimal moves : {}  (mean {:.2}; Korf 1985 ~52-53)",
        total_optimal, total_optimal as f64 / n as f64);
    println!("total nodes         : {}", total_nodes);
    println!("mean nodes/instance : {}", total_nodes / n);
    println!("mean iterations     : {:.1}", total_iters as f64 / n as f64);
    println!("total wall-clock    : {:.2?}", total_time);
    println!("mean wall-clock     : {:.2?}", total_time / n as u32);
    println!("throughput          : {:.3} Mnodes/s", nps / 1.0e6);
    println!("heaviest (nodes)    : #{} -> {} nodes, {:.2?}",
        heaviest.index, heaviest.nodes, heaviest.elapsed);
    println!("slowest (time)      : #{} -> {:.2?}, {} nodes",
        slowest.index, slowest.elapsed, slowest.nodes);

    if mismatches.is_empty() {
        println!("optimality          : OK — all {} match Korf 1985 Table 1", n);
    } else {
        println!("optimality          : {} MISMATCH(es):", mismatches.len());
        for (idx, found, want) in mismatches {
            println!("    instance #{}: solver {} vs Korf {}", idx, found, want);
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e == "help" {
                eprintln!("usage: korf100_bench [--pdb-dir DIR] [--instances FILE] \
                           [--heuristic korf|korf-plus|additive|manhattan] [--limit N] [--quiet] [--scratch]");
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let mut instances = match parse_instances(&args.instances) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Some(limit) = args.limit {
        instances.truncate(limit);
    }
    println!("Loaded {} instances from {}", instances.len(), args.instances.display());

    let (rows, mismatches, name) = if args.heuristic == HeuristicChoice::Manhattan {
        let (rows, mm) = run(|s| idastar_with_stats(s, &ManhattanHeuristic), &instances, args.quiet);
        (rows, mm, "manhattan".to_string())
    } else {
        let (p7_db, p8_db) = match load_pdbs(&args.pdb_dir) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("error: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let dbs = [p7_db, p8_db];
        let h_add = AdditivePdbHeuristic::new(&dbs);
        match args.heuristic {
            HeuristicChoice::Additive => {
                let (rows, mm) = run(|s| idastar_with_stats(s, &h_add), &instances, args.quiet);
                (rows, mm, "additive 7-8".to_string())
            }
            HeuristicChoice::Korf if !args.scratch => {
                // Default: incremental evaluator (OPTIMIZATION.md #1).
                let inc = KorfPdbInc::new([&dbs[0], &dbs[1]]);
                let (rows, mm) = run(|s| idastar_inc_with_stats(s, &inc), &instances, args.quiet);
                (rows, mm, "korf incremental".to_string())
            }
            HeuristicChoice::Korf => {
                // --scratch: from-scratch combinator path, for A/B comparison.
                let h_refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs));
                let h_korf = MaxHeuristic::new(&h_add as &dyn Heuristic, &h_refl as &dyn Heuristic);
                let (rows, mm) = run(|s| idastar_with_stats(s, &h_korf), &instances, args.quiet);
                (rows, mm, "korf scratch max(additive, reflected)".to_string())
            }
            HeuristicChoice::KorfPlus => {
                let h_refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&dbs));
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
                let (rows, mm) = run(|s| idastar_with_stats(s, &h_plus), &instances, args.quiet);
                (rows, mm, "korf-plus".to_string())
            }
            HeuristicChoice::Manhattan => unreachable!(),
        }
    };

    print_summary(&name, &rows, &mismatches);

    if mismatches.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
