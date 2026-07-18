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
//!   four canonical 6-6-6-6 additive PDBs (`pdb24_*.bin`).
//! - `zpdb`      : same composition on **zero-aware 1-bit PDBs** (`*.zbin`).
//!   Strictly dominates `korf`. Default.
//! - `zpdb-plus` : `max(zpdb, refl-zpdb, MD+LinearConflict, WalkingDistance)` —
//!   the strongest admissible heuristic here, all advanced incrementally per node.
//! - `lc` / `wd` : standalone Linear-Conflict / Walking-Distance (WD is the
//!   strongest term on deep boards like the 180° rotation `R`).
//! - `cwd-zpdb-lazy`  : `max(cWD, k6-zpdb)` with the zPDB advanced only where cWD
//!   fails to prune (`LazyMaxInc`). The settled deep-board combiner.
//! - `cwd-zpdb8-lazy` : three-tier lazy cascade `max(cWD, k6-zpdb, k8-zpdb)` —
//!   `LazyMaxInc(LazyMaxInc(cWD, k6), k8)`. cWD every node, k6 only where cWD fails
//!   to prune, the 30.5 GiB three-group 8-tile ZPDB (`pdb24_k8_{a,b,c}.zbin`) only
//!   where `max(cWD,k6)` still fails. Node-identical to the eager three-way max;
//!   measured −43% nodes @thr 142 / −52% @144 on c1 over cWD+k6 (SUGGESTIONS_R §2).
//! - `select`    : per-board 3-way auto-pick — cheap `max(LC,WD)` / pure `zpdb` /
//!   `zpdb-plus` from the root heuristics (Phase 1C policy: WD on deep boards,
//!   zpdb on general). `--combine-slack` tunes it.
//!
//! PDB set (for `zpdb`/`zpdb-plus`):
//! - `k6` : the 6-6-6-6 partition `pdb24_{a,b,c,d}.zbin` (default).
//! - `k7` : the 7-7-7-3 partition `pdb24_k7_{a,b,c,d}.zbin` (larger, tighter).
//!
//! Modes:
//! - default        : optimal solve (first solution found is optimal).
//! - `--max-bound T` : bounded / lower-bound search capped at threshold `T`.
//!   Prints `Lower bound: depth >= K` if it exhausts `T`, or the optimal solution
//!   if found at/below `T`.
//! - `--prove-at-least T` : sugar for `--max-bound (T-1)` — succeeds in proving
//!   `depth >= T` iff the search exhausts every threshold < T.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::pdb::{KorfPdbInc, PatternDb, ZPatternDb, ZpdbInc, ZpdbPlusInc};
use puzzle8::puzzle24::search::{
    Cwd, IncHeuristic, IncHeuristicMut, IncManhattan, LadderOutcome, LazyMaxInc, LinearConflictInc,
    MaxInc, MoveDfa, Search, SearchStats, WalkingDistanceHeuristic, WalkingDistanceInc,
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
/// The three Candidate-1 8-tile ZPDB groups (SUGGESTIONS_R §2); the second,
/// finest tier of the `cwd-zpdb8-lazy` cascade. Fixed partition (not a
/// `--pdb-set` choice): A=2×4 top-left, B=corner-L (owns the blank corner),
/// C=bottom-left. 10.17 GiB each (30.5 GiB total), mmap'd.
const ZPDB_FILES_K8: [&str; 3] = ["pdb24_k8_a.zbin", "pdb24_k8_b.zbin", "pdb24_k8_c.zbin"];

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
    /// Per-node `max(cWD, zpdb)` — the always-on combiner. cWD's escape-constrained
    /// WD maxed against the zero-aware PDB Korf-max on every node. Retains cWD's
    /// neighbor-prune (forwarded through `MaxInc::child_h_lb`), so it runs at full
    /// parity with plain `cwd` plus the zPDB term.
    CwdZpdb,
    /// [`CwdZpdb`](Self::CwdZpdb) with lazy zPDB evaluation (`LazyMaxInc`): the
    /// zPDB is advanced only on children cWD fails to prune, skipping the
    /// dominant per-node probe cost on the (frontier-heavy) cWD-pruned children.
    /// Node-identical to `cwd-zpdb`; per-node cost only.
    CwdZpdbLazy,
    /// Three-tier lazy cascade `max(cWD, k6-zpdb, k8-zpdb)` via nested
    /// `LazyMaxInc(LazyMaxInc(cWD, k6), k8)`: cWD every node, the k6 zPDB probed
    /// only on children cWD fails to prune, and the three-group 8-tile ZPDB
    /// (`pdb24_k8_{a,b,c}.zbin`, SUGGESTIONS_R §2) probed only when `max(cWD,k6)`
    /// still fails — minimizing the expensive 30.5 GiB k8 mmap probes. The middle
    /// tier honors `--pdb-set` (default k6). Node-identical to the eager three-way
    /// max; per-node/probe cost only. The measured k8 pruning win on c1 is
    /// −43% nodes @thr 142, −52% @144 over the cWD+k6 baseline.
    CwdZpdb8Lazy,
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
    /// heuristic. Applies to both the parallel and sequential drivers; sound for
    /// lower-bound proofs. **Default on**; disable with `--no-move-dfa`.
    move_dfa: bool,
    /// For `--heuristic cwd`: let the search bound a child from its parent's
    /// cached neighbor-WD and prune it before probing. Sound. **Default on**;
    /// disable with `--no-cwd-neighbor-prune`.
    cwd_neighbor_prune: bool,
    /// For `--heuristic select`: pick zpdb-plus (over pure zpdb) when the classical
    /// terms are within this many of the PDB root. `0` = never combine (default).
    combine_slack: u8,
    /// Root-orbit-split: on a σ-symmetric start board (`reflect(start) == start`,
    /// e.g. board R), search one representative per σ-orbit of the root's children
    /// — a sound ~2× reduction for lower-bound proofs. Applies to both the parallel
    /// and sequential drivers. `None` = auto (on when the board is symmetric);
    /// `Some(true)` = force on (warns + ignored if the board is not symmetric);
    /// `Some(false)` = off (`--no-root-orbit-split`, for A/B measurement).
    root_orbit_split: Option<bool>,
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
        "usage: {prog} --pdb-dir DIR [--position \"...\"] [--from FILE]\n         \
         [--heuristic manhattan|lc|wd|cwd|korf|zpdb|zpdb-plus|cwd-zpdb|cwd-zpdb-lazy|cwd-zpdb8-lazy|select] (default: cwd) [--pdb-set k6|k7]\n         \
         [--max-bound T | --prove-at-least T] [--parallel]\n         \
         [--no-move-dfa] [--no-cwd-neighbor-prune]  (both default ON) [--combine-slack S]\n         \
         [--no-root-orbit-split]  (auto-on for σ-symmetric boards)"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = None;
    let mut position = None;
    let mut from = None;
    let mut heuristic = HeuristicChoice::Cwd;
    let mut pdb_set = PdbSet::K6;
    let mut max_bound = None;
    let mut parallel = false;
    let mut move_dfa = true;
    let mut cwd_neighbor_prune = true;
    let mut combine_slack = 0u8;
    let mut root_orbit_split: Option<bool> = None;

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
                    "cwd-zpdb" => HeuristicChoice::CwdZpdb,
                    "cwd-zpdb-lazy" => HeuristicChoice::CwdZpdbLazy,
                    "cwd-zpdb8-lazy" => HeuristicChoice::CwdZpdb8Lazy,
                    "select" => HeuristicChoice::Select,
                    other => return Err(format!("unknown heuristic {other:?}")),
                };
            }
            "--pdb-set" => {
                i += 1;
                pdb_set = match argv.get(i).ok_or("--pdb-set needs a value")?.as_str() {
                    "k6" => PdbSet::K6,
                    "k7" => PdbSet::K7,
                    other => return Err(format!("unknown pdb-set {other:?}")),
                };
            }
            "--max-bound" => {
                i += 1;
                let t: u8 = argv
                    .get(i)
                    .ok_or("--max-bound needs a value")?
                    .parse()
                    .map_err(|e| format!("--max-bound: {e}"))?;
                max_bound = Some(t);
            }
            "--prove-at-least" => {
                i += 1;
                let t: u8 = argv
                    .get(i)
                    .ok_or("--prove-at-least needs a value")?
                    .parse()
                    .map_err(|e| format!("--prove-at-least: {e}"))?;
                if t == 0 {
                    return Err("--prove-at-least must be >= 1".into());
                }
                // Exhausting every threshold < T proves depth >= T.
                max_bound = Some(t - 1);
            }
            "--parallel" => parallel = true,
            // move-DFA and cWD neighbor-prune default ON; the positive flags are
            // kept as accepted no-ops for back-compat, with `--no-*` to disable.
            "--move-dfa" => move_dfa = true,
            "--no-move-dfa" => move_dfa = false,
            "--cwd-neighbor-prune" => cwd_neighbor_prune = true,
            "--no-cwd-neighbor-prune" => cwd_neighbor_prune = false,
            "--root-orbit-split" => root_orbit_split = Some(true),
            "--no-root-orbit-split" => root_orbit_split = Some(false),
            "--combine-slack" => {
                i += 1;
                combine_slack = argv
                    .get(i)
                    .ok_or("--combine-slack needs a value")?
                    .parse()
                    .map_err(|e| format!("--combine-slack: {e}"))?;
            }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }
    Ok(Args {
        pdb_dir,
        position,
        from,
        heuristic,
        pdb_set,
        max_bound,
        parallel,
        move_dfa,
        cwd_neighbor_prune,
        combine_slack,
        root_orbit_split,
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
            tok.parse::<u8>().map_err(|e| format!("token {tok:?}: {e}"))?
        };
        if v > 24 {
            return Err(format!("value {v} out of range 0..=24"));
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
            return Err(format!("value {v} appears more than once"));
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

/// Load the three Candidate-1 8-tile ZPDB groups ([`ZPDB_FILES_K8`]).
fn load_zpdbs_k8(dir: &Path) -> Result<Vec<ZPatternDb>, String> {
    let mut dbs = Vec::with_capacity(3);
    for name in ZPDB_FILES_K8 {
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
    println!("Wall-clock     : {elapsed:.2?}");

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

/// Wall-clock timestamp `HH:MM:SS.mmm UTC` for log lines. Dependency-free
/// (formats `SystemTime` since the epoch; no calendar, so UTC time-of-day only).
fn now_hms() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    format!(
        "{:02}:{:02}:{:02}.{:03} UTC",
        (secs / 3600) % 24,
        (secs / 60) % 60,
        secs % 60,
        d.subsec_millis()
    )
}

/// Run the search `f`, bracketing it with `search: start`/`search: end`
/// timestamp logs on stderr, and return its result alongside the search-only
/// duration (excludes table load / DFA build — those are `setup`). This makes
/// the time actually spent in search explicit instead of buried in wall-clock.
fn timed_search<R>(setup: std::time::Duration, f: impl FnOnce() -> R) -> (R, std::time::Duration) {
    eprintln!("[{}] search: start  (setup {:.2?})", now_hms(), setup);
    let t = Instant::now();
    let r = f();
    let dt = t.elapsed();
    eprintln!("[{}] search: end    ({:.2?} in search)", now_hms(), dt);
    (r, dt)
}

/// `search` is the search-only duration (from [`timed_search`]); throughput is
/// reported against it, not total wall-clock, so table load never dilutes it.
fn print_stats(st: &SearchStats, search: std::time::Duration) {
    println!("Nodes          : {}", st.nodes);
    println!("Iterations     : {}", st.iterations);
    println!("Search time    : {search:.2?}");
    let secs = search.as_secs_f64();
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
    orbit_split: bool,
    t0: Instant,
) -> ExitCode
where
    <E as IncHeuristic>::Ctx: Send + Sync,
{
    // Build the move-pruning DFA (Taylor–Korf duplicate elimination) when
    // requested — sound for lower-bound proofs; build() self-verifies.
    let dfa = if move_dfa { Some(MoveDfa::build_default()) } else { None };
    if let Some(dfa) = &dfa {
        eprintln!(
            "move-pruning DFA: {} states, {} KiB (L2-resident)",
            dfa.states(),
            dfa.table_bytes() / 1024
        );
    }
    // One declarative search; the (parallel, pruner) runtime flags pick the arm
    // (each is a distinct type-state, all returning the same outcome + stats).
    let ((outcome, st), search_dt) = timed_search(t0.elapsed(), || {
        let s = Search::new(start, e).maybe_bound(max_bound).orbit_split(orbit_split);
        match (parallel, &dfa) {
            (true, Some(dfa)) => s.pruner(dfa).parallel().run(),
            (true, None) => s.parallel().run(),
            (false, Some(dfa)) => s.pruner(dfa).run(),
            (false, None) => s.run(),
        }
    });
    let elapsed = t0.elapsed();
    match outcome {
        LadderOutcome::Solved(s) => {
            if let Some(mb) = max_bound {
                println!("Found within bound {mb} (optimal):");
            }
            print_solution(start, &s, elapsed);
            print_stats(&st, search_dt);
            ExitCode::SUCCESS
        }
        LadderOutcome::ProvedAtLeast(k) => {
            println!("Lower bound: depth >= {k}");
            println!("Wall-clock     : {elapsed:.2?}");
            print_stats(&st, search_dt);
            ExitCode::SUCCESS
        }
        LadderOutcome::TimedOut(k) => {
            // No deadline is passed here, so this is not expected.
            println!("Lower bound: depth >= {k} (timed out)");
            print_stats(&st, search_dt);
            ExitCode::SUCCESS
        }
        LadderOutcome::Unsolvable => {
            eprintln!("position is unsolvable");
            ExitCode::FAILURE
        }
    }
}

fn require_dir(args: &Args, label: &str) -> Result<PathBuf, ExitCode> {
    match &args.pdb_dir {
        Some(d) => Ok(d.clone()),
        None => {
            eprintln!("error: --pdb-dir required for {label} heuristic");
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
            eprintln!("error: {e}");
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
            eprintln!("error: position parse: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !start.is_solvable() {
        eprintln!("error: position is not solvable");
        return ExitCode::FAILURE;
    }

    // Root-orbit-split: auto-on when the start board is σ-symmetric. Sound only on
    // a σ-fixed board, so we gate on the symmetry test and never apply it otherwise
    // (a forced `--root-orbit-split` on an asymmetric board is refused with a
    // warning). Both the parallel driver (split filter) and the sequential mut
    // drivers (root `g == 0` filter) honor it, node-identically. See
    // puzzle24::symmetry.
    let symmetric = puzzle8::puzzle24::symmetry::is_symmetric(&start);
    let wants_orbit = args.root_orbit_split.unwrap_or(true);
    let orbit = wants_orbit && symmetric;
    if args.root_orbit_split == Some(true) && !symmetric {
        eprintln!("warning: --root-orbit-split ignored: board is not σ-symmetric (reflect(start) != start)");
    } else if orbit {
        eprintln!(
            "root-orbit-split: board is σ-symmetric; searching one representative per σ-orbit of the root's children ({} driver)",
            if args.parallel { "parallel" } else { "sequential" }
        );
    }

    let t0 = Instant::now();
    match args.heuristic {
        HeuristicChoice::Manhattan => run_inc(&start, &IncManhattan, args.max_bound, args.parallel, args.move_dfa, orbit, t0),
        HeuristicChoice::Lc => run_inc(&start, &LinearConflictInc, args.max_bound, args.parallel, args.move_dfa, orbit, t0),
        HeuristicChoice::Wd => {
            WalkingDistanceHeuristic::warm_up_verbose();
            run_inc(&start, &WalkingDistanceInc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::Cwd => {
            eprintln!("cWD: loading tables…");
            let cwd = Cwd::new().with_neighbor_prune(args.cwd_neighbor_prune);
            eprintln!(
                "cWD ready: {} path{}",
                if cwd.has_overlay() { "fast single-line-max table" } else { "reference per-node A*" },
                if args.cwd_neighbor_prune { ", neighbor-prune ON" } else { "" }
            );
            run_inc(&start, &cwd, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::Korf => {
            let dir = match require_dir(&args, "korf") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_pdbs(&dir) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let inc = KorfPdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::Zpdb => {
            let dir = match require_dir(&args, "zpdb") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::ZpdbPlus => {
            let dir = match require_dir(&args, "zpdb-plus") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            // Walking Distance needs its (heavy) table built before the search.
            WalkingDistanceHeuristic::warm_up_verbose();
            let inc = ZpdbPlusInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::CwdZpdb => {
            let dir = match require_dir(&args, "cwd-zpdb") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!("cWD+zpdb: loading cWD tables…");
            let cwd = Cwd::new().with_neighbor_prune(args.cwd_neighbor_prune);
            eprintln!(
                "cWD+zpdb ready: cWD {} path{} maxed per-node with zpdb (neighbor-prune forwarded)",
                if cwd.has_overlay() { "fast single-line-max table" } else { "reference per-node A*" },
                if args.cwd_neighbor_prune { ", neighbor-prune ON" } else { "" },
            );
            let zpdb = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            let inc = MaxInc::new(cwd, zpdb);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::CwdZpdbLazy => {
            let dir = match require_dir(&args, "cwd-zpdb-lazy") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!("cWD+zpdb (lazy): loading cWD tables…");
            let cwd = Cwd::new().with_neighbor_prune(args.cwd_neighbor_prune);
            eprintln!(
                "cWD+zpdb (lazy) ready: zpdb advanced only on children cWD fails to prune{}",
                if args.cwd_neighbor_prune { "; neighbor-prune ON" } else { "" },
            );
            let zpdb = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            let inc = LazyMaxInc::new(cwd, zpdb);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::CwdZpdb8Lazy => {
            let dir = match require_dir(&args, "cwd-zpdb8-lazy") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let k6 = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let k8 = match load_zpdbs_k8(&dir) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!("cWD+k6+k8 (lazy cascade): loading cWD tables…");
            let cwd = Cwd::new().with_neighbor_prune(args.cwd_neighbor_prune);
            eprintln!(
                "cWD+k6+k8 (lazy cascade) ready: k6 advanced only where cWD fails to prune, \
                 k8 (30.5 GiB) only where max(cWD,k6) fails{}",
                if args.cwd_neighbor_prune { "; neighbor-prune ON" } else { "" },
            );
            let z6 = ZpdbInc::new([&k6[0], &k6[1], &k6[2], &k6[3]]);
            let z8 = ZpdbInc::new([&k8[0], &k8[1], &k8[2]]);
            // Nested lazy: cWD (cheap, every node) → k6 (mid) → k8 (finest).
            let inc = LazyMaxInc::new(LazyMaxInc::new(cwd, z6), z8);
            run_inc(&start, &inc, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
        }
        HeuristicChoice::Select => {
            let dir = match require_dir(&args, "select") {
                Ok(d) => d,
                Err(c) => return c,
            };
            let dbs = match load_zpdbs(&dir, args.pdb_set) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
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
                    println!("Selected       : max(LC,WD)  (cheap_h {cheap_root} >= zpdb_h {zpdb_root})");
                    run_inc(&start, &cheap, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
                }
                Pick::Zpdb => {
                    println!("Selected       : zpdb  (zpdb_h {zpdb_root} > cheap_h {cheap_root})");
                    run_inc(&start, &zpdb, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
                }
                Pick::ZpdbPlus => {
                    println!("Selected       : zpdb-plus  (classical terms within slack of zpdb_h {zpdb_root})");
                    let zplus = ZpdbPlusInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
                    run_inc(&start, &zplus, args.max_bound, args.parallel, args.move_dfa, orbit, t0)
                }
            }
        }
    }
}
