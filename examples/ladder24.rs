//! `ladder24` — Phase 1B calibration harness for the 24-puzzle (PUZZLE24.md).
//!
//! Measures the feasibility frontier of the IDA\* solver on the 24-puzzle in two
//! regimes, modeled on `examples/korf100_bench.rs`:
//!
//! 1. **Optimal-solve cost vs depth** — random walks from GOAL of increasing
//!    length, solved optimally with a per-board wall-clock budget. Answers "to
//!    what depth can we optimally solve within T?" (random instances saturate
//!    ~100–114, Korf & Taylor 1996).
//! 2. **Bounded lower-bound cost vs threshold** — structured deep boards (the
//!    180° rotation `R`, its reflection, and any `--from` boards) run through the
//!    bounded driver up to `--max-bound`, with a per-board budget. Answers "how
//!    high a threshold can we exhaust within T?" — the realistic frontier for the
//!    deepest boards (this is how Rokicki proved `R ≥ 152`).
//!
//! Both regimes use the unified deadline-aware [`idastar_inc_ladder`] driver.
//!
//! Run:
//! ```text
//! cargo run --release --features mmap --example ladder24 -- \
//!     --pdb-dir data --heuristics manhattan,zpdb-k6,zpdb-plus-k7 \
//!     --mode both --walk-lengths 20,40,60,80,100,120 --reps 2 \
//!     --budget-secs 60 --bound-budget-secs 300 --max-bound 140
//! ```
//!
//! Heuristics: `manhattan`, `lc`, `wd`, `zpdb-k6`, `zpdb-k7`, `zpdb-plus-k6`,
//! `zpdb-plus-k7`, and the per-board 3-way selectors `select-k6`/`select-k7`
//! (auto-pick cheap `max(LC,WD)` / pure `zpdb` / `zpdb-plus`; `--combine-slack`
//! tunes when zpdb-plus is chosen). `--top-n N` cheaply screens all candidates by
//! `max(LC,WD)` root h and runs only the N hardest. Modes: `optimal`, `bounded`,
//! `both`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use puzzle8::puzzle24::pdb::{ZPatternDb, ZpdbInc, ZpdbPlusInc};
use puzzle8::puzzle24::search::{
    idastar_inc_ladder, IncHeuristic, IncManhattan, LadderOutcome, LinearConflictInc, MaxInc,
    ManhattanHeuristic, SearchStats, WalkingDistanceHeuristic, WalkingDistanceInc,
};
use puzzle8::puzzle24::search::Heuristic;
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};
use puzzle8::puzzle24::symmetry::reflect;

const ZPDB_FILES_K6: [&str; 4] = ["pdb24_a.zbin", "pdb24_b.zbin", "pdb24_c.zbin", "pdb24_d.zbin"];
const ZPDB_FILES_K7: [&str; 4] = [
    "pdb24_k7_a.zbin",
    "pdb24_k7_b.zbin",
    "pdb24_k7_c.zbin",
    "pdb24_k7_d.zbin",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Heur {
    Manhattan,
    Lc,
    Wd,
    ZpdbK6,
    ZpdbK7,
    ZpdbPlusK6,
    ZpdbPlusK7,
    /// Per-board 3-way selector over the k6 PDBs: automatically pick cheap
    /// `max(LC,WD)`, pure `zpdb`, or `zpdb-plus` per [`pick_heuristic`] (tuned by
    /// `--combine-slack`).
    SelectK6,
    /// Same 3-way selector with the k7 PDBs.
    SelectK7,
}

impl Heur {
    fn name(self) -> &'static str {
        match self {
            Heur::Manhattan => "manhattan",
            Heur::Lc => "lc",
            Heur::Wd => "wd",
            Heur::ZpdbK6 => "zpdb-k6",
            Heur::ZpdbK7 => "zpdb-k7",
            Heur::ZpdbPlusK6 => "zpdb-plus-k6",
            Heur::ZpdbPlusK7 => "zpdb-plus-k7",
            Heur::SelectK6 => "select-k6",
            Heur::SelectK7 => "select-k7",
        }
    }
    fn parse(s: &str) -> Result<Heur, String> {
        Ok(match s {
            "manhattan" => Heur::Manhattan,
            "lc" => Heur::Lc,
            "wd" => Heur::Wd,
            "zpdb-k6" => Heur::ZpdbK6,
            "zpdb-k7" => Heur::ZpdbK7,
            "zpdb-plus-k6" => Heur::ZpdbPlusK6,
            "zpdb-plus-k7" => Heur::ZpdbPlusK7,
            "select-k6" => Heur::SelectK6,
            "select-k7" => Heur::SelectK7,
            other => return Err(format!("unknown heuristic {:?}", other)),
        })
    }
    fn needs_k6(self) -> bool {
        matches!(self, Heur::ZpdbK6 | Heur::ZpdbPlusK6 | Heur::SelectK6)
    }
    fn needs_k7(self) -> bool {
        matches!(self, Heur::ZpdbK7 | Heur::ZpdbPlusK7 | Heur::SelectK7)
    }
    fn needs_wd(self) -> bool {
        matches!(
            self,
            Heur::Wd | Heur::ZpdbPlusK6 | Heur::ZpdbPlusK7 | Heur::SelectK6 | Heur::SelectK7
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Optimal,
    Bounded,
    Both,
}

struct Args {
    pdb_dir: PathBuf,
    heuristics: Vec<Heur>,
    mode: Mode,
    walk_lengths: Vec<u16>,
    reps: u32,
    seed: u64,
    budget_secs: u64,
    bound_budget_secs: u64,
    max_bound: u8,
    from: Option<PathBuf>,
    /// Catalog triage (#2): cheaply score *all* boards by max(LC,WD) root h, then
    /// run the (selective) solve only on the `top_n` hardest. `None` = run all.
    top_n: Option<usize>,
    /// 3-way selector slack: choose zpdb-plus (over pure zpdb) when the classical
    /// terms are within this many of the PDB root. `0` = never combine (default).
    combine_slack: u8,
    quiet: bool,
}

/// One measured (board × heuristic) result.
struct Row {
    label: String,
    root_h: u8,
    nodes: u64,
    iters: u32,
    elapsed: Duration,
    /// `Some(depth)` if optimally solved, else `None`.
    solved_depth: Option<u8>,
    /// Proven lower bound (`= depth` if solved).
    lower_bound: u8,
    outcome: String,
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = PathBuf::from("data");
    let mut heuristics = vec![Heur::Manhattan, Heur::ZpdbK6];
    let mut mode = Mode::Both;
    let mut walk_lengths = vec![20u16, 40, 60, 80, 100, 120];
    let mut reps = 2u32;
    let mut seed = 0x24_5151_C0FFEE_24u64;
    let mut budget_secs = 60u64;
    let mut bound_budget_secs = 300u64;
    let mut max_bound = 140u8;
    let mut from = None;
    let mut top_n: Option<usize> = None;
    let mut combine_slack = 0u8;
    let mut quiet = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => {
                i += 1;
                pdb_dir = PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?);
            }
            "--heuristics" => {
                i += 1;
                let s = argv.get(i).ok_or("--heuristics needs a value")?;
                heuristics = s.split(',').map(Heur::parse).collect::<Result<_, _>>()?;
            }
            "--mode" => {
                i += 1;
                mode = match argv.get(i).ok_or("--mode needs a value")?.as_str() {
                    "optimal" => Mode::Optimal,
                    "bounded" => Mode::Bounded,
                    "both" => Mode::Both,
                    other => return Err(format!("unknown mode {:?}", other)),
                };
            }
            "--walk-lengths" => {
                i += 1;
                let s = argv.get(i).ok_or("--walk-lengths needs a value")?;
                walk_lengths = s
                    .split(',')
                    .map(|t| t.trim().parse::<u16>().map_err(|e| format!("walk len {:?}: {}", t, e)))
                    .collect::<Result<_, _>>()?;
            }
            "--reps" => {
                i += 1;
                reps = argv.get(i).ok_or("--reps needs a value")?.parse().map_err(|e| format!("--reps: {}", e))?;
            }
            "--seed" => {
                i += 1;
                seed = argv.get(i).ok_or("--seed needs a value")?.parse().map_err(|e| format!("--seed: {}", e))?;
            }
            "--budget-secs" => {
                i += 1;
                budget_secs = argv.get(i).ok_or("--budget-secs needs a value")?.parse().map_err(|e| format!("--budget-secs: {}", e))?;
            }
            "--bound-budget-secs" => {
                i += 1;
                bound_budget_secs = argv.get(i).ok_or("--bound-budget-secs needs a value")?.parse().map_err(|e| format!("--bound-budget-secs: {}", e))?;
            }
            "--max-bound" => {
                i += 1;
                max_bound = argv.get(i).ok_or("--max-bound needs a value")?.parse().map_err(|e| format!("--max-bound: {}", e))?;
            }
            "--from" => {
                i += 1;
                from = Some(PathBuf::from(argv.get(i).ok_or("--from needs a value")?));
            }
            "--top-n" => {
                i += 1;
                top_n = Some(argv.get(i).ok_or("--top-n needs a value")?.parse().map_err(|e| format!("--top-n: {}", e))?);
            }
            "--combine-slack" => {
                i += 1;
                combine_slack = argv.get(i).ok_or("--combine-slack needs a value")?.parse().map_err(|e| format!("--combine-slack: {}", e))?;
            }
            "--quiet" => quiet = true,
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    Ok(Args {
        pdb_dir, heuristics, mode, walk_lengths, reps, seed, budget_secs, bound_budget_secs,
        max_bound, from, top_n, combine_slack, quiet,
    })
}

fn load_zpdbs(dir: &Path, files: &[&str; 4]) -> Result<Vec<ZPatternDb>, String> {
    let mut dbs = Vec::with_capacity(4);
    for name in files {
        let path = dir.join(name);
        let db = ZPatternDb::load_mmap(&path).map_err(|e| format!("loading {}: {}", path.display(), e))?;
        dbs.push(db);
    }
    Ok(dbs)
}

/// `R` = the 180° board rotation: blank fixed at cell 0, tile `25-i` at cell `i`.
/// The canonical hard board (Rokicki LB 152); 276 inversions → solvable.
fn r_board() -> State {
    let mut c = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        c[i] = (25 - i) as u8;
    }
    State(c)
}

/// A no-immediate-undo random walk of `len` steps from GOAL (xorshift). The
/// result is solvable by construction (reachable from GOAL).
fn random_walk(seed: u64, len: u16) -> State {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = GOAL;
    let mut last: Option<Move> = None;
    for _ in 0..len {
        let opts: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| last.map_or(true, |p: Move| m != p.inverse()))
            .collect();
        let m = opts[(next() as usize) % opts.len()];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

fn parse_board_line(line: &str) -> Option<State> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let mut cells = [0u8; N_CELLS];
    let mut n = 0;
    for tok in line.split_whitespace() {
        if n >= N_CELLS {
            return None;
        }
        let v = if tok == "_" || tok == "." { 0 } else { tok.parse::<u8>().ok()? };
        if v > 24 {
            return None;
        }
        cells[n] = v;
        n += 1;
    }
    if n != N_CELLS {
        return None;
    }
    Some(State(cells))
}

fn root_h<E: IncHeuristic>(e: &E, s: &State) -> u8 {
    let mut st = SearchStats::default();
    e.root(s, &mut st).0
}

/// Run one board through the deadline-aware driver and record a [`Row`].
fn run_board<E: IncHeuristic>(
    e: &E,
    label: String,
    s: &State,
    max_bound: u8,
    budget: Duration,
) -> Row {
    let rh = root_h(e, s);
    let deadline = Instant::now() + budget;
    let t = Instant::now();
    let (outcome, stats) = idastar_inc_ladder(s, e, max_bound, Some(deadline));
    let elapsed = t.elapsed();
    let (solved_depth, lower_bound, outcome_str) = match outcome {
        LadderOutcome::Solved(sol) => {
            // Replay to confirm validity.
            let mut cur = *s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL, "board {}: solution does not reach GOAL", label);
            let d = sol.len() as u8;
            (Some(d), d, format!("solved d={}", d))
        }
        LadderOutcome::ProvedAtLeast(k) => (None, k, format!("LB>={}", k)),
        LadderOutcome::TimedOut(k) => (None, k, format!("timeout LB>={}", k)),
        LadderOutcome::Unsolvable => (None, 0, "unsolvable".to_string()),
    };
    Row { label, root_h: rh, nodes: stats.nodes, iters: stats.iterations, elapsed, solved_depth, lower_bound, outcome: outcome_str }
}

fn print_table(rows: &[Row]) {
    println!(
        "  {:<14} {:>5} {:>16} {:>5} {:>11} {:>13}  {}",
        "board", "h0", "nodes", "iters", "time", "nodes/s", "outcome"
    );
    for r in rows {
        let secs = r.elapsed.as_secs_f64();
        let nps = if secs > 0.0 { r.nodes as f64 / secs } else { 0.0 };
        println!(
            "  {:<14} {:>5} {:>16} {:>5} {:>11} {:>13.0}  {}",
            r.label, r.root_h, r.nodes, r.iters, format!("{:.2?}", r.elapsed), nps, r.outcome
        );
    }
}

/// Run the optimal-solve regime (random walks) for one heuristic.
fn run_optimal<E: IncHeuristic>(hname: &str, e: &E, boards: &[(String, State)], budget: Duration, quiet: bool) {
    println!("\n--- OPTIMAL-SOLVE  [{}]  (per-board budget {:?}) ---", hname, budget);
    let mut rows = Vec::new();
    for (label, s) in boards {
        rows.push(run_board(e, label.clone(), s, u8::MAX, budget));
    }
    if !quiet {
        print_table(&rows);
    }
    let deepest = rows.iter().filter_map(|r| r.solved_depth).max();
    let total_nodes: u64 = rows.iter().map(|r| r.nodes).sum();
    let total_time: f64 = rows.iter().map(|r| r.elapsed.as_secs_f64()).sum();
    let nps = if total_time > 0.0 { total_nodes as f64 / total_time } else { 0.0 };
    match deepest {
        Some(d) => println!("  => deepest optimally solved within budget: d={}  ({:.2} Mnodes/s agg)", d, nps / 1e6),
        None => println!("  => no board solved within budget  ({:.2} Mnodes/s agg)", nps / 1e6),
    }
}

/// Run the bounded lower-bound regime (structured boards) for one heuristic.
fn run_bounded<E: IncHeuristic>(hname: &str, e: &E, boards: &[(String, State)], max_bound: u8, budget: Duration, quiet: bool) {
    println!("\n--- BOUNDED-LB  [{}]  (max-bound {}, per-board budget {:?}) ---", hname, max_bound, budget);
    let mut rows = Vec::new();
    for (label, s) in boards {
        rows.push(run_board(e, label.clone(), s, max_bound, budget));
    }
    if !quiet {
        print_table(&rows);
    }
    if let Some(r) = rows.iter().find(|r| r.label.starts_with('R') && r.label != "reflect(R)") {
        println!("  => best proven lower bound on R: depth >= {}", r.lower_bound);
    }
    let best = rows.iter().map(|r| r.lower_bound).max().unwrap_or(0);
    println!("  => highest lower bound across structured boards: {}", best);
}

/// Which heuristic the 3-way selector chose for a board.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    Cheap,
    Zpdb,
    ZpdbPlus,
}

impl Pick {
    fn tag(self) -> &'static str {
        match self {
            Pick::Cheap => "cheap",
            Pick::Zpdb => "zpdb",
            Pick::ZpdbPlus => "zpdb+",
        }
    }
}

/// The 3-way decision (automatic, per board), from the root heuristics:
/// - `zpdb_root ≤ cheap_root`  → **cheap** (`max(LC,WD)`): the PDB adds nothing
///   at the root (e.g. R, where WD dominates), so take the fastest term.
/// - PDB wins at root, classical terms weak (`cheap_root < zpdb_root − slack`)
///   → **pure zpdb**: same root as zpdb-plus but cheaper per node.
/// - PDB wins at root, classical terms competitive (`cheap_root ≥ zpdb_root −
///   slack`) → **zpdb-plus**: LC/WD are strong enough that their max with the PDB
///   is likely to prune better *below* the root, justifying the extra per-node cost.
///
/// `zpdb-plus` is *root-dominated* (its root is `max(zpdb,cheap)`), so it can only
/// be chosen on the deeper-combination proxy — never by raw root comparison. With
/// `slack = 0` the selector reduces to cheap-vs-pure-zpdb (the regression-free
/// default); raise `slack` to opt into combining when the classical terms are close.
fn pick_heuristic(cheap_root: u8, zpdb_root: u8, slack: u8) -> Pick {
    if zpdb_root <= cheap_root {
        Pick::Cheap
    } else if cheap_root + slack >= zpdb_root {
        Pick::ZpdbPlus
    } else {
        Pick::Zpdb
    }
}

/// Per-board 3-way selector (#1, refined): choose automatically among cheap
/// `max(LC,WD)`, pure `zpdb`, and `zpdb-plus` per [`pick_heuristic`], then run the
/// chosen one. Returns the row and the choice.
#[allow(clippy::too_many_arguments)]
fn run_board_select<C: IncHeuristic, Z: IncHeuristic, X: IncHeuristic>(
    cheap: &C,
    zpdb: &Z,
    zpdb_plus: &X,
    slack: u8,
    label: String,
    s: &State,
    max_bound: u8,
    budget: Duration,
) -> (Row, Pick) {
    let cheap_root = root_h(cheap, s);
    let zpdb_root = root_h(zpdb, s);
    let pick = pick_heuristic(cheap_root, zpdb_root, slack);
    let deadline = Instant::now() + budget;
    let t = Instant::now();
    let (outcome, stats) = match pick {
        Pick::Cheap => idastar_inc_ladder(s, cheap, max_bound, Some(deadline)),
        Pick::Zpdb => idastar_inc_ladder(s, zpdb, max_bound, Some(deadline)),
        Pick::ZpdbPlus => idastar_inc_ladder(s, zpdb_plus, max_bound, Some(deadline)),
    };
    let elapsed = t.elapsed();
    let tag = pick.tag();
    let (solved_depth, lower_bound, outcome_str) = match outcome {
        LadderOutcome::Solved(sol) => {
            let mut cur = *s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL, "board {}: solution does not reach GOAL", label);
            let d = sol.len() as u8;
            (Some(d), d, format!("[{}] solved d={}", tag, d))
        }
        LadderOutcome::ProvedAtLeast(k) => (None, k, format!("[{}] LB>={}", tag, k)),
        LadderOutcome::TimedOut(k) => (None, k, format!("[{}] timeout LB>={}", tag, k)),
        LadderOutcome::Unsolvable => (None, 0, "unsolvable".to_string()),
    };
    let row = Row {
        label,
        root_h: cheap_root.max(zpdb_root),
        nodes: stats.nodes,
        iters: stats.iterations,
        elapsed,
        solved_depth,
        lower_bound,
        outcome: outcome_str,
    };
    (row, pick)
}

fn pick_counts(picks: &[Pick]) -> String {
    let c = picks.iter().filter(|p| **p == Pick::Cheap).count();
    let z = picks.iter().filter(|p| **p == Pick::Zpdb).count();
    let zp = picks.iter().filter(|p| **p == Pick::ZpdbPlus).count();
    format!("picks: cheap={} zpdb={} zpdb+={}", c, z, zp)
}

#[allow(clippy::too_many_arguments)]
fn run_optimal_select<C: IncHeuristic, Z: IncHeuristic, X: IncHeuristic>(
    hname: &str, cheap: &C, zpdb: &Z, zpdb_plus: &X, slack: u8, boards: &[(String, State)], budget: Duration, quiet: bool,
) {
    println!("\n--- OPTIMAL-SOLVE  [{}]  (per-board budget {:?}, combine-slack {}) ---", hname, budget, slack);
    let mut rows = Vec::new();
    let mut picks = Vec::new();
    for (label, s) in boards {
        let (row, pick) = run_board_select(cheap, zpdb, zpdb_plus, slack, label.clone(), s, u8::MAX, budget);
        picks.push(pick);
        rows.push(row);
    }
    if !quiet {
        print_table(&rows);
    }
    let deepest = rows.iter().filter_map(|r| r.solved_depth).max();
    let total_nodes: u64 = rows.iter().map(|r| r.nodes).sum();
    let total_time: f64 = rows.iter().map(|r| r.elapsed.as_secs_f64()).sum();
    let nps = if total_time > 0.0 { total_nodes as f64 / total_time } else { 0.0 };
    match deepest {
        Some(d) => println!("  => deepest optimally solved within budget: d={}  ({:.2} Mnodes/s agg); {}", d, nps / 1e6, pick_counts(&picks)),
        None => println!("  => no board solved within budget; {}", pick_counts(&picks)),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bounded_select<C: IncHeuristic, Z: IncHeuristic, X: IncHeuristic>(
    hname: &str, cheap: &C, zpdb: &Z, zpdb_plus: &X, slack: u8, boards: &[(String, State)], max_bound: u8, budget: Duration, quiet: bool,
) {
    println!("\n--- BOUNDED-LB  [{}]  (max-bound {}, per-board budget {:?}, combine-slack {}) ---", hname, max_bound, budget, slack);
    let mut rows = Vec::new();
    let mut picks = Vec::new();
    for (label, s) in boards {
        let (row, pick) = run_board_select(cheap, zpdb, zpdb_plus, slack, label.clone(), s, max_bound, budget);
        picks.push(pick);
        rows.push(row);
    }
    if !quiet {
        print_table(&rows);
    }
    if let Some(r) = rows.iter().find(|r| r.label.starts_with('R') && r.label != "reflect(R)") {
        println!("  => best proven lower bound on R: depth >= {}", r.lower_bound);
    }
    let best = rows.iter().map(|r| r.lower_bound).max().unwrap_or(0);
    println!("  => highest lower bound across structured boards: {}; {}", best, pick_counts(&picks));
}

/// Cheap catalog score (#2): `max(LinearConflict, WalkingDistance)` root h.
fn cheap_score(s: &State) -> u8 {
    let mut st = SearchStats::default();
    let lc = LinearConflictInc.root(s, &mut st).0;
    let wd = WalkingDistanceInc.root(s, &mut st).0;
    lc.max(wd)
}

/// Catalog triage (#2): score every board cheaply, print the ranking, and keep
/// only the `top_n` hardest for the (expensive) solve.
fn screen_top_n(set_name: &str, boards: Vec<(String, State)>, top_n: usize) -> Vec<(String, State)> {
    let mut scored: Vec<(u8, String, State)> =
        boards.into_iter().map(|(l, s)| (cheap_score(&s), l, s)).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    println!("\n=== SCREEN [{}] — cheap max(LC,WD) root over all {} candidates ===", set_name, scored.len());
    for (h, l, _) in &scored {
        println!("  {:<14} cheap-h={}", l, h);
    }
    let keep = top_n.min(scored.len());
    println!("  => keeping top {} hardest for the (selective) solve", keep);
    scored.into_iter().take(keep).map(|(_, l, s)| (l, s)).collect()
}

/// Build heuristic `h` (borrowing the loaded PDBs) and run the requested modes.
#[allow(clippy::too_many_arguments)]
fn dispatch(
    h: Heur,
    k6: &Option<Vec<ZPatternDb>>,
    k7: &Option<Vec<ZPatternDb>>,
    walks: &[(String, State)],
    structured: &[(String, State)],
    mode: Mode,
    max_bound: u8,
    opt_budget: Duration,
    bnd_budget: Duration,
    combine_slack: u8,
    quiet: bool,
) {
    macro_rules! run_modes {
        ($e:expr) => {{
            let e = $e;
            if matches!(mode, Mode::Optimal | Mode::Both) {
                run_optimal(h.name(), &e, walks, opt_budget, quiet);
            }
            if matches!(mode, Mode::Bounded | Mode::Both) {
                run_bounded(h.name(), &e, structured, max_bound, bnd_budget, quiet);
            }
        }};
    }
    match h {
        Heur::Manhattan => run_modes!(IncManhattan),
        Heur::Lc => run_modes!(LinearConflictInc),
        Heur::Wd => run_modes!(WalkingDistanceInc),
        Heur::ZpdbK6 => {
            let d = k6.as_ref().unwrap();
            run_modes!(ZpdbInc::new([&d[0], &d[1], &d[2], &d[3]]));
        }
        Heur::ZpdbK7 => {
            let d = k7.as_ref().unwrap();
            run_modes!(ZpdbInc::new([&d[0], &d[1], &d[2], &d[3]]));
        }
        Heur::ZpdbPlusK6 => {
            let d = k6.as_ref().unwrap();
            run_modes!(ZpdbPlusInc::new([&d[0], &d[1], &d[2], &d[3]]));
        }
        Heur::ZpdbPlusK7 => {
            let d = k7.as_ref().unwrap();
            run_modes!(ZpdbPlusInc::new([&d[0], &d[1], &d[2], &d[3]]));
        }
        Heur::SelectK6 | Heur::SelectK7 => {
            let d = if matches!(h, Heur::SelectK6) { k6.as_ref().unwrap() } else { k7.as_ref().unwrap() };
            let cheap = MaxInc::new(LinearConflictInc, WalkingDistanceInc);
            let zpdb = ZpdbInc::new([&d[0], &d[1], &d[2], &d[3]]);
            let zpdb_plus = ZpdbPlusInc::new([&d[0], &d[1], &d[2], &d[3]]);
            if matches!(mode, Mode::Optimal | Mode::Both) {
                run_optimal_select(h.name(), &cheap, &zpdb, &zpdb_plus, combine_slack, walks, opt_budget, quiet);
            }
            if matches!(mode, Mode::Bounded | Mode::Both) {
                run_bounded_select(h.name(), &cheap, &zpdb, &zpdb_plus, combine_slack, structured, max_bound, bnd_budget, quiet);
            }
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e == "help" {
                eprintln!(
                    "usage: ladder24 [--pdb-dir DIR] [--heuristics csv] [--mode optimal|bounded|both]\n         \
                     [--walk-lengths csv] [--reps N] [--seed S] [--budget-secs T]\n         \
                     [--bound-budget-secs T] [--max-bound K] [--from FILE] [--top-n N] [--quiet]\n       \
                     heuristics: manhattan,lc,wd,zpdb-k6,zpdb-k7,zpdb-plus-k6,zpdb-plus-k7,select-k6,select-k7"
                );
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Board sets.
    let mut walks: Vec<(String, State)> = Vec::new();
    for &len in &args.walk_lengths {
        for rep in 0..args.reps {
            let seed = args.seed.wrapping_add((len as u64) << 32).wrapping_add(rep as u64 * 0x9E3779B97F4A7C15);
            walks.push((format!("walk{}#{}", len, rep), random_walk(seed, len)));
        }
    }
    let mut structured: Vec<(String, State)> = vec![
        ("R".to_string(), r_board()),
        ("reflect(R)".to_string(), reflect(&r_board())),
    ];
    if let Some(path) = &args.from {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                for (i, line) in text.lines().enumerate() {
                    if let Some(s) = parse_board_line(line) {
                        if s.is_solvable() {
                            structured.push((format!("from#{}", i), s));
                        } else {
                            eprintln!("warning: --from line {} not solvable, skipped", i + 1);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("error reading {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
        }
    }

    // Load only the PDB sets that the requested heuristics need.
    let need_k6 = args.heuristics.iter().any(|h| h.needs_k6());
    let need_k7 = args.heuristics.iter().any(|h| h.needs_k7());
    let need_wd = args.heuristics.iter().any(|h| h.needs_wd());
    let k6 = if need_k6 {
        match load_zpdbs(&args.pdb_dir, &ZPDB_FILES_K6) {
            Ok(d) => Some(d),
            Err(e) => { eprintln!("error: {}", e); return ExitCode::FAILURE; }
        }
    } else { None };
    let k7 = if need_k7 {
        match load_zpdbs(&args.pdb_dir, &ZPDB_FILES_K7) {
            Ok(d) => Some(d),
            Err(e) => { eprintln!("error: {}", e); return ExitCode::FAILURE; }
        }
    } else { None };
    // WD table is needed by any WD/zpdb-plus/select heuristic and by `--top-n`
    // screening (the cheap catalog score is max(LC,WD)).
    if need_wd || args.top_n.is_some() {
        eprintln!("building Walking Distance table (one-off, ~65.65M states)…");
        let t = Instant::now();
        WalkingDistanceHeuristic::warm_up();
        eprintln!("  WD table built in {:?}", t.elapsed());
    }

    // Catalog triage (#2): cheaply score all candidates, keep the top-N hardest.
    if let Some(n) = args.top_n {
        walks = screen_top_n("walks", walks, n);
    }

    println!(
        "ladder24: {} heuristics, {} walk boards, {} structured boards",
        args.heuristics.len(), walks.len(), structured.len()
    );
    // Sanity: root h of R under plain Manhattan (the documented 112 baseline).
    println!("  root h(R) Manhattan = {}", ManhattanHeuristic.h(&r_board()));

    let opt_budget = Duration::from_secs(args.budget_secs);
    let bnd_budget = Duration::from_secs(args.bound_budget_secs);
    for &h in &args.heuristics {
        dispatch(h, &k6, &k7, &walks, &structured, args.mode, args.max_bound, opt_budget, bnd_budget, args.combine_slack, args.quiet);
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_way_pick_logic() {
        // PDB useless at root (R: WD 140 > zpdb 120) → cheap, regardless of slack.
        assert!(matches!(pick_heuristic(140, 120, 0), Pick::Cheap));
        assert!(matches!(pick_heuristic(140, 120, 8), Pick::Cheap));
        // Ties go to cheap (cheapest at equal root).
        assert!(matches!(pick_heuristic(50, 50, 0), Pick::Cheap));
        // PDB wins at root, classical weak → pure zpdb (default slack 0).
        assert!(matches!(pick_heuristic(62, 68, 0), Pick::Zpdb));
        assert!(matches!(pick_heuristic(50, 80, 8), Pick::Zpdb));
        assert!(matches!(pick_heuristic(50, 51, 0), Pick::Zpdb));
        // PDB wins at root, classical competitive (within slack) → zpdb-plus.
        assert!(matches!(pick_heuristic(62, 68, 8), Pick::ZpdbPlus));
        assert!(matches!(pick_heuristic(50, 51, 1), Pick::ZpdbPlus));
    }

    #[test]
    fn r_board_is_solvable_and_180_rotation() {
        let r = r_board();
        assert!(r.is_solvable());
        assert_eq!(r.0[0], 0); // blank fixed
        assert_eq!(r.0[1], 24);
        assert_eq!(r.0[24], 1);
    }
}
