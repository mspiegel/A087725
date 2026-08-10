//! Solve a 24-puzzle position optimally, or prove a lower bound on its optimal
//! depth, with the (iterative, arena-based) search engine over cWD.
//!
//! ```text
//! solve24 --position "<25 tokens>" [--prove-at-least T] [tier flags] [--parallel]
//! ```
//!
//! Position format: 25 whitespace-separated tokens in row-major order, `_`/`.`/`0`
//! for the blank and `1..=24` for tiles. The board must be solvable.
//!
//! The base stack is fixed: **cWD + move-DFA + neighbour-WD child pre-prune +
//! root σ-orbit split**, the configuration the `R` lower-bound program runs.
//! The flags select what runs on top of it:
//!
//! - `--prove-at-least T` caps the search at threshold `T-1`; exhausting it
//!   *proves* `dist ≥ T`. This is a threshold, not a guess: cWD(R) = 144, so on
//!   `R` anything at or below 145 returns immediately having searched nothing.
//!   Omit it to search unbounded until the optimum is found.
//! - The heuristic tiers `--lm` / `--lm2` / `--clm2` (one at most) and
//!   `--zpdb8`, all sound. The production stack for deep `R` bounds is the
//!   cascade `--clm2 --zpdb8` (consult order cWD → cLM2 → k8); per-tier
//!   semantics, artifacts and measured ratios are in `--help`.
//! - `--parallel` runs each threshold on the tree-splitting rayon driver,
//!   node-identical to the sequential engine.
//! - `--no-root-orbit-split` disables the σ-orbit split, which is otherwise
//!   auto-enabled on σ-symmetric boards. That split is *soundness*-critical
//!   rather than merely fast — it discards half the root's children on the
//!   strength of the symmetry argument — so running a proof both ways and
//!   confirming the same bound is a cheap end-to-end check on that reasoning.
//!
//! Tables: the merged cWD artifact and the LM/LM2 artifact default to
//! `data/cwd_mm.bin` and `data/cwd_lm_mm.bin` (`--cwd-mm` / `--cwd-lm-mm`
//! override them, e.g. for dense-repack A/Bs); the clm2 joint table and the
//! three k8 zPDBs are fixed at `data/cwd_lm1l_mm.bin` and
//! `data/pdb24_k8_{a,b,c}.zbin`. Without `data/cwd_mm.bin`, `Cwd::new()` falls
//! back to `data/wd24.bin` + `data/cwd_single.bin`, building via BFS as a last
//! resort.

use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::engine::{bounded_tiers, SearchOpts, Tiers};
use puzzle8::puzzle24::search::move_dfa::MoveDfa;
use puzzle8::puzzle24::search::{BoundedOutcome, SearchStats};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

/// Solve a 24-puzzle position optimally, or prove a lower bound on its optimal
/// depth, with the (iterative, arena-based) search engine over cWD.
#[derive(Parser)]
#[command(name = "solve24")]
struct Args {
    /// Board: 25 whitespace-separated tokens, row-major. Blank is 0, _ or .;
    /// tiles are 1..=24.
    #[arg(long, value_name = "P")]
    position: String,

    /// Cap the search at threshold T-1; exhausting it proves dist >= T. Omit
    /// to solve optimally with no cap. This is a threshold, not a guess:
    /// cWD(R) = 144, so on R anything at or below 145 returns immediately
    /// having searched nothing.
    #[arg(long, value_name = "T", value_parser = clap::value_parser!(u8).range(1..))]
    prove_at_least: Option<u8>,

    /// Disable the σ-orbit split (auto-on when the board is σ-symmetric). Use
    /// to cross-check a proof: the split is soundness-critical rather than
    /// merely fast, so running a proof both ways and confirming the same bound
    /// is a cheap end-to-end check on the symmetry argument.
    #[arg(long)]
    no_root_orbit_split: bool,

    /// Split each threshold into subtrees and search them on rayon workers.
    /// Node counts are identical to the sequential driver. Thread count from
    /// RAYON_NUM_THREADS (default: core count).
    #[arg(long, conflicts_with = "max_nodes")]
    parallel: bool,

    /// Add the Last-Move tier (cwd-lm): max(cWD, last-move branch min) from
    /// data/cwd_lm.bin. Sound; not node-identical to plain cWD (stronger
    /// heuristic). Composes with --zpdb8 (consult order cWD → LM → k8),
    /// though --lm2/--clm2 dominate it there.
    #[arg(long, conflicts_with_all = ["lm2", "clm2"])]
    lm: bool,

    /// Add the last-two-moves tier (cwd-lm2): 4-branch endgame min from
    /// data/cwd_lm.bin + data/cwd_lm2.bin.
    #[arg(long, conflicts_with = "clm2")]
    lm2: bool,

    /// The constrained-LM2 tier: LM2's endgame branches lifted by
    /// single-demanded-line escape constraints (cWD + last-two-moves, jointly
    /// priced), read from data/cwd_lm_mm.bin + data/cwd_lm1l_mm.bin (~13.5 GB
    /// mmap'd). Standalone — use at most one of --lm / --lm2 / --clm2.
    /// Composes with --zpdb8 (consult order cWD → cLM2 → k8): 1.39x fewer
    /// nodes and faster than --lm2 --zpdb8 at 146.
    #[arg(long)]
    clm2: bool,

    /// Add the lazy k8 tier: max(cWD, k8) from the three 8-tile ZPDBs
    /// (data/pdb24_k8_*.zbin, ~30.5 GB mmap'd), consulted only at children
    /// the tiers above fail to prune, through a shared packed-position cache.
    /// With --lm2 this is the cascade: 1.98x fewer nodes than --lm2 alone at
    /// threshold 148, and the cut grows with depth (1.21x @144, 1.48x @146).
    #[arg(long)]
    zpdb8: bool,

    /// Disable the neighbour-WD child pre-prune.
    #[arg(long = "no-cwd-neighbor-prune")]
    no_neighbor_prune: bool,

    /// Merged-table artifact, e.g. a dense repack, for footprint/throughput
    /// A/Bs.
    #[arg(long, value_name = "PATH", default_value = "data/cwd_mm.bin")]
    cwd_mm: String,

    /// LM/LM2 artifact, used by --lm / --lm2 / --clm2.
    #[arg(long, value_name = "PATH", default_value = "data/cwd_lm_mm.bin")]
    cwd_lm_mm: String,

    /// Stop after N nodes ('_' separators allowed). BENCHMARKING ONLY — the
    /// result is not a proof, and is reported as such. Useful because the
    /// engine is node-identical across variants, so a budget cuts every A/B
    /// arm at the same tree position. Not supported with --parallel: the
    /// budget would cut each worker independently, so the truncation point
    /// would not be reproducible.
    #[arg(long, value_name = "N", value_parser = parse_max_nodes)]
    max_nodes: Option<u64>,

    /// Copy each table into anonymous MADV_HUGEPAGE memory instead of mapping
    /// its file, so probes are translated through 2 MiB pages. The tables are
    /// probed randomly across ~49 GB, so at 4 KiB nearly every probe pays a
    /// page walk; huge pages cut the TLB footprint ~512x. Costs real RSS —
    /// the tables become resident anonymous memory rather than evictable page
    /// cache, so this needs a machine with RAM to spare (fine at 500 GB,
    /// fatal at 32). Requires transparent_hugepage/enabled = madvise|always.
    #[arg(long)]
    hugepages: bool,

    /// Checkpoint long proofs into DIR: each rayon worker appends completed
    /// work units to its own file (no cross-thread synchronization) and the
    /// driver records exhausted thresholds. Interrupted? Re-run the same
    /// command with the same DIR — completed thresholds and units restore,
    /// only unfinished work re-searches (at most one in-flight unit per
    /// worker is lost). Records are keyed to the position + flag
    /// configuration, so a stale DIR is ignored rather than misapplied.
    /// Requires --parallel.
    #[arg(long, value_name = "DIR", requires = "parallel")]
    checkpoint: Option<String>,
}

/// `--max-nodes` accepts `_` separators (e.g. `2_000_000_000`).
fn parse_max_nodes(s: &str) -> Result<u64, String> {
    let n: u64 = s.replace('_', "").parse().map_err(|e| format!("{e}"))?;
    if n == 0 {
        return Err("must be >= 1".into());
    }
    Ok(n)
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

/// Run the search `f`, bracketing it with `search: start`/`search: end` markers
/// so a sampling profiler can be aimed at the search and not the table load.
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
        println!(
            "Throughput     : {:.2} Mnodes/s",
            st.nodes as f64 / secs / 1e6
        );
    }
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
    assert_eq!(cur, GOAL, "solution must reach GOAL");
}

// dyld's ASLR slide for the main executable. Sampling tools report raw runtime
// PCs, and deriving the slide by searching for the value that best fits the
// symbol table is unreliable at low sample counts — it produced a profile
// attributing 14% of time to a once-per-threshold callback. Printing it makes
// symbolisation exact. macOS-only (dyld); Linux profilers get the slide from
// /proc/self/maps.
#[cfg(target_os = "macos")]
extern "C" {
    fn _dyld_get_image_vmaddr_slide(image_index: u32) -> isize;
}

fn main() -> ExitCode {
    let t0 = Instant::now();
    #[cfg(target_os = "macos")]
    eprintln!("image slide: {:#x}", unsafe {
        _dyld_get_image_vmaddr_slide(0)
    });
    let args = Args::parse();
    // `--prove-at-least T` caps at threshold `T-1`: the deepest to exhaust.
    let max_bound = args.prove_at_least.map(|t| t - 1);
    let max_nodes = args.max_nodes.unwrap_or(u64::MAX);

    let start = match parse_position(&args.position) {
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

    // The σ-orbit split is sound only on a σ-fixed root, so it is auto-detected
    // rather than requested; the flag exists only to turn it off.
    let symmetric = puzzle8::puzzle24::symmetry::is_symmetric(&start);
    let orbit = !args.no_root_orbit_split && symmetric;
    if orbit {
        eprintln!(
            "root-orbit-split: board is σ-symmetric; searching one representative \
             per σ-orbit of the root's children"
        );
    } else if symmetric {
        eprintln!("root-orbit-split: σ-symmetric board, split disabled by flag");
    }

    // Set before any table is opened: the loaders read this globally.
    puzzle8::puzzle24::hugemap::set_hugepages(args.hugepages);
    if args.hugepages {
        eprintln!("hugepages: tables will be copied into anonymous MADV_HUGEPAGE memory");
    }

    let mm_path = &args.cwd_mm;
    let cwd = if std::path::Path::new(&mm_path).exists() {
        eprintln!("cWD: mmapping {mm_path}…");
        puzzle8::puzzle24::search::cwd::Cwd::mm_only(std::path::Path::new(&mm_path))
            .expect("cwd_mm artifact")
            .with_neighbor_prune(!args.no_neighbor_prune)
    } else {
        eprintln!("cWD: loading tables… (build data/cwd_mm.bin with `build_cwd_artifacts mm` for fast setup)");
        Cwd::new().with_neighbor_prune(!args.no_neighbor_prune)
    };
    let lm_mm_path = &args.cwd_lm_mm;
    let lmt = if args.lm {
        eprintln!("cwd-lm: mmapping {lm_mm_path}…");
        match puzzle8::puzzle24::search::engine::load_cwd_lm_mm(std::path::Path::new(&lm_mm_path)) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("error: --lm: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let lm1l = if args.clm2 {
        let path = "data/cwd_lm1l_mm.bin";
        eprintln!("clm2: mmapping {path}…");
        match puzzle8::puzzle24::search::cwd_lm1l::CwdLm1lMm::load(std::path::Path::new(path)) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("error: --clm2 (build {path} with `build_cwd_artifacts lm1l`): {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    // --clm2 subsumes the LM2 tier: it needs the same branch tables.
    let lm2t = if args.lm2 || args.clm2 {
        let flag = if args.clm2 { "--clm2" } else { "--lm2" };
        eprintln!("cwd-lm2: mmapping {lm_mm_path}…");
        match puzzle8::puzzle24::search::engine::load_cwd_lm_mm(std::path::Path::new(&lm_mm_path)) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!(
                    "error: {flag} (build data/cwd_lm_mm.bin with `build_cwd_artifacts lm-mm`): {e}"
                );
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let k8 = if args.zpdb8 {
        eprintln!("k8: mmapping the three 8-tile ZPDBs (32.8 GB)…");
        match puzzle8::puzzle24::search::engine::K8Ctx::load_mmap(std::path::Path::new("data")) {
            Ok(ctx) => {
                eprintln!(
                    "k8 ready: lazy max(cWD, k8) at survivors, both σ-views, \
                     shared packed-position cache"
                );
                Some(ctx)
            }
            Err(e) => {
                eprintln!("error: --zpdb8: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    eprintln!(
        "cWD ready: {} path, neighbor-prune ON",
        if cwd.has_overlay() {
            "fast single-line-max table"
        } else {
            "reference per-node A*"
        }
    );

    let dfa = MoveDfa::build_default();
    eprintln!(
        "engine: iterative arena, move-pruning DFA {} states, {} KiB",
        dfa.states(),
        dfa.table_bytes() / 1024
    );

    // Per-threshold telemetry. Cumulative totals hide that the rate FALLS with
    // depth — 31.50 Mn/s exhausting 144 vs ~26.7 exhausting 146 (FINDINGS §8x) —
    // and a ladder run's cumulative average silently blends the two.

    if args.parallel {
        eprintln!(
            "parallel: tree-splitting driver on {} rayon workers (RAYON_NUM_THREADS to change)",
            rayon::current_num_threads()
        );
    }

    let ((outcome, st), search_dt) = timed_search(t0.elapsed(), || {
        let mut prev_nodes = 0u64;
        let on_iter = |bound: u8, stats: &SearchStats, iter_dt: std::time::Duration| {
            let nodes = stats.nodes - prev_nodes;
            prev_nodes = stats.nodes;
            let secs = iter_dt.as_secs_f64();
            let rate = if secs > 0.0 {
                nodes as f64 / secs / 1e6
            } else {
                0.0
            };
            eprintln!(
                "[{}] threshold {bound:>3} exhausted: {nodes:>15} nodes in {iter_dt:.2?} \
                     ({rate:.2} Mn/s)",
                now_hms()
            );
            #[cfg(feature = "probe-cache-stats")]
            puzzle8::puzzle24::search::engine::lm2_slack_report_reset();
            #[cfg(feature = "search-census")]
            puzzle8::puzzle24::search::engine::lm1l_report_reset();
            // Per-node event rates for this threshold alone. Counts are
            // frequency-independent, so these are comparable across
            // thresholds even on battery.
            // Start recording when the FIRST threshold finishes — that is
            // the moment the deeper pass begins. Keying on the deeper bound
            // instead would never fire, because a budgeted run never
            // exhausts it.
        };

        let tiers = Tiers {
            k8: k8.as_ref(),
            lm: lmt.as_ref(),
            lm2: lm2t.as_ref(),
            lm1l: lm1l.as_ref(),
        };
        // `budget` stays u64::MAX under --parallel, and `checkpoint` stays
        // None without it: clap rejects both combinations.
        let opts = SearchOpts {
            orbit_split: orbit,
            max_bound: max_bound.unwrap_or(u8::MAX),
            budget: max_nodes,
            checkpoint: args.checkpoint.as_ref().map(std::path::PathBuf::from),
        };
        if args.parallel {
            puzzle8::puzzle24::search::engine::bounded_parallel(
                &start, &cwd, &dfa, tiers, opts, on_iter,
            )
        } else {
            bounded_tiers(&start, &cwd, &dfa, tiers, opts, on_iter)
        }
    });
    let elapsed = t0.elapsed();

    #[cfg(feature = "probe-cache-stats")]
    puzzle8::puzzle24::search::engine::lm_cache_stats_report();
    #[cfg(feature = "probe-cache-stats")]
    puzzle8::puzzle24::search::engine::lm2_cache_stats_report();
    #[cfg(feature = "probe-cache-stats")]
    puzzle8::puzzle24::search::engine::k8_cache_stats_report();
    #[cfg(feature = "search-census")]
    puzzle8::puzzle24::search::engine::k8_prune_stats_report(st.nodes);
    #[cfg(feature = "search-census")]
    puzzle8::puzzle24::search::engine::lm2_prune_stats_report(st.nodes);
    #[cfg(feature = "search-census")]
    puzzle8::puzzle24::search::engine::surviving_children_report();

    // The budget-truncated threshold never "exhausts", so the per-iteration hook
    // never fires for it — and that is precisely the threshold under study.

    match outcome {
        BoundedOutcome::Solved(s) => {
            if let Some(mb) = max_bound {
                println!("Found within bound {mb} (optimal):");
            }
            print_solution(&start, &s, elapsed);
            print_stats(&st, search_dt);
            ExitCode::SUCCESS
        }
        BoundedOutcome::ProvedAtLeast(k) => {
            println!("Lower bound: depth >= {k}");
            println!("Wall-clock     : {elapsed:.2?}");
            print_stats(&st, search_dt);
            ExitCode::SUCCESS
        }
        BoundedOutcome::Unsolvable => {
            println!("Unsolvable from this start.");
            print_stats(&st, search_dt);
            ExitCode::FAILURE
        }
        BoundedOutcome::BudgetExhausted(t) => {
            println!("*** NOT A PROOF — node budget reached inside threshold {t} ***");
            println!("Threshold {t} was NOT exhausted, so no lower bound follows.");
            println!("Wall-clock     : {elapsed:.2?}");
            print_stats(&st, search_dt);
            ExitCode::SUCCESS
        }
    }
}
