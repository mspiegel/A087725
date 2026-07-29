//! Solve a 24-puzzle position optimally, or prove a lower bound on its optimal
//! depth, with the flat (iterative) IDA\* engine over cWD.
//!
//! ```text
//! solve24 --position "<25 tokens>" [--prove-at-least T] [--no-root-orbit-split]
//! ```
//!
//! Position format: 25 whitespace-separated tokens in row-major order, `_`/`.`/`0`
//! for the blank and `1..=24` for tiles. The board must be solvable.
//!
//! - Omit `--prove-at-least` to search unbounded until the optimum is found.
//! - `--prove-at-least T` caps the search at threshold `T-1`; exhausting it
//!   *proves* `dist ≥ T`. This is a threshold, not a guess: cWD(R) = 144, so on
//!   `R` anything at or below 145 returns immediately having searched nothing.
//! - `--no-root-orbit-split` disables the σ-orbit split, which is otherwise
//!   auto-enabled on σ-symmetric boards. That split is *soundness*-critical
//!   rather than merely fast — it discards half the root's children on the
//!   strength of the symmetry argument — so running a proof both ways and
//!   confirming the same bound is a cheap end-to-end check on that reasoning.
//!
//! # Why there are no other flags
//!
//! This binary drives exactly one configuration: **cWD + move-DFA + neighbour-WD
//! child pre-prune + root σ-orbit split**, the stack the `R` lower-bound program
//! runs. None of it is selectable, because there is nothing to select between —
//! the PDB heuristics and the recursive engine are gone, and the DFA is folded
//! into the flat engine's per-node candidate mask rather than applied as a
//! separate filter.
//!
//! `Cwd::new()` locates its own tables at `data/wd24.bin` and
//! `data/cwd_single.bin`, which is why there is no `--pdb-dir`.

use std::process::ExitCode;
use std::time::Instant;

use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::flat::flat_bounded_budgeted;
use puzzle8::puzzle24::search::move_dfa::MoveDfa;
use puzzle8::puzzle24::search::{BoundedOutcome, SearchStats};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

struct Args {
    position: Option<String>,
    /// `--prove-at-least T` stores `T-1`: the deepest threshold to exhaust.
    max_bound: Option<u8>,
    /// `None` = auto (on iff the board is σ-symmetric).
    root_orbit_split: Option<bool>,
    /// Stop after this many nodes. Benchmarking only — a truncated run proves
    /// nothing. `u64::MAX` = no budget.
    max_nodes: u64,
}

const USAGE: &str = "\
usage: solve24 --position \"<25 tokens>\" [--prove-at-least T] [--no-root-orbit-split]

  --position P            board, 25 whitespace-separated tokens, row-major.
                          blank is 0, _ or .; tiles are 1..=24.
  --prove-at-least T      cap the search at threshold T-1; exhausting it proves
                          dist >= T. omit to solve optimally with no cap.
  --no-root-orbit-split   disable the σ-orbit split (auto-on when the board is
                          σ-symmetric). use to cross-check a proof.
  --max-nodes N           stop after N nodes. BENCHMARKING ONLY — the result is
                          not a proof, and is reported as such. Useful because
                          the engine is node-identical across variants, so a
                          budget cuts every A/B arm at the same tree position.
  --help                  this message.";

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut position = None;
    let mut max_bound = None;
    let mut root_orbit_split = None;
    let mut max_nodes = u64::MAX;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--position" => {
                i += 1;
                position = Some(argv.get(i).ok_or("--position needs a value")?.clone());
            }
            "--prove-at-least" => {
                i += 1;
                let t: u8 = argv
                    .get(i)
                    .ok_or("--prove-at-least needs a value")?
                    .parse()
                    .map_err(|e| format!("--prove-at-least: {e}"))?;
                if t < 1 {
                    return Err("--prove-at-least must be >= 1".into());
                }
                max_bound = Some(t - 1);
            }
            "--no-root-orbit-split" => root_orbit_split = Some(false),
            "--max-nodes" => {
                i += 1;
                max_nodes = argv
                    .get(i)
                    .ok_or("--max-nodes needs a value")?
                    .replace('_', "")
                    .parse()
                    .map_err(|e| format!("--max-nodes: {e}"))?;
                if max_nodes == 0 {
                    return Err("--max-nodes must be >= 1".into());
                }
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}")),
        }
        i += 1;
    }
    Ok(Args {
        position,
        max_bound,
        root_orbit_split,
        max_nodes,
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
    #[cfg(feature = "demand-histogram")]
    println!(
        "{}",
        puzzle8::puzzle24::search::flat::demand_histogram::report()
    );
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

/// Events to count when built with `pmu-counters`. Chosen to separate the two
/// live hypotheses for the depth slowdown: cache locality vs page walks. The
/// TLB pair is the interesting one — those pages are all resident, so TLB
/// pressure is completely invisible to the FAULTS counter in `top`.
#[cfg(feature = "pmu-counters")]
const PMU_EVENTS: &[&str] = &[
    "CORE_ACTIVE_CYCLE",
    "INST_ALL",
    "L1D_CACHE_MISS_LD_NONSPEC",
    "L2_TLB_MISS_DATA",
];

// dyld's ASLR slide for the main executable. Sampling tools report raw runtime
// PCs, and deriving the slide by searching for the value that best fits the
// symbol table is unreliable at low sample counts — it produced a profile
// attributing 14% of time to a once-per-threshold callback. Printing it makes
// symbolisation exact.
extern "C" {
    fn _dyld_get_image_vmaddr_slide(image_index: u32) -> isize;
}

fn main() -> ExitCode {
    let t0 = Instant::now();
    eprintln!("image slide: {:#x}", unsafe {
        _dyld_get_image_vmaddr_slide(0)
    });
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let Some(position_str) = args.position.as_ref() else {
        eprintln!("error: --position is required\n\n{USAGE}");
        return ExitCode::FAILURE;
    };
    let start = match parse_position(position_str) {
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
    let orbit = args.root_orbit_split.unwrap_or(true) && symmetric;
    if orbit {
        eprintln!(
            "root-orbit-split: board is σ-symmetric; searching one representative \
             per σ-orbit of the root's children"
        );
    } else if symmetric {
        eprintln!("root-orbit-split: σ-symmetric board, split disabled by flag");
    }

    eprintln!("cWD: loading tables…");
    let cwd = Cwd::new().with_neighbor_prune(true);
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
        "flat engine: iterative arena, move-pruning DFA {} states, {} KiB",
        dfa.states(),
        dfa.table_bytes() / 1024
    );

    // Per-threshold telemetry. Cumulative totals hide that the rate FALLS with
    // depth — 31.50 Mn/s exhausting 144 vs ~26.7 exhausting 146 (FINDINGS §8x) —
    // and a ladder run's cumulative average silently blends the two.
    #[cfg(feature = "pmu-counters")]
    let pmu = match puzzle8::puzzle24::pmu::Pmu::new(PMU_EVENTS) {
        Ok(p) => {
            eprintln!("pmu: counting {}", PMU_EVENTS.join(", "));
            Some(p)
        }
        Err(e) => {
            eprintln!("pmu: disabled ({e})");
            None
        }
    };

    #[cfg(feature = "pmu-counters")]
    let pmu_state = std::cell::RefCell::new((
        pmu.as_ref().map(|p| p.read()).unwrap_or_default(),
        0u64, // nodes at the last threshold boundary
    ));

    let ((outcome, st), search_dt) = timed_search(t0.elapsed(), || {
        let mut prev_nodes = 0u64;
        flat_bounded_budgeted(
            &start,
            &cwd,
            &dfa,
            orbit,
            args.max_bound.unwrap_or(u8::MAX),
            args.max_nodes,
            |bound, stats, iter_dt| {
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
                // Per-node event rates for this threshold alone. Counts are
                // frequency-independent, so these are comparable across
                // thresholds even on battery.
                // Start recording when the FIRST threshold finishes — that is
                // the moment the deeper pass begins. Keying on the deeper bound
                // instead would never fire, because a budgeted run never
                // exhausts it.
                #[cfg(feature = "probe-locality")]
                if prev_nodes == nodes {
                    eprintln!("    probe-locality: recording the next 40M probes of the next pass");
                    puzzle8::puzzle24::probe_locality::start(40_000_000);
                }
                #[cfg(feature = "pmu-counters")]
                if let Some(p) = pmu.as_ref() {
                    let now = p.read();
                    let mut st = pmu_state.borrow_mut();
                    for (i, name) in p.names().iter().enumerate() {
                        let d =
                            now.get(i).copied().unwrap_or(0) - st.0.get(i).copied().unwrap_or(0);
                        eprintln!(
                            "    pmu {name:<28} {d:>18}  {:>8.3} /node",
                            d as f64 / nodes as f64
                        );
                    }
                    st.0 = now;
                    st.1 = stats.nodes;
                }
            },
        )
    });
    let elapsed = t0.elapsed();

    // The budget-truncated threshold never "exhausts", so the per-iteration hook
    // never fires for it — and that is precisely the threshold under study.
    #[cfg(feature = "pmu-counters")]
    if let Some(p) = pmu.as_ref() {
        let st_pmu = pmu_state.borrow();
        let partial = st.nodes.saturating_sub(st_pmu.1);
        if partial > 0 {
            let now = p.read();
            eprintln!("    -- partial (unexhausted) threshold: {partial} nodes --");
            for (i, name) in p.names().iter().enumerate() {
                let d = now.get(i).copied().unwrap_or(0) - st_pmu.0.get(i).copied().unwrap_or(0);
                eprintln!(
                    "    pmu {name:<28} {d:>18}  {:>8.3} /node",
                    d as f64 / partial as f64
                );
            }
        }
    }

    #[cfg(feature = "probe-locality")]
    eprintln!("{}", puzzle8::puzzle24::probe_locality::report());

    match outcome {
        BoundedOutcome::Solved(s) => {
            if let Some(mb) = args.max_bound {
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
