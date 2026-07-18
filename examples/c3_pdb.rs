//! Direction #3: Pattern Database (PDB) + IDA\* compression study.
//!
//! Builds several PDB partitions, runs IDA\* with each as the heuristic
//! across a representative sample of the 8-puzzle state space, and emits a
//! uniform `Report` for each — comparable apples-to-apples against the
//! Manhattan and full-table baselines.
//!
//! Run with:
//!     cargo run --release --example c3_pdb
//!
//! Reports `bytes_stored`, mean `solve_ns`, mean `h_calls` (a proxy for
//! nodes expanded), and `max_error_plies` (must be 0 for any correct PDB).

use std::cell::Cell;
use std::time::Instant;

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::pdb::pattern::Pattern;
use puzzle8::puzzle8::pdb::{AdditivePdbHeuristic, PatternDb, PdbHeuristic};
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{idastar, Heuristic, ManhattanHeuristic, TableHeuristic};
use puzzle8::puzzle8::state::N_STATES;

/// Heuristic that counts how many times `h()` is invoked. Used to compare
/// the IDA\* search effort between heuristics.
struct CountingHeuristic<'a, H: Heuristic> {
    inner: &'a H,
    count: Cell<u64>,
}

impl<'a, H: Heuristic> CountingHeuristic<'a, H> {
    fn new(inner: &'a H) -> Self {
        Self { inner, count: Cell::new(0) }
    }
    fn count(&self) -> u64 {
        self.count.get()
    }
    fn reset(&self) {
        self.count.set(0);
    }
}

impl<'a, H: Heuristic> Heuristic for CountingHeuristic<'a, H> {
    fn h(&self, s: &puzzle8::puzzle8::state::State) -> u8 {
        self.count.set(self.count.get() + 1);
        self.inner.h(s)
    }
}

/// Uniform report shape across compression directions. See DESIGN.md.
#[derive(Clone)]
struct Report {
    name: String,
    bytes_stored: usize,
    mean_solve_ns: u64,
    mean_h_calls: u64,
    max_error_plies: u8,
    samples: usize,
}

impl Report {
    fn print(&self) {
        println!(
            "  {:<32}  bytes={:>9}  mean_ns={:>10}  mean_h_calls={:>9}  max_err={}  (n={})",
            self.name,
            self.bytes_stored,
            self.mean_solve_ns,
            self.mean_h_calls,
            self.max_error_plies,
            self.samples,
        );
    }
}

/// Sample one state for every `stride` ranks. Stride 100 gives ~1815 samples
/// uniformly distributed across the state space — enough for stable mean
/// timings, cheap enough for an interactive run.
fn sample_states(stride: u32) -> Vec<u32> {
    (0..N_STATES).step_by(stride as usize).collect()
}

fn benchmark<H: Heuristic>(
    name: &str,
    h: &H,
    bytes_stored: usize,
    table: &DistanceTable,
    samples: &[u32],
) -> Report {
    let counter = CountingHeuristic::new(h);
    let mut total_solve_ns: u64 = 0;
    let mut total_h_calls: u64 = 0;
    let mut max_error: u8 = 0;

    for &r in samples {
        let s = unrank(r);
        counter.reset();
        let start = Instant::now();
        let sol = idastar(&s, &counter).expect("solvable");
        total_solve_ns += start.elapsed().as_nanos() as u64;
        total_h_calls += counter.count();
        let truth = table.dist(&s);
        let len = sol.len() as u8;
        let err = len.abs_diff(truth);
        if err > max_error {
            max_error = err;
        }
    }

    let n = samples.len() as u64;
    Report {
        name: name.to_string(),
        bytes_stored,
        mean_solve_ns: total_solve_ns / n,
        mean_h_calls: total_h_calls / n,
        max_error_plies: max_error,
        samples: samples.len(),
    }
}

fn main() {
    eprintln!("building reference distance table...");
    let table = DistanceTable::build();
    eprintln!("done: {} states, diameter {}.\n", table.visited_count(), table.diameter());

    let samples = sample_states(100);
    println!("== Direction #3: PDB compression study ==");
    println!("8-puzzle, single-threaded IDA\\*. Sample of {} states.\n", samples.len());

    let mut reports: Vec<Report> = Vec::new();

    // Baseline 1: Manhattan heuristic, no PDB at all.
    {
        let h = ManhattanHeuristic;
        let r = benchmark("baseline:Manhattan", &h, 0, &table, &samples);
        r.print();
        reports.push(r);
    }

    // Baseline 2: full distance table as the heuristic (exact, no search).
    {
        let h = TableHeuristic::new(&table);
        let r = benchmark(
            "baseline:full-table",
            &h,
            puzzle8::puzzle8::io::FILE_SIZE,
            &table,
            &samples,
        );
        r.print();
        reports.push(r);
    }

    // Single PDBs of varying size.
    for tiles in [
        &[1u8, 2][..],
        &[1, 2, 3],
        &[1, 2, 3, 4],
        &[1, 2, 3, 4, 5],
        &[1, 2, 3, 4, 5, 6, 7],
    ] {
        let pdb = PatternDb::build(Pattern::new(tiles));
        let label = format!("PDB{tiles:?}");
        let h = PdbHeuristic::new(&pdb);
        let r = benchmark(&label, &h, pdb.bytes_stored(), &table, &samples);
        r.print();
        reports.push(r);
    }

    // Additive PDB partitions.
    let partitions: &[&[&[u8]]] = &[
        &[&[1, 2, 3, 4], &[5, 6, 7, 8]],
        &[&[1, 2, 3], &[4, 5, 6, 7, 8]],
        &[&[1, 2, 3, 4, 5], &[6, 7, 8]],
        &[&[1, 2, 3], &[4, 5], &[6, 7, 8]],
    ];
    for parts in partitions {
        let dbs: Vec<PatternDb> =
            parts.iter().map(|tiles| PatternDb::build(Pattern::new(tiles))).collect();
        let label = format!(
            "Additive[{}]",
            parts.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" | ")
        );
        let h = AdditivePdbHeuristic::new(&dbs);
        let bytes = h.bytes_stored();
        let r = benchmark(&label, &h, bytes, &table, &samples);
        r.print();
        reports.push(r);
    }

    println!();
    println!("== Summary ==");
    println!("Goal: produce optimal solutions at smaller `bytes_stored` than");
    println!("the 181,448-byte full-table baseline. PDB compositions below it");
    println!("with `max_err == 0` are the lossless wins.");
    println!();
    let baseline_bytes = puzzle8::puzzle8::io::FILE_SIZE;
    for r in &reports {
        if r.max_error_plies != 0 {
            println!("  {:<32}  NOT OPTIMAL (max_err = {})", r.name, r.max_error_plies);
            continue;
        }
        let ratio = if r.bytes_stored == 0 {
            f64::INFINITY
        } else {
            baseline_bytes as f64 / r.bytes_stored as f64
        };
        println!(
            "  {:<32}  {:>10} bytes  ({:.2}× compression vs baseline)",
            r.name, r.bytes_stored, ratio
        );
    }
}
