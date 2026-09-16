//! Sphere-stratified estimation of heuristic quality η on the 24-puzzle
//! (Clausecker & Schintke, SoCS 2021). See `puzzle24::eta` for the method.
//!
//! ```text
//! eta24 yield --k-min 10 --k-max 64 --attempts 10000 --moribund both --verifier zpdb
//! ```
//!
//! `yield` measures, per sphere depth `k`, how often a `k`-step walk from GOAL
//! ends at distance exactly `k`, and what the rejection search costs. Output is
//! tab-separated on stdout; progress goes to stderr.
//!
//! Verifiers for "is there a solution shorter than k":
//! - `zpdb`: recursive IDA\* with a zero-aware PDB partition and its diagonal
//!   reflection, 6-6-6-6 (`--zpdb-set k6`) or 7-7-7-3 (`k7`);
//! - `cwd`: the flat engine's bounded search over plain cWD
//!   (`data/cwd_mm.bin`), one `engine::bounded` call per board.
//!
//! Measured at k = 64 (moribund walker, 2000 attempts, 12 threads on a
//! non-idle Mac): k6 224k nodes / 33 ms per attempt, k7 671k / 110 ms, cwd
//! 4.9M / 199 ms. All three classify identically; k6 is the default.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use puzzle8::puzzle24::eta::sphere::{attempt_with, reject_shorter, Attempt};
use puzzle8::puzzle24::eta::{Rng, Walker, Z95};
use puzzle8::puzzle24::pdb::{ZPatternDb, ZpdbInc};
use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::engine;
use puzzle8::puzzle24::search::{BoundedOutcome, MoveDfa};
use puzzle8::puzzle24::state::State;
use rayon::prelude::*;

const ZPDB_K6_FILES: [&str; 4] = [
    "pdb24_a.zbin",
    "pdb24_b.zbin",
    "pdb24_c.zbin",
    "pdb24_d.zbin",
];

const ZPDB_K7_FILES: [&str; 4] = [
    "pdb24_k7_a.zbin",
    "pdb24_k7_b.zbin",
    "pdb24_k7_c.zbin",
    "pdb24_k7_d.zbin",
];

/// Attempts per parallel work unit; each unit owns one RNG stream.
const CHUNK: u64 = 256;

#[derive(Parser)]
#[command(name = "eta24")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum MoribundArg {
    On,
    Off,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum VerifierArg {
    Zpdb,
    Cwd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ZpdbSetArg {
    /// 6-6-6-6, pdb24_{a,b,c,d}.zbin (22.6 MB each).
    K6,
    /// 7-7-7-3, pdb24_k7_{a,b,c,d}.zbin (508 MB ×3).
    K7,
}

#[derive(clap::Args)]
struct VerifierOpts {
    /// Search used to reject walks that end closer than k.
    #[arg(long, value_enum, default_value_t = VerifierArg::Zpdb)]
    verifier: VerifierArg,
    /// Zero-aware PDB partition for `--verifier zpdb`.
    #[arg(long, value_enum, default_value_t = ZpdbSetArg::K6)]
    zpdb_set: ZpdbSetArg,
    /// Directory holding the zero-aware PDB files.
    #[arg(long, value_name = "DIR", default_value = "data")]
    pdb_dir: PathBuf,
    /// Merged cWD artifact for `--verifier cwd`.
    #[arg(long, value_name = "PATH", default_value = "data/cwd_mm.bin")]
    cwd_mm: PathBuf,
}

#[derive(Subcommand)]
enum Cmd {
    /// Walk yield and rejection-search cost per sphere depth.
    Yield {
        #[arg(long, default_value_t = 10)]
        k_min: u32,
        #[arg(long, default_value_t = 64)]
        k_max: u32,
        #[arg(long, default_value_t = 2)]
        k_step: u32,
        /// Walks per depth.
        #[arg(long, default_value_t = 10_000)]
        attempts: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Moribund pruning in the walk rule.
        #[arg(long, value_enum, default_value_t = MoribundArg::Both)]
        moribund: MoribundArg,
        /// Prefix checkpoint spacings to compare (0 = final board only).
        #[arg(long, value_delimiter = ',', default_value = "0")]
        check_every: Vec<u32>,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
}

/// Loaded verifier resources, shared read-only across worker threads.
enum Verifier {
    Zpdb(Vec<ZPatternDb>),
    Cwd { cwd: Box<Cwd>, dfa: MoveDfa },
}

impl Verifier {
    fn load(opts: &VerifierOpts) -> Result<Verifier, String> {
        match opts.verifier {
            VerifierArg::Zpdb => match opts.zpdb_set {
                ZpdbSetArg::K6 => ZPDB_K6_FILES,
                ZpdbSetArg::K7 => ZPDB_K7_FILES,
            }
            .iter()
            .map(|name| {
                ZPatternDb::load_mmap(&opts.pdb_dir.join(name)).map_err(|e| format!("{name}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Verifier::Zpdb),
            VerifierArg::Cwd => {
                let cwd = Cwd::mm_only(Path::new(&opts.cwd_mm))
                    .map_err(|e| format!("{}: {e}", opts.cwd_mm.display()))?;
                Ok(Verifier::Cwd {
                    cwd: Box::new(cwd),
                    dfa: MoveDfa::build_default(),
                })
            }
        }
    }

    /// Run `f` with a rejection test `(board, k) -> (shorter, nodes)`.
    fn with_reject<R>(&self, f: impl FnOnce(&mut dyn FnMut(&State, u32) -> (bool, u64)) -> R) -> R {
        match self {
            Verifier::Zpdb(dbs) => {
                let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
                f(&mut |v, k| reject_shorter(v, k, &inc))
            }
            Verifier::Cwd { cwd, dfa } => f(&mut |v, k| {
                if k < 2 {
                    return (false, 0);
                }
                let cap = u8::try_from(k - 2).expect("sphere depth exceeds u8 search bounds");
                let (outcome, stats) = engine::bounded(v, cwd, dfa, false, cap);
                match outcome {
                    BoundedOutcome::Solved(_) => (true, stats.nodes),
                    BoundedOutcome::ProvedAtLeast(_) => (false, stats.nodes),
                    other => panic!("engine::bounded on a walk endpoint returned {other:?}"),
                }
            }),
        }
    }
}

#[derive(Default, Clone, Copy)]
struct YieldTally {
    attempts: u64,
    dead: u64,
    accepted: u64,
    nodes: u64,
    nanos: u128,
}

impl YieldTally {
    fn merge(mut self, o: YieldTally) -> YieldTally {
        self.attempts += o.attempts;
        self.dead += o.dead;
        self.accepted += o.accepted;
        self.nodes += o.nodes;
        self.nanos += o.nanos;
        self
    }
}

struct YieldRun {
    k_min: u32,
    k_max: u32,
    k_step: u32,
    attempts: u64,
    seed: u64,
    moribund: MoribundArg,
    check_every: Vec<u32>,
}

fn run_yield(run: YieldRun, verifier: &Verifier) {
    let dfa = MoveDfa::build_default();
    let modes: &[bool] = match run.moribund {
        MoribundArg::On => &[true],
        MoribundArg::Off => &[false],
        MoribundArg::Both => &[false, true],
    };
    println!(
        "moribund\tcheck_every\tk\tattempts\tdead_end_rate\tyield\tyield_ci95\tnodes_per_attempt\tus_per_attempt\tus_per_accepted"
    );
    let settings: Vec<(bool, u32)> = modes
        .iter()
        .flat_map(|&mb| run.check_every.iter().map(move |&c| (mb, c)))
        .collect();
    for (mb, check_every) in settings {
        let walker = Walker::new(&dfa, mb);
        eprintln!(
            "walker moribund={mb}: {} nodes, {} doomed; check_every={check_every}",
            walker.node_count(),
            walker.doomed_nodes()
        );
        let mut k = run.k_min;
        while k <= run.k_max {
            let t0 = Instant::now();
            let chunks = run.attempts.div_ceil(CHUNK);
            let tally = (0..chunks)
                .into_par_iter()
                .map(|c| {
                    let mut rng = Rng::stream(run.seed, (k as u64) << 1 | mb as u64, c);
                    let mut path = Vec::with_capacity(k as usize);
                    let n = CHUNK.min(run.attempts - c * CHUNK);
                    let start = Instant::now();
                    let mut t = YieldTally {
                        attempts: n,
                        ..Default::default()
                    };
                    verifier.with_reject(|reject| {
                        for _ in 0..n {
                            let (a, nodes) = attempt_with(
                                &walker,
                                k,
                                check_every,
                                &mut rng,
                                &mut path,
                                &mut *reject,
                            );
                            t.nodes += nodes;
                            match a {
                                Attempt::DeadEnd => t.dead += 1,
                                Attempt::Rejected => {}
                                Attempt::Accepted { .. } => t.accepted += 1,
                            }
                        }
                    });
                    t.nanos = start.elapsed().as_nanos();
                    t
                })
                .reduce(YieldTally::default, YieldTally::merge);
            let n = tally.attempts as f64;
            let y = tally.accepted as f64 / n;
            let us = tally.nanos as f64 / 1e3;
            println!(
                "{}\t{check_every}\t{k}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.1}\t{:.2}\t{:.2}",
                if mb { "on" } else { "off" },
                tally.attempts,
                tally.dead as f64 / n,
                y,
                Z95 * (y * (1.0 - y) / n).sqrt(),
                tally.nodes as f64 / n,
                us / n,
                if tally.accepted > 0 {
                    us / tally.accepted as f64
                } else {
                    f64::NAN
                },
            );
            eprintln!(
                "k={k} moribund={mb}: yield {y:.4} in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            k += run.k_step;
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match args.cmd {
        Cmd::Yield {
            k_min,
            k_max,
            k_step,
            attempts,
            seed,
            moribund,
            check_every,
            verifier,
        } => Verifier::load(&verifier).map(|v| {
            run_yield(
                YieldRun {
                    k_min,
                    k_max,
                    k_step,
                    attempts,
                    seed,
                    moribund,
                    check_every,
                },
                &v,
            )
        }),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
