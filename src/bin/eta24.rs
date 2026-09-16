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
//! ```text
//! eta24 probe --k 30,45,64 --attempts 20000 --window 14
//! ```
//!
//! `probe` additionally computes the reach probability `P(v)` of every
//! accepted board and reports, per depth: stage costs, interval sizes, path
//! multiplicity `P(v) / walk probability`, Horvitz–Thompson estimates of
//! `|V_k|` and of the Manhattan-distance stratum quality `η_k` with 95%
//! intervals and weight dispersion, and the attempts a ±5% interval on `η_k`
//! would need. `|V_k|` is compared with A090031 (`k ≤ 30`) or Clausecker's
//! sampled sizes (ZIB Report 20-17, App. B, `k ≤ 64`). Requires `--verifier
//! zpdb`.
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
use puzzle8::puzzle24::eta::reach::reach_probability;
use puzzle8::puzzle24::eta::sphere::{attempt, attempt_with, reject_shorter, Attempt};
use puzzle8::puzzle24::eta::walker::Choice;
use puzzle8::puzzle24::eta::{
    blank_weights, branching_factor, Accum, Rng, Walker, Weighting, A090031, Z95,
};
use puzzle8::puzzle24::pdb::{ZPatternDb, ZpdbInc};
use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::engine;
use puzzle8::puzzle24::search::move_dfa::DEFAULT_WINDOW;
use puzzle8::puzzle24::search::{
    BoundedOutcome, Heuristic, LongMoveDfa, ManhattanHeuristic, MoveDfa,
};
use puzzle8::puzzle24::state::{State, N_STATES};
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
enum ChoiceArg {
    /// Every allowed move equally likely.
    Uniform,
    /// Moves weighted by the number of moves allowed after them.
    Lookahead,
}

impl From<ChoiceArg> for Choice {
    fn from(c: ChoiceArg) -> Choice {
        match c {
            ChoiceArg::Uniform => Choice::Uniform,
            ChoiceArg::Lookahead => Choice::Lookahead,
        }
    }
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
        /// Move-DFA windows to compare; the rule covers sequences of up to
        /// window + 1 moves. 11 uses MoveDfa, anything else LongMoveDfa.
        #[arg(long, value_delimiter = ',', default_value = "11")]
        window: Vec<u8>,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
    /// Reach probabilities, estimator dispersion and projected cost per depth.
    Probe {
        /// Sphere depths to probe.
        #[arg(long, value_delimiter = ',', default_value = "30,45,64")]
        k: Vec<u32>,
        /// Walks per depth.
        #[arg(long, default_value_t = 20_000)]
        attempts: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Prefix checkpoint spacing (0 = final board only).
        #[arg(long, default_value_t = 8)]
        check_every: u32,
        /// Move-DFA window; the rule covers sequences of up to window + 1 moves.
        #[arg(long, default_value_t = 14)]
        window: u8,
        /// Moribund pruning in the walk rule.
        #[arg(long, value_enum, default_value_t = MoribundArg::On)]
        moribund: MoribundArg,
        /// Target half-width of the 95% interval on η_k, relative, for the
        /// projected attempt count.
        #[arg(long, default_value_t = 0.05)]
        target_rel: f64,
        /// Report where the 1/P(v) weight dispersion comes from instead of
        /// the estimates.
        #[arg(long)]
        diagnose: bool,
        /// How a walk chooses among allowed moves.
        #[arg(long, value_enum, default_value_t = ChoiceArg::Uniform)]
        choice: ChoiceArg,
        /// Tilt toward low Manhattan distance: a move raising it is weighted
        /// b^-λ, one lowering it b^λ (0 = no tilt).
        #[arg(long, value_name = "λ", default_value_t = 0.0)]
        md_tilt: f64,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
}

/// Sphere sizes estimated by Clausecker (ZIB Report 20-17, App. B) for
/// `k = 31..=64`, where A090031 has no exact terms.
const THESIS_SPHERE_SIZES: [f64; 34] = [
    5.6194e10, 1.1994e11, 2.3783e11, 5.0202e11, 9.8134e11, 2.0533e12, 3.9619e12, 8.1914e12,
    1.5588e13, 3.1794e13, 5.9655e13, 1.2043e14, 2.2266e14, 4.4187e14, 8.0513e14, 1.5792e15,
    2.8374e15, 5.4745e15, 9.6311e15, 1.8397e16, 3.1874e16, 5.9307e16, 1.0118e17, 1.8691e17,
    3.1078e17, 5.6396e17, 9.2789e17, 1.6497e18, 2.6358e18, 4.5888e18, 7.1572e18, 1.2410e19,
    1.9020e19, 3.0484e19,
];

/// Reference size of sphere `k`: exact from A090031 or Clausecker's estimate.
fn reference_sphere_size(k: u32) -> Option<(f64, &'static str)> {
    match k {
        0..=30 => Some((A090031[k as usize] as f64, "A090031 exact")),
        31..=64 => Some((THESIS_SPHERE_SIZES[k as usize - 31], "thesis App. B")),
        _ => None,
    }
}

/// Walker over the move DFA for `window`: the engine's [`MoveDfa`] at its
/// default window, [`LongMoveDfa`] otherwise.
fn walker_for(window: u8, moribund: bool) -> Walker {
    if window == DEFAULT_WINDOW {
        Walker::new(&MoveDfa::build_default(), moribund)
    } else {
        Walker::new(&LongMoveDfa::build(window), moribund)
    }
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
    window: Vec<u8>,
}

fn run_yield(run: YieldRun, verifier: &Verifier) {
    let modes: &[bool] = match run.moribund {
        MoribundArg::On => &[true],
        MoribundArg::Off => &[false],
        MoribundArg::Both => &[false, true],
    };
    println!(
        "window\tmoribund\tcheck_every\tk\tattempts\tdead_end_rate\tyield\tyield_ci95\tnodes_per_attempt\tus_per_attempt\tus_per_accepted"
    );
    let mut settings: Vec<(u8, bool, u32)> = Vec::new();
    for &w in &run.window {
        for &mb in modes {
            for &c in &run.check_every {
                settings.push((w, mb, c));
            }
        }
    }
    for (window, mb, check_every) in settings {
        let t_build = Instant::now();
        let walker = walker_for(window, mb);
        eprintln!(
            "walker window={window} moribund={mb}: {} nodes, {} doomed, built in {:.1}s; check_every={check_every}",
            walker.node_count(),
            walker.doomed_nodes(),
            t_build.elapsed().as_secs_f64()
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
                "{window}\t{}\t{check_every}\t{k}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.1}\t{:.2}\t{:.2}",
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

/// Diagnostics for one accepted sample.
#[derive(Clone, Copy)]
struct Sample {
    prob: f64,
    walk_prob: f64,
    interval: usize,
    layer_states: usize,
    back_nodes: u64,
    stage_b_ns: u64,
    md: u8,
    blank: u8,
    /// Steps of the accepted walk by number of allowed moves (index 1..=3).
    steps: [u16; 4],
}

#[derive(Default)]
struct ProbeTally {
    attempts: u64,
    stage_a_nodes: u64,
    stage_a_ns: u128,
    samples: Vec<Sample>,
}

impl ProbeTally {
    fn merge(mut self, mut o: ProbeTally) -> ProbeTally {
        self.attempts += o.attempts;
        self.stage_a_nodes += o.stage_a_nodes;
        self.stage_a_ns += o.stage_a_ns;
        self.samples.append(&mut o.samples);
        self
    }
}

struct ProbeRun {
    k: Vec<u32>,
    attempts: u64,
    seed: u64,
    check_every: u32,
    window: u8,
    moribund: bool,
    target_rel: f64,
    /// Report the weight-dispersion diagnosis instead of the estimates.
    diagnose: bool,
    choice: Choice,
    md_tilt: f64,
}

fn run_probe(run: ProbeRun, verifier: &Verifier) -> Result<(), String> {
    let Verifier::Zpdb(dbs) = verifier else {
        return Err(
            "probe needs --verifier zpdb (the reach probability uses its heuristic)".into(),
        );
    };
    let t_build = Instant::now();
    let walker = walker_for(run.window, run.moribund)
        .with_choice(run.choice)
        .with_md_tilt(run.md_tilt);
    eprintln!(
        "walker window={} moribund={} choice={:?} md_tilt={}: {} nodes, built in {:.1}s",
        run.window,
        run.moribund,
        run.choice,
        run.md_tilt,
        walker.node_count(),
        t_build.elapsed().as_secs_f64()
    );
    let b = branching_factor();
    let weights: Vec<(Weighting, [f64; 25])> = Weighting::ALL
        .iter()
        .map(|&w| (w, blank_weights(w)))
        .collect();
    println!(
        "# probe: window {} (covers {} moves), moribund {}, choice {:?}, md_tilt {}, check_every {}, {} attempts per depth, seed {}, b = {b:.9}",
        run.window,
        run.window + 1,
        run.moribund,
        run.choice,
        run.md_tilt,
        run.check_every,
        run.attempts,
        run.seed
    );
    for &k in &run.k {
        let t0 = Instant::now();
        let tally = sample_depth(&walker, dbs, k, &run);
        if run.diagnose {
            report_diagnosis(k, &tally, b, t0.elapsed());
        } else {
            report_probe(k, &tally, b, &weights, run.target_rel, t0.elapsed());
        }
    }
    Ok(())
}

/// Run `run.attempts` walks at depth `k` in parallel and compute the reach
/// probability of every accepted board.
fn sample_depth(walker: &Walker, dbs: &[ZPatternDb], k: u32, run: &ProbeRun) -> ProbeTally {
    let chunks = run.attempts.div_ceil(CHUNK);
    (0..chunks)
        .into_par_iter()
        .map(|c| {
            let inc = ZpdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);
            let mut rng = Rng::stream(run.seed, k as u64, c);
            let mut path = Vec::with_capacity(k as usize);
            let n = CHUNK.min(run.attempts - c * CHUNK);
            let mut t = ProbeTally {
                attempts: n,
                ..Default::default()
            };
            for _ in 0..n {
                let ta = Instant::now();
                let (a, nodes) = attempt(walker, k, run.check_every, &mut rng, &inc, &mut path);
                t.stage_a_ns += ta.elapsed().as_nanos();
                t.stage_a_nodes += nodes;
                if let Attempt::Accepted { board, walk_prob } = a {
                    let tb = Instant::now();
                    let r = reach_probability(walker, &board, k, &inc);
                    let stage_b_ns = tb.elapsed().as_nanos() as u64;
                    assert!(
                        r.prob >= walk_prob * (1.0 - 1e-9),
                        "P(v) {} below the walk's own probability {walk_prob}",
                        r.prob
                    );
                    let mut steps = [0u16; 4];
                    let mut node = walker.root();
                    for (i, &m) in path.iter().enumerate() {
                        steps[walker.allowed(node, k - i as u32).len() as usize] += 1;
                        node = walker.next(node, m);
                    }
                    t.samples.push(Sample {
                        prob: r.prob,
                        walk_prob,
                        interval: r.interval,
                        layer_states: r.max_layer_states,
                        back_nodes: r.nodes,
                        stage_b_ns,
                        md: ManhattanHeuristic.h(&board),
                        blank: board.blank_pos(),
                        steps,
                    });
                }
            }
            t
        })
        .reduce(ProbeTally::default, ProbeTally::merge)
}

/// Kish effective sample size of positive terms.
fn ess(xs: impl Iterator<Item = f64>) -> f64 {
    let (s, s2) = xs.fold((0.0, 0.0), |(s, s2), x| (s + x, s2 + x * x));
    if s2 == 0.0 {
        0.0
    } else {
        s * s / s2
    }
}

/// Mean, standard deviation and Pearson correlation helpers over samples.
fn mean_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, v.sqrt())
}

fn correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let (mx, sx) = mean_sd(xs);
    let (my, sy) = mean_sd(ys);
    let n = xs.len() as f64;
    let cov = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    cov / (sx * sy)
}

/// Symmetry class of a blank cell on the 5×5 board.
fn blank_class(cell: u8) -> &'static str {
    let (r, c) = (cell / 5, cell % 5);
    let (a, b) = (r.min(4 - r), c.min(4 - c));
    match (a.min(b), a.max(b)) {
        (0, 0) => "corner",
        (0, 1) => "edge-near-corner",
        (0, 2) => "edge-middle",
        (1, 1) => "inner-corner",
        (1, 2) => "inner-edge",
        _ => "centre",
    }
}

/// Where the 1/P(v) weight dispersion comes from.
fn report_diagnosis(k: u32, t: &ProbeTally, b: f64, wall: std::time::Duration) {
    let acc = t.samples.len();
    println!(
        "== k = {k}: {acc} accepted of {} attempts, wall {:.1} s",
        t.attempts,
        wall.as_secs_f64()
    );
    if acc < 10 {
        println!("  too few accepted samples");
        return;
    }
    let w: Vec<f64> = t.samples.iter().map(|s| 1.0 / s.prob).collect();
    let inv_walk: Vec<f64> = t.samples.iter().map(|s| 1.0 / s.walk_prob).collect();
    let mult: Vec<f64> = t.samples.iter().map(|s| s.prob / s.walk_prob).collect();
    let total: f64 = w.iter().sum();

    let mut sorted = w.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| sorted[((acc - 1) as f64 * p) as usize];
    let median = q(0.5);
    let top1: f64 = sorted[acc - acc.div_ceil(100)..].iter().sum();
    let top10: f64 = sorted[acc.saturating_sub(10)..].iter().sum();
    println!(
        "  weight W = 1/P: W/median at p10 {:.2}, p90 {:.1}, p99 {:.1}, max {:.0}; top 1% of samples carry {:.1}% of sum W, top 10 carry {:.1}%",
        q(0.1) / median,
        q(0.9) / median,
        q(0.99) / median,
        sorted[acc - 1] / median,
        100.0 * top1 / total,
        100.0 * top10 / total
    );

    let lw: Vec<f64> = w.iter().map(|x| x.log10()).collect();
    let lwalk: Vec<f64> = inv_walk.iter().map(|x| x.log10()).collect();
    let lmult: Vec<f64> = mult.iter().map(|x| x.log10()).collect();
    println!(
        "  log10 spread (sd): W {:.2}; choice product 1/walk {:.2}; multiplicity M {:.2}; corr(log 1/walk, log M) {:+.2}",
        mean_sd(&lw).1,
        mean_sd(&lwalk).1,
        mean_sd(&lmult).1,
        correlation(&lwalk, &lmult)
    );
    let mean_mult = mult.iter().sum::<f64>() / acc as f64;
    println!(
        "  ESS: W {:.0}; if only the choice product varied (M fixed at its mean) {:.0}; if only M varied (walk prob fixed) {:.0}; of {acc}",
        ess(w.iter().copied()),
        ess(inv_walk.iter().map(|x| x / mean_mult)),
        ess(mult.iter().map(|m| 1.0 / m))
    );

    let three: Vec<f64> = t.samples.iter().map(|s| s.steps[3] as f64).collect();
    let (two_m, _) = mean_sd(
        &t.samples
            .iter()
            .map(|s| s.steps[2] as f64)
            .collect::<Vec<_>>(),
    );
    let (one_m, _) = mean_sd(
        &t.samples
            .iter()
            .map(|s| s.steps[1] as f64)
            .collect::<Vec<_>>(),
    );
    let (three_m, three_sd) = mean_sd(&three);
    println!(
        "  steps per walk: 1-way {one_m:.1}, 2-way {two_m:.1}, 3-way {three_m:.1} (sd {three_sd:.1}); corr(#3-way, log W) {:+.2}, corr(#3-way, log M) {:+.2}",
        correlation(&three, &lw),
        correlation(&three, &lmult)
    );

    println!("  by end-blank class:            samples   share of sum W   median W/median");
    for class in [
        "corner",
        "edge-near-corner",
        "edge-middle",
        "inner-corner",
        "inner-edge",
        "centre",
    ] {
        let mut ws: Vec<f64> = t
            .samples
            .iter()
            .zip(&w)
            .filter(|(s, _)| blank_class(s.blank) == class)
            .map(|(_, &x)| x)
            .collect();
        if ws.is_empty() {
            continue;
        }
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "    {class:<18} {:>8.1}%   {:>12.1}%   {:>12.2}",
            100.0 * ws.len() as f64 / acc as f64,
            100.0 * ws.iter().sum::<f64>() / total,
            ws[ws.len() / 2] / median
        );
    }

    let md: Vec<f64> = t.samples.iter().map(|s| s.md as f64).collect();
    let (md_mean, md_sd) = mean_sd(&md);
    let eta_terms: Vec<f64> = t
        .samples
        .iter()
        .map(|s| b.powi(-(s.md as i32)) / s.prob)
        .collect();
    let eta_total: f64 = eta_terms.iter().sum();
    println!(
        "  MD of accepted: mean {md_mean:.1}, sd {md_sd:.1}; corr(MD, log W) {:+.2}; ESS of eta terms {:.0}",
        correlation(&md, &lw),
        ess(eta_terms.iter().copied())
    );
    println!("  by MD relative to mean:        samples   share of sum W   share of sum eta terms");
    for (lo, hi, label) in [
        (f64::NEG_INFINITY, -2.0, "< mean-2sd"),
        (-2.0, -1.0, "mean-2sd..-1sd"),
        (-1.0, 0.0, "mean-1sd..mean"),
        (0.0, 1.0, "mean..+1sd"),
        (1.0, f64::INFINITY, "> mean+1sd"),
    ] {
        let idx: Vec<usize> = (0..acc)
            .filter(|&i| {
                let z = (md[i] - md_mean) / md_sd;
                z >= lo && z < hi
            })
            .collect();
        println!(
            "    {label:<18} {:>8.1}%   {:>12.1}%   {:>12.1}%",
            100.0 * idx.len() as f64 / acc as f64,
            100.0 * idx.iter().map(|&i| w[i]).sum::<f64>() / total,
            100.0 * idx.iter().map(|&i| eta_terms[i]).sum::<f64>() / eta_total
        );
    }

    let mut order: Vec<usize> = (0..acc).collect();
    order.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap());
    println!(
        "  heaviest samples: W/median  log2(1/walk)  M      3-way  blank class        MD  interval"
    );
    for &i in order.iter().take(5) {
        let s = &t.samples[i];
        println!(
            "    {:>10.0}  {:>12.1}  {:>6.2}  {:>5}  {:<18} {:>3}  {:>8}",
            w[i] / median,
            -s.walk_prob.log2(),
            mult[i],
            s.steps[3],
            blank_class(s.blank),
            s.md,
            s.interval
        );
    }
    println!(
        "  for comparison, median sample: log2(1/walk) {:.1}, M {:.2}, 3-way {:.0}",
        {
            let mut v = lwalk.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[acc / 2] / 2f64.log10()
        },
        {
            let mut v = mult.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[acc / 2]
        },
        {
            let mut v = three.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[acc / 2]
        }
    );
}

fn report_probe(
    k: u32,
    t: &ProbeTally,
    b: f64,
    weights: &[(Weighting, [f64; 25])],
    target_rel: f64,
    wall: std::time::Duration,
) {
    let n = t.attempts as f64;
    let acc = t.samples.len();
    let n_states = N_STATES as f64;
    println!(
        "== k = {k}: {acc} accepted of {} attempts (yield {:.5}), wall {:.1} s",
        t.attempts,
        acc as f64 / n,
        wall.as_secs_f64()
    );
    println!(
        "  stage A: {:.3} ms/attempt, {:.0} nodes/attempt",
        t.stage_a_ns as f64 / 1e6 / n,
        t.stage_a_nodes as f64 / n
    );
    if acc == 0 {
        println!("  no accepted samples");
        return;
    }
    let mean = |f: fn(&Sample) -> f64| t.samples.iter().map(f).sum::<f64>() / acc as f64;
    let max = |f: fn(&Sample) -> f64| t.samples.iter().map(f).fold(0.0, f64::max);
    let stage_b_ns: u128 = t.samples.iter().map(|s| s.stage_b_ns as u128).sum();
    println!(
        "  stage B: {:.2} ms/accepted (max {:.1}); backward nodes {:.0} (max {:.0}); interval {:.0} boards (max {:.0}); forward layer states max {:.0}",
        mean(|s| s.stage_b_ns as f64) / 1e6,
        max(|s| s.stage_b_ns as f64) / 1e6,
        mean(|s| s.back_nodes as f64),
        max(|s| s.back_nodes as f64),
        mean(|s| s.interval as f64),
        max(|s| s.interval as f64),
        max(|s| s.layer_states as f64),
    );
    let mut mult: Vec<f64> = t.samples.iter().map(|s| s.prob / s.walk_prob).collect();
    mult.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  P(v) / walk probability: mean {:.2}, median {:.2}, max {:.1}; mean MD of accepted {:.1}",
        mean(|s| s.prob / s.walk_prob),
        mult[acc / 2],
        mult[acc - 1],
        mean(|s| s.md as f64)
    );

    let mut size = Accum::default();
    for s in &t.samples {
        size.add(1.0 / s.prob);
    }
    size.add_zeros(t.attempts - acc as u64);
    let (sz, sz_half) = (size.mean(), Z95 * size.std_error());
    let reference = reference_sphere_size(k).map_or(String::new(), |(r, src)| {
        format!(
            "; reference {r:.4e} ({src}): {:+.2}% = {:+.1} SE",
            100.0 * (sz - r) / r,
            (sz - r) / size.std_error()
        )
    });
    println!(
        "  |V_k| = {sz:.4e} ± {sz_half:.2e} ({:.2}%), ESS {:.0} of {acc}, largest weight {:.2}%{reference}",
        100.0 * sz_half / sz,
        size.effective_n(),
        100.0 * size.max_share()
    );

    let thread_us_per_attempt = (t.stage_a_ns + stage_b_ns) as f64 / 1e3 / n;
    for (wt, w) in weights {
        let mut eta = Accum::default();
        for s in &t.samples {
            eta.add(w[s.blank as usize] * b.powi(-(s.md as i32)) / s.prob);
        }
        eta.add_zeros(t.attempts - acc as u64);
        let value = eta.mean() / n_states;
        let half = Z95 * eta.std_error() / n_states;
        let rel = half / value;
        println!(
            "  eta_k MD {:<7} = {value:.4e} ± {half:.2e} ({:.2}%), ESS {:.0}, largest term {:.2}%",
            wt.name(),
            100.0 * rel,
            eta.effective_n(),
            100.0 * eta.max_share()
        );
        if *wt == Weighting::Uniform {
            let need = n * (rel / target_rel).powi(2);
            let thread_h = need * thread_us_per_attempt / 1e6 / 3600.0;
            println!(
                "  for ±{:.0}% on eta_k (uniform): ~{need:.3e} attempts, {:.3} ms thread time per attempt, ~{thread_h:.2} thread-hours (~{:.2} h on 12 threads)",
                100.0 * target_rel,
                thread_us_per_attempt / 1e3,
                thread_h / 12.0
            );
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
            window,
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
                    window,
                },
                &v,
            )
        }),
        Cmd::Probe {
            k,
            attempts,
            seed,
            check_every,
            window,
            moribund,
            target_rel,
            diagnose,
            choice,
            md_tilt,
            verifier,
        } => {
            let moribund = match moribund {
                MoribundArg::On => Ok(true),
                MoribundArg::Off => Ok(false),
                MoribundArg::Both => Err("probe takes --moribund on or off".to_string()),
            };
            moribund.and_then(|moribund| {
                Verifier::load(&verifier).and_then(|v| {
                    run_probe(
                        ProbeRun {
                            k,
                            attempts,
                            seed,
                            check_every,
                            window,
                            moribund,
                            target_rel,
                            diagnose,
                            choice: choice.into(),
                            md_tilt,
                        },
                        &v,
                    )
                })
            })
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
