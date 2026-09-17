//! Stratified estimation of heuristic quality η on the 24-puzzle (Clausecker &
//! Schintke, SoCS 2021). See `puzzle24::eta` for the method and
//! records/eta24_*.txt for the campaigns run with it.
//!
//! `campaign layers` gives exact spheres for small `k`, `campaign sample`
//! stores sphere samples for larger `k`, `campaign tail` and `campaign tail-md`
//! sample the tail `d ≥ L`, `campaign total` estimates η by Manhattan levels,
//! and `campaign score` combines everything for one heuristic.
//!
//! "Is there a solution shorter than k" is decided by recursive IDA\* with the
//! 6-6-6-6 zero-aware PDB partition and its diagonal reflection
//! (records/eta24_yield.txt, Result 2).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use puzzle8::puzzle24::eta::reach::reach_probability;
use puzzle8::puzzle24::eta::sphere::{attempt, Attempt};
use puzzle8::puzzle24::eta::tail::{at_least, uniform_solvable};
use puzzle8::puzzle24::eta::{
    blank_weights, branching_factor, Accum, Rng, Walker, Weighting, A090031, Z95,
};
use puzzle8::puzzle24::pdb::{ZPatternDb, ZpdbInc};
use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::engine;
use puzzle8::puzzle24::search::{
    Heuristic, LongMoveDfa, ManhattanHeuristic, WalkingDistanceHeuristic,
};
use puzzle8::puzzle24::state::{State, N_STATES};
use rayon::prelude::*;

const ZPDB_FILES: [&str; 4] = [
    "pdb24_a.zbin",
    "pdb24_b.zbin",
    "pdb24_c.zbin",
    "pdb24_d.zbin",
];

/// Attempts per parallel work unit; each unit owns one RNG stream.
const CHUNK: u64 = 256;

#[derive(Parser)]
#[command(name = "eta24")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args)]
struct VerifierOpts {
    /// Directory holding the zero-aware PDB files.
    #[arg(long, value_name = "DIR", default_value = "data")]
    pdb_dir: PathBuf,
}

#[derive(Subcommand)]
enum Cmd {
    /// A stratified sampling campaign: exact layers, stored samples, scoring.
    #[command(subcommand)]
    Campaign(Campaign),
}

#[derive(Subcommand)]
enum Campaign {
    /// Exact |V_k| and exact eta_k for k <= max-k, from breadth-first layers.
    Layers {
        /// Layer 24 is 3.2e8 boards: ~14 GiB peak, 40-60 s for a table-free
        /// heuristic.
        #[arg(long, default_value_t = 24)]
        max_k: usize,
        /// Campaign directory; writes exact_layers_<heuristic>.tsv there.
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        #[arg(long, value_enum, default_value_t = HeuristicArg::Md)]
        heuristic: HeuristicArg,
    },
    /// Draw sphere samples for k-min..=k-max into DIR, extending what is there.
    Sample {
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        #[arg(long, default_value_t = 23)]
        k_min: u32,
        #[arg(long, default_value_t = 64)]
        k_max: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Target thread-seconds for stratum k-min (including samples already
        /// on disk).
        #[arg(long, default_value_t = 60.0)]
        thread_seconds: f64,
        /// Budget growth per stratum: k gets thread_seconds × growth^(k − k_min).
        #[arg(long, default_value_t = 1.0)]
        growth: f64,
        /// Cap on any stratum's thread-second budget.
        #[arg(long, default_value_t = f64::INFINITY)]
        max_thread_seconds: f64,
        /// Move-DFA window.
        #[arg(long, default_value_t = 14)]
        window: u8,
        /// Prefix checkpoint spacing (does not change the sample distribution).
        #[arg(long, default_value_t = 8)]
        check_every: u32,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
    /// Sample the tail V_>=min-distance by uniform random solvable states,
    /// storing the accepted boards for scoring; extends what is in DIR.
    Tail {
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        /// Accept a state when it is proven at least this far from GOAL.
        #[arg(long, default_value_t = 65)]
        min_distance: u8,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Target thread-seconds for the tail (including draws already on disk).
        #[arg(long, default_value_t = 600.0)]
        thread_seconds: f64,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
    /// Sample the tail V_>=min-distance stratified by Manhattan distance
    /// m <= md-max: uniform placements within each level, weighted by the
    /// level's exact size (eta::md_levels). Scoring takes m > md-max from
    /// `campaign tail`. Extends what is in DIR.
    TailMd {
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        /// Count a state when it is proven at least this far from GOAL.
        #[arg(long, default_value_t = 65)]
        min_distance: u8,
        /// Highest Manhattan distance sampled by level, and the split scoring
        /// takes the uniform tail above. The level table takes
        /// 8·(2^25 − 1)·(md-max + 1) bytes: 10.2 GiB at 40. A campaign's meta
        /// file pins this, so an existing directory keeps the value it was
        /// started with.
        #[arg(long, default_value_t = 40)]
        md_max: u8,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Target thread-seconds (including draws already on disk).
        #[arg(long, default_value_t = 3600.0)]
        thread_seconds: f64,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
    /// Total eta of a heuristic by Manhattan levels, without distance proofs:
    /// uniform placements within each level m <= md-max weighted by the level's
    /// exact size, plus uniform solvable states with Manhattan distance above
    /// md-max.
    Total {
        #[arg(long, value_enum, default_value_t = HeuristicArg::Md)]
        heuristic: HeuristicArg,
        /// Highest Manhattan distance sampled by level. The table takes
        /// 8·(2^25 − 1)·(md-max + 1) bytes, 12.2 GiB at 48, and is freed
        /// before the heuristic loads its own; above MD 48 the uniform part
        /// is 0.4% of eta for cWD, where above 40 it was 7% and heavy-tailed.
        #[arg(long, default_value_t = 48)]
        md_max: u8,
        /// Draws per level.
        #[arg(long, default_value_t = 4_000_000)]
        level_draws: u64,
        /// Uniform solvable draws for Manhattan distance above md-max.
        #[arg(long, default_value_t = 1_000_000_000)]
        uniform_draws: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Count only states proven at least this far from GOAL.
        #[arg(long, default_value_t = 0)]
        distance_min: u8,
        /// Count only states proven at most this far from GOAL. At or below
        /// md-max, no state above md-max qualifies and no uniform draws are made.
        #[arg(long)]
        distance_max: Option<u8>,
        #[command(flatten)]
        verifier: VerifierOpts,
    },
    /// Score the campaign in DIR for a heuristic.
    Score {
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        #[arg(long, value_enum, default_value_t = HeuristicArg::Md)]
        heuristic: HeuristicArg,
        /// Batches for batch-means standard errors.
        #[arg(long, default_value_t = 20)]
        batches: usize,
        /// Published per-stratum eta_k to compare with (TSV: k, plotted value,
        /// value used; e.g. records/eta24_fig53_digitized.tsv).
        #[arg(long, value_name = "PATH")]
        published_histogram: Option<PathBuf>,
        /// When rescoring the uniform tail, score boards with Manhattan
        /// distance above this only one in --rescore-stride, weighted by the
        /// stride.
        #[arg(long, default_value_t = TAIL_STORE_MAX_MD)]
        rescore_full_md: u8,
        #[arg(long, default_value_t = 1)]
        rescore_stride: u64,
    },
}

/// Heuristic scored by a campaign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum HeuristicArg {
    /// Manhattan distance.
    Md,
    /// Walking distance (row WD + column WD, data/wd24.bin).
    Wd,
    /// The solver's cWD (data/cwd_mm.bin).
    Cwd,
    /// The solver's `--lm2` value, max(cWD, last-two-moves branches)
    /// (adds data/cwd_lm_mm.bin).
    Lm2,
    /// The solver's `--clm2` value, max(LM2, single-demanded-line joint value)
    /// (adds data/cwd_lm1l_mm.bin).
    Clm2,
    /// The solver's k8 tier alone: max of normal and reflected sums of the
    /// three 8-tile zPDBs (data/pdb24_k8_{a,b,c}.zbin, 30.5 GB).
    K8,
    /// The `--clm2 --zpdb8` cascade the solver prunes with, max(cLM2, k8).
    #[value(name = "clm2k8")]
    Clm2K8,
}

static K8_TABLES: std::sync::OnceLock<engine::K8Ctx> = std::sync::OnceLock::new();

/// Tables of the solver's walking-distance tiers, loaded once per process by
/// [`HeuristicArg::prepare`] for the first tier heuristic prepared.
struct TierTables {
    cwd: Cwd,
    lm: Option<puzzle8::puzzle24::search::cwd_lm::CwdLmMm>,
    lm1l: Option<puzzle8::puzzle24::search::cwd_lm1l::CwdLm1lMm>,
}

static TIER_TABLES: std::sync::OnceLock<TierTables> = std::sync::OnceLock::new();

thread_local! {
    /// Per-thread evaluator over [`TIER_TABLES`] (it holds front caches).
    static TIER_EVAL: std::cell::RefCell<Option<engine::TierEval<'static>>> =
        const { std::cell::RefCell::new(None) };
}

impl HeuristicArg {
    fn name(self) -> &'static str {
        match self {
            HeuristicArg::Md => "md",
            HeuristicArg::Wd => "wd",
            HeuristicArg::Cwd => "cwd",
            HeuristicArg::Lm2 => "lm2",
            HeuristicArg::Clm2 => "clm2",
            HeuristicArg::K8 => "k8",
            HeuristicArg::Clm2K8 => "clm2k8",
        }
    }

    /// Load any tables the heuristic needs before parallel evaluation.
    fn prepare(self) {
        use puzzle8::puzzle24::search::{cwd_lm::CwdLmMm, cwd_lm1l::CwdLm1lMm};
        let load = |path: &'static str| -> &'static Path {
            eprintln!("{}: mapping {path}", self.name());
            Path::new(path)
        };
        if matches!(self, HeuristicArg::K8 | HeuristicArg::Clm2K8) {
            K8_TABLES.get_or_init(|| {
                eprintln!("{}: mapping data/pdb24_k8_{{a,b,c}}.zbin", self.name());
                engine::K8Ctx::load_mmap(Path::new("data"), engine::EngineConfig::Standard)
                    .expect("data/pdb24_k8_{a,b,c}.zbin")
            });
        }
        match self {
            HeuristicArg::Md | HeuristicArg::K8 => {}
            HeuristicArg::Wd => WalkingDistanceHeuristic::warm_up_verbose(),
            HeuristicArg::Cwd | HeuristicArg::Lm2 | HeuristicArg::Clm2 | HeuristicArg::Clm2K8 => {
                TIER_TABLES.get_or_init(|| {
                    let t0 = Instant::now();
                    let joint = matches!(self, HeuristicArg::Clm2 | HeuristicArg::Clm2K8);
                    let tables = TierTables {
                        cwd: Cwd::mm_only(load("data/cwd_mm.bin")).expect("data/cwd_mm.bin"),
                        lm: (self != HeuristicArg::Cwd).then(|| {
                            CwdLmMm::load(load("data/cwd_lm_mm.bin")).expect("data/cwd_lm_mm.bin")
                        }),
                        lm1l: joint.then(|| {
                            CwdLm1lMm::load(load("data/cwd_lm1l_mm.bin"))
                                .expect("data/cwd_lm1l_mm.bin")
                        }),
                    };
                    eprintln!("{}: tables ready in {:.1?}", self.name(), t0.elapsed());
                    tables
                });
            }
        }
    }

    fn k8(s: &State) -> u8 {
        K8_TABLES
            .get()
            .expect("k8 evaluated before prepare()")
            .eval(s)
    }

    fn h(self, s: &State) -> u8 {
        match self {
            HeuristicArg::Md => ManhattanHeuristic.h(s),
            HeuristicArg::Wd => WalkingDistanceHeuristic.h(s),
            HeuristicArg::K8 => Self::k8(s),
            HeuristicArg::Clm2K8 => HeuristicArg::Clm2.h(s).max(Self::k8(s)),
            HeuristicArg::Cwd | HeuristicArg::Lm2 | HeuristicArg::Clm2 => TIER_EVAL.with(|cell| {
                let mut slot = cell.borrow_mut();
                let ev = slot.get_or_insert_with(|| {
                    let t = TIER_TABLES
                        .get()
                        .expect("tier heuristic evaluated before prepare()");
                    engine::TierEval::new(&t.cwd, t.lm.as_ref(), t.lm1l.as_ref())
                });
                let v = ev.eval(s);
                let missing =
                    || -> u8 { panic!("{} needs tables this process did not load", self.name()) };
                match self {
                    HeuristicArg::Cwd => v.cwd,
                    HeuristicArg::Lm2 => v.lm2.unwrap_or_else(missing),
                    _ => v.clm2.unwrap_or_else(missing),
                }
            }),
        }
    }

    /// Exact eta over the whole puzzle per weighting, where known.
    fn exact_total(self) -> Option<[f64; 3]> {
        match self {
            HeuristicArg::Md => Some(EXACT_MD_ETA),
            _ => None,
        }
    }
}

fn exact_layers_path(dir: &Path, heuristic: HeuristicArg) -> PathBuf {
    dir.join(format!("exact_layers_{}.tsv", heuristic.name()))
}

/// Chunks per parallel round in `sample`.
const SAMPLE_ROUND_CHUNKS: u64 = 96;

/// Version of the walk rule recorded in stratum meta files. Bump it when the
/// walker's move probabilities change for the same settings.
const WALKER_VERSION: &str = "1";

/// Exact Manhattan-distance quality of the whole 24-puzzle at the branching
/// factor `weights::branching_factor()` = 2.367604543724, per weighting
/// (uniform, tree, degree): (perm(M) + det(M′)) / 2 / |V| over the 25×25 matrix
/// of b^−md(tile, cell) with the blank row holding w(cell), computed outside
/// the repo and checked against brute force on the 8-puzzle
/// (records/eta24_md.txt).
const EXACT_MD_ETA: [f64; 3] = [1.0009629701e-19, 9.7305423082e-20, 9.7565321490e-20];

/// Clausecker's published Manhattan-distance figures (ZIB Report 20-17,
/// Table 5.1 and Fig. 5.3).
const PUBLISHED_MD_ETA: f64 = 9.926e-20;
const PUBLISHED_MD_ETA_HALF95: f64 = 9.013e-21;
const PUBLISHED_MD_ETA_GE65: f64 = 2.364e-20;

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

/// The 6-6-6-6 zero-aware PDBs the distance proofs use, shared read-only
/// across worker threads.
struct Verifier {
    dbs: Vec<ZPatternDb>,
}

impl Verifier {
    fn load(opts: &VerifierOpts) -> Result<Verifier, String> {
        ZPDB_FILES
            .iter()
            .map(|name| {
                ZPatternDb::load_mmap(&opts.pdb_dir.join(name)).map_err(|e| format!("{name}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|dbs| Verifier { dbs })
    }

    /// Incremental Korf-max heuristic over the four PDBs and their reflection.
    fn inc(&self) -> ZpdbInc<'_, 4> {
        let d = &self.dbs;
        ZpdbInc::new([&d[0], &d[1], &d[2], &d[3]])
    }
}

/// `campaign layers`: exact |V_k| and Σ w·b^−h / |V| per weighting for k ≤ max_k.
fn run_layers(max_k: usize, dir: &Path, heuristic: HeuristicArg) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    heuristic.prepare();
    let b = branching_factor();
    let weights: Vec<[f64; 25]> = Weighting::ALL.iter().map(|&w| blank_weights(w)).collect();
    let n_states = N_STATES as f64;
    let path = exact_layers_path(dir, heuristic);
    let mut out = format!(
        "# heuristic {}\n# k\tsize\teta_uniform\teta_tree\teta_degree\n",
        heuristic.name()
    );
    let t0 = Instant::now();
    puzzle8::puzzle24::eta::for_each_layer(max_k, |k, layer| {
        let sums = layer
            .par_iter()
            .map(|&key| {
                let s = puzzle8::puzzle24::eta::unpack(key);
                let base = b.powi(-(heuristic.h(&s) as i32));
                let blank = s.blank_pos() as usize;
                [
                    weights[0][blank] * base,
                    weights[1][blank] * base,
                    weights[2][blank] * base,
                ]
            })
            .reduce(|| [0.0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
        out.push_str(&format!(
            "{k}\t{}\t{:.12e}\t{:.12e}\t{:.12e}\n",
            layer.len(),
            sums[0] / n_states,
            sums[1] / n_states,
            sums[2] / n_states
        ));
        eprintln!(
            "layer {k}: {} boards ({:.1}s)",
            layer.len(),
            t0.elapsed().as_secs_f64()
        );
    });
    std::fs::write(&path, out).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

struct SampleCampaign {
    dir: PathBuf,
    k_min: u32,
    k_max: u32,
    seed: u64,
    thread_seconds: f64,
    growth: f64,
    max_thread_seconds: f64,
    window: u8,
    check_every: u32,
}

/// `campaign sample`: extend each stratum until its thread-time budget is met.
fn run_sample(c: SampleCampaign, verifier: &Verifier) -> Result<(), String> {
    use puzzle8::puzzle24::eta::layers::pack;
    use puzzle8::puzzle24::eta::samples::{
        append_chunks, append_samples, chunks_path, meta_path, read_chunks, samples_path,
        write_or_check_meta, ChunkRecord, SampleRecord,
    };
    std::fs::create_dir_all(&c.dir).map_err(|e| e.to_string())?;
    let walker = Walker::new(&LongMoveDfa::build(c.window));
    eprintln!("walker window={}: {} nodes", c.window, walker.node_count());
    for k in c.k_min..=c.k_max {
        let settings: std::collections::BTreeMap<String, String> = [
            ("k", k.to_string()),
            ("window", c.window.to_string()),
            ("moribund", "true".to_string()),
            ("choice", "Lookahead".to_string()),
            ("md_tilt", "0".to_string()),
            ("walker_version", WALKER_VERSION.to_string()),
        ]
        .into_iter()
        .map(|(a, b)| (a.to_string(), b))
        .collect();
        write_or_check_meta(&meta_path(&c.dir, k), &settings).map_err(|e| e.to_string())?;
        let existing = read_chunks(&chunks_path(&c.dir, k)).map_err(|e| e.to_string())?;
        let mut next_chunk = existing.iter().map(|r| r.chunk + 1).max().unwrap_or(0);
        let mut spent_ns: u128 = existing.iter().map(|r| r.thread_ns as u128).sum();
        let budget_ns = (c.thread_seconds * c.growth.powi((k - c.k_min) as i32))
            .min(c.max_thread_seconds)
            * 1e9;
        let t0 = Instant::now();
        let (mut attempts, mut accepted) = (0u64, 0u64);
        while (spent_ns as f64) < budget_ns {
            let round: Vec<(ChunkRecord, Vec<SampleRecord>)> = (next_chunk
                ..next_chunk + SAMPLE_ROUND_CHUNKS)
                .into_par_iter()
                .map(|chunk| {
                    let inc = verifier.inc();
                    let mut rng = Rng::stream(c.seed, k as u64, chunk);
                    let mut path = Vec::with_capacity(k as usize);
                    let start = Instant::now();
                    let mut samples = Vec::new();
                    for _ in 0..CHUNK {
                        let (a, _) = attempt(&walker, k, c.check_every, &mut rng, &inc, &mut path);
                        if let Attempt::Accepted { board, walk_prob } = a {
                            let r = reach_probability(&walker, &board, k, &inc);
                            assert!(r.prob >= walk_prob * (1.0 - 1e-9));
                            samples.push(SampleRecord {
                                board: pack(&board),
                                prob: r.prob,
                                chunk: u32::try_from(chunk).expect("chunk index fits u32"),
                            });
                        }
                    }
                    let rec = ChunkRecord {
                        seed: c.seed,
                        chunk,
                        attempts: CHUNK,
                        accepted: samples.len() as u64,
                        thread_ns: start.elapsed().as_nanos() as u64,
                    };
                    (rec, samples)
                })
                .collect();
            let all: Vec<SampleRecord> =
                round.iter().flat_map(|(_, s)| s.iter().copied()).collect();
            append_samples(&samples_path(&c.dir, k), &all).map_err(|e| e.to_string())?;
            let recs: Vec<ChunkRecord> = round.iter().map(|(r, _)| *r).collect();
            append_chunks(&chunks_path(&c.dir, k), &recs).map_err(|e| e.to_string())?;
            spent_ns += recs.iter().map(|r| r.thread_ns as u128).sum::<u128>();
            attempts += recs.iter().map(|r| r.attempts).sum::<u64>();
            accepted += all.len() as u64;
            next_chunk += SAMPLE_ROUND_CHUNKS;
        }
        eprintln!(
            "k={k}: +{attempts} attempts, +{accepted} accepted this run; {:.0} of {:.0} thread-s; wall {:.0}s",
            spent_ns as f64 / 1e9,
            budget_ns / 1e9,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Uniform draws per tail chunk.
const TAIL_CHUNK: u64 = 1 << 16;

/// Accepted tail states with Manhattan distance at most this are stored as
/// boards, which is what scoring reads: every other accepted state has
/// `h ≥ MD > this` for the heuristics scored here, so it contributes at most
/// `b^−(this + 1)`.
///
/// 72 costs 28 B per stored board — 6.4 GB for the 7.1×10^8-draw run of
/// records/eta24_md.txt — and bounds the unstored part at 4.7e-28, below
/// every tail measured so far (the smallest, k8's, is 4.7e-25). Lowering it
/// saves storage but weakens that bound geometrically: at 56 the bound is
/// 4.5e-22, which would swamp the tails of the stronger heuristics.
const TAIL_STORE_MAX_MD: u8 = 72;

fn tail_chunks_path(dir: &Path, min_distance: u8) -> PathBuf {
    dir.join(format!("tail_ge{min_distance}.chunks.tsv"))
}

fn tail_samples_path(dir: &Path, min_distance: u8) -> PathBuf {
    dir.join(format!("tail_ge{min_distance}.samples"))
}

/// One tail chunk: attempts, accepted, thread ns and the stored boards.
struct TailChunk {
    chunk: u64,
    attempts: u64,
    accepted: u64,
    thread_ns: u64,
    stored: Vec<puzzle8::puzzle24::eta::samples::SampleRecord>,
}

/// `campaign tail`: uniform draws until the tail's thread-second budget is met.
fn run_tail(
    dir: &Path,
    min_distance: u8,
    seed: u64,
    thread_seconds: f64,
    verifier: &Verifier,
) -> Result<(), String> {
    use puzzle8::puzzle24::eta::layers::pack;
    use puzzle8::puzzle24::eta::samples::{append_samples, SampleRecord};
    use std::io::Write;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    // Walking distance only as an acceptance shortcut: it is often already at
    // least min_distance, which skips the search.
    HeuristicArg::Wd.prepare();
    let chunks_file = tail_chunks_path(dir, min_distance);
    let existing = read_tail_chunks(&chunks_file)?;
    let mut next_chunk = existing.iter().map(|c| c.chunk + 1).max().unwrap_or(0);
    let mut spent_ns: u128 = existing.iter().map(|c| c.thread_ns as u128).sum();
    let budget_ns = thread_seconds * 1e9;
    let t0 = Instant::now();
    let (mut attempts, mut accepted) = (0u64, 0u64);
    while (spent_ns as f64) < budget_ns {
        let round: Vec<TailChunk> = (next_chunk..next_chunk + SAMPLE_ROUND_CHUNKS)
            .into_par_iter()
            .map(|chunk| {
                let inc = verifier.inc();
                let mut rng = Rng::stream(seed, 1000 + min_distance as u64, chunk);
                let start = Instant::now();
                let mut t = TailChunk {
                    chunk,
                    attempts: TAIL_CHUNK,
                    accepted: 0,
                    thread_ns: 0,
                    stored: Vec::new(),
                };
                for _ in 0..TAIL_CHUNK {
                    let s = uniform_solvable(&mut rng);
                    let md = ManhattanHeuristic.h(&s);
                    // A uniform state has no parity tied to min_distance, so
                    // the proof must exhaust threshold min_distance − 1.
                    let far = md.max(WalkingDistanceHeuristic.h(&s)) >= min_distance
                        || at_least(&s, min_distance, &inc);
                    if !far {
                        continue;
                    }
                    t.accepted += 1;
                    if md <= TAIL_STORE_MAX_MD {
                        t.stored.push(SampleRecord {
                            board: pack(&s),
                            prob: 1.0,
                            chunk: u32::try_from(chunk).expect("chunk index fits u32"),
                        });
                    }
                }
                t.thread_ns = start.elapsed().as_nanos() as u64;
                t
            })
            .collect();
        let stored: Vec<SampleRecord> = round
            .iter()
            .flat_map(|t| t.stored.iter().copied())
            .collect();
        append_samples(&tail_samples_path(dir, min_distance), &stored)
            .map_err(|e| e.to_string())?;
        let fresh = !chunks_file.exists();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&chunks_file)
            .map_err(|e| e.to_string())?;
        let mut text = String::new();
        if fresh {
            text.push_str("# seed\tchunk\tattempts\taccepted\tthread_ns\tstored\n");
        }
        for t in &round {
            text.push_str(&format!(
                "{seed}\t{}\t{}\t{}\t{}\t{}\n",
                t.chunk,
                t.attempts,
                t.accepted,
                t.thread_ns,
                t.stored.len()
            ));
        }
        f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        spent_ns += round.iter().map(|t| t.thread_ns as u128).sum::<u128>();
        attempts += round.iter().map(|t| t.attempts).sum::<u64>();
        accepted += round.iter().map(|t| t.accepted).sum::<u64>();
        next_chunk += SAMPLE_ROUND_CHUNKS;
        eprintln!(
            "tail >= {min_distance}: +{attempts} draws, +{accepted} accepted this run; {:.0} of {:.0} thread-s; wall {:.0}s",
            spent_ns as f64 / 1e9,
            budget_ns / 1e9,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// A parsed tail chunk line. Runs before the heuristic sums moved out of the
/// file wrote six more columns; they are ignored.
struct TailChunkRecord {
    chunk: u64,
    attempts: u64,
    accepted: u64,
    thread_ns: u64,
    stored: u64,
}

fn read_tail_chunks(path: &Path) -> Result<Vec<TailChunkRecord>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 6 && f.len() != 12 {
            break;
        }
        let Ok(ints) = f[1..6]
            .iter()
            .map(|x| x.parse::<u64>())
            .collect::<Result<Vec<_>, _>>()
        else {
            break;
        };
        out.push(TailChunkRecord {
            chunk: ints[0],
            attempts: ints[1],
            accepted: ints[2],
            thread_ns: ints[3],
            stored: ints[4],
        });
    }
    Ok(out)
}

/// A uniform tail run rescored for a heuristic it was not scored with, from
/// its stored boards (Manhattan distance <= [`TAIL_STORE_MAX_MD`]): per chunk,
/// Σ w(blank)·b^−h over stored boards with Manhattan distance <= `split` and
/// above it.
struct RescoredTail {
    split: u8,
    /// chunk → (attempts, accepted − stored, sums at MD <= split, sums above).
    chunks: BTreeMap<u64, (u64, u64, [f64; 3], [f64; 3])>,
}

impl RescoredTail {
    /// Upper bound on the part of eta from accepted boards that were not
    /// stored: each has Manhattan distance above [`TAIL_STORE_MAX_MD`], so for
    /// a heuristic at least Manhattan distance it contributes at most
    /// max w · b^−(TAIL_STORE_MAX_MD + 1).
    fn unstored_bound(&self) -> f64 {
        let w_max = Weighting::ALL
            .iter()
            .flat_map(|&w| blank_weights(w))
            .fold(0.0f64, f64::max);
        let attempts: u64 = self.chunks.values().map(|c| c.0).sum();
        let unstored: u64 = self.chunks.values().map(|c| c.1).sum();
        unstored as f64 * w_max * branching_factor().powi(-(TAIL_STORE_MAX_MD as i32 + 1))
            / attempts as f64
    }

    /// (mean, batch SE) per weighting over the chosen side of the split.
    fn eta(&self, above_only: bool, batches: usize) -> [(f64, f64); 3] {
        use puzzle8::puzzle24::eta::batch_means;
        [0, 1, 2].map(|w| {
            let per: Vec<(u64, f64)> = self
                .chunks
                .values()
                .map(|c| (c.0, if above_only { c.3[w] } else { c.2[w] + c.3[w] }))
                .collect();
            batch_means(&per, batches)
        })
    }
}

/// Rescore the uniform tail run for `heuristic`, or `None` without one.
/// Boards with Manhattan distance above `stride.0` are scored one in
/// `stride.1` (by file position) and weighted by `stride.1`, an unbiased
/// estimate for heuristics too slow to score every stored board.
fn rescore_uniform_tail(
    dir: &Path,
    min_distance: u8,
    heuristic: HeuristicArg,
    split: u8,
    stride: (u8, u64),
) -> Result<Option<RescoredTail>, String> {
    use puzzle8::puzzle24::eta::samples::{for_each_sample, SampleRecord};
    use puzzle8::puzzle24::eta::unpack;
    let uniform = read_tail_chunks(&tail_chunks_path(dir, min_distance))?;
    if uniform.is_empty() {
        return Ok(None);
    }
    let b = branching_factor();
    let weights: Vec<[f64; 25]> = Weighting::ALL.iter().map(|&w| blank_weights(w)).collect();
    let mut chunks: BTreeMap<u64, (u64, u64, [f64; 3], [f64; 3])> = uniform
        .iter()
        .map(|c| {
            (
                c.chunk,
                (c.attempts, c.accepted - c.stored, [0.0; 3], [0.0; 3]),
            )
        })
        .collect();
    let t0 = Instant::now();
    let mut seen = 0u64;
    let mut buffer: Vec<SampleRecord> = Vec::with_capacity(1 << 20);
    type Sides = (u64, u64, [f64; 3], [f64; 3]);
    let (stride_md, stride) = (stride.0, stride.1.max(1));
    let mut flush = |buffer: &mut Vec<SampleRecord>, chunks: &mut BTreeMap<u64, Sides>| {
        let first = seen;
        let scored: Vec<(u64, bool, [f64; 3])> = buffer
            .par_iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let s = unpack(r.board);
                let md = ManhattanHeuristic.h(&s);
                let scale = if md <= stride_md {
                    1.0
                } else if (first + i as u64) % stride == 0 {
                    stride as f64
                } else {
                    return None;
                };
                let base = scale * b.powi(-(heuristic.h(&s) as i32));
                let blank = s.blank_pos() as usize;
                Some((
                    r.chunk as u64,
                    md > split,
                    [0, 1, 2].map(|w| weights[w][blank] * base),
                ))
            })
            .collect();
        for (chunk, above, x) in scored {
            let Some(entry) = chunks.get_mut(&chunk) else {
                continue;
            };
            let side = if above { &mut entry.3 } else { &mut entry.2 };
            for w in 0..3 {
                side[w] += x[w];
            }
        }
        seen += buffer.len() as u64;
        buffer.clear();
    };
    let path = tail_samples_path(dir, min_distance);
    for_each_sample(&path, |r| {
        buffer.push(r);
        if buffer.len() == buffer.capacity() {
            flush(&mut buffer, &mut chunks);
        }
    })
    .map_err(|e| format!("{}: {e}", path.display()))?;
    flush(&mut buffer, &mut chunks);
    eprintln!(
        "rescored {seen} stored uniform-tail boards for {} in {:.1?}",
        heuristic.name(),
        t0.elapsed()
    );
    Ok(Some(RescoredTail { split, chunks }))
}

/// Sampled tail estimate for one heuristic: (mean, batch SE) per weighting.
struct TailEstimate {
    attempts: u64,
    accepted: u64,
    thread_s: f64,
    eta: [(f64, f64); 3],
}

/// The uniform tail, scored from `rescored`'s stored boards.
fn read_tail(
    dir: &Path,
    min_distance: u32,
    batches: usize,
    rescored: &RescoredTail,
) -> Result<Option<TailEstimate>, String> {
    let Ok(md) = u8::try_from(min_distance) else {
        return Ok(None);
    };
    let chunks = read_tail_chunks(&tail_chunks_path(dir, md))?;
    if chunks.is_empty() {
        return Ok(None);
    }
    let eta = rescored.eta(false, batches);
    Ok(Some(TailEstimate {
        attempts: chunks.iter().map(|c| c.attempts).sum(),
        accepted: chunks.iter().map(|c| c.accepted).sum(),
        thread_s: chunks.iter().map(|c| c.thread_ns as f64).sum::<f64>() / 1e9,
        eta,
    }))
}

/// Version of the level sampler recorded in `tail_md_ge{L}.meta`. Bump it when
/// the draws for the same seed and chunk change.
const TAIL_MD_VERSION: &str = "1";

/// Draws per Manhattan level before draws are allocated by estimated error.
const TAIL_MD_PILOT_DRAWS: u64 = 96;

/// Target thread time of one level chunk, and the cap on its draws.
const TAIL_MD_CHUNK_NS: f64 = 20e9;
const TAIL_MD_MAX_CHUNK_DRAWS: u64 = 1 << 14;

fn tail_md_path(dir: &Path, min_distance: u8, ext: &str) -> PathBuf {
    dir.join(format!("tail_md_ge{min_distance}.{ext}"))
}

/// Draws at one Manhattan level: how many were drawn, were solvable, and were
/// proven at least `L` from GOAL (the hits, every one of them stored as a
/// board). `count` is the level's placement count, so a per-draw mean `x`
/// scales to `count/|V| · x` in eta.
#[derive(Clone, Default)]
struct LevelTally {
    count: f64,
    chunks: u64,
    draws: u64,
    solvable: u64,
    hits: u64,
    thread_ns: u64,
}

impl LevelTally {
    fn merge(&mut self, o: &LevelTally) {
        self.chunks += o.chunks;
        self.draws += o.draws;
        self.solvable += o.solvable;
        self.hits += o.hits;
        self.thread_ns += o.thread_ns;
    }

    /// The level's share of eta with every hit worth `value`, as
    /// (contribution, standard error): the hit rate is a per-draw Bernoulli.
    /// Used for Manhattan distance, where `value = b^−m` is the same for every
    /// hit; other heuristics score their stored hit boards ([`LevelMoments`]).
    fn eta_at(&self, value: f64) -> (f64, f64) {
        if self.draws < 2 {
            return (0.0, 0.0);
        }
        let n = self.draws as f64;
        let p = self.hits as f64 / n;
        let scale = self.count / N_STATES as f64;
        (
            scale * p * value,
            scale * value * (p * (1.0 - p) / (n - 1.0)).sqrt(),
        )
    }
}

fn read_tail_md_chunks(path: &Path) -> Result<BTreeMap<u8, LevelTally>, String> {
    let mut out: BTreeMap<u8, LevelTally> = BTreeMap::new();
    for (m, _, chunk) in read_tail_md_chunk_lines(path)? {
        let level = out.entry(m).or_default();
        level.count = chunk.count;
        level.merge(&chunk);
    }
    Ok(out)
}

/// The chunk lines of a `tail_md_ge{L}.tsv` as (level, chunk index, tally);
/// a line that does not parse (an interrupted append) ends the read.
fn read_tail_md_chunk_lines(path: &Path) -> Result<Vec<(u8, u64, LevelTally)>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        // Runs before the heuristic sums moved out of the file wrote twelve
        // more columns; they are ignored.
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 8 && f.len() != 20 {
            break;
        }
        let ints: Result<Vec<u64>, _> = f[1..7].iter().map(|x| x.parse::<u64>()).collect();
        let floats: Result<Vec<f64>, _> = f[7..8].iter().map(|x| x.parse::<f64>()).collect();
        let (Ok(ints), Ok(floats)) = (ints, floats) else {
            break;
        };
        let m = u8::try_from(ints[0]).map_err(|e| format!("{line}: {e}"))?;
        let chunk = LevelTally {
            count: floats[0],
            chunks: 1,
            draws: ints[2],
            solvable: ints[3],
            hits: ints[4],
            thread_ns: ints[5],
        };
        out.push((m, ints[1], chunk));
    }
    Ok(out)
}

/// Next round's chunks as (level, draws). Levels below their pilot draws come
/// first. After that, draws go toward the Neyman allocation n_m ∝ σ_m/√c_m
/// (c_m thread ns per draw) for the thread time spent after this round, with
/// σ_m the level's per-draw standard deviation of its Manhattan-distance
/// share of the tail, relative to the current total. A level's σ is at least
/// that of one hit in its next draw, so levels without hits keep being
/// sampled. Heuristics other than Manhattan distance are scored from the
/// stored hit boards afterwards, and gain from the same draws.
fn allocate_tail_md(levels: &BTreeMap<u8, LevelTally>, b: f64) -> Vec<(u8, u64)> {
    let round = SAMPLE_ROUND_CHUNKS as usize;
    let mut out: Vec<(u8, u64)> = levels
        .iter()
        .filter(|(_, t)| t.count > 0.0 && t.draws < TAIL_MD_PILOT_DRAWS)
        .map(|(&m, t)| (m, TAIL_MD_PILOT_DRAWS - t.draws))
        .take(round)
        .collect();
    if !out.is_empty() {
        return out;
    }
    let total: f64 = levels
        .iter()
        .map(|(&m, t)| t.eta_at(b.powi(-(m as i32))).0)
        .sum();
    // (level, σ, thread ns per draw, draws so far)
    let stats: Vec<(u8, f64, f64, u64)> = levels
        .iter()
        .filter(|(_, t)| t.count > 0.0)
        .map(|(&m, t)| {
            let n = t.draws as f64;
            let value = b.powi(-(m as i32));
            let one_hit = t.count / N_STATES as f64 * value / (n + 1.0).sqrt();
            let sd = if total > 0.0 {
                (t.eta_at(value).1 * n.sqrt()).max(one_hit) / total
            } else {
                0.0
            };
            (m, sd, t.thread_ns as f64 / n, t.draws)
        })
        .collect();
    let spent: f64 = levels.values().map(|t| t.thread_ns as f64).sum();
    let budget = spent + round as f64 * TAIL_MD_CHUNK_NS;
    let norm: f64 = stats.iter().map(|s| s.1 * s.2.sqrt()).sum();
    // (level, thread ns short of its target, thread ns per draw)
    let deficits: Vec<(u8, f64, f64)> = stats
        .iter()
        .map(|&(m, sd, cost, draws)| {
            let target = if norm > 0.0 {
                budget * sd / (cost.sqrt() * norm)
            } else {
                0.0
            };
            (m, (target - draws as f64).max(0.0) * cost, cost)
        })
        .collect();
    let total: f64 = deficits.iter().map(|d| d.1).sum();
    for &(m, deficit, cost) in &deficits {
        if deficit <= 0.0 {
            continue;
        }
        let chunks = (round as f64 * deficit / total).round() as usize;
        let draws = ((TAIL_MD_CHUNK_NS / cost).round() as u64)
            .clamp(1, TAIL_MD_MAX_CHUNK_DRAWS)
            .min(((deficit / cost).ceil() as u64).max(1));
        out.extend(std::iter::repeat_n((m, draws), chunks));
    }
    if out.is_empty() {
        let &(m, _, cost) = deficits
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least one level");
        let draws = ((TAIL_MD_CHUNK_NS / cost).round() as u64).clamp(1, TAIL_MD_MAX_CHUNK_DRAWS);
        out.push((m, draws));
    }
    out
}

struct TailMdCampaign {
    dir: PathBuf,
    min_distance: u8,
    md_max: u8,
    seed: u64,
    thread_seconds: f64,
}

/// `campaign tail-md`: level-stratified tail draws until the budget is met.
fn run_tail_md(c: TailMdCampaign, verifier: &Verifier) -> Result<(), String> {
    use puzzle8::puzzle24::eta::layers::pack;
    use puzzle8::puzzle24::eta::md_levels::MdLevels;
    use puzzle8::puzzle24::eta::samples::{append_samples, write_or_check_meta, SampleRecord};
    use std::io::Write;
    let (dir, l) = (c.dir.as_path(), c.min_distance);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let settings: BTreeMap<String, String> = [
        ("min_distance", l.to_string()),
        ("md_max", c.md_max.to_string()),
        ("seed", c.seed.to_string()),
        ("levels", TAIL_MD_VERSION.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    write_or_check_meta(&tail_md_path(dir, l, "meta"), &settings).map_err(|e| e.to_string())?;
    let t_build = Instant::now();
    let table = MdLevels::build(c.md_max);
    eprintln!(
        "level table: cap {}, {:.1} GiB in {:.1?}",
        c.md_max,
        table.table_bytes() as f64 / (1u64 << 30) as f64,
        t_build.elapsed()
    );
    let b = branching_factor();
    let chunks_file = tail_md_path(dir, l, "tsv");
    let mut levels = read_tail_md_chunks(&chunks_file)?;
    for m in 0..=c.md_max {
        let count = table.count(m);
        let level = levels.entry(m).or_default();
        if level.draws > 0 && (level.count - count).abs() > 1e-12 * count {
            return Err(format!(
                "level {m}: count {} on disk, {count} from the table",
                level.count
            ));
        }
        level.count = count;
    }
    let budget_ns = c.thread_seconds * 1e9;
    let t0 = Instant::now();
    loop {
        let spent: f64 = levels.values().map(|t| t.thread_ns as f64).sum();
        if spent >= budget_ns {
            break;
        }
        let mut next_chunk: BTreeMap<u8, u64> =
            levels.iter().map(|(&m, t)| (m, t.chunks)).collect();
        let jobs: Vec<(u8, u64, u64)> = allocate_tail_md(&levels, b)
            .into_iter()
            .map(|(m, draws)| {
                let chunk = next_chunk.get_mut(&m).expect("allocated level exists");
                *chunk += 1;
                (m, *chunk - 1, draws)
            })
            .collect();
        let results: Vec<(u8, u64, LevelTally, Vec<SampleRecord>)> = jobs
            .into_par_iter()
            .map(|(m, chunk, draws)| {
                let inc = verifier.inc();
                let stream = (1 << 20) | (l as u64) << 8 | m as u64;
                let mut rng = Rng::stream(c.seed, stream, chunk);
                let start = Instant::now();
                let mut t = LevelTally {
                    count: table.count(m),
                    chunks: 1,
                    draws,
                    ..LevelTally::default()
                };
                let mut hits = Vec::new();
                for _ in 0..draws {
                    let s = table.sample(m, &mut rng);
                    if !s.is_solvable() {
                        continue;
                    }
                    t.solvable += 1;
                    if !at_least(&s, l, &inc) {
                        continue;
                    }
                    t.hits += 1;
                    hits.push(SampleRecord {
                        board: pack(&s),
                        prob: 1.0,
                        chunk: u32::try_from(chunk).expect("chunk index fits u32"),
                    });
                }
                t.thread_ns = start.elapsed().as_nanos() as u64;
                (m, chunk, t, hits)
            })
            .collect();
        let hits: Vec<SampleRecord> = results.iter().flat_map(|r| r.3.iter().copied()).collect();
        append_samples(&tail_md_path(dir, l, "hits"), &hits).map_err(|e| e.to_string())?;
        let fresh = !chunks_file.exists();
        let mut text = String::new();
        if fresh {
            text.push_str("# seed\tm\tchunk\tdraws\tsolvable\thits\tthread_ns\tcount\n");
        }
        for (m, chunk, t, _) in &results {
            text.push_str(&format!(
                "{}\t{m}\t{chunk}\t{}\t{}\t{}\t{}\t{:.17e}\n",
                c.seed, t.draws, t.solvable, t.hits, t.thread_ns, t.count
            ));
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&chunks_file)
            .and_then(|mut f| f.write_all(text.as_bytes()))
            .map_err(|e| format!("{}: {e}", chunks_file.display()))?;
        for (m, _, t, _) in &results {
            levels.get_mut(m).expect("level exists").merge(t);
        }
        let spent: f64 = levels.values().map(|t| t.thread_ns as f64).sum();
        let (sum, var) = levels.iter().fold((0.0, 0.0), |(s, v), (&m, t)| {
            let (e, se) = t.eta_at(b.powi(-(m as i32)));
            (s + e, v + se * se)
        });
        eprintln!(
            "tail-md >= {l}: {} chunks, {} hits this round; levels 0..={} md {sum:.4e} ± {:.1}%; {:.0} of {:.0} thread-s; wall {:.0}s",
            results.len(),
            hits.len(),
            c.md_max,
            100.0 * Z95 * var.sqrt() / sum,
            spent / 1e9,
            budget_ns / 1e9,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Level-stratified tail for one heuristic: the levels m ≤ md_max, and the
/// uniform tail restricted to m > md_max.
struct TailMdEstimate {
    md_max: u8,
    /// Per level: its draws, and the moments of the scored heuristic over them
    /// (hits contribute w(blank)·b^−h, other draws zero).
    levels: BTreeMap<u8, (LevelTally, Moments)>,
    /// (mean, batch SE) per weighting of the uniform tail's draws with Manhattan
    /// distance above md_max, when a uniform tail is present.
    rest: Option<[(f64, f64); 3]>,
}

impl TailMdEstimate {
    /// (contribution to eta, standard error) of one level under weighting `w`.
    fn eta(&self, level: &(LevelTally, Moments), w: usize) -> (f64, f64) {
        let (tally, moments) = level;
        if moments.draws < 2 {
            return (0.0, 0.0);
        }
        let scale = tally.count / N_STATES as f64;
        let (mean, se) = moments.mean(w);
        (scale * mean, scale * se)
    }
}

/// Manhattan-distance cap of a level-stratified tail campaign, if one exists.
fn tail_md_max(dir: &Path, min_distance: u8) -> Result<Option<u8>, String> {
    use puzzle8::puzzle24::eta::samples::read_meta;
    let meta_path = tail_md_path(dir, min_distance, "meta");
    if !meta_path.exists() {
        return Ok(None);
    }
    let meta = read_meta(&meta_path).map_err(|e| format!("{}: {e}", meta_path.display()))?;
    meta.get("md_max")
        .and_then(|v| v.parse().ok())
        .map(Some)
        .ok_or_else(|| format!("{}: no md_max", meta_path.display()))
}

/// The level-stratified tail for `heuristic`: each level's stored hit boards
/// scored (every hit is stored), and the part above md_max from `rescored`,
/// which must be split at md_max.
fn read_tail_md(
    dir: &Path,
    min_distance: u32,
    heuristic: HeuristicArg,
    batches: usize,
    rescored: &RescoredTail,
) -> Result<Option<TailMdEstimate>, String> {
    use puzzle8::puzzle24::eta::samples::read_samples;
    use puzzle8::puzzle24::eta::unpack;
    let Ok(l) = u8::try_from(min_distance) else {
        return Ok(None);
    };
    let tallies = read_tail_md_chunks(&tail_md_path(dir, l, "tsv"))?;
    if tallies.is_empty() {
        return Ok(None);
    }
    let md_max = tail_md_max(dir, l)?.ok_or("level tail without its meta file")?;
    if md_max > TAIL_STORE_MAX_MD {
        return Err(format!(
            "levels end at MD {md_max}, above the uniform tail's stored boards (MD <= {TAIL_STORE_MAX_MD}), so the part above the levels cannot be scored"
        ));
    }
    if rescored.split != md_max {
        return Err(format!(
            "uniform tail rescored at MD {} but the levels end at {md_max}",
            rescored.split
        ));
    }
    let lines: std::collections::BTreeSet<(u8, u64)> =
        read_tail_md_chunk_lines(&tail_md_path(dir, l, "tsv"))?
            .into_iter()
            .map(|(m, chunk, _)| (m, chunk))
            .collect();
    let b = branching_factor();
    let weights: Vec<[f64; 25]> = Weighting::ALL.iter().map(|&w| blank_weights(w)).collect();
    let mut levels: BTreeMap<u8, (LevelTally, Moments)> = tallies
        .into_iter()
        .map(|(m, t)| {
            let draws = t.draws;
            (
                m,
                (
                    t,
                    Moments {
                        draws,
                        ..Moments::default()
                    },
                ),
            )
        })
        .collect();
    let hits_path = tail_md_path(dir, l, "hits");
    let hits = read_samples(&hits_path).map_err(|e| format!("{}: {e}", hits_path.display()))?;
    let scored: Vec<(u8, u64, usize, u8)> = hits
        .par_iter()
        .map(|r| {
            let s = unpack(r.board);
            (
                ManhattanHeuristic.h(&s),
                r.chunk as u64,
                s.blank_pos() as usize,
                heuristic.h(&s),
            )
        })
        .collect();
    for (m, chunk, blank, h) in scored {
        if !lines.contains(&(m, chunk)) {
            continue;
        }
        let (_, moments) = levels.get_mut(&m).expect("a hit's level has chunk lines");
        let base = b.powi(-(h as i32));
        for w in 0..3 {
            let x = weights[w][blank] * base;
            moments.sums[w] += x;
            moments.squares[w] += x * x;
        }
    }
    Ok(Some(TailMdEstimate {
        md_max,
        levels,
        rest: Some(rescored.eta(true, batches)),
    }))
}

/// Draws per chunk in `campaign total`: large for heuristic lookups alone,
/// small when each draw also proves its distance, so levels still spread
/// over all threads.
const TOTAL_CHUNK: u64 = 1 << 16;
const TOTAL_BAND_CHUNK: u64 = 64;

/// Σx and Σx² per weighting over `draws` draws.
#[derive(Clone, Copy, Default)]
struct Moments {
    draws: u64,
    sums: [f64; 3],
    squares: [f64; 3],
}

impl Moments {
    fn merge(mut self, o: Moments) -> Moments {
        self.draws += o.draws;
        for w in 0..3 {
            self.sums[w] += o.sums[w];
            self.squares[w] += o.squares[w];
        }
        self
    }

    /// (mean, standard error) for weighting `w`.
    fn mean(&self, w: usize) -> (f64, f64) {
        let n = self.draws as f64;
        let mean = self.sums[w] / n;
        let var = (self.squares[w] / n - mean * mean).max(0.0) * n / (n - 1.0);
        (mean, (var / n).sqrt())
    }
}

/// Moments of x = w(blank)·b^−h(s) over `draws` draws, a draw that yields
/// `None` counting as zero, in parallel chunks of `chunk_draws`. Each chunk
/// starts a cursor (a generator, or an index) with `start(chunk)` and takes
/// its draws from it in order.
fn moments<C>(
    draws: u64,
    chunk_draws: u64,
    heuristic: HeuristicArg,
    start: impl Fn(u64) -> C + Sync,
    draw: impl Fn(&mut C) -> Option<State> + Sync,
) -> Moments {
    let b = branching_factor();
    let weights: Vec<[f64; 25]> = Weighting::ALL.iter().map(|&w| blank_weights(w)).collect();
    (0..draws.div_ceil(chunk_draws))
        .into_par_iter()
        .map(|chunk| {
            let mut rng = start(chunk);
            let n = chunk_draws.min(draws - chunk * chunk_draws);
            let mut m = Moments {
                draws: n,
                ..Moments::default()
            };
            for _ in 0..n {
                let Some(s) = draw(&mut rng) else {
                    continue;
                };
                let base = b.powi(-(heuristic.h(&s) as i32));
                let blank = s.blank_pos() as usize;
                for w in 0..3 {
                    let x = weights[w][blank] * base;
                    m.sums[w] += x;
                    m.squares[w] += x * x;
                }
            }
            m
        })
        .reduce(Moments::default, Moments::merge)
}

/// `campaign total`. `band` = (least, greatest) distance counted; distances
/// are proven with the verifier's zero-aware PDBs when the band is not
/// (0, None).
fn run_total(
    heuristic: HeuristicArg,
    md_max: u8,
    level_draws: u64,
    uniform_draws: u64,
    seed: u64,
    band: (u8, Option<u8>),
    verifier: Option<&Verifier>,
) -> Result<(), String> {
    use puzzle8::puzzle24::eta::md_levels::MdLevels;
    use puzzle8::puzzle24::eta::{pack, unpack};
    if level_draws < 2 || uniform_draws < 2 {
        return Err("campaign total needs at least 2 draws per stratum".into());
    }
    if band.1.is_some_and(|hi| hi == u8::MAX) {
        return Err("--distance-max must be below 255".into());
    }
    let in_band = |s: &State| -> bool {
        let Some(verifier) = verifier else {
            return true;
        };
        let inc = verifier.inc();
        at_least(s, band.0, &inc) && band.1.is_none_or(|hi| !at_least(s, hi + 1, &inc))
    };
    let t0 = Instant::now();
    let table = MdLevels::build(md_max);
    eprintln!(
        "level table: cap {md_max}, {:.1} GiB in {:.1?}",
        table.table_bytes() as f64 / (1u64 << 30) as f64,
        t0.elapsed()
    );
    let n_states = N_STATES as f64;
    let chunk_draws = if band == (0, None) {
        TOTAL_CHUNK
    } else {
        TOTAL_BAND_CHUNK
    };
    // Draw every level first and free the level table before the heuristic
    // loads its own tables, so the two never occupy memory together.
    let drawn: Vec<(u8, f64, Vec<u128>)> = (0..=md_max)
        .filter(|&m| table.count(m) > 0.0)
        .map(|m| {
            let boards = (0..level_draws.div_ceil(chunk_draws))
                .into_par_iter()
                .flat_map_iter(|chunk| {
                    let mut rng = Rng::stream(seed, (2 << 20) | m as u64, chunk);
                    let n = chunk_draws.min(level_draws - chunk * chunk_draws);
                    let table = &table;
                    (0..n).map(move |_| pack(&table.sample(m, &mut rng)))
                })
                .collect();
            (m, table.count(m), boards)
        })
        .collect();
    drop(table);
    eprintln!(
        "level draws: {} boards in {:.1?}",
        drawn.iter().map(|d| d.2.len()).sum::<usize>(),
        t0.elapsed()
    );
    heuristic.prepare();
    let band_text = match band {
        (0, None) => String::from("all distances"),
        (lo, None) => format!("distance >= {lo}"),
        (lo, Some(hi)) => format!("distance {lo}..={hi}"),
    };
    let rest_empty = band.1.is_some_and(|hi| hi <= md_max);
    println!(
        "# eta of {} over {band_text} by Manhattan levels 0..={md_max} ({level_draws} draws each) and uniform draws above ({}); seed {seed}; errors 95%",
        heuristic.name(),
        if rest_empty {
            String::from("none: Manhattan distance never exceeds distance")
        } else {
            uniform_draws.to_string()
        }
    );
    println!("#  m   count/|V|    eta_m uniform ± 95%       eta_m tree    eta_m degree");
    let mut levels = [(0.0f64, 0.0f64); 3];
    for (m, count, boards) in &drawn {
        let scale = count / n_states;
        let mo = moments(
            boards.len() as u64,
            chunk_draws,
            heuristic,
            |chunk| (chunk * chunk_draws) as usize,
            |i| {
                let s = unpack(boards[*i]);
                *i += 1;
                Some(s).filter(|s| s.is_solvable() && in_band(s))
            },
        );
        let est = [0, 1, 2].map(|w| {
            let (mean, se) = mo.mean(w);
            (scale * mean, scale * se)
        });
        for w in 0..3 {
            levels[w].0 += est[w].0;
            levels[w].1 += est[w].1 * est[w].1;
        }
        println!(
            "  {m:>2}  {:.4e}   {:.4e} ± {:>6}   {:.4e}    {:.4e}",
            scale,
            est[0].0,
            if est[0].0 > 0.0 {
                format!("{:.2}%", 100.0 * Z95 * est[0].1 / est[0].0)
            } else {
                String::from("-")
            },
            est[1].0,
            est[2].0
        );
    }
    let t_uniform = Instant::now();
    let rest = if rest_empty {
        Moments {
            draws: 2,
            ..Moments::default()
        }
    } else {
        moments(
            uniform_draws,
            chunk_draws,
            heuristic,
            |chunk| Rng::stream(seed, 3 << 20, chunk),
            |rng| {
                Some(uniform_solvable(rng))
                    .filter(|s| ManhattanHeuristic.h(s) > md_max && in_band(s))
            },
        )
    };
    eprintln!("uniform draws: {:.1?}", t_uniform.elapsed());
    for (w, name) in ["uniform", "tree", "degree"].iter().enumerate() {
        let (lm, lse) = (levels[w].0, levels[w].1.sqrt());
        let (rm, rse) = rest.mean(w);
        let total = lm + rm;
        let se = (lse * lse + rse * rse).sqrt();
        let mut line = format!(
            "  {name:<7}: levels MD <= {md_max} = {lm:.4e} ± {:.2e}; uniform MD > {md_max} = {rm:.4e} ± {:.2e}; total = {total:.4e} ± {:.2e} ({:.2}%)",
            Z95 * lse,
            Z95 * rse,
            Z95 * se,
            100.0 * Z95 * se / total
        );
        if let (Some(exact), (0, None)) = (heuristic.exact_total(), band) {
            line += &format!(
                " vs exact {:.4e}: {:+.3}% = {:+.1} SE",
                exact[w],
                100.0 * (total - exact[w]) / exact[w],
                (total - exact[w]) / se
            );
        }
        println!("{line}");
    }
    println!("# wall {:.1?}", t0.elapsed());
    Ok(())
}

fn report_tail_md_levels(t: &TailMdEstimate, min_distance: u32) {
    let total: f64 = t.levels.values().map(|x| t.eta(x, 0).0).sum();
    let draws: u64 = t.levels.values().map(|(x, _)| x.draws).sum();
    let thread_s: f64 = t
        .levels
        .values()
        .map(|(x, _)| x.thread_ns as f64)
        .sum::<f64>()
        / 1e9;
    println!(
        "# level-stratified tail d >= {min_distance}, Manhattan levels 0..={}: {draws} draws, {thread_s:.0} thread-s",
        t.md_max
    );
    println!("#  m   count/|V|    draws  solvable     hits  hit rate  thread-s   eta_m uniform ± 95%     share");
    for (m, level) in &t.levels {
        let (e, se) = t.eta(level, 0);
        let x = &level.0;
        println!(
            "  {m:>2}  {:.4e} {:>8} {:>9} {:>8}  {:>8.4} {:>9.0}   {e:.4e} ± {:>6}  {:>5.1}%",
            x.count / N_STATES as f64,
            x.draws,
            x.solvable,
            x.hits,
            x.hits as f64 / x.solvable.max(1) as f64,
            x.thread_ns as f64 / 1e9,
            if e > 0.0 {
                format!("{:.1}%", 100.0 * Z95 * se / e)
            } else {
                String::from("-")
            },
            100.0 * e / total
        );
    }
}

/// Stratum estimate: (mean, batch standard error) per quantity.
struct StratumScore {
    k: u32,
    attempts: u64,
    accepted: u64,
    size: (f64, f64),
    size_ess: f64,
    eta: [(f64, f64); 3],
    eta_ess: [f64; 3],
    eta_max_share: f64,
    exact: bool,
}

/// `campaign score`: combine exact layers, sampled strata and, if present, the
/// sampled tail, for one heuristic.
fn run_score(
    dir: &Path,
    heuristic: HeuristicArg,
    batches: usize,
    published: Option<&Path>,
    rescore_stride: (u8, u64),
) -> Result<(), String> {
    use puzzle8::puzzle24::eta::samples::{chunks_path, read_chunks, read_samples, samples_path};
    use puzzle8::puzzle24::eta::{batch_means, unpack};
    heuristic.prepare();
    let b = branching_factor();
    let weights: Vec<[f64; 25]> = Weighting::ALL.iter().map(|&w| blank_weights(w)).collect();
    let n_states = N_STATES as f64;

    let mut strata: Vec<StratumScore> = Vec::new();
    let exact_path = exact_layers_path(dir, heuristic);
    let exact_text = std::fs::read_to_string(&exact_path).map_err(|e| {
        format!(
            "{}: {e} (run `campaign layers` first)",
            exact_path.display()
        )
    })?;
    for line in exact_text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        let k: u32 = f[0].parse().map_err(|e| format!("{line}: {e}"))?;
        let size: f64 = f[1].parse().map_err(|e| format!("{line}: {e}"))?;
        let eta: Vec<f64> = f[2..5].iter().map(|x| x.parse().unwrap()).collect();
        strata.push(StratumScore {
            k,
            attempts: 0,
            accepted: 0,
            size: (size, 0.0),
            size_ess: f64::NAN,
            eta: [(eta[0], 0.0), (eta[1], 0.0), (eta[2], 0.0)],
            eta_ess: [f64::NAN; 3],
            eta_max_share: f64::NAN,
            exact: true,
        });
    }
    let exact_max = strata.iter().map(|s| s.k).max().unwrap_or(0);

    for k in exact_max + 1..=200 {
        let chunks = read_chunks(&chunks_path(dir, k)).map_err(|e| e.to_string())?;
        if chunks.is_empty() {
            if k > exact_max + 1 && strata.last().is_some_and(|s| s.k < k - 1) {
                break;
            }
            continue;
        }
        let mut per_chunk: std::collections::BTreeMap<u64, (u64, [f64; 4])> = chunks
            .iter()
            .map(|r| (r.chunk, (r.attempts, [0.0; 4])))
            .collect();
        let (mut size_acc, mut eta_acc) = (Accum::default(), [Accum::default(); 3]);
        let mut accepted = 0u64;
        let records = read_samples(&samples_path(dir, k)).map_err(|e| e.to_string())?;
        let hs: Vec<u8> = records
            .par_iter()
            .map(|s| heuristic.h(&unpack(s.board)))
            .collect();
        for (s, &h) in records.iter().zip(&hs) {
            let Some(entry) = per_chunk.get_mut(&(s.chunk as u64)) else {
                continue;
            };
            accepted += 1;
            let board = unpack(s.board);
            let base = b.powi(-(h as i32)) / s.prob;
            let blank = board.blank_pos() as usize;
            entry.1[0] += 1.0 / s.prob;
            size_acc.add(1.0 / s.prob);
            for w in 0..3 {
                let x = weights[w][blank] * base;
                entry.1[w + 1] += x;
                eta_acc[w].add(x);
            }
        }
        let attempts: u64 = per_chunk.values().map(|v| v.0).sum();
        let column =
            |i: usize| -> Vec<(u64, f64)> { per_chunk.values().map(|v| (v.0, v.1[i])).collect() };
        let size = batch_means(&column(0), batches);
        let eta = [1, 2, 3].map(|i| {
            let (m, se) = batch_means(&column(i), batches);
            (m / n_states, se / n_states)
        });
        strata.push(StratumScore {
            k,
            attempts,
            accepted,
            size,
            size_ess: size_acc.effective_n(),
            eta,
            eta_ess: [0, 1, 2].map(|w| eta_acc[w].effective_n()),
            eta_max_share: eta_acc[0].max_share(),
            exact: false,
        });
    }

    println!(
        "# heuristic {}, campaign {}; b = {b:.12}; errors are 95% batch-means intervals ({batches} batches)",
        heuristic.name(),
        dir.display()
    );
    println!("#  k  source    attempts  accepted   |V_k|                     vs reference              eta_k uniform              (ESS, top term)   eta_k tree    eta_k degree");
    for s in &strata {
        let reference = reference_sphere_size(s.k).map_or(String::from("-"), |(r, src)| {
            if s.exact {
                format!("{:+.0e} ({src})", s.size.0 - r)
            } else {
                format!(
                    "{:+.2}% = {:+.1} SE ({src})",
                    100.0 * (s.size.0 - r) / r,
                    (s.size.0 - r) / s.size.1
                )
            }
        });
        let pm = |(m, se): (f64, f64)| {
            if s.exact {
                format!("{m:.4e} exact")
            } else {
                format!("{m:.4e} ± {:.1}%", 100.0 * Z95 * se / m)
            }
        };
        println!(
            "  {:>2}  {:<7} {:>10} {:>9}   {:<24} {:<25} {:<26} ({:>6.0}, {:>5.1}%)  {:<22} {}",
            s.k,
            if s.exact { "layers" } else { "sampled" },
            s.attempts,
            s.accepted,
            if s.exact {
                pm(s.size)
            } else {
                format!("{} ESS {:.0}", pm(s.size), s.size_ess)
            },
            reference,
            pm(s.eta[0]),
            s.eta_ess[0],
            100.0 * s.eta_max_share,
            pm(s.eta[1]),
            pm(s.eta[2])
        );
    }

    let k_max = strata.iter().map(|s| s.k).max().unwrap_or(0);
    let contiguous = strata.iter().enumerate().all(|(i, s)| s.k as usize == i);
    println!();
    println!(
        "# strata 0..={k_max} present{}",
        if contiguous {
            ""
        } else {
            " WITH GAPS — sums below are incomplete"
        }
    );
    let rescored = match u8::try_from(k_max + 1) {
        Ok(l) => rescore_uniform_tail(
            dir,
            l,
            heuristic,
            tail_md_max(dir, l)?.unwrap_or(0),
            rescore_stride,
        )?,
        Err(_) => None,
    };
    let tail = match &rescored {
        Some(r) => read_tail(dir, k_max + 1, batches, r)?,
        None => None,
    };
    if let Some(t) = &tail {
        println!(
            "# sampled tail d >= {}: {} uniform draws, {} accepted, {:.0} thread-s",
            k_max + 1,
            t.attempts,
            t.accepted,
            t.thread_s
        );
    }
    if let Some(r) = &rescored {
        println!(
            "# {} rescored from the uniform tail's stored boards (MD <= {TAIL_STORE_MAX_MD}); unstored accepted boards add at most {:.2e}",
            heuristic.name(),
            r.unstored_bound()
        );
    }
    let tail_md = match &rescored {
        Some(r) => read_tail_md(dir, k_max + 1, heuristic, batches, r)?,
        None => None,
    };
    if let Some(t) = &tail_md {
        report_tail_md_levels(t, k_max + 1);
    }
    for (w, name) in ["uniform", "tree", "degree"].iter().enumerate() {
        let sum: f64 = strata.iter().map(|s| s.eta[w].0).sum();
        let se = strata
            .iter()
            .map(|s| s.eta[w].1.powi(2))
            .sum::<f64>()
            .sqrt();
        let mut line = format!(
            "  {name:<7}: sum eta_0..{k_max} = {sum:.4e} ± {:.2e} ({:.1}%)",
            Z95 * se,
            100.0 * Z95 * se / sum
        );
        if let Some(exact) = heuristic.exact_total() {
            line += &format!(
                "; exact total {:.4e}; tail by subtraction eta_>={} = {:.4e} ± {:.2e}",
                exact[w],
                k_max + 1,
                exact[w] - sum,
                Z95 * se
            );
        }
        if let Some(t) = &tail {
            let (tm, tse) = t.eta[w];
            let total_se = (se * se + tse * tse).sqrt();
            line += &format!(
                "; sampled tail {tm:.4e} ± {:.2e} ({:.1}%); total = sum + sampled tail = {:.4e} ± {:.2e} ({:.1}%)",
                Z95 * tse,
                100.0 * Z95 * tse / tm,
                sum + tm,
                Z95 * total_se,
                100.0 * Z95 * total_se / (sum + tm)
            );
            if let Some(exact) = heuristic.exact_total() {
                line += &format!(
                    " vs exact {:+.2}% = {:+.1} SE",
                    100.0 * (sum + tm - exact[w]) / exact[w],
                    (sum + tm - exact[w]) / total_se
                );
            }
        }
        println!("{line}");
        if let Some(t) = &tail_md {
            let (lm, lvar) = t.levels.values().fold((0.0f64, 0.0f64), |(s, v), level| {
                let (e, e_se) = t.eta(level, w);
                (s + e, v + e_se * e_se)
            });
            let mut line = format!(
                "  {name:<7}: levels MD <= {} = {lm:.4e} ± {:.2e} ({:.1}%)",
                t.md_max,
                Z95 * lvar.sqrt(),
                100.0 * Z95 * lvar.sqrt() / lm
            );
            if let Some(rest) = &t.rest {
                let (rm, rse) = rest[w];
                let tail_m = lm + rm;
                let tail_se = (lvar + rse * rse).sqrt();
                let total = sum + tail_m;
                let total_se = (se * se + tail_se * tail_se).sqrt();
                line += &format!(
                    "; uniform draws MD > {} = {rm:.4e} ± {:.2e}; tail = {tail_m:.4e} ± {:.2e} ({:.1}%); total = sum + tail = {total:.4e} ± {:.2e} ({:.1}%)",
                    t.md_max,
                    Z95 * rse,
                    Z95 * tail_se,
                    100.0 * Z95 * tail_se / tail_m,
                    Z95 * total_se,
                    100.0 * Z95 * total_se / total
                );
                if let Some(exact) = heuristic.exact_total() {
                    line += &format!(
                        " vs exact {:+.2}% = {:+.1} SE",
                        100.0 * (total - exact[w]) / exact[w],
                        (total - exact[w]) / total_se
                    );
                }
            }
            println!("{line}");
        }
    }
    if heuristic == HeuristicArg::Md {
        println!(
            "# published (thesis Table 5.1, Fig. 5.3): eta = {PUBLISHED_MD_ETA:.3e} ± {PUBLISHED_MD_ETA_HALF95:.3e}; eta_>=65 = {PUBLISHED_MD_ETA_GE65:.3e}"
        );
    }

    if let Some(path) = published {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let paper: std::collections::BTreeMap<u32, f64> = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                (f[0].parse().unwrap(), f[2].parse().unwrap())
            })
            .collect();
        println!();
        println!(
            "# per-stratum eta_k (uniform) against {}; SE = batch standard error",
            path.display()
        );
        println!("#  k   ours          ± 95%       paper        ours/paper   (ours - paper)/SE");
        for s in strata.iter().filter(|s| paper.contains_key(&s.k)) {
            let p = paper[&s.k];
            let (m, se) = s.eta[0];
            let z = if s.exact || se == 0.0 {
                String::from("exact")
            } else {
                format!("{:+.1}", (m - p) / se)
            };
            println!(
                "  {:>2}   {m:.4e}   {:>6}   {p:.4e}   {:>6.3}       {z}",
                s.k,
                if s.exact {
                    String::from("exact")
                } else {
                    format!("{:.1}%", 100.0 * Z95 * se / m)
                },
                if p > 0.0 { m / p } else { f64::NAN }
            );
        }
        println!("#  band      ours                 paper        ours/paper   (ours - paper)/SE");
        for (lo, hi) in [
            (0, 29),
            (30, 39),
            (40, 49),
            (50, 54),
            (55, 59),
            (60, 64),
            (0, 64),
        ] {
            let band: Vec<&StratumScore> = strata
                .iter()
                .filter(|s| (lo..=hi).contains(&s.k) && paper.contains_key(&s.k))
                .collect();
            if band.is_empty() {
                continue;
            }
            let ours: f64 = band.iter().map(|s| s.eta[0].0).sum();
            let se = band.iter().map(|s| s.eta[0].1.powi(2)).sum::<f64>().sqrt();
            let theirs: f64 = band.iter().map(|s| paper[&s.k]).sum();
            println!(
                "  {lo:>2}..{hi:<2}   {ours:.4e} ± {:>5.1}%   {theirs:.4e}   {:>6.3}       {:+.1}",
                100.0 * Z95 * se / ours,
                ours / theirs,
                (ours - theirs) / se
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match args.cmd {
        Cmd::Campaign(Campaign::Layers {
            max_k,
            dir,
            heuristic,
        }) => run_layers(max_k, &dir, heuristic),
        Cmd::Campaign(Campaign::Tail {
            dir,
            min_distance,
            seed,
            thread_seconds,
            verifier,
        }) => Verifier::load(&verifier)
            .and_then(|v| run_tail(&dir, min_distance, seed, thread_seconds, &v)),
        Cmd::Campaign(Campaign::Total {
            heuristic,
            md_max,
            level_draws,
            uniform_draws,
            seed,
            distance_min,
            distance_max,
            verifier,
        }) => {
            let band = (distance_min, distance_max);
            if band == (0, None) {
                run_total(
                    heuristic,
                    md_max,
                    level_draws,
                    uniform_draws,
                    seed,
                    band,
                    None,
                )
            } else {
                Verifier::load(&verifier).and_then(|v| {
                    run_total(
                        heuristic,
                        md_max,
                        level_draws,
                        uniform_draws,
                        seed,
                        band,
                        Some(&v),
                    )
                })
            }
        }
        Cmd::Campaign(Campaign::TailMd {
            dir,
            min_distance,
            md_max,
            seed,
            thread_seconds,
            verifier,
        }) => Verifier::load(&verifier).and_then(|v| {
            run_tail_md(
                TailMdCampaign {
                    dir,
                    min_distance,
                    md_max,
                    seed,
                    thread_seconds,
                },
                &v,
            )
        }),
        Cmd::Campaign(Campaign::Sample {
            dir,
            k_min,
            k_max,
            seed,
            thread_seconds,
            growth,
            max_thread_seconds,
            window,
            check_every,
            verifier,
        }) => Verifier::load(&verifier).and_then(|v| {
            run_sample(
                SampleCampaign {
                    dir,
                    k_min,
                    k_max,
                    seed,
                    thread_seconds,
                    growth,
                    max_thread_seconds,
                    window,
                    check_every,
                },
                &v,
            )
        }),
        Cmd::Campaign(Campaign::Score {
            dir,
            heuristic,
            batches,
            published_histogram,
            rescore_full_md,
            rescore_stride,
        }) => run_score(
            &dir,
            heuristic,
            batches,
            published_histogram.as_deref(),
            (rescore_full_md, rescore_stride),
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
