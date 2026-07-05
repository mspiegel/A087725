//! The adversarial co-training loop.
//!
//! Each round: (1) train the solver's value net for `solver_steps_per_round`
//! DAVI steps on a batch that mixes uniform-random scrambles with boards from
//! the current (frozen) generator; then (2) freeze the solver and train the
//! generator for `generator_steps_per_round` REINFORCE rounds against it.
//! Periodically evaluate (mid-depth excess-over-optimal, or deep
//! excess-over-beam — see `EvalSpec`) and checkpoint both networks.

use std::path::PathBuf;

use candle_core::{Device, Result};

use super::checkpoint;
use super::davi::{Davi, DaviConfig};
use super::eval::{self, DeepEvalConfig, EvalConfig};
use super::generator::{Generator, GeneratorConfig};
use super::scramble::{scramble, Rng};
use crate::puzzle24::state::State;

/// Which in-loop eval to run each checkpoint: mid-depth (excess-over-optimal via
/// IDA\*, feasible ≤~depth 50) or deep (excess-over-beam, for the scale-to-`R`
/// regime where no optimal solver exists).
pub enum EvalSpec {
    MidDepth(EvalConfig),
    Deep(DeepEvalConfig),
}

/// Linear walk-length ramp — the depth curriculum. `k` grows from `k_start`
/// (round 0) to `k_end` (final round), pacing both generator board depth and
/// uniform-scramble depth to the solver's learned frontier. Deep boards that
/// outrun the frontier give DAVI no usable bootstrap gradient, so the ramp keeps
/// depth just ahead of what the solver has learned.
pub struct Curriculum {
    pub k_start: u32,
    pub k_end: u32,
}

/// The ramped walk length `k` for `round` of `rounds` (linear, endpoints
/// inclusive; `round=0`→`k_start`, `round=rounds-1`→`k_end`), clamped ≥ 1.
fn ramped_k(c: &Curriculum, round: u32, rounds: u32) -> u32 {
    if rounds <= 1 {
        return c.k_end.max(1);
    }
    let f = round as f64 / (rounds - 1) as f64;
    (c.k_start as f64 + (c.k_end as f64 - c.k_start as f64) * f).round().max(1.0) as u32
}

pub struct AlternationConfig {
    pub rounds: u32,
    pub solver_steps_per_round: u32,
    pub generator_steps_per_round: u32,
    /// States per DAVI step.
    pub solver_batch: usize,
    /// Fraction of each solver batch drawn from the generator (rest uniform).
    pub generator_frac: f32,
    pub davi: DaviConfig,
    pub generator: GeneratorConfig,
    /// Evaluate + checkpoint every this many rounds (0 = only at the end).
    pub eval_every: u32,
    pub eval: EvalSpec,
    /// Optional depth curriculum (walk-length ramp). `None` = fixed `k`
    /// (`generator.k_max` / `davi.k_max` unchanged across rounds).
    pub curriculum: Option<Curriculum>,
    pub checkpoint_dir: PathBuf,
    pub seed: u64,
    pub verbose: bool,
    /// If true, load `value_latest`/`policy_latest` from `checkpoint_dir` (if
    /// present) into the networks before training, continuing from those
    /// weights. Round numbering and the generator's EMA baseline still reset.
    pub resume: bool,
    /// On resume, keep the trained solver but reset the generator to a fresh
    /// (random-init) policy instead of loading `policy_latest`. Use when
    /// switching the generator reward (a collapsed policy from the old reward is
    /// a poor, low-entropy start). Ignored unless `resume`.
    pub reset_generator: bool,
}

const MID_METRICS_HEADER: &str =
    "round\tsolver_loss\tgen_reward\tgen_baseline\tholdout_solved\tholdout_mean_len\toptimal_labeled\tmean_excess";
const DEEP_METRICS_HEADER: &str =
    "round\tsolver_loss\tgen_reward\tgen_baseline\tlearned_solved\tbeam_solved\tmean_len_learned\tmean_len_beam\tmean_excess_beam\tlearned_wins\tr_learned/beam";

/// Run the full co-training loop. Returns after `rounds` rounds, having written
/// checkpoints and metrics to `checkpoint_dir`.
pub fn run(cfg: &AlternationConfig, device: Device) -> Result<()> {
    let mut davi = Davi::new(&cfg.davi, device.clone())?;
    let mut generator = Generator::new(&cfg.generator, device)?;
    let mut rng = Rng::new(cfg.seed);

    if cfg.resume {
        let vpath = checkpoint::value_latest_path(&cfg.checkpoint_dir);
        let ppath = checkpoint::policy_latest_path(&cfg.checkpoint_dir);
        if vpath.exists() {
            davi.load_online(&vpath)?;
            let loaded_gen = if !cfg.reset_generator && ppath.exists() {
                generator.load(&ppath)?;
                true
            } else {
                false
            };
            if cfg.verbose {
                eprintln!(
                    "resumed solver from {}{}",
                    cfg.checkpoint_dir.display(),
                    if loaded_gen { " + generator" } else { " (generator fresh)" }
                );
            }
        } else if cfg.verbose {
            eprintln!(
                "resume requested but no value checkpoint in {}; starting fresh",
                cfg.checkpoint_dir.display()
            );
        }
    }

    let n_gen = ((cfg.solver_batch as f32) * cfg.generator_frac).round() as usize;
    let n_gen = n_gen.min(cfg.solver_batch);
    let n_uni = cfg.solver_batch - n_gen;

    for round in 0..cfg.rounds {
        // ---- Depth curriculum: ramp the generator + uniform walk length ----
        // Set BEFORE the pool build so the frozen-for-this-round generator emits
        // boards at the ramped depth; uniform scrambles use the same `uni_k`.
        let uni_k = match &cfg.curriculum {
            Some(c) => {
                let k = ramped_k(c, round, cfg.rounds);
                generator.set_k_max(k);
                k
            }
            None => cfg.davi.k_max,
        };

        // ---- Solver phase (generator frozen; boards mix uniform + generator) ----
        // The generator is frozen for the whole solver phase, so roll its boards
        // ONCE into a pool (batched lockstep) and sample the per-step batch from
        // it — re-rolling per step dominated per-step cost. The pool is
        // regenerated each round so it tracks the updated generator.
        let gen_pool: Vec<State> = if n_gen > 0 {
            let pool_size = (cfg.solver_batch * 8).max(1024);
            let t = std::time::Instant::now();
            let pool = generator.sample_pool(
                pool_size,
                |s| davi.value_of(s).expect("solver value_of failed"),
                &mut rng,
            )?;
            if cfg.verbose {
                eprintln!(
                    "    [pool] {} gen boards in {:.0} ms",
                    pool_size,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            pool
        } else {
            Vec::new()
        };

        let mut loss_sum = 0.0f32;
        let mut win = std::time::Instant::now();
        // ~10 timing windows per round regardless of round length.
        let win_every = (cfg.solver_steps_per_round / 10).max(1);
        for i in 0..cfg.solver_steps_per_round {
            let mut batch: Vec<State> = Vec::with_capacity(cfg.solver_batch);
            for _ in 0..n_gen {
                let idx = rng.gen_range(0, (gen_pool.len() - 1) as u32) as usize;
                batch.push(gen_pool[idx]);
            }
            for _ in 0..n_uni {
                batch.push(scramble(&mut rng, uni_k).0);
            }
            loss_sum += davi.train_step(&batch)?;
            if cfg.verbose && (i + 1) % win_every == 0 {
                let ms = win.elapsed().as_secs_f64() * 1000.0 / win_every as f64;
                eprintln!("    [solver {}/{}] {:.0} ms/step", i + 1, cfg.solver_steps_per_round, ms);
                win = std::time::Instant::now();
            }
        }
        let solver_loss = loss_sum / cfg.solver_steps_per_round.max(1) as f32;

        // ---- Generator phase (solver frozen: value_of is read-only) ----
        let mut reward_sum = 0.0f32;
        for _ in 0..cfg.generator_steps_per_round {
            let r = generator
                .train_round(|s| davi.value_of(s).expect("solver value_of failed"), &mut rng)?;
            reward_sum += r;
        }
        let gen_reward = reward_sum / cfg.generator_steps_per_round.max(1) as f32;

        if cfg.verbose {
            eprintln!(
                "round {:>3}: solver_loss {:.3}  gen_reward {:+.2}  baseline {:+.2}",
                round,
                solver_loss,
                gen_reward,
                generator.reward_baseline()
            );
        }

        // ---- Periodic eval + checkpoint ----
        let is_last = round + 1 == cfg.rounds;
        let do_eval = is_last || (cfg.eval_every > 0 && (round + 1) % cfg.eval_every == 0);
        if do_eval {
            let prefix = format!(
                "{}\t{:.4}\t{:.3}\t{:.3}",
                round,
                solver_loss,
                gen_reward,
                generator.reward_baseline(),
            );
            let opt = |v: Option<f32>| v.map(|x| format!("{:.2}", x)).unwrap_or_else(|| "-".into());
            let optsign = |v: Option<f32>| v.map(|x| format!("{:+.2}", x)).unwrap_or_else(|| "-".into());
            let (header, line) = match &cfg.eval {
                EvalSpec::MidDepth(ec) => {
                    let r = eval::run(|s| davi.value_of(s).expect("solver value_of failed"), ec);
                    if cfg.verbose {
                        r.print();
                    }
                    let line = format!(
                        "{}\t{}\t{}\t{}\t{}",
                        prefix,
                        r.holdout_solved,
                        opt(r.holdout_mean_len),
                        r.optimal_labeled_n,
                        optsign(r.mean_excess_over_optimal),
                    );
                    (MID_METRICS_HEADER, line)
                }
                EvalSpec::Deep(dc) => {
                    let r = eval::run_deep(|s| davi.value_of(s).expect("solver value_of failed"), dc);
                    if cfg.verbose {
                        r.print();
                    }
                    let rcol = r
                        .r_line
                        .as_ref()
                        .map(|rr| {
                            let l = rr.learned.map(|v| v.to_string()).unwrap_or_else(|| "x".into());
                            let b = rr.beam.map(|v| v.to_string()).unwrap_or_else(|| "x".into());
                            format!("{}/{}", l, b)
                        })
                        .unwrap_or_else(|| "-".into());
                    let line = format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        prefix,
                        r.learned_solved,
                        r.beam_solved,
                        opt(r.mean_learned_len),
                        opt(r.mean_beam_len),
                        optsign(r.mean_excess_over_beam),
                        r.learned_wins,
                        rcol,
                    );
                    (DEEP_METRICS_HEADER, line)
                }
            };
            checkpoint::save(&cfg.checkpoint_dir, round, davi.online_varmap(), generator.varmap())?;
            checkpoint::append_metrics(&cfg.checkpoint_dir, header, &line)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::beam::BeamConfig;
    use super::super::bwas::BwasConfig;
    use super::super::eval::LabelHeuristic;
    use super::super::generator::{BaselineHeuristic, GeneratorReward, GeneratorSource};
    use super::*;

    #[test]
    fn tiny_loop_runs_and_checkpoints() {
        let dir = std::env::temp_dir().join(format!("ml24_alt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let small_bwas = BwasConfig { weight: 2.0, batch_size: 16, node_budget: 20_000 };
        let cfg = AlternationConfig {
            rounds: 2,
            solver_steps_per_round: 5,
            generator_steps_per_round: 5,
            solver_batch: 32,
            generator_frac: 0.5,
            davi: DaviConfig { k_max: 8, hidden: 32, blocks: 1, lr: 1e-3, target_sync_every: 10 },
            generator: GeneratorConfig {
                k_max: 8,
                hidden: 32,
                lr: 1e-3,
                baseline_decay: 0.9,
                solver_bwas: small_bwas,
                beam: BeamConfig { width: 100, max_depth: 60, node_budget: 200_000 },
                baseline: BaselineHeuristic::Manhattan,
                fail_penalty: 400.0,
                // Regret keeps this test table-free (no WD warm-up); entropy on.
                reward: GeneratorReward::Regret,
                entropy_beta: 0.01,
                adv_lambda: 1.0,
                source: GeneratorSource::PolicyRollout,
            },
            eval_every: 1,
            eval: EvalSpec::MidDepth(EvalConfig {
                bwas: small_bwas,
                holdout_n: 6,
                depth_min: 4,
                depth_max: 8,
                seed: 1,
                optimal_max_bound: 16,
                label_heuristic: LabelHeuristic::Lc,
            }),
            // Exercise the ramp (Regret + curriculum is table-free).
            curriculum: Some(Curriculum { k_start: 4, k_end: 8 }),
            checkpoint_dir: dir.clone(),
            seed: 1,
            verbose: false,
            resume: false,
            reset_generator: false,
        };

        run(&cfg, Device::Cpu).unwrap();

        // Resuming from the just-written checkpoint must run without error.
        let mut cfg_resume = AlternationConfig { resume: true, ..cfg };
        cfg_resume.rounds = 1;
        run(&cfg_resume, Device::Cpu).unwrap();

        // Checkpoints + metrics were written across the round boundary.
        assert!(checkpoint::value_latest_path(&dir).exists(), "no latest checkpoint written");
        assert!(dir.join("metrics.tsv").exists(), "no metrics written");

        // The latest checkpoint reloads into a fresh net.
        use crate::puzzle24::ml::value_net::ValueNet;
        use candle_core::DType;
        use candle_nn::{VarBuilder, VarMap};
        let mut vm = VarMap::new();
        let _net =
            ValueNet::new(VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), 32, 1).unwrap();
        vm.load(checkpoint::value_latest_path(&dir)).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_deep_loop_writes_deep_metrics() {
        // Same tiny loop but with the deep (excess-over-beam) in-loop eval;
        // Manhattan baseline so no WD table is needed. Verifies the Deep branch
        // runs and writes the deep metrics header.
        let dir = std::env::temp_dir().join(format!("ml24_altdeep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let small_bwas = BwasConfig { weight: 2.0, batch_size: 16, node_budget: 50_000 };
        let small_beam = BeamConfig { width: 100, max_depth: 60, node_budget: 200_000 };
        let cfg = AlternationConfig {
            rounds: 1,
            solver_steps_per_round: 5,
            generator_steps_per_round: 3,
            solver_batch: 32,
            generator_frac: 0.5,
            davi: DaviConfig { k_max: 8, hidden: 32, blocks: 1, lr: 1e-3, target_sync_every: 10 },
            generator: GeneratorConfig {
                k_max: 8,
                hidden: 32,
                lr: 1e-3,
                baseline_decay: 0.9,
                solver_bwas: small_bwas,
                beam: small_beam,
                baseline: BaselineHeuristic::Manhattan,
                fail_penalty: 400.0,
                reward: GeneratorReward::Regret,
                entropy_beta: 0.01,
                adv_lambda: 1.0,
                source: GeneratorSource::PolicyRollout,
            },
            eval_every: 1,
            eval: EvalSpec::Deep(DeepEvalConfig {
                bwas: small_bwas,
                beam: small_beam,
                baseline: BaselineHeuristic::Manhattan,
                holdout_n: 5,
                depth_min: 10,
                depth_max: 20,
                seed: 3,
                include_r: false,
            }),
            curriculum: None,
            checkpoint_dir: dir.clone(),
            seed: 2,
            verbose: false,
            resume: false,
            reset_generator: false,
        };

        run(&cfg, Device::Cpu).unwrap();

        let metrics = std::fs::read_to_string(dir.join("metrics.tsv")).unwrap();
        assert!(metrics.contains("mean_excess_beam"), "deep metrics header missing: {metrics}");
        assert!(checkpoint::value_latest_path(&dir).exists(), "no checkpoint written");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ramped_k_endpoints_monotonic_and_clamped() {
        let c = Curriculum { k_start: 50, k_end: 160 };
        assert_eq!(ramped_k(&c, 0, 10), 50, "round 0 should be k_start");
        assert_eq!(ramped_k(&c, 9, 10), 160, "last round should be k_end");
        let mut prev = 0;
        for round in 0..10 {
            let k = ramped_k(&c, round, 10);
            assert!(k >= prev, "not monotonic at round {round}");
            prev = k;
        }
        // rounds <= 1 → k_end; clamp ≥ 1.
        assert_eq!(ramped_k(&c, 0, 1), 160);
        assert_eq!(ramped_k(&c, 0, 0), 160);
        assert_eq!(ramped_k(&Curriculum { k_start: 0, k_end: 0 }, 0, 5), 1);
    }
}
