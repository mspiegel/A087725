//! The adversarial co-training loop.
//!
//! Each round: (1) train the solver's value net for `solver_steps_per_round`
//! DAVI steps on a batch that mixes uniform-random scrambles with boards from
//! the current (frozen) generator; then (2) freeze the solver and train the
//! generator for `generator_steps_per_round` REINFORCE rounds against it.
//! Periodically evaluate against the holdout + antipodes and checkpoint both
//! networks. See TRAINING.md.

use std::path::PathBuf;

use candle_core::{Device, Result};

use super::checkpoint;
use super::davi::{Davi, DaviConfig};
use super::eval::{self, EvalConfig};
use super::generator::{Generator, GeneratorConfig};
use super::scramble::{scramble, Rng};
use crate::puzzle15::state::State;

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
    pub eval: EvalConfig,
    pub checkpoint_dir: PathBuf,
    pub seed: u64,
    pub verbose: bool,
    /// If true, load `value_latest`/`policy_latest` from `checkpoint_dir` (if
    /// present) into the networks before training, continuing from those
    /// weights. Round numbering and the generator's EMA baseline still reset.
    pub resume: bool,
}

const METRICS_HEADER: &str =
    "round\tsolver_loss\tgen_reward\tgen_baseline\tholdout_solved\tholdout_mean_len\tantipode_solved\tantipode_excess";

/// Run the full co-training loop. Returns after `rounds` rounds, having written
/// checkpoints and metrics to `checkpoint_dir`.
pub fn run(cfg: &AlternationConfig, device: Device) -> Result<()> {
    let mut davi = Davi::new(&cfg.davi, device.clone())?;
    let mut generator = Generator::new(&cfg.generator, device)?;
    let mut rng = Rng::new(cfg.seed);

    if cfg.resume {
        let vpath = checkpoint::value_latest_path(&cfg.checkpoint_dir);
        let ppath = checkpoint::policy_latest_path(&cfg.checkpoint_dir);
        if vpath.exists() && ppath.exists() {
            davi.load_online(&vpath)?;
            generator.load(&ppath)?;
            if cfg.verbose {
                eprintln!("resumed weights from {}", cfg.checkpoint_dir.display());
            }
        } else if cfg.verbose {
            eprintln!(
                "resume requested but no latest checkpoint in {}; starting fresh",
                cfg.checkpoint_dir.display()
            );
        }
    }

    let n_gen = ((cfg.solver_batch as f32) * cfg.generator_frac).round() as usize;
    let n_gen = n_gen.min(cfg.solver_batch);
    let n_uni = cfg.solver_batch - n_gen;

    for round in 0..cfg.rounds {
        // ---- Solver phase (generator frozen; boards mix uniform + generator) ----
        let mut loss_sum = 0.0f32;
        for _ in 0..cfg.solver_steps_per_round {
            let mut batch: Vec<State> = Vec::with_capacity(cfg.solver_batch);
            for _ in 0..n_gen {
                batch.push(generator.sample_board(&mut rng));
            }
            for _ in 0..n_uni {
                batch.push(scramble(&mut rng, cfg.davi.k_max).0);
            }
            loss_sum += davi.train_step(&batch)?;
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
            let report = eval::run(|s| davi.value_of(s).expect("solver value_of failed"), &cfg.eval);
            if cfg.verbose {
                report.print();
            }
            checkpoint::save(&cfg.checkpoint_dir, round, davi.online_varmap(), generator.varmap())?;
            let line = format!(
                "{}\t{:.4}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}",
                round,
                solver_loss,
                gen_reward,
                generator.reward_baseline(),
                report.holdout_solved,
                report.holdout_mean_len.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
                report.antipode_solved,
                report
                    .antipode_mean_excess
                    .map(|v| format!("{:+.2}", v))
                    .unwrap_or_else(|| "-".into()),
            );
            checkpoint::append_metrics(&cfg.checkpoint_dir, METRICS_HEADER, &line)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::bwas::BwasConfig;

    #[test]
    fn tiny_loop_runs_and_checkpoints() {
        let dir = std::env::temp_dir().join(format!("ml_alt_{}", std::process::id()));
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
                fail_penalty: 200.0,
            },
            eval_every: 1,
            eval: EvalConfig {
                bwas: small_bwas,
                antipodes_path: PathBuf::from("data/pdb15_antipodes.txt"),
                holdout_n: 6,
                holdout_k_max: 8,
                holdout_seed: 1,
            },
            checkpoint_dir: dir.clone(),
            seed: 1,
            verbose: false,
            resume: false,
        };

        run(&cfg, Device::Cpu).unwrap();

        // Resuming from the just-written checkpoint must run without error.
        let mut cfg_resume = AlternationConfig { resume: true, ..cfg };
        cfg_resume.rounds = 1;
        run(&cfg_resume, Device::Cpu).unwrap();

        // Checkpoints + metrics were written across the round boundary.
        assert!(checkpoint::value_latest_path(&dir).exists(), "no latest checkpoint written");
        assert!(dir.join("metrics.tsv").exists(), "no metrics written");

        // The latest checkpoint reloads into a fresh net (verifies save/load across rounds).
        use crate::puzzle15::ml::value_net::ValueNet;
        use candle_core::DType;
        use candle_nn::{VarBuilder, VarMap};
        let mut vm = VarMap::new();
        let _net =
            ValueNet::new(VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), 32, 1).unwrap();
        vm.load(checkpoint::value_latest_path(&dir)).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
