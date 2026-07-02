//! The adversarial board generator, trained by REINFORCE.
//!
//! A rollout walks from GOAL for `k ~ Uniform(1, k_max)` steps, choosing each
//! move from the policy network (`super::policy_net`). The resulting board `b`
//! is scored by **regret** against a fixed, non-learned baseline (GANCO): run
//! the current learned solver (BWAS over the value net) and an exact
//! `idastar` + Walking-Distance solve on `b`, and reward the generator by
//!   `reward = cost_learned(b) − cost_baseline(b)`.
//! High reward = boards where the *learned* solver underperforms a simple fixed
//! reference (the informative training signal). The REINFORCE loss
//!   `-(reward − baseline) · Σ log π(move_t)`
//! is backpropagated through the accumulated per-step log-probabilities; a
//! running EMA `reward` baseline reduces gradient variance.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use super::bwas::{search, BwasConfig, BwasOutcome};
use super::policy_net::{sample_move, PolicyNet, DEFAULT_HIDDEN};
use super::scramble::Rng;
use crate::puzzle15::search::{idastar, WalkingDistanceHeuristic};
use crate::puzzle15::state::{Move, State, GOAL};

pub struct GeneratorConfig {
    /// Fixed rollout-length range `[1, k_max]`.
    pub k_max: u32,
    pub hidden: usize,
    pub lr: f64,
    /// EMA decay for the reward baseline (variance reduction).
    pub baseline_decay: f32,
    /// BWAS config for the learned solver during reward evaluation.
    pub solver_bwas: BwasConfig,
    /// Cost charged when the learned solver exceeds its budget (must be strictly
    /// worse than any real 15-puzzle solution length; the diameter is 80).
    pub fail_penalty: f32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            k_max: 25,
            hidden: DEFAULT_HIDDEN,
            lr: 1e-3,
            baseline_decay: 0.95,
            solver_bwas: BwasConfig { weight: 2.0, batch_size: 1000, node_budget: 100_000 },
            fail_penalty: 200.0,
        }
    }
}

pub struct Generator {
    varmap: VarMap,
    net: PolicyNet,
    opt: AdamW,
    device: Device,
    k_max: u32,
    baseline_decay: f32,
    solver_bwas: BwasConfig,
    fail_penalty: f32,
    reward_baseline: f32,
}

impl Generator {
    pub fn new(cfg: &GeneratorConfig, device: Device) -> Result<Self> {
        // The exact baseline uses Walking Distance; build its table once.
        WalkingDistanceHeuristic::warm_up();

        let varmap = VarMap::new();
        let net = PolicyNet::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), cfg.hidden)?;
        let opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: cfg.lr, ..Default::default() })?;
        Ok(Self {
            varmap,
            net,
            opt,
            device,
            k_max: cfg.k_max.max(1),
            baseline_decay: cfg.baseline_decay,
            solver_bwas: cfg.solver_bwas,
            fail_penalty: cfg.fail_penalty,
            reward_baseline: 0.0,
        })
    }

    /// Roll out a board from GOAL, returning the board and the per-step chosen
    /// log-probabilities (graph tensors, for the policy-gradient backward pass).
    fn rollout(&self, rng: &mut Rng) -> Result<(State, Vec<Tensor>)> {
        let k = rng.gen_range(1, self.k_max);
        let mut s = GOAL;
        let mut blank = s.blank_pos();
        let mut last: Option<Move> = None;
        let mut log_probs = Vec::with_capacity(k as usize);
        for _ in 0..k {
            let banned = last.map(|m| m.inverse());
            let (m, lp) = sample_move(&self.net, &s, blank, banned, &self.device, rng)?;
            log_probs.push(lp);
            let (ns, nb) = s.apply_at(m, blank);
            s = ns;
            blank = nb;
            last = Some(m);
        }
        Ok((s, log_probs))
    }

    /// One REINFORCE update against the (frozen) learned solver `solver_value_of`.
    /// Returns the regret reward for this rollout.
    pub fn train_round<F>(&mut self, solver_value_of: F, rng: &mut Rng) -> Result<f32>
    where
        F: Fn(&[State]) -> Vec<f32>,
    {
        let (board, log_probs) = self.rollout(rng)?;

        // Learned solver cost under its node budget.
        let cost_learned = match search(&board, &self.solver_bwas, &solver_value_of) {
            BwasOutcome::Solved { moves, .. } => moves.len() as f32,
            BwasOutcome::BudgetExceeded { .. } => self.fail_penalty,
        };
        // Fixed baseline: exact optimal via idastar + Walking Distance.
        let t = std::time::Instant::now();
        let cost_baseline = idastar(&board, &WalkingDistanceHeuristic)
            .expect("generated board is solvable by construction")
            .len() as f32;
        super::profile::record_if("gen/baseline_idastar", t);

        let reward = cost_learned - cost_baseline;
        let advantage = reward - self.reward_baseline;

        // REINFORCE loss = -advantage * sum(log_probs).
        let mut sum_lp = log_probs[0].clone();
        for lp in &log_probs[1..] {
            sum_lp = sum_lp.add(lp)?;
        }
        let loss = sum_lp.affine(-(advantage as f64), 0.0)?;
        self.opt.backward_step(&loss)?;

        // EMA baseline update.
        self.reward_baseline =
            self.baseline_decay * self.reward_baseline + (1.0 - self.baseline_decay) * reward;
        Ok(reward)
    }

    /// Inference-only rollout: sample a board from the current policy (log-probs
    /// discarded). Used to feed generator boards into the solver's DAVI training.
    pub fn sample_board(&self, rng: &mut Rng) -> State {
        let k = rng.gen_range(1, self.k_max);
        let mut s = GOAL;
        let mut blank = s.blank_pos();
        let mut last: Option<Move> = None;
        for _ in 0..k {
            let banned = last.map(|m| m.inverse());
            let (m, _lp) =
                sample_move(&self.net, &s, blank, banned, &self.device, rng).expect("policy sample");
            let (ns, nb) = s.apply_at(m, blank);
            s = ns;
            blank = nb;
            last = Some(m);
        }
        s
    }

    /// Resume: load policy weights from a safetensors checkpoint (by name into
    /// the existing vars). The EMA reward baseline is not persisted and resets.
    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        self.varmap.load(path)
    }

    pub fn reward_baseline(&self) -> f32 {
        self.reward_baseline
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A "solver" heuristic that makes BWAS explore like BFS (constant h).
    fn constant_value(states: &[State]) -> Vec<f32> {
        vec![0.0; states.len()]
    }

    fn cpu_generator(k_max: u32, budget: u64) -> Generator {
        Generator::new(
            &GeneratorConfig {
                k_max,
                hidden: 32,
                lr: 1e-3,
                baseline_decay: 0.9,
                solver_bwas: BwasConfig { weight: 1.0, batch_size: 8, node_budget: budget },
                fail_penalty: 200.0,
            },
            Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn train_round_no_nan_and_finite_reward() {
        let mut g = cpu_generator(10, 100_000);
        let mut rng = Rng::new(3);
        for _ in 0..20 {
            let r = g.train_round(constant_value, &mut rng).unwrap();
            assert!(r.is_finite(), "reward not finite: {}", r);
        }
        assert!(g.reward_baseline().is_finite());
    }

    #[test]
    fn tiny_solver_budget_forces_positive_regret() {
        // With node_budget=1 the learned solver essentially always fails, so
        // cost_learned = fail_penalty (200) and reward = 200 - optimal_cost.
        // Optimal cost of a length<=10 board is <= 10, so a non-trivial board
        // gives reward ~190. A rollout can occasionally cycle back to GOAL
        // (reward 0) even with immediate-undo banned, so require most rounds —
        // not all — to show the large positive regret, and reward always finite.
        let mut g = cpu_generator(10, 1);
        let mut rng = Rng::new(11);
        let mut big = 0;
        let rounds = 20;
        for _ in 0..rounds {
            let r = g.train_round(constant_value, &mut rng).unwrap();
            assert!(r.is_finite() && r >= 0.0, "reward should be finite & >=0, got {}", r);
            if r > 180.0 {
                big += 1;
            }
        }
        assert!(big >= 15, "expected most rounds to show large regret, got {}/{}", big, rounds);
    }

    #[test]
    fn sample_board_is_solvable() {
        let g = cpu_generator(20, 100_000);
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            assert!(g.sample_board(&mut rng).is_solvable());
        }
    }
}
