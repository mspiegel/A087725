//! The adversarial board generator, trained by REINFORCE.
//!
//! A rollout walks from GOAL for `k ~ Uniform(1, k_max)` steps, choosing each
//! move from the policy network (`super::policy_net`). The resulting board `b`
//! is scored by **regret** against a fixed, non-learned baseline (GANCO): run
//! the current learned solver (BWAS over the value net) and a fast **beam-search
//! suboptimal solver** (`super::beam`, admissible heuristic — Walking Distance by
//! default) on `b`, and reward the generator by
//!   `reward = cost_learned(b) − cost_baseline(b)`.
//! High reward = boards where the *learned* solver underperforms the fixed
//! reference solver. The REINFORCE loss `-(reward − baseline) · Σ log π(move_t)`
//! is backpropagated through the accumulated per-step log-probabilities; a
//! running EMA `reward` baseline reduces gradient variance.
//!
//! Unlike the 15-puzzle, there is no exact-optimal baseline (the 24-puzzle has no
//! feasible optimal solver past ~depth 50), so the baseline is the beam solver.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use super::beam::{beam_search, BeamConfig};
use super::bwas::{search, BwasConfig, BwasOutcome};
use super::encoding::encode_batch;
use super::policy_net::{sample_move, PolicyNet, DEFAULT_HIDDEN};
use super::scramble::Rng;
use crate::puzzle24::search::{Heuristic, ManhattanHeuristic, WalkingDistanceHeuristic};
use crate::puzzle24::state::{Move, State, GOAL};

/// Which admissible heuristic the beam baseline uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaselineHeuristic {
    /// Walking Distance — the strongest single term on deep boards (loads the
    /// `data/wd24.bin` table via `warm_up`). The production default.
    Wd,
    /// Manhattan — table-free (used in tests to avoid the WD warm-up).
    Manhattan,
}

pub struct GeneratorConfig {
    /// Fixed rollout-length range `[1, k_max]`.
    pub k_max: u32,
    pub hidden: usize,
    pub lr: f64,
    /// EMA decay for the reward baseline (variance reduction).
    pub baseline_decay: f32,
    /// BWAS config for the learned solver during reward evaluation.
    pub solver_bwas: BwasConfig,
    /// Beam-search config for the baseline solver.
    pub beam: BeamConfig,
    /// Admissible heuristic for the beam baseline.
    pub baseline: BaselineHeuristic,
    /// Cost charged when a solver exceeds its budget (must be strictly worse than
    /// any real 24-puzzle solution length; the diameter upper bound is 205).
    pub fail_penalty: f32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            k_max: 40,
            hidden: DEFAULT_HIDDEN,
            lr: 1e-3,
            baseline_decay: 0.95,
            solver_bwas: BwasConfig { weight: 2.0, batch_size: 1000, node_budget: 300_000 },
            beam: BeamConfig::default(),
            baseline: BaselineHeuristic::Wd,
            fail_penalty: 400.0,
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
    beam_cfg: BeamConfig,
    beam_h: Box<dyn Heuristic + Send + Sync>,
    fail_penalty: f32,
    reward_baseline: f32,
}

impl Generator {
    pub fn new(cfg: &GeneratorConfig, device: Device) -> Result<Self> {
        // Resolve the beam baseline heuristic; WD loads its table once (OnceLock).
        let beam_h: Box<dyn Heuristic + Send + Sync> = match cfg.baseline {
            BaselineHeuristic::Wd => {
                WalkingDistanceHeuristic::warm_up();
                Box::new(WalkingDistanceHeuristic)
            }
            BaselineHeuristic::Manhattan => Box::new(ManhattanHeuristic),
        };

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
            beam_cfg: cfg.beam,
            beam_h,
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
        // Fixed baseline: fast suboptimal beam search with an admissible heuristic.
        let t = std::time::Instant::now();
        let cost_baseline = match beam_search(&board, self.beam_h.as_ref(), &self.beam_cfg) {
            Some(mv) => mv.len() as f32,
            None => self.fail_penalty,
        };
        super::profile::record_if("gen/baseline_beam", t);

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

    /// Inference-only batched rollout: sample `n` boards from the current policy
    /// in **lockstep** — one batched policy forward per move-step over all still-
    /// active boards — instead of `n` separate move-by-move rollouts. Turns
    /// ~`n·k` tiny batch-1 forwards into ~`k` big batched forwards (≈100× faster
    /// pool build; this fix is why the 24-puzzle training loop is feasible).
    pub fn sample_pool(&self, n: usize, rng: &mut Rng) -> Result<Vec<State>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let lens: Vec<u32> = (0..n).map(|_| rng.gen_range(1, self.k_max)).collect();
        let max_k = *lens.iter().max().unwrap();

        let mut states = vec![GOAL; n];
        let mut blanks: Vec<u8> = states.iter().map(|s| s.blank_pos()).collect();
        let mut last: Vec<Option<Move>> = vec![None; n];

        for step in 0..max_k {
            let active: Vec<usize> = (0..n).filter(|&i| step < lens[i]).collect();
            if active.is_empty() {
                break;
            }
            let active_states: Vec<State> = active.iter().map(|&i| states[i]).collect();
            let x = encode_batch(&active_states, &self.device)?; // [A, 625]
            let logits = self.net.forward(&x)?; // [A, 4]
            let rows: Vec<Vec<f32>> = logits.to_vec2::<f32>()?;

            for (a, &i) in active.iter().enumerate() {
                let blank = blanks[i];
                let banned = last[i].map(|m| m.inverse());
                let legal = State::legal_moves_at(blank);
                let row = &rows[a];
                let allowed = |k: usize| legal.contains(Move::ALL[k]) && Some(Move::ALL[k]) != banned;

                let mut mx = f32::NEG_INFINITY;
                for k in 0..4 {
                    if allowed(k) {
                        mx = mx.max(row[k]);
                    }
                }
                let mut w = [0f32; 4];
                let mut sum = 0.0f32;
                for k in 0..4 {
                    if allowed(k) {
                        let e = (row[k] - mx).exp();
                        w[k] = e;
                        sum += e;
                    }
                }
                let mut acc = rng.gen_f32() * sum;
                let mut chosen = (0..4).find(|&k| allowed(k)).unwrap();
                for k in 0..4 {
                    if w[k] > 0.0 {
                        chosen = k;
                        acc -= w[k];
                        if acc <= 0.0 {
                            break;
                        }
                    }
                }
                let (ns, nb) = states[i].apply_at(Move::ALL[chosen], blank);
                states[i] = ns;
                blanks[i] = nb;
                last[i] = Some(Move::ALL[chosen]);
            }
        }
        Ok(states)
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

    /// Test generator: Manhattan baseline (no WD warm-up), small nets/beam.
    fn cpu_generator(k_max: u32, budget: u64) -> Generator {
        Generator::new(
            &GeneratorConfig {
                k_max,
                hidden: 32,
                lr: 1e-3,
                baseline_decay: 0.9,
                solver_bwas: BwasConfig { weight: 1.0, batch_size: 8, node_budget: budget },
                beam: BeamConfig { width: 200, max_depth: 80, node_budget: 500_000 },
                baseline: BaselineHeuristic::Manhattan,
                fail_penalty: 400.0,
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
        // node_budget=1 => the learned solver always fails => cost_learned =
        // fail_penalty (400); the beam baseline solves the shallow (k<=10) board
        // in ~10-15 moves, so reward ≈ 385+. A rollout can occasionally cycle back
        // to GOAL (reward 0), so require most rounds — not all — to be large.
        let mut g = cpu_generator(10, 1);
        let mut rng = Rng::new(11);
        let mut big = 0;
        let rounds = 20;
        for _ in 0..rounds {
            let r = g.train_round(constant_value, &mut rng).unwrap();
            assert!(r.is_finite() && r >= 0.0, "reward should be finite & >=0, got {}", r);
            if r > 350.0 {
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

    #[test]
    fn sample_pool_is_solvable_and_correct_count() {
        let g = cpu_generator(20, 100_000);
        let mut rng = Rng::new(9);
        let pool = g.sample_pool(200, &mut rng).unwrap();
        assert_eq!(pool.len(), 200);
        for s in &pool {
            assert!(s.is_solvable());
        }
        assert!(g.sample_pool(0, &mut rng).unwrap().is_empty());
    }

    #[test]
    fn wd_baseline_reward_is_finite_when_table_present() {
        // Opt-in coverage of the production WD beam baseline + the Box<dyn
        // Heuristic> dispatch. Runs only when the 590 MB `data/wd24.bin` artifact
        // is present (absent on CI / a fresh checkout), so it never breaks the
        // default suite.
        if !std::path::Path::new("data/wd24.bin").exists() {
            return;
        }
        let mut g = Generator::new(
            &GeneratorConfig {
                k_max: 12,
                hidden: 32,
                lr: 1e-3,
                baseline_decay: 0.9,
                solver_bwas: BwasConfig { weight: 2.0, batch_size: 8, node_budget: 50_000 },
                beam: BeamConfig { width: 200, max_depth: 100, node_budget: 500_000 },
                baseline: BaselineHeuristic::Wd,
                fail_penalty: 400.0,
            },
            Device::Cpu,
        )
        .unwrap();
        let mut rng = Rng::new(17);
        for _ in 0..5 {
            let r = g.train_round(constant_value, &mut rng).unwrap();
            assert!(r.is_finite(), "WD-baseline reward not finite: {}", r);
        }
    }
}
