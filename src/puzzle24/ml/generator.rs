//! The adversarial board generator, trained by REINFORCE.
//!
//! A rollout walks from GOAL for `k ~ Uniform(1, k_max)` steps, choosing each
//! move from the policy network (`super::policy_net`). The resulting board `b` is
//! scored by one of two rewards ([`GeneratorReward`]):
//!
//! - **`WdDepth` (default):** `WD(b) + λ·max(0, WD(b) − V(b))` — the admissible
//!   Walking-Distance lower bound (certifies genuine depth, hack-proof) plus an
//!   adversarial term that rewards boards the learned solver *provably*
//!   underestimates. WD is computed incrementally along the walk; only a value
//!   forward `V(b)` is needed (no BWAS/beam solve). This drives the generator to
//!   produce genuinely deep boards near the solver's frontier.
//! - **`Regret` (legacy):** GANCO regret `cost_learned(b) − cost_beam(b)` vs a
//!   fast beam baseline (`super::beam`) — kept for A/B.
//!
//! The REINFORCE loss `-(reward − baseline)·Σ log π(move_t) − β·Σ H_t` (policy
//! gradient + an entropy bonus for diversity) is backpropagated through the
//! per-step log-probabilities; a running EMA `reward` baseline reduces variance.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use super::beam::{beam_search, BeamConfig};
use super::bwas::{search, BwasConfig, BwasOutcome};
use super::encoding::encode_batch;
use super::policy_net::{sample_move, sample_move_full, PolicyNet, DEFAULT_HIDDEN};
use super::scramble::Rng;
use crate::puzzle24::search::{
    Heuristic, IncHeuristic, ManhattanHeuristic, SearchStats, WalkingDistanceHeuristic,
    WalkingDistanceInc,
};
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

/// How the generator's per-rollout reward is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratorReward {
    /// Legacy GANCO regret vs the beam baseline: `cost_learned − cost_beam`.
    /// (Runs a BWAS solve + a beam solve per rollout.)
    Regret,
    /// Hack-proof depth composite: `WD(board) + λ·max(0, WD − V(board))` — the
    /// admissible WD lower bound (certifies depth) plus an adversarial term on
    /// boards the learned solver *provably* underestimates. Needs only a value
    /// forward, no BWAS/beam solve.
    WdDepth,
}

/// The `WdDepth` reward: WD floor + adversarial term. `wd` = Walking-Distance
/// lower bound of the board, `v` = the learned solver's value estimate, `lambda`
/// = adversarial weight. `max(0, wd−v)` is nonzero only when the solver
/// underestimates below the admissible floor (a certified error).
fn composite_reward(wd: f32, v: f32, lambda: f32) -> f32 {
    wd + lambda * (wd - v).max(0.0)
}

/// Policy entropy `H = -Σ_i p_i log p_i` from a `[4]` log-prob tensor (graph).
/// Masked entries are ≈ `-1e30`, so `exp()→0` and `p·log p→-0.0` — NaN-safe.
fn entropy_of(logp4: &Tensor) -> Result<Tensor> {
    let p = logp4.exp()?; // [4] probs
    let plogp = p.mul(logp4)?; // [4], Σ ≤ 0
    plogp.sum_all()?.neg()
}

/// One rollout step's data for the policy-gradient + entropy loss.
struct Step {
    /// Chosen move's log-prob (scalar graph tensor).
    lp: Tensor,
    /// Full `[4]` masked log-prob distribution (graph tensor, for entropy).
    logp4: Tensor,
    /// WD lower bound of the board *after* this move (0 in `Regret` mode, unused).
    wd_child: u8,
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
    /// How the per-rollout reward is computed.
    pub reward: GeneratorReward,
    /// Entropy-bonus weight in the REINFORCE loss (0 = off).
    pub entropy_beta: f32,
    /// Adversarial-term weight λ in the `WdDepth` composite (0 = pure WD).
    pub adv_lambda: f32,
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
            reward: GeneratorReward::WdDepth,
            entropy_beta: 0.01,
            adv_lambda: 1.0,
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
    reward: GeneratorReward,
    entropy_beta: f32,
    adv_lambda: f32,
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
        // The WdDepth reward evaluates incremental WD during every rollout, so its
        // table must be up even when the beam baseline is Manhattan.
        if cfg.reward == GeneratorReward::WdDepth {
            WalkingDistanceHeuristic::warm_up();
        }

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
            reward: cfg.reward,
            entropy_beta: cfg.entropy_beta,
            adv_lambda: cfg.adv_lambda,
            reward_baseline: 0.0,
        })
    }

    /// Update the rollout walk-length cap (the curriculum knob). Clamped to ≥ 1.
    pub fn set_k_max(&mut self, k: u32) {
        self.k_max = k.max(1);
    }

    /// Draw a walk length. `WdDepth` uses a **fixed** length (`= k_max`): with the
    /// terminal WD reward, a random `[1, k_max]` length correlates with the reward
    /// but is outside the policy's control, so its variance swamps the
    /// policy-gradient signal about *which moves* raise WD (and the entropy bonus
    /// then collapses the policy to a folding random walk). A fixed length makes
    /// reward variance come only from move quality. `Regret` keeps the legacy
    /// uniform range.
    fn draw_k(&self, rng: &mut Rng) -> u32 {
        match self.reward {
            GeneratorReward::WdDepth => self.k_max,
            GeneratorReward::Regret => rng.gen_range(1, self.k_max),
        }
    }

    /// Roll out a board from GOAL, returning the board and per-step [`Step`]
    /// records (chosen log-prob, full distribution for entropy, and the board's
    /// WD after each move). Incremental WD is only threaded in `WdDepth` mode, so
    /// `Regret` rollouts never touch the WD table (`wd_child` stays 0 there).
    fn rollout(&self, rng: &mut Rng) -> Result<(State, Vec<Step>)> {
        let k = self.draw_k(rng);
        let mut s = GOAL;
        let mut blank = s.blank_pos();
        let mut last: Option<Move> = None;
        let track_wd = self.reward == GeneratorReward::WdDepth;
        let wd = WalkingDistanceInc;
        let mut stats = SearchStats::default();
        let mut ctx = if track_wd { Some(wd.root(&s, &mut stats).1) } else { None };
        let mut steps = Vec::with_capacity(k as usize);
        for _ in 0..k {
            let banned = last.map(|m| m.inverse());
            let (m, lp, logp4) =
                sample_move_full(&self.net, &s, blank, banned, &self.device, rng)?;
            let (ns, nb) = s.apply_at(m, blank);
            let wd_child = if let Some(c) = ctx {
                let (h, cc) = wd.advance(&c, &ns, m, &mut stats);
                ctx = Some(cc);
                h
            } else {
                0
            };
            steps.push(Step { lp, logp4, wd_child });
            s = ns;
            blank = nb;
            last = Some(m);
        }
        Ok((s, steps))
    }

    /// One REINFORCE update against the (frozen) learned solver `solver_value_of`.
    /// Returns the per-rollout reward (regret, or the WD-depth composite).
    pub fn train_round<F>(&mut self, solver_value_of: F, rng: &mut Rng) -> Result<f32>
    where
        F: Fn(&[State]) -> Vec<f32>,
    {
        let (board, steps) = self.rollout(rng)?;

        let reward = match self.reward {
            GeneratorReward::Regret => {
                // Learned solver cost under its node budget.
                let cost_learned = match search(&board, &self.solver_bwas, &solver_value_of) {
                    BwasOutcome::Solved { moves, .. } => moves.len() as f32,
                    BwasOutcome::BudgetExceeded { .. } => self.fail_penalty,
                };
                // Fixed suboptimal beam baseline with an admissible heuristic.
                let t = std::time::Instant::now();
                let cost_baseline = match beam_search(&board, self.beam_h.as_ref(), &self.beam_cfg) {
                    Some(mv) => mv.len() as f32,
                    None => self.fail_penalty,
                };
                super::profile::record_if("gen/baseline_beam", t);
                cost_learned - cost_baseline
            }
            GeneratorReward::WdDepth => {
                // WD floor (admissible ⇒ certifies depth) + adversarial term on
                // boards the solver provably underestimates. Only a value forward.
                let wd = steps.last().map(|s| s.wd_child).unwrap_or(0) as f32;
                let v = solver_value_of(std::slice::from_ref(&board))[0];
                composite_reward(wd, v, self.adv_lambda)
            }
        };
        let advantage = reward - self.reward_baseline;

        // Policy-gradient term: -advantage · Σ log π(move_t).
        let mut sum_lp = steps[0].lp.clone();
        for st in &steps[1..] {
            sum_lp = sum_lp.add(&st.lp)?;
        }
        let pg = sum_lp.affine(-(advantage as f64), 0.0)?;

        // Entropy bonus: subtract β·Σ H_t to *maximise* policy entropy (diversity).
        let loss = if self.entropy_beta > 0.0 {
            let mut sum_h = entropy_of(&steps[0].logp4)?;
            for st in &steps[1..] {
                sum_h = sum_h.add(&entropy_of(&st.logp4)?)?;
            }
            pg.sub(&sum_h.affine(self.entropy_beta as f64, 0.0)?)?
        } else {
            pg
        };
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
        let lens: Vec<u32> = (0..n).map(|_| self.draw_k(rng)).collect();
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
        let k = self.draw_k(rng);
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

    /// Test generator: Regret + Manhattan baseline (no WD warm-up), small nets.
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
                reward: GeneratorReward::Regret,
                entropy_beta: 0.01,
                adv_lambda: 1.0,
            },
            Device::Cpu,
        )
        .unwrap()
    }

    /// Test generator in `WdDepth` mode (warms the WD table — gate callers on
    /// `data/wd24.bin`). Manhattan beam baseline is unused in this mode.
    fn wd_generator(k_max: u32) -> Generator {
        Generator::new(
            &GeneratorConfig {
                k_max,
                hidden: 32,
                lr: 1e-3,
                baseline_decay: 0.9,
                solver_bwas: BwasConfig { weight: 1.0, batch_size: 8, node_budget: 10_000 },
                beam: BeamConfig { width: 100, max_depth: 80, node_budget: 200_000 },
                baseline: BaselineHeuristic::Manhattan,
                fail_penalty: 400.0,
                reward: GeneratorReward::WdDepth,
                entropy_beta: 0.01,
                adv_lambda: 1.0,
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
                reward: GeneratorReward::Regret,
                entropy_beta: 0.01,
                adv_lambda: 1.0,
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

    #[test]
    fn composite_reward_properties() {
        // Floor: reward ≥ wd. Self-clear: v ≥ wd ⇒ reward == wd. And the reward is
        // non-decreasing in wd (the derivative 1 + λ·[wd>v] ≥ 0).
        for &lambda in &[0.0f32, 0.5, 1.0, 2.0] {
            for &wd in &[0.0f32, 10.0, 50.0, 140.0] {
                assert!(composite_reward(wd, 0.0, lambda) >= wd, "floor violated");
                assert_eq!(composite_reward(wd, wd, lambda), wd, "not self-cleared at v==wd");
                assert_eq!(composite_reward(wd, wd + 5.0, lambda), wd, "not clamped at v>wd");
            }
        }
        let (v, lambda) = (20.0f32, 1.0);
        let mut prev = f32::NEG_INFINITY;
        for wd in 0..=140 {
            let r = composite_reward(wd as f32, v, lambda);
            assert!(r >= prev - 1e-6, "not non-decreasing at wd={}", wd);
            prev = r;
        }
    }

    #[test]
    fn entropy_is_bounded_and_finite() {
        let dev = Device::Cpu;
        // Uniform over 4 legal moves ⇒ H = ln 4 (the max).
        let uniform = Tensor::from_vec(vec![(0.25f32).ln(); 4], (4,), &dev).unwrap();
        let h = entropy_of(&uniform).unwrap().to_scalar::<f32>().unwrap();
        assert!((h - (4f32).ln()).abs() < 1e-4, "uniform entropy {} != ln4", h);
        // Two moves masked to -1e30 (as log_softmax would): H = ln 2, and finite
        // (the `0·(-1e30)` terms are -0.0, not NaN).
        let masked =
            Tensor::from_vec(vec![(0.5f32).ln(), (0.5f32).ln(), -1e30, -1e30], (4,), &dev).unwrap();
        let h2 = entropy_of(&masked).unwrap().to_scalar::<f32>().unwrap();
        assert!(h2.is_finite() && (0.0..=(4f32).ln() + 1e-3).contains(&h2), "bad masked H {}", h2);
        assert!((h2 - (2f32).ln()).abs() < 1e-3, "masked entropy {} != ln2", h2);
    }

    #[test]
    fn set_k_max_stays_solvable() {
        let mut g = cpu_generator(50, 100_000);
        g.set_k_max(1);
        let mut rng = Rng::new(3);
        for s in g.sample_pool(50, &mut rng).unwrap() {
            assert!(s.is_solvable());
        }
    }

    #[test]
    fn wd_reward_matches_oracle_and_grows_with_depth() {
        // WD-gated. incremental `wd_child` must equal the from-scratch oracle, and
        // mean WD must increase with the walk-length k.
        if !std::path::Path::new("data/wd24.bin").exists() {
            return;
        }
        use crate::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
        let g = wd_generator(40);
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let (board, steps) = g.rollout(&mut rng).unwrap();
            assert_eq!(
                steps.last().unwrap().wd_child,
                WalkingDistanceHeuristic.h(&board),
                "incremental WD != oracle on {:?}",
                board.0
            );
        }
        let mean_wd = |k: u32, seed: u64| {
            let g = wd_generator(k);
            let mut rng = Rng::new(seed);
            let n = 40u32;
            let sum: u32 =
                (0..n).map(|_| g.rollout(&mut rng).unwrap().1.last().unwrap().wd_child as u32).sum();
            sum as f32 / n as f32
        };
        let (m4, m40, m120) = (mean_wd(4, 1), mean_wd(40, 2), mean_wd(120, 3));
        assert!(m4 < m40 && m40 < m120, "WD not increasing with k: {} {} {}", m4, m40, m120);
    }

    #[test]
    fn wd_train_round_finite_and_above_floor() {
        // WD-gated. WdDepth reward with a constant (V=0) solver ⇒ reward = 2·wd.
        if !std::path::Path::new("data/wd24.bin").exists() {
            return;
        }
        let mut g = wd_generator(30);
        let mut rng = Rng::new(7);
        for _ in 0..10 {
            let r = g.train_round(constant_value, &mut rng).unwrap();
            assert!(r.is_finite() && r >= 0.0, "WdDepth reward bad: {}", r);
        }
        assert!(g.reward_baseline().is_finite());
    }
}
