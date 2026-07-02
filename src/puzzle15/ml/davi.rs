//! Deep Approximate Value Iteration (DAVI) trainer for the solver's value net.
//!
//! Per DeepCubeA (Agostinelli et al. 2019): regress the online network `V(s)`
//! toward the bootstrapped Bellman target
//!   `V_target(s) = min over legal neighbors s' of (1 + V_target_net(s'))`,
//! with the goal clamped to `0` as the fixed boundary condition, using a
//! periodically-synced *target* network for stability. The target values are
//! materialized to plain `f32` and rebuilt as a detached tensor, so gradients
//! flow **only** through the online forward pass.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use std::time::Instant;

use super::encoding::encode_batch;
use super::profile;
use super::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use crate::puzzle15::state::{State, GOAL};

pub struct DaviConfig {
    /// Fixed scramble-depth range `[1, k_max]` for training-state generation.
    pub k_max: u32,
    pub hidden: usize,
    /// Number of residual blocks in the value net (0 = plain MLP).
    pub blocks: usize,
    pub lr: f64,
    /// Steps between hard syncs of `target <- online`.
    pub target_sync_every: usize,
}

impl Default for DaviConfig {
    fn default() -> Self {
        Self {
            k_max: 30,
            hidden: DEFAULT_HIDDEN,
            blocks: DEFAULT_BLOCKS,
            lr: 1e-3,
            target_sync_every: 1000,
        }
    }
}

pub struct Davi {
    online_varmap: VarMap,
    target_varmap: VarMap,
    online: ValueNet,
    target: ValueNet,
    opt: AdamW,
    device: Device,
    steps: usize,
    sync_every: usize,
}

impl Davi {
    pub fn new(cfg: &DaviConfig, device: Device) -> Result<Self> {
        let online_varmap = VarMap::new();
        let online = ValueNet::new(
            VarBuilder::from_varmap(&online_varmap, DType::F32, &device),
            cfg.hidden,
            cfg.blocks,
        )?;
        let target_varmap = VarMap::new();
        let target = ValueNet::new(
            VarBuilder::from_varmap(&target_varmap, DType::F32, &device),
            cfg.hidden,
            cfg.blocks,
        )?;

        let opt = AdamW::new(
            online_varmap.all_vars(),
            ParamsAdamW { lr: cfg.lr, ..Default::default() },
        )?;

        let mut davi = Self {
            online_varmap,
            target_varmap,
            online,
            target,
            opt,
            device,
            steps: 0,
            sync_every: cfg.target_sync_every.max(1),
        };
        // Start with target == online so the first bootstrap targets are sane.
        davi.sync_target()?;
        Ok(davi)
    }

    /// One gradient step over a batch of (already-scrambled, non-goal) states.
    /// Returns the scalar MSE loss for logging.
    pub fn train_step(&mut self, states: &[State]) -> Result<f32> {
        debug_assert!(!states.is_empty());

        // Flatten every state's legal neighbors into one list, tracking ranges.
        let t = Instant::now();
        let mut flat: Vec<State> = Vec::with_capacity(states.len() * 4);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(states.len());
        for s in states {
            let start = flat.len();
            let blank = s.blank_pos();
            for m in State::legal_moves_at(blank).iter() {
                flat.push(s.apply_at(m, blank).0);
            }
            ranges.push((start, flat.len() - start));
        }
        profile::record_if("davi/neighbors", t);

        // One target-net forward over all neighbors (no grad — detached below).
        // `values` ends in a `to_vec1` which itself syncs the device.
        let t = Instant::now();
        let vt = self.target.values(&flat, &self.device)?;
        profile::record_if("davi/target_eval", t);

        // Bellman target per state, with the goal clamped to 0 everywhere.
        let t = Instant::now();
        let mut targets = Vec::with_capacity(states.len());
        for (i, s) in states.iter().enumerate() {
            if *s == GOAL {
                targets.push(0.0f32);
                continue;
            }
            let (start, count) = ranges[i];
            let mut best = f32::INFINITY;
            for j in 0..count {
                let nb = &flat[start + j];
                let v = if *nb == GOAL { 0.0 } else { vt[start + j] };
                best = best.min(1.0 + v);
            }
            targets.push(best);
        }
        profile::record_if("davi/host_reduce", t);

        // Detached target tensor; only the online forward is in the autograd graph.
        let t = Instant::now();
        let target_t = Tensor::from_vec(targets, (states.len(), 1), &self.device)?;
        let x = encode_batch(states, &self.device)?;
        let preds = self.online.forward(&x)?; // [n, 1]
        let loss = loss::mse(&preds, &target_t)?;
        profile::sync(&self.device); // force the forward to execute for accurate attribution
        profile::record_if("davi/online_fwd", t);

        let t = Instant::now();
        self.opt.backward_step(&loss)?;
        profile::sync(&self.device);
        profile::record_if("davi/backward", t);

        let t = Instant::now();
        let loss_val = loss.to_scalar::<f32>()?;
        profile::record_if("davi/loss_scalar", t);

        self.steps += 1;
        if self.steps % self.sync_every == 0 {
            self.sync_target()?;
        }
        Ok(loss_val)
    }

    /// Hard-copy the online parameters into the target network, in place (the
    /// target `ValueNet`'s tensors share storage with `target_varmap`, so they
    /// see the update).
    pub fn sync_target(&mut self) -> Result<()> {
        let pairs: Vec<(String, Tensor)> = {
            let data = self.online_varmap.data().lock().unwrap();
            data.iter().map(|(n, v)| (n.clone(), v.as_tensor().clone())).collect()
        };
        for (name, tensor) in pairs {
            self.target_varmap.set_one(&name, &tensor)?;
        }
        Ok(())
    }

    /// Resume: load online-network weights from a safetensors checkpoint (by
    /// name into the existing vars), then re-sync the target network so both
    /// match the loaded weights. Architecture (hidden size) must match the
    /// checkpoint, or `load` errors.
    pub fn load_online(&mut self, path: &std::path::Path) -> Result<()> {
        self.online_varmap.load(path)?;
        self.sync_target()?;
        Ok(())
    }

    /// Online-network cost-to-go for a batch of states (the solver's heuristic).
    pub fn value_of(&self, states: &[State]) -> Result<Vec<f32>> {
        self.online.values(states, &self.device)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Borrow the online varmap (for checkpointing).
    pub fn online_varmap(&self) -> &VarMap {
        &self.online_varmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::scramble::{scramble, Rng};

    fn cpu_davi(k_max: u32) -> Davi {
        Davi::new(
            &DaviConfig { k_max, hidden: 64, blocks: 2, lr: 1e-3, target_sync_every: 50 },
            Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn train_step_no_nan_on_tiny_batch() {
        let mut davi = cpu_davi(3);
        let mut rng = Rng::new(1);
        let states: Vec<State> = (0..32).map(|_| scramble(&mut rng, 3).0).collect();
        let loss = davi.train_step(&states).unwrap();
        assert!(loss.is_finite(), "loss was not finite: {}", loss);
        assert!(loss >= 0.0);
    }

    #[test]
    fn load_online_resumes_identical_value_function() {
        // Train a bit, save online weights, then load them into a fresh Davi and
        // confirm value_of matches exactly — the core of resume support.
        let mut davi = cpu_davi(5);
        let mut rng = Rng::new(21);
        for _ in 0..30 {
            let states: Vec<State> = (0..32).map(|_| scramble(&mut rng, 5).0).collect();
            davi.train_step(&states).unwrap();
        }
        let probe: Vec<State> = (0..16).map(|_| scramble(&mut rng, 5).0).collect();
        let before = davi.value_of(&probe).unwrap();

        let path = std::env::temp_dir().join(format!("davi_resume_{}.safetensors", std::process::id()));
        davi.online_varmap().save(&path).unwrap();

        let mut davi2 = cpu_davi(5);
        let fresh = davi2.value_of(&probe).unwrap();
        davi2.load_online(&path).unwrap();
        let resumed = davi2.value_of(&probe).unwrap();

        assert_ne!(before, fresh, "fresh net unexpectedly equals trained net");
        assert_eq!(before, resumed, "load_online did not reproduce the trained value function");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loss_trends_down_on_shallow_scrambles() {
        // On k_max<=5 scrambles the optimal cost-to-go is small and learnable
        // fast; the running loss should drop substantially over a short run.
        // candle's CPU weight-init RNG can't be seeded, so use a stable config
        // (lower lr, less frequent target sync) that converges regardless of
        // init, and — since DAVI's loss is inherently sawtooth (it jumps at each
        // target-net sync) — compare the *best* loss reached in the second half
        // against the initial window rather than a sync-sensitive end-window.
        let mut davi = Davi::new(
            &DaviConfig { k_max: 5, hidden: 96, blocks: 2, lr: 5e-4, target_sync_every: 100 },
            Device::Cpu,
        )
        .unwrap();
        let mut rng = Rng::new(7);
        let mut first_window = 0.0f32;
        let mut best_second_half = f32::INFINITY;
        let window = 20;
        let steps = 600;
        for step in 0..steps {
            let states: Vec<State> = (0..64).map(|_| scramble(&mut rng, 5).0).collect();
            let l = davi.train_step(&states).unwrap();
            if step < window {
                first_window += l;
            }
            if step >= steps / 2 {
                best_second_half = best_second_half.min(l);
            }
        }
        let first = first_window / window as f32;
        assert!(
            best_second_half < first * 0.6,
            "loss did not trend down enough: first-window {:.3} -> best-second-half {:.3}",
            first,
            best_second_half
        );
    }
}
