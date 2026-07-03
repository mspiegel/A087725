//! The solver's learned cost-to-go value network `V(s)`.
//!
//! A DeepCubeA-style residual MLP: one-hot board (`625`) → input projection →
//! a stack of `blocks` residual blocks → single scalar (the estimated number of
//! moves to the goal). No admissible heuristic feeds it — it sees only the raw
//! encoded board. Trained by DAVI (`super::davi`) and consumed by the weighted
//! search (`super::bwas`).
//!
//! **Normalization = a hand-written RmsNorm** (`x / sqrt(mean(x^2)+eps) * weight`,
//! per-sample, no bias): batch-invariant for the variable-size BWAS/target-net
//! forwards, and — unlike candle 0.9's fused layer-norm/rms-norm — runs on Metal.
//! See the 15-puzzle version / `ml_bench` for the full rationale.

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{linear, Init, Linear, VarBuilder};

use super::encoding::{encode_batch, ENCODED_DIM};
use crate::puzzle24::state::State;

/// RmsNorm built from primitive ops (Metal-compatible): a learnable per-feature
/// scale, no bias, no mean subtraction. `y = x / sqrt(mean(x^2) + eps) * weight`.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dim(D::Minus1)? as f64;
        let ms = (x.sqr()?.sum_keepdim(D::Minus1)? / d)?;
        let x_normed = x.broadcast_div(&(ms + self.eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

/// Default hidden width — larger than the 15-puzzle (the 24-puzzle is much
/// harder; see the plan's §6 hyperparameters).
pub const DEFAULT_HIDDEN: usize = 1024;
/// Default number of residual blocks in the body.
pub const DEFAULT_BLOCKS: usize = 6;

const LN_EPS: f64 = 1e-5;

/// A residual block operating at a fixed `width`:
/// `y = relu( x + norm2(l2( relu(norm1(l1(x))) )) )`.
struct ResBlock {
    l1: Linear,
    n1: RmsNorm,
    l2: Linear,
    n2: RmsNorm,
}

impl ResBlock {
    fn new(width: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: linear(width, width, vb.pp("l1"))?,
            n1: RmsNorm::new(width, LN_EPS, vb.pp("n1"))?,
            l2: linear(width, width, vb.pp("l2"))?,
            n2: RmsNorm::new(width, LN_EPS, vb.pp("n2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.n1.forward(&self.l1.forward(x)?)?.relu()?;
        let h = self.n2.forward(&self.l2.forward(&h)?)?;
        (x + h)?.relu()
    }
}

/// Cost-to-go residual MLP: `625 -> hidden` input projection, then `blocks`
/// residual blocks at `hidden`, then `hidden -> 1` (raw move-count output).
pub struct ValueNet {
    in_proj: Linear,
    in_norm: RmsNorm,
    blocks: Vec<ResBlock>,
    out: Linear,
}

impl ValueNet {
    /// Build the network, registering parameters under `vb` (names `in_proj.*`,
    /// `in_norm.*`, `block{i}.*`, `out.*`). Both the online and target nets must
    /// be built with the *same* `hidden`/`blocks` so their parameter names line
    /// up for the target sync.
    pub fn new(vb: VarBuilder, hidden: usize, blocks: usize) -> Result<Self> {
        let in_proj = linear(ENCODED_DIM, hidden, vb.pp("in_proj"))?;
        let in_norm = RmsNorm::new(hidden, LN_EPS, vb.pp("in_norm"))?;
        let mut body = Vec::with_capacity(blocks);
        for i in 0..blocks {
            body.push(ResBlock::new(hidden, vb.pp(format!("block{i}")))?);
        }
        let out = linear(hidden, 1, vb.pp("out"))?;
        Ok(Self { in_proj, in_norm, blocks: body, out })
    }

    /// Convenience: encode `states`, run a forward pass, return one scalar
    /// cost-to-go per state. Used by the search driver and evaluation harness.
    pub fn values(&self, states: &[State], device: &Device) -> Result<Vec<f32>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let x = encode_batch(states, device)?;
        let y = self.forward(&x)?; // [n, 1]
        y.flatten_all()?.to_vec1::<f32>()
    }
}

impl Module for ValueNet {
    /// `x: [batch, 625]` → `[batch, 1]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.in_norm.forward(&self.in_proj.forward(x)?)?.relu()?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        self.out.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::GOAL;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ml24_{}_{}", std::process::id(), name))
    }

    #[test]
    fn forward_shape_is_batch_by_one() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let net = ValueNet::new(vb, 32, 2).unwrap();
        let x = encode_batch(&[GOAL, GOAL], &dev).unwrap();
        let y = net.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 1]);
    }

    #[test]
    fn single_and_large_batch_are_consistent() {
        // RmsNorm makes batch=1 and a large batch give identical per-row outputs
        // (the property that motivates it over BatchNorm for search).
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let net = ValueNet::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), 48, 3).unwrap();
        let states = [GOAL, GOAL.apply(crate::puzzle24::state::Move::Up)];
        let batched = net.values(&states, &dev).unwrap();
        let one_by_one: Vec<f32> =
            states.iter().flat_map(|s| net.values(std::slice::from_ref(s), &dev).unwrap()).collect();
        for (a, b) in batched.iter().zip(one_by_one.iter()) {
            assert!((a - b).abs() < 1e-4, "batch vs single differ: {a} vs {b}");
        }
    }

    #[test]
    fn save_load_round_trip_reproduces_output() {
        let dev = Device::Cpu;
        let states = [GOAL, GOAL.apply(crate::puzzle24::state::Move::Up)];

        let vm_a = VarMap::new();
        let net_a = ValueNet::new(VarBuilder::from_varmap(&vm_a, DType::F32, &dev), 32, 2).unwrap();
        let out_a = net_a.values(&states, &dev).unwrap();

        let path = tmp_path("value_net_roundtrip.safetensors");
        vm_a.save(&path).unwrap();

        let mut vm_b = VarMap::new();
        let net_b = ValueNet::new(VarBuilder::from_varmap(&vm_b, DType::F32, &dev), 32, 2).unwrap();
        let out_b_before = net_b.values(&states, &dev).unwrap();
        vm_b.load(&path).unwrap();
        let out_b_after = net_b.values(&states, &dev).unwrap();

        assert_ne!(out_a, out_b_before, "distinct random inits unexpectedly equal");
        assert_eq!(out_a, out_b_after, "loaded weights did not reproduce A's output");

        let _ = std::fs::remove_file(&path);
    }
}
