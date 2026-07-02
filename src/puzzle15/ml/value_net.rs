//! The solver's learned cost-to-go value network `V(s)`.
//!
//! A small MLP: one-hot board (`256`) → hidden ReLU stack → single scalar (the
//! estimated number of moves to the goal). No admissible heuristic feeds it —
//! it sees only the raw encoded board (see TRAINING.md). Trained by DAVI
//! (`super::davi`) and consumed by the weighted search (`super::bwas`).

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use super::encoding::{encode_batch, ENCODED_DIM};
use crate::puzzle15::state::State;

/// Default hidden width for the PoC (see TRAINING.md §hyperparameters).
pub const DEFAULT_HIDDEN: usize = 512;

/// Cost-to-go MLP: `256 -> hidden -> hidden -> hidden -> 1`, ReLU between layers,
/// no activation on the scalar output (DAVI regresses a raw move count).
pub struct ValueNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    out: Linear,
}

impl ValueNet {
    /// Build the network, registering parameters under `vb` (names `l1.*`,
    /// `l2.*`, `l3.*`, `out.*`).
    pub fn new(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            l1: linear(ENCODED_DIM, hidden, vb.pp("l1"))?,
            l2: linear(hidden, hidden, vb.pp("l2"))?,
            l3: linear(hidden, hidden, vb.pp("l3"))?,
            out: linear(hidden, 1, vb.pp("out"))?,
        })
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
    /// `x: [batch, 256]` → `[batch, 1]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.out.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::state::GOAL;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}_{}", std::process::id(), name))
    }

    #[test]
    fn forward_shape_is_batch_by_one() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let net = ValueNet::new(vb, 32).unwrap();
        let x = encode_batch(&[GOAL, GOAL], &dev).unwrap();
        let y = net.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 1]);
    }

    #[test]
    fn save_load_round_trip_reproduces_output() {
        let dev = Device::Cpu;
        let states = [GOAL, GOAL.apply(crate::puzzle15::state::Move::Up)];

        // Net A with random init.
        let vm_a = VarMap::new();
        let net_a = ValueNet::new(VarBuilder::from_varmap(&vm_a, DType::F32, &dev), 32).unwrap();
        let out_a = net_a.values(&states, &dev).unwrap();

        let path = tmp_path("value_net_roundtrip.safetensors");
        vm_a.save(&path).unwrap();

        // Net B: fresh random init, then load A's weights by name.
        let mut vm_b = VarMap::new();
        let net_b = ValueNet::new(VarBuilder::from_varmap(&vm_b, DType::F32, &dev), 32).unwrap();
        let out_b_before = net_b.values(&states, &dev).unwrap();
        vm_b.load(&path).unwrap();
        let out_b_after = net_b.values(&states, &dev).unwrap();

        // Before load: B differs from A (different random init). After: identical.
        assert_ne!(out_a, out_b_before, "distinct random inits unexpectedly equal");
        assert_eq!(out_a, out_b_after, "loaded weights did not reproduce A's output");

        let _ = std::fs::remove_file(&path);
    }
}
