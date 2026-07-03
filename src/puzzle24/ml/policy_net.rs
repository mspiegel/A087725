//! The generator's move-selection policy network.
//!
//! Maps a one-hot board (`625`) to 4 move logits (order = `Move::ALL`). During a
//! rollout the generator walks from GOAL, sampling one move per step from this
//! policy with illegal moves masked out. `sample_move` returns both the chosen
//! move and its **log-probability as a graph tensor**, so REINFORCE can
//! backpropagate the policy-gradient loss through it (`super::generator`).

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, ops, Linear, VarBuilder};

use super::encoding::{encode_batch, ENCODED_DIM};
use super::scramble::Rng;
use crate::puzzle24::state::{Move, State};

/// Default hidden width for the policy net.
pub const DEFAULT_HIDDEN: usize = 512;

/// Large negative masking constant (a finite stand-in for `-inf` that keeps
/// `log_softmax` numerically safe while driving illegal-move probs to ~0).
const MASK_NEG: f32 = -1.0e30;

pub struct PolicyNet {
    l1: Linear,
    l2: Linear,
    out: Linear,
}

impl PolicyNet {
    pub fn new(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            l1: linear(ENCODED_DIM, hidden, vb.pp("l1"))?,
            l2: linear(hidden, hidden, vb.pp("l2"))?,
            out: linear(hidden, 4, vb.pp("out"))?,
        })
    }
}

impl Module for PolicyNet {
    /// `x: [batch, 625]` → `[batch, 4]` raw move logits.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        self.out.forward(&x)
    }
}

/// Sample one legal move from the policy at state `s` (blank at `blank`).
///
/// Illegal moves — and the optional `banned` move (forbidding an immediate undo)
/// — are masked to ~0 probability before sampling. Returns the chosen move and
/// its log-probability **as a graph tensor** (for REINFORCE).
pub fn sample_move(
    net: &PolicyNet,
    s: &State,
    blank: u8,
    banned: Option<Move>,
    device: &Device,
    rng: &mut Rng,
) -> Result<(Move, Tensor)> {
    let x = encode_batch(std::slice::from_ref(s), device)?; // [1, 625]
    let logits = net.forward(&x)?.reshape((4,))?; // [4]

    let legal = State::legal_moves_at(blank);
    let allowed = |m: Move| legal.contains(m) && Some(m) != banned;
    let mask_vec: Vec<f32> =
        Move::ALL.iter().map(|&m| if allowed(m) { 0.0 } else { MASK_NEG }).collect();
    let mask = Tensor::from_vec(mask_vec, (4,), device)?;
    let masked = logits.add(&mask)?;
    let log_probs = ops::log_softmax(&masked, 0)?; // [4], in the autograd graph

    // Detached probabilities for categorical sampling.
    let probs: Vec<f32> = log_probs.exp()?.to_vec1::<f32>()?;
    let mut idx = 0usize;
    let mut acc = rng.gen_f32();
    for (i, &p) in probs.iter().enumerate() {
        idx = i;
        acc -= p;
        if acc <= 0.0 {
            break;
        }
    }
    // Guard against float underflow picking a masked (near-zero) entry.
    if !allowed(Move::ALL[idx]) {
        idx = (0..4)
            .filter(|&i| allowed(Move::ALL[i]))
            .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
            .expect("at least one legal non-banned move");
    }

    let chosen = Move::ALL[idx];
    let log_prob = log_probs.get(idx)?; // scalar tensor, graph-connected
    Ok((chosen, log_prob))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::GOAL;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn forward_shape_is_batch_by_four() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let net = PolicyNet::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), 32).unwrap();
        let x = encode_batch(&[GOAL, GOAL], &dev).unwrap();
        assert_eq!(net.forward(&x).unwrap().dims(), &[2, 4]);
    }

    #[test]
    fn only_legal_moves_are_sampled() {
        // GOAL's blank is at the bottom-right corner (cell 24 of the 5x5): only Up
        // and Left are legal. Over many samples we must never see Down or Right.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let net = PolicyNet::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), 32).unwrap();
        let mut rng = Rng::new(2024);
        let blank = GOAL.blank_pos();
        assert_eq!(blank, 24);
        for _ in 0..2000 {
            let (m, lp) = sample_move(&net, &GOAL, blank, None, &dev, &mut rng).unwrap();
            assert!(
                m == Move::Up || m == Move::Left,
                "sampled illegal move {:?} from GOAL corner",
                m
            );
            let lpv = lp.to_scalar::<f32>().unwrap();
            assert!(lpv <= 1e-4 && lpv.is_finite(), "bad log-prob {}", lpv);
        }
    }

    #[test]
    fn banned_move_is_never_sampled() {
        // Move the blank to an interior cell so all 4 moves are legal; banning one
        // must ensure it's never chosen.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let net = PolicyNet::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), 32).unwrap();
        let mut rng = Rng::new(77);
        let mut s = GOAL;
        for m in [Move::Left, Move::Left, Move::Up, Move::Up] {
            if State::legal_moves_at(s.blank_pos()).contains(m) {
                s = s.apply(m);
            }
        }
        let blank = s.blank_pos();
        assert_eq!(State::legal_moves_at(blank).len(), 4, "need interior blank");
        for _ in 0..2000 {
            let (m, _) = sample_move(&net, &s, blank, Some(Move::Up), &dev, &mut rng).unwrap();
            assert_ne!(m, Move::Up, "banned move was sampled");
        }
    }

    #[test]
    fn masked_illegal_probs_are_negligible() {
        // The two illegal moves from GOAL's corner must carry ~0 probability.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let net = PolicyNet::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), 32).unwrap();
        let x = encode_batch(&[GOAL], &dev).unwrap();
        let logits = net.forward(&x).unwrap().reshape((4,)).unwrap();
        let legal = State::legal_moves_at(GOAL.blank_pos());
        let mask_vec: Vec<f32> =
            Move::ALL.iter().map(|&m| if legal.contains(m) { 0.0 } else { MASK_NEG }).collect();
        let mask = Tensor::from_vec(mask_vec, (4,), &dev).unwrap();
        let probs = ops::log_softmax(&logits.add(&mask).unwrap(), 0)
            .unwrap()
            .exp()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, &m) in Move::ALL.iter().enumerate() {
            if !legal.contains(m) {
                assert!(probs[i] < 1e-6, "illegal move {:?} had prob {}", m, probs[i]);
            }
        }
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "probs sum to {}", total);
    }
}
