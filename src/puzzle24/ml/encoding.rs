//! One-hot encoding of a 24-puzzle [`State`] for the value/policy networks.
//!
//! Each of the 25 cells is one-hot over its 25 possible values (tiles `1..=24`
//! plus the blank `0`), giving a flat `25 * 25 = 625`-dimensional `f32` vector:
//! `v[cell * 25 + value] = 1.0`. This is the standard DeepCubeA-style encoding —
//! it lets the network see "which tile is at which cell" without imposing any
//! ordinal structure on tile identities.

use candle_core::{Device, Result, Tensor};

use crate::puzzle24::state::{State, N_CELLS};

/// Input dimensionality of the one-hot encoding: `N_CELLS * N_CELLS = 625`.
pub const ENCODED_DIM: usize = N_CELLS * N_CELLS;

/// One-hot encode a single board into a `[625]` `f32` array.
pub fn encode_one(s: &State) -> [f32; ENCODED_DIM] {
    let mut out = [0.0f32; ENCODED_DIM];
    for (cell, &value) in s.0.iter().enumerate() {
        out[cell * N_CELLS + value as usize] = 1.0;
    }
    out
}

/// Encode a batch of boards into a `[n, 625]` `f32` tensor on `device`.
pub fn encode_batch(states: &[State], device: &Device) -> Result<Tensor> {
    let n = states.len();
    let mut flat = Vec::with_capacity(n * ENCODED_DIM);
    for s in states {
        flat.extend_from_slice(&encode_one(s));
    }
    Tensor::from_vec(flat, (n, ENCODED_DIM), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::{Move, GOAL};

    #[test]
    fn goal_has_one_hot_per_cell() {
        let enc = encode_one(&GOAL);
        // Exactly one hot bit per cell.
        let ones: f32 = enc.iter().sum();
        assert_eq!(ones as usize, N_CELLS);
        // GOAL is [1,2,3,...,24,0]; cell `c` holds value `c+1` for c<24, blank at 24.
        for cell in 0..N_CELLS {
            let value = GOAL.0[cell] as usize;
            assert_eq!(enc[cell * N_CELLS + value], 1.0, "cell {} value {}", cell, value);
        }
    }

    #[test]
    fn distinct_states_encode_distinctly() {
        let a = encode_one(&GOAL);
        let b = encode_one(&GOAL.apply(Move::Up));
        assert_ne!(a, b);
    }

    #[test]
    fn batch_shape_and_content() {
        let dev = Device::Cpu;
        let states = [GOAL, GOAL.apply(Move::Up), GOAL.apply(Move::Left)];
        let t = encode_batch(&states, &dev).unwrap();
        assert_eq!(t.dims(), &[3, ENCODED_DIM]);
        // Row 0 must match encode_one(GOAL).
        let row0 = t.get(0).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(row0, encode_one(&GOAL).to_vec());
        // Each row sums to 25 (one hot per cell).
        let sums = t.sum(1).unwrap().to_vec1::<f32>().unwrap();
        for s in sums {
            assert_eq!(s as usize, N_CELLS);
        }
    }
}
