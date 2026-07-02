//! Adversarial generator/solver co-training for the 15-puzzle (proof of concept).
//!
//! See `TRAINING.md` at the project root for the full design. In brief:
//!
//! - A learned cost-to-go value network (`value_net`) trained by Deep Approximate
//!   Value Iteration (`davi`), deployed via a weighted batch A\* search (`bwas`) —
//!   the *solver*.
//! - A learned move-sequence policy (`policy_net`) trained by REINFORCE
//!   (`generator`) to construct boards that are hard for the current solver,
//!   rewarded by regret against a fixed `idastar` + Walking-Distance baseline.
//! - An alternating co-training loop (`alternate`) and an evaluation harness
//!   (`eval`) that scores the solver against the 15-puzzle's known depth-80
//!   antipodes and enumerated deep layers.
//!
//! The whole module requires the `ml` feature (it cannot build without
//! `candle-core`/`candle-nn`).

pub mod alternate;
pub mod bwas;
pub mod checkpoint;
pub mod davi;
pub mod device;
pub mod encoding;
pub mod eval;
pub mod generator;
pub mod policy_net;
pub mod profile;
pub mod scramble;
pub mod value_net;
