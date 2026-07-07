//! Adversarial generator/solver co-training for the 24-puzzle.
//!
//! A parallel port of `crate::puzzle15::ml` (see `TRAINING.md`). In brief:
//!
//! - A learned cost-to-go value network (`value_net`) trained by Deep Approximate
//!   Value Iteration (`davi`), deployed via a weighted batch A\* search (`bwas`) —
//!   the *solver*.
//! - A learned move-sequence policy (`policy_net`) trained by REINFORCE
//!   (`generator`) to construct boards that are hard for the current solver,
//!   rewarded by regret against a fixed **beam-search** suboptimal baseline
//!   (`beam`) using an admissible heuristic (Walking Distance by default).
//! - An alternating co-training loop (`alternate`) and an evaluation harness
//!   (`eval`) that scores the solver on mid-depth boards labeled with their true
//!   optimal length by the admissible `idastar` (feasible to ~depth 50).
//!
//! Unlike the 15-puzzle, the 24-puzzle has no known optimal beyond ~depth 50 and
//! no enumerated deep-board ground truth, so success is measured as
//! excess-over-optimal on mid-depth boards, then solution length vs. the beam
//! baseline on deep boards / `R`.
//!
//! The whole module requires the `ml` feature (it cannot build without
//! `candle-core`/`candle-nn`).

pub mod alternate;
pub mod beam;
pub mod bidirectional;
pub mod bwas;
pub mod checkpoint;
pub mod corridor;
pub mod davi;
pub mod device;
pub mod encoding;
pub mod eval;
pub mod generator;
pub mod policy_net;
pub mod profile;
pub mod scramble;
pub mod value_net;
pub mod wdsearch;
