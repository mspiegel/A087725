//! Heuristic quality η for the 24-puzzle (Clausecker & Schintke, SoCS 2021;
//! Clausecker, ZIB Report 20-17).
//!
//! η = |V|⁻¹ Σ_v w(v)·b^−h(v) is the constant factor by which a consistent
//! heuristic `h` reduces the number of nodes IDA\* expands, relative to
//! uninformed iterative deepening with the same branching factor `b`. It is
//! dominated by the rare states close to [`GOAL`](crate::puzzle24::state::GOAL),
//! so a uniform sample cannot estimate it; instead the state space is
//! stratified by distance `k` from GOAL:
//!
//! - small `k`: exact breadth-first layers ([`layers`]);
//! - moderate `k`: random walks of `k` steps from GOAL under a pruned move rule
//!   ([`walker`]), accepted when the end board is proven to lie at distance
//!   exactly `k`, reweighted by the probability of reaching it;
//! - the remainder `d ≥ L`: uniform random states proven to lie at least `L`
//!   away ([`tail`]), and, for the states with small Manhattan distance that
//!   carry most of a Manhattan-like heuristic's tail but that uniform draws
//!   almost never reach, uniform draws within each Manhattan level weighted by
//!   its exact size ([`md_levels`]).
//!
//! [`estimate`] holds the Horvitz–Thompson estimators, [`weights`] the branching
//! factor and the equilibrium weightings `w`.

pub mod estimate;
pub mod layers;
pub mod md_levels;
pub mod reach;
pub mod rng;
pub mod samples;
pub mod sphere;
pub mod tail;
pub mod walker;
pub mod weights;

pub use estimate::{batch_means, Accum, PairAccum, Z95};
pub use layers::{for_each_layer, pack, unpack, A090031};
pub use rng::Rng;
pub use walker::Walker;
pub use weights::{blank_weights, branching_factor, tree_equilibrium, Weighting};
