//! 24-puzzle (5×5) optimal-search infrastructure — Milestone 5.
//!
//! Ports the verified 15-puzzle machinery to the 5×5 board (`25!/2 ≈ 7.76 ×
//! 10²⁴` states). The full distance table is hopeless at this scale, so the
//! heuristic path is IDA\* + **zero-aware** additive pattern databases
//! (Clausecker & Reinefeld, SOCS 2019) with 1-bit compression; see
//! `docs/zpdb-codec-spec.md`. Grown out incrementally per the Phase plan.
//!
//! Unlike the even-width 15-puzzle, the 24-puzzle is **odd-width**, so it shares
//! the 8-puzzle's blank-independent solvability rule.

pub mod frame;
pub mod pdb;
#[cfg(feature = "pmu-counters")]
pub mod pmu;
#[cfg(feature = "probe-locality")]
pub mod probe_locality;
pub mod rank;
pub mod search;
pub mod state;
pub mod symmetry;

#[cfg(feature = "ml")]
pub mod ml;

pub use rank::{rank, unrank};
pub use state::{
    Move, MoveSet, State, DIAMETER_LOWER, DIAMETER_UPPER, GOAL, N_CELLS, N_STATES, N_TILES, W,
};
