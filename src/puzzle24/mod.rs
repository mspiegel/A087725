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

pub mod state;
pub mod rank;
pub mod symmetry;
pub mod search;
pub mod pdb;

pub use rank::{rank, unrank};
pub use state::{
    DIAMETER_LOWER, DIAMETER_UPPER, Move, MoveSet, State, GOAL, N_CELLS, N_STATES, N_TILES, W,
};
