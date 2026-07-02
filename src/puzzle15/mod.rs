//! 15-puzzle (4×4) optimal solver.
//!
//! Lifts the verified 8-puzzle techniques to the 4×4 board at ~10.46 trillion
//! states. The full distance table is infeasible at this scale (~5 TB), so the
//! baseline is IDA\* + additive pattern databases per Korf (1997). Modules are
//! grown out incrementally per Milestone 3 of `DESIGN.md`.

pub mod state;
pub mod rank;
pub mod symmetry;
pub mod search;
pub mod pdb;
pub mod enumerate;
#[cfg(feature = "ml")]
pub mod ml;

pub use state::{
    GOAL, N_STATES, N_CELLS, N_TILES, W, DIAMETER, Move, MoveSet, State,
};
pub use rank::{rank, unrank};
