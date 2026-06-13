//! 8-puzzle (3×3) optimal solver and compression study.
//!
//! See `DESIGN.md` at the project root for the long-term goal (15-puzzle, then
//! 24-puzzle) and the seven compression directions under investigation. The
//! 8-puzzle code in this module is the verified reference implementation and
//! is not expected to change as the project scales up.

pub mod state;
pub mod rank;
pub mod bfs;
pub mod io;
pub mod moves;
pub mod search;
pub mod symmetry;
pub mod pdb;

pub use state::{GOAL, N_STATES, DIAMETER, Move, MoveSet, State};
pub use rank::{rank, unrank};
pub use bfs::{DistanceTable, UNVISITED};
