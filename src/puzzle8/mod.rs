//! 8-puzzle (3×3) optimal solver and compression study.
//!
//! See `DESIGN.md` at the project root for the long-term goal (15-puzzle, then
//! 24-puzzle) and the seven compression directions under investigation. The
//! 8-puzzle code in this module is the verified reference implementation and
//! is not expected to change as the project scales up.

pub mod bfs;
pub mod io;
pub mod moves;
pub mod pdb;
pub mod rank;
pub mod search;
pub mod state;
pub mod symmetry;

pub use bfs::{DistanceTable, UNVISITED};
pub use rank::{rank, unrank};
pub use state::{Move, MoveSet, State, DIAMETER, GOAL, N_STATES};
