//! Optimal search algorithms over [`crate::puzzle8::state::State`].

pub mod heuristic;
pub mod idastar;

pub use heuristic::{Heuristic, ManhattanHeuristic, TableHeuristic};
pub use idastar::idastar;
