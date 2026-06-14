//! Optimal search algorithms over [`crate::puzzle8::state::State`].

pub mod heuristic;
pub mod idastar;
pub mod linear_conflict;
pub mod max;
pub mod walking_distance;

pub use heuristic::{Heuristic, ManhattanHeuristic, TableHeuristic};
pub use idastar::idastar;
pub use linear_conflict::LinearConflictHeuristic;
pub use max::MaxHeuristic;
pub use walking_distance::WalkingDistanceHeuristic;
