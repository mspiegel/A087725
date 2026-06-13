//! Pattern databases (PDBs) for the 15-puzzle.
//!
//! A PDB stores, for each *projection* of a 15-puzzle state onto a chosen
//! subset of tiles (the [`pattern::Pattern`]), the minimum number of moves
//! needed to bring those pattern tiles to their goal positions. The
//! non-pattern tiles are treated as anonymous "filler" — moves that swap the
//! blank with a filler tile cost zero (free in the projected accounting);
//! moves that swap the blank with a pattern tile cost one.
//!
//! By construction this yields an **admissible** lower bound on the full
//! puzzle's distance: any solution must also bring the pattern tiles home,
//! taking at least the PDB-stored cost. Plugging it into IDA\* (via the
//! [`crate::puzzle15::search::Heuristic`] trait) gives an optimal solver
//! whose storage is `O(projected state space size)` rather than
//! `O(full state space size)`.
//!
//! **Additive PDBs** (Korf & Felner 2002): if two patterns are *disjoint*
//! (share no tiles), then summing their PDB values is still admissible —
//! every full-puzzle move increments at most one pattern's PDB. The Korf 7-8
//! partition is the standard 15-puzzle baseline.
//!
//! At 15-puzzle scale the PDB engine is the *only* path to optimal solutions
//! within reasonable resources: the full distance table is ~5 TB.

pub mod pattern;
pub mod build;
pub mod db;
pub mod heuristic;

pub use pattern::{Pattern, ProjectedState, ANON};
pub use build::{build, UNVISITED};
pub use db::{LoadError, PatternDb};
pub use heuristic::{AdditivePdbHeuristic, MaxHeuristic, PdbHeuristic, ReflectedHeuristic};
