//! Pattern databases (PDBs) for the 8-puzzle.
//!
//! A PDB stores, for each *projection* of an 8-puzzle state onto a chosen
//! subset of tiles (the [`Pattern`]), the minimum number of moves needed to
//! bring those pattern tiles to their goal positions. The non-pattern tiles
//! are treated as anonymous "filler" — moves that swap the blank with a
//! filler tile cost zero (they're free in the projected accounting); moves
//! that swap the blank with a pattern tile cost one.
//!
//! By construction this yields an **admissible** lower bound on the full
//! puzzle's distance: any solution must also bring the pattern tiles home,
//! taking at least the PDB-stored cost. Plugging it into IDA\* (via the
//! [`crate::search::Heuristic`] trait) gives an optimal solver whose storage
//! is `O(projected state space size)` rather than `O(full state space size)`.
//!
//! **Additive PDBs** (Korf & Felner 2002): if two patterns are *disjoint*
//! (share no tiles), then summing their PDB values is still admissible —
//! every full-puzzle move increments at most one pattern's PDB. We expose
//! [`AdditivePdbHeuristic`] for this case.
//!
//! On the 8-puzzle, PDBs are pedagogically valuable but not dramatic
//! storage-wise — the full table is only 181 KB to begin with. The same
//! infrastructure transfers (algorithmically, not codewise) to the 15-puzzle,
//! where PDBs compress storage by ~10^5× and are the canonical optimal
//! solver.

pub mod pattern;
pub mod build;
pub mod db;
pub mod heuristic;

pub use db::{LoadError, PatternDb};
pub use heuristic::{AdditivePdbHeuristic, PdbHeuristic};
pub use pattern::{Pattern, ProjectedState, ANON};
