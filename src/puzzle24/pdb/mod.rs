//! Additive pattern databases (PDBs) for the 24-puzzle.
//!
//! Standard Korf-Felner additive PDBs over a disjoint tile partition: each PDB
//! stores, for a projection onto its pattern tiles, the minimum pattern-tile
//! moves to home them (anon moves free). Summing disjoint patterns stays
//! admissible. This is the directly usable Korf-max heuristic for the
//! 24-puzzle, and the admissibility oracle for the zero-aware PDBs
//! (`docs/zpdb-codec-spec.md`): the zero-aware value is pointwise ≥ this one.
//!
//! The full 6-tile build (P(25,6) = 127,512,000 entries, ~127 MB) is produced by
//! the `build_pdb24` binary; this module is verified on small patterns in tests.

pub mod build;
pub mod db;
pub mod heuristic;
pub mod pattern;
pub mod zbuild;
pub mod zcodec;
pub mod zdb;
pub mod zheuristic;
pub mod zpdb;

pub use build::{build, UNVISITED};
#[cfg(feature = "parallel")]
pub use build::build_parallel;
pub use db::{file_size_for, LoadError, PatternDb};
pub use heuristic::{
    AdditivePdbHeuristic, KorfCtx, KorfPdbInc, MaxHeuristic, PdbHeuristic, ReflectedHeuristic,
    ZpdbCtx, ZpdbInc,
};
pub use pattern::{Pattern, ProjectedState, ANON};
pub use zdb::ZPatternDb;
pub use zheuristic::{AdditiveZpdbHeuristic, ZpdbHeuristic, ZpdbPlusCtx, ZpdbPlusInc};
