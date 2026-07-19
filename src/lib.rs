//! Sliding-puzzle optimal solvers and compression study.
//!
//! See `DESIGN.md` at the project root for the long-term goal (15-puzzle, then
//! 24-puzzle) and the seven compression directions under investigation.
//!
//! The crate is organized by puzzle size:
//!
//! - [`puzzle8`] — the verified 8-puzzle (3×3) implementation. Frozen
//!   reference; not expected to change as the project scales up.
//! - [`puzzle15`] — the 15-puzzle (4×4) at ~10.46 trillion states: IDA\* +
//!   additive pattern databases (Milestone 3).
//! - [`puzzle24`] — the 24-puzzle (5×5), Milestone 5: zero-aware PDBs toward
//!   tightening the open `[152, 205]` STM diameter bounds.

pub mod puzzle15;
pub mod puzzle24;
pub mod puzzle8;
