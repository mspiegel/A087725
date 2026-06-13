//! Sliding-puzzle optimal solvers and compression study.
//!
//! See `DESIGN.md` at the project root for the long-term goal (15-puzzle, then
//! 24-puzzle) and the seven compression directions under investigation.
//!
//! The crate is organized by puzzle size:
//!
//! - [`puzzle8`] — the verified 8-puzzle (3×3) implementation. Frozen
//!   reference; not expected to change as the project scales up.
//! - `puzzle15` — coming in Milestone 3, lifting the proven 8-puzzle
//!   techniques to the 15-puzzle (4×4) at ~10.46 trillion states.

pub mod puzzle8;
pub mod puzzle15;
