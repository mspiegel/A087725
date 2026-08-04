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
// The 24-puzzle is mmap-only by design: its tables (cwd_mm.bin 4.29 GB,
// cwd_lm_mm.bin 4.29 GB, three 10.92 GB zPDBs) are mapped, never read into
// owned memory, so there is no non-mmap code path to maintain. Building
// without `mmap` therefore omits the module rather than failing to compile;
// every puzzle24 target declares `required-features = ["mmap"]`.
#[cfg(feature = "mmap")]
pub mod puzzle24;
pub mod puzzle8;
