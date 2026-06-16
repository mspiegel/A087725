//! Complete enumeration of deep 15-puzzle boards (optimal depth ≥ T) by a
//! top-down frontier from the antipodes, gated by the exact A087725 counts.
//!
//! See `ENUMERATION.md` at the repo root for the method and correctness
//! argument. The frontier itself is solve-free; the strong PDB heuristic is used
//! only by the optional pocket fallback in [`frontier::process_layer`].

pub mod antipodes;
pub mod cache;
pub mod frontier;
pub mod histogram;
pub mod store;

pub use frontier::{run, LayerReport};
pub use histogram::Histogram;
pub use store::Store;
