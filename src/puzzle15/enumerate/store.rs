//! The `rank → exact-depth` map plus per-depth rank buckets, and per-layer
//! file output.
//!
//! Membership and depth lookups go through a single `HashMap<u64, u8>`; the
//! per-depth `Vec<u64>` buckets let the frontier iterate a whole layer (to
//! expand it) and write it out. A board is stored at most once — the first
//! `insert` wins and pushes into its bucket.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::puzzle15::state::DIAMETER;

/// In-memory enumeration state: every discovered board with its exact depth.
pub struct Store {
    depth: std::collections::HashMap<u64, u8>,
    /// `layers[d]` holds the ranks of every board known to be at depth `d`.
    layers: Vec<Vec<u64>>,
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

impl Store {
    pub fn new() -> Self {
        Self {
            depth: std::collections::HashMap::new(),
            layers: vec![Vec::new(); DIAMETER as usize + 1],
        }
    }

    /// Reserve capacity for the rank map (avoids rehashing on big runs).
    pub fn reserve(&mut self, n: usize) {
        self.depth.reserve(n);
    }

    #[inline]
    pub fn contains(&self, r: u64) -> bool {
        self.depth.contains_key(&r)
    }

    #[inline]
    pub fn depth_of(&self, r: u64) -> Option<u8> {
        self.depth.get(&r).copied()
    }

    /// Record `r` at depth `d` if not already present. Returns `true` if it was
    /// newly inserted.
    #[inline]
    pub fn insert(&mut self, r: u64, d: u8) -> bool {
        if self.depth.insert(r, d).is_none() {
            self.layers[d as usize].push(r);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn layer(&self, d: u8) -> &[u64] {
        &self.layers[d as usize]
    }

    #[inline]
    pub fn count(&self, d: u8) -> usize {
        self.layers[d as usize].len()
    }

    /// Total boards stored across all depths.
    pub fn total(&self) -> usize {
        self.depth.len()
    }

    /// Write layer `d` to `path` as little-endian 6-byte (u48) ranks, sorted
    /// ascending so the file is deterministic and `sha256`-stable.
    pub fn write_layer(&self, d: u8, path: &Path) -> io::Result<()> {
        let mut ranks: Vec<u64> = self.layers[d as usize].clone();
        ranks.sort_unstable();
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        for r in ranks {
            // 16!/2 < 2^44, so the low 6 bytes hold the whole rank.
            w.write_all(&r.to_le_bytes()[..6])?;
        }
        w.flush()
    }
}
