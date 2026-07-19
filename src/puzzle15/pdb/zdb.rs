//! A built zero-aware 15-puzzle PDB, plus its on-disk format.
//!
//! Unlike the 24-puzzle ZPDB (which 1-bit-packs `h` and recovers parity from
//! the bipartite cell-sum invariant), this stores **raw distance bytes** — one
//! `u8` per `(m, p, r)` entry. The 1-bit codec is unsound on the 4×4 board
//! (vertical moves are ±4 = even offsets, so the cell-sum parity argument
//! fails), and at 15-puzzle PDB scale the raw form is cheap enough.
//!
//! Binary format (little-endian): 4-byte magic `Z15D`, `u32` version, `u32`
//! pattern bitmask (bits 1..=15), `u32` reserved (0), `u64` entry total
//! (`ZpdbLayout::total()`), then `total` raw distance bytes. The pattern and
//! total are self-describing and cross-checked at load.
//!
//! Lookups:
//! - [`ZPatternDb::value`] — O(1) read of `raw[rank(proj)]` (the hot path; the
//!   incremental heuristic keeps `proj` updated and calls this).
//! - [`ZPatternDb::cold_lookup`] — O(1) absolute `h(state)` from a full state.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use super::pattern::{Pattern, ProjectedState};
use super::zbuild;
use super::zpdb::ZpdbLayout;
use crate::puzzle15::state::State;

pub const MAGIC: &[u8; 4] = b"Z15D";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 24;

/// Header size + raw-byte length for a built ZPDB of the given pattern.
pub fn file_size_for(pattern: Pattern) -> u64 {
    let total = ZpdbLayout::new(pattern).total();
    HEADER_BYTES as u64 + total
}

enum Storage {
    Owned(Vec<u8>),
    #[cfg(feature = "mmap")]
    Mmapped(memmap2::Mmap),
}

impl Storage {
    #[inline]
    fn raw(&self) -> &[u8] {
        match self {
            Storage::Owned(v) => v,
            #[cfg(feature = "mmap")]
            Storage::Mmapped(m) => &m[HEADER_BYTES..],
        }
    }
}

/// A built zero-aware 15-puzzle PDB, indexed by `(m, p, r)` via [`ZpdbLayout`].
pub struct ZPatternDb {
    pattern: Pattern,
    layout: ZpdbLayout,
    storage: Storage,
}

impl ZPatternDb {
    /// Wrap a freshly-built raw byte distance array.
    pub fn from_dist(pattern: Pattern, dist: Vec<u8>) -> Self {
        let layout = ZpdbLayout::new(pattern);
        assert_eq!(
            dist.len() as u64,
            layout.total(),
            "dist length {} != layout total {}",
            dist.len(),
            layout.total()
        );
        Self {
            pattern,
            layout,
            storage: Storage::Owned(dist),
        }
    }

    /// Build the ZPDB end-to-end (single-threaded BFS).
    pub fn build(pattern: Pattern) -> Self {
        let (dist, _) = zbuild::build_zpdb(pattern);
        Self::from_dist(pattern, dist)
    }

    /// Build the ZPDB end-to-end (multi-threaded BFS).
    #[cfg(feature = "parallel")]
    pub fn build_parallel(pattern: Pattern) -> Self {
        let (dist, _) = zbuild::build_zpdb_parallel(pattern);
        Self::from_dist(pattern, dist)
    }

    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    pub fn layout(&self) -> &ZpdbLayout {
        &self.layout
    }

    /// The raw distance payload (`HEADER_BYTES`-stripped).
    pub fn raw(&self) -> &[u8] {
        self.storage.raw()
    }

    /// Raw distance at a global `(m, p, r)` index.
    #[inline]
    pub fn raw_at(&self, idx: u64) -> u8 {
        self.storage.raw()[idx as usize]
    }

    /// O(1) zero-aware value for an already-projected state: rank `proj` against
    /// this pattern's layout and read the stored distance. The hot path.
    #[inline]
    pub fn value(&self, proj: &ProjectedState) -> u8 {
        self.storage.raw()[self.layout.rank(proj, self.pattern) as usize]
    }

    /// Differential lookup. With raw storage the absolute `h` is stored
    /// directly, so this ignores `old_h` and returns the neighbor's raw value.
    /// Kept for API parity with the 24-puzzle 1-bit codec.
    #[inline]
    pub fn diff_lookup(&self, neighbor_idx: u64, _old_h: u8) -> u8 {
        self.storage.raw()[neighbor_idx as usize]
    }

    /// Absolute lookup of `h(state)` from a full state.
    pub fn cold_lookup(&self, state: &State) -> u8 {
        let proj = ProjectedState::from_state(state, self.pattern);
        self.cold_lookup_proj(&proj)
    }

    /// Absolute lookup of `h` from a projected state — a direct table read.
    #[inline]
    pub fn cold_lookup_proj(&self, proj: &ProjectedState) -> u8 {
        self.value(proj)
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&self.pattern.0.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?;
        f.write_all(&self.layout.total().to_le_bytes())?;
        f.write_all(self.storage.raw())?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let mut f = File::open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        f.read_exact(&mut header)?;
        let (pattern, total) = parse_header(&header)?;
        let layout = ZpdbLayout::new(pattern);
        if layout.total() != total {
            return Err(LoadError::TotalMismatch {
                in_file: total,
                expected: layout.total(),
            });
        }
        let mut raw = vec![0u8; total as usize];
        f.read_exact(&mut raw)?;
        let mut tail = [0u8; 1];
        match f.read(&mut tail)? {
            0 => Ok(Self {
                pattern,
                layout,
                storage: Storage::Owned(raw),
            }),
            _ => Err(LoadError::TrailingBytes),
        }
    }

    #[cfg(feature = "mmap")]
    pub fn load_mmap(path: &Path) -> Result<Self, LoadError> {
        let f = File::open(path)?;
        // SAFETY: ZPDB files are write-once-then-immutable build artifacts.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() < HEADER_BYTES {
            return Err(LoadError::ShortFile { got: map.len() });
        }
        let header: [u8; HEADER_BYTES] = map[..HEADER_BYTES].try_into().unwrap();
        let (pattern, total) = parse_header(&header)?;
        let layout = ZpdbLayout::new(pattern);
        if layout.total() != total {
            return Err(LoadError::TotalMismatch {
                in_file: total,
                expected: layout.total(),
            });
        }
        let expected = HEADER_BYTES as u64 + total;
        if (map.len() as u64) != expected {
            return Err(LoadError::SizeMismatch {
                got: map.len() as u64,
                expected,
            });
        }
        Ok(Self {
            pattern,
            layout,
            storage: Storage::Mmapped(map),
        })
    }
}

fn parse_header(header: &[u8; HEADER_BYTES]) -> Result<(Pattern, u64), LoadError> {
    let magic: &[u8; 4] = header[0..4].try_into().unwrap();
    if magic != MAGIC {
        return Err(LoadError::BadMagic(*magic));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    let bits = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let reserved = u32::from_le_bytes(header[12..16].try_into().unwrap());
    if reserved != 0 {
        return Err(LoadError::ReservedNonZero);
    }
    let total = u64::from_le_bytes(header[16..24].try_into().unwrap());
    Ok((Pattern(bits), total))
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    ReservedNonZero,
    TrailingBytes,
    ShortFile { got: usize },
    SizeMismatch { got: u64, expected: u64 },
    TotalMismatch { in_file: u64, expected: u64 },
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {e}"),
            LoadError::BadMagic(m) => write!(f, "bad magic: {m:?}, expected {MAGIC:?}"),
            LoadError::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            LoadError::ReservedNonZero => write!(f, "reserved bytes must be zero"),
            LoadError::TrailingBytes => write!(f, "file has trailing bytes after payload"),
            LoadError::ShortFile { got } => write!(f, "file too short: {got} bytes"),
            LoadError::SizeMismatch { got, expected } => {
                write!(f, "file size {got} != expected {expected}")
            }
            LoadError::TotalMismatch { in_file, expected } => {
                write!(
                    f,
                    "entry total mismatch: file {in_file} vs layout {expected}"
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::state::GOAL;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}_{}", std::process::id(), name))
    }

    #[test]
    fn build_then_cold_lookup_goal_is_zero() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        assert_eq!(zdb.cold_lookup(&GOAL), 0);
    }

    #[test]
    fn save_load_round_trip_owned() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        let path = tmp_path("p15_zpdb_roundtrip.bin");
        zdb.save(&path).unwrap();
        let loaded = ZPatternDb::load(&path).unwrap();
        assert_eq!(loaded.pattern().0, zdb.pattern().0);
        assert_eq!(loaded.raw(), zdb.raw());
        assert_eq!(loaded.layout().total(), zdb.layout().total());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_size_matches_format() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        let path = tmp_path("p15_zpdb_filesize.bin");
        zdb.save(&path).unwrap();
        let actual = std::fs::metadata(&path).unwrap().len();
        assert_eq!(actual, file_size_for(zdb.pattern()));
        assert_eq!(actual, HEADER_BYTES as u64 + zdb.layout().total());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("p15_zpdb_bad_magic.bin");
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&zdb.pattern().0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&zdb.layout().total().to_le_bytes()).unwrap();
        f.write_all(zdb.raw()).unwrap();
        drop(f);
        assert!(matches!(
            ZPatternDb::load(&path),
            Err(LoadError::BadMagic(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_total_mismatch_in_header() {
        let path = tmp_path("p15_zpdb_total_mismatch.bin");
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        let mut f = File::create(&path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&zdb.pattern().0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(zdb.layout().total() + 8).to_le_bytes())
            .unwrap();
        f.write_all(zdb.raw()).unwrap();
        drop(f);
        assert!(matches!(
            ZPatternDb::load(&path),
            Err(LoadError::TotalMismatch { .. })
        ));
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_load_matches_owned() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        let path = tmp_path("p15_zpdb_mmap.bin");
        zdb.save(&path).unwrap();
        let owned = ZPatternDb::load(&path).unwrap();
        let mapped = ZPatternDb::load_mmap(&path).unwrap();
        assert_eq!(owned.raw(), mapped.raw());
        assert_eq!(owned.layout().total(), mapped.layout().total());
        std::fs::remove_file(&path).ok();
    }

    /// Cold lookup at every reachable entry equals `dist[idx]`.
    #[test]
    fn cold_lookup_matches_dist_at_every_entry_k3() {
        use super::super::pattern::ANON;
        use super::super::zbuild::{build_zpdb, gen_moves};
        use super::super::zpdb::{regions, OCCUPIED};

        let pattern = Pattern::new(&[1, 7, 13]);
        let (dist, layout) = build_zpdb(pattern);
        let zdb = ZPatternDb::from_dist(pattern, dist.clone());

        // BFS over the abstract graph, checking cold_lookup == dist at each entry.
        let goal = ProjectedState::goal(pattern);
        let mut occ = 0u32;
        for (c, &v) in goal.cells.iter().enumerate() {
            if v != 0 && v != ANON {
                occ |= 1u32 << c;
            }
        }
        let (count, labels) = regions(occ);
        let mut base = [ANON; 16];
        for (c, &v) in goal.cells.iter().enumerate() {
            if v != 0 && v != ANON {
                base[c] = v;
            }
        }
        let mut rep = vec![usize::MAX; count as usize];
        for (c, &l) in labels.iter().enumerate() {
            if l != OCCUPIED && rep[l as usize] == usize::MAX {
                rep[l as usize] = c;
            }
        }
        let mut visited = vec![false; layout.total() as usize];
        let mut frontier: Vec<ProjectedState> = Vec::new();
        for &rc in &rep {
            let mut nc = base;
            nc[rc] = 0;
            let ps = ProjectedState::from_projection(nc);
            let i = layout.rank(&ps, pattern) as usize;
            if !visited[i] {
                visited[i] = true;
                frontier.push(ps);
            }
        }
        let mut succ: Vec<ProjectedState> = Vec::new();
        while let Some(s) = frontier.pop() {
            let idx = layout.rank(&s, pattern) as usize;
            assert_eq!(zdb.cold_lookup_proj(&s), dist[idx]);
            succ.clear();
            gen_moves(&layout, &s, &mut succ);
            for ns in succ.drain(..) {
                let i = layout.rank(&ns, pattern) as usize;
                if !visited[i] {
                    visited[i] = true;
                    frontier.push(ns);
                }
            }
        }
    }
}
