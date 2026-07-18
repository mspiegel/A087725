//! [`PatternDb`]: a built 15-puzzle PDB plus its on-disk format.
//!
//! Binary format (little-endian):
//! ```text
//!   offset  bytes   description
//!   -------------------------------------------------------
//!   0       4       magic "P15D"
//!   4       4       version (u32)
//!   8       4       pattern bitmask (u32; bits 1..=15)
//!   12      4       reserved (must be 0)
//!   16      N       distance bytes (N = num_projected_states)
//! ```
//!
//! `N` depends on the pattern size. For the Korf 7-8 partition, `N` is
//! 57,657,600 (P7) or 518,918,400 (P8). The file is self-describing: the
//! pattern bitmask lets us know which tiles the entries are indexed against.
//!
//! Storage backends (selected via the `mmap` Cargo feature):
//!
//! - Owned `Vec<u8>` — used for in-memory builds and for the `mmap`-less
//!   `load` path. The full payload is read into RAM on load.
//! - mmap'd file — when the `mmap` feature is enabled, [`PatternDb::load_mmap`]
//!   maps the file into the process's address space. Page-in is on demand,
//!   the OS shares pages across processes, and re-launching the solver hits
//!   a warm page cache.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use super::build;
use super::pattern::{Pattern, ProjectedState};
use crate::puzzle15::state::State;

pub const MAGIC: &[u8; 4] = b"P15D";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 16;

/// Storage backing a [`PatternDb`].
enum Storage {
    /// Owned bytes (fresh build, or non-mmap load).
    Owned(Vec<u8>),
    /// Memory-mapped file. The `Mmap` includes the 16-byte header; the
    /// distance slice starts at offset [`HEADER_BYTES`].
    #[cfg(feature = "mmap")]
    Mmapped(memmap2::Mmap),
}

impl Storage {
    #[inline]
    fn dist(&self) -> &[u8] {
        match self {
            Storage::Owned(v) => v,
            #[cfg(feature = "mmap")]
            Storage::Mmapped(m) => &m[HEADER_BYTES..],
        }
    }
}

/// A built 15-puzzle pattern database. Stores the pattern definition plus a
/// distance table indexed by [`ProjectedState::rank`].
pub struct PatternDb {
    pattern: Pattern,
    storage: Storage,
}

impl PatternDb {
    /// Build a PDB for the given pattern via 0/1 BFS in projected space.
    pub fn build(pattern: Pattern) -> Self {
        let dist = build::build(pattern);
        debug_assert_eq!(dist.len() as u64, pattern.num_projected_states());
        Self { pattern, storage: Storage::Owned(dist) }
    }

    /// Build using multi-threaded BFS (rayon). Available with the `parallel`
    /// feature. Byte-identical output to [`build`](Self::build).
    #[cfg(feature = "parallel")]
    pub fn build_parallel(pattern: Pattern) -> Self {
        let dist = build::build_parallel(pattern);
        debug_assert_eq!(dist.len() as u64, pattern.num_projected_states());
        Self { pattern, storage: Storage::Owned(dist) }
    }

    /// Construct from an owned distance vector (e.g. produced by the
    /// build module directly). For tests and reuse.
    pub fn from_dist(pattern: Pattern, dist: Vec<u8>) -> Self {
        assert_eq!(dist.len() as u64, pattern.num_projected_states());
        Self { pattern, storage: Storage::Owned(dist) }
    }

    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    /// Number of bytes the distance table occupies (matches the on-disk
    /// payload size, excluding the 16-byte header).
    pub fn bytes_stored(&self) -> usize {
        self.storage.dist().len()
    }

    /// PDB heuristic value for `s`: the projected distance from `s` to the
    /// projected goal. Always `≤ dist(s, GOAL)` (admissible).
    #[inline]
    pub fn h(&self, s: &State) -> u8 {
        let proj = ProjectedState::from_state(s, self.pattern);
        let r = proj.rank(self.pattern) as usize;
        let d = self.storage.dist()[r];
        debug_assert!(
            d != build::UNVISITED,
            "PDB queried at unreachable projection for state {:?}",
            s.0
        );
        d
    }

    /// PDB value for an already-projected state: ranks `proj` against this PDB's
    /// pattern and reads the stored distance. Unlike [`h`](Self::h), it does
    /// **not** re-project from a full [`State`] — the caller maintains `proj`
    /// incrementally, which is the hot path for the incremental IDA\* evaluator.
    #[inline]
    pub fn value(&self, proj: &ProjectedState) -> u8 {
        self.storage.dist()[proj.rank(self.pattern) as usize]
    }

    /// Raw distance entry at a PDB index. Mostly useful for tests.
    pub fn raw_distance(&self, rank: u64) -> u8 {
        self.storage.dist()[rank as usize]
    }

    pub fn raw(&self) -> &[u8] {
        self.storage.dist()
    }

    /// Write to `path`, overwriting any existing file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&self.pattern.0.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?; // reserved
        f.write_all(self.storage.dist())?;
        f.flush()?;
        Ok(())
    }

    /// Load from `path` into RAM, verifying header and payload length.
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let mut f = File::open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        f.read_exact(&mut header)?;
        let (pattern, _) = parse_header(&header)?;
        let n = pattern.num_projected_states() as usize;
        let mut dist = vec![0u8; n];
        f.read_exact(&mut dist)?;
        // Verify no trailing bytes.
        let mut tail = [0u8; 1];
        match f.read(&mut tail)? {
            0 => Ok(Self { pattern, storage: Storage::Owned(dist) }),
            _ => Err(LoadError::TrailingBytes),
        }
    }

    /// Load from `path` by mmap'ing the file. Requires the `mmap` feature.
    /// The file remains mapped for the lifetime of the returned `PatternDb`.
    #[cfg(feature = "mmap")]
    pub fn load_mmap(path: &Path) -> Result<Self, LoadError> {
        let f = File::open(path)?;
        // SAFETY: We treat the file as read-only. If another process mutates
        // the file while we hold the mapping, we'd see torn reads — but the
        // PDB files are write-once-then-immutable build artifacts.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() < HEADER_BYTES {
            return Err(LoadError::ShortFile { got: map.len() });
        }
        let header: [u8; HEADER_BYTES] = map[..HEADER_BYTES].try_into().unwrap();
        let (pattern, _) = parse_header(&header)?;
        let expected = HEADER_BYTES as u64 + pattern.num_projected_states();
        if (map.len() as u64) != expected {
            return Err(LoadError::SizeMismatch {
                got: map.len() as u64,
                expected,
            });
        }
        Ok(Self { pattern, storage: Storage::Mmapped(map) })
    }
}

/// Parse the 16-byte header. Reads via `from_le_bytes` to avoid alignment UB.
fn parse_header(header: &[u8; HEADER_BYTES]) -> Result<(Pattern, u32), LoadError> {
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
    Ok((Pattern(bits), version))
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    ReservedNonZero,
    TrailingBytes,
    /// mmap path: file shorter than the header.
    ShortFile { got: usize },
    /// mmap path: file size doesn't match `HEADER_BYTES + num_projected_states`.
    SizeMismatch { got: u64, expected: u64 },
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
            LoadError::UnsupportedVersion(v) => {
                write!(f, "unsupported version: {v} (this build expects {VERSION})")
            }
            LoadError::ReservedNonZero => write!(f, "reserved bytes must be zero"),
            LoadError::TrailingBytes => write!(f, "file has trailing bytes after expected payload"),
            LoadError::ShortFile { got } => {
                write!(f, "file too short for header: got {got} bytes, need {HEADER_BYTES}")
            }
            LoadError::SizeMismatch { got, expected } => {
                write!(f, "file size {got} doesn't match expected {expected} (header + payload)")
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// File size in bytes for a built PDB given its pattern.
pub fn file_size_for(pattern: Pattern) -> u64 {
    HEADER_BYTES as u64 + pattern.num_projected_states()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::state::GOAL;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}_{}", std::process::id(), name))
    }

    #[test]
    fn build_then_query_goal_is_zero() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
        assert_eq!(pdb.h(&GOAL), 0);
    }

    #[test]
    fn save_load_roundtrip() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let path = tmp_path("p15_pdb_roundtrip.bin");
        pdb.save(&path).unwrap();
        let loaded = PatternDb::load(&path).unwrap();
        assert_eq!(loaded.pattern.0, pdb.pattern.0);
        assert_eq!(loaded.raw(), pdb.raw());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_size_matches_format() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2]));
        let path = tmp_path("p15_pdb_filesize.bin");
        pdb.save(&path).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), file_size_for(pdb.pattern));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn header_bytes_constant_matches_layout() {
        assert_eq!(HEADER_BYTES, 16);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("p15_pdb_bad_magic.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern.0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(pdb.raw()).unwrap();
        drop(f);
        match PatternDb::load(&path) {
            Err(LoadError::BadMagic(_)) => {}
            other => panic!("expected BadMagic, got Ok = {}", other.is_ok()),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_version() {
        let path = tmp_path("p15_pdb_bad_version.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&999u32.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern.0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(pdb.raw()).unwrap();
        drop(f);
        match PatternDb::load(&path) {
            Err(LoadError::UnsupportedVersion(999)) => {}
            other => panic!("expected UnsupportedVersion, got Ok = {}", other.is_ok()),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_trailing_bytes() {
        let path = tmp_path("p15_pdb_trailing.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern.0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(pdb.raw()).unwrap();
        f.write_all(b"extra").unwrap();
        drop(f);
        match PatternDb::load(&path) {
            Err(LoadError::TrailingBytes) => {}
            other => panic!("expected TrailingBytes, got Ok = {}", other.is_ok()),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn build_is_deterministic() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let b = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        assert_eq!(a.raw(), b.raw());
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_load_matches_owned_load() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let path = tmp_path("p15_pdb_mmap.bin");
        pdb.save(&path).unwrap();
        let owned = PatternDb::load(&path).unwrap();
        let mapped = PatternDb::load_mmap(&path).unwrap();
        assert_eq!(owned.raw(), mapped.raw());
        assert_eq!(owned.pattern.0, mapped.pattern.0);
        // h() must produce identical results.
        assert_eq!(owned.h(&GOAL), mapped.h(&GOAL));
        let s = GOAL.apply(crate::puzzle15::state::Move::Up);
        assert_eq!(owned.h(&s), mapped.h(&s));
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_rejects_bad_magic() {
        let path = tmp_path("p15_pdb_mmap_bad_magic.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern.0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(pdb.raw()).unwrap();
        drop(f);
        match PatternDb::load_mmap(&path) {
            Err(LoadError::BadMagic(_)) => {}
            other => panic!("expected BadMagic, got Ok = {}", other.is_ok()),
        }
        std::fs::remove_file(&path).ok();
    }
}
