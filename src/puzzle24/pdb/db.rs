//! [`PatternDb`]: a built 24-puzzle additive PDB plus its on-disk format.
//!
//! Binary format (little-endian): 4-byte magic `P24D`, `u32` version, `u32`
//! pattern bitmask (bits 1..=24), `u32` reserved (0), then `N` distance bytes
//! (`N = num_projected_states`). The pattern bitmask makes the file
//! self-describing.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use super::build;
use super::pattern::{Pattern, ProjectedState};
use crate::puzzle24::state::State;

pub const MAGIC: &[u8; 4] = b"P24D";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 16;

enum Storage {
    Owned(Vec<u8>),
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

/// A built 24-puzzle additive pattern database, indexed by
/// [`ProjectedState::rank`].
pub struct PatternDb {
    pattern: Pattern,
    storage: Storage,
}

impl PatternDb {
    /// Build via single-threaded 0/1 BFS.
    pub fn build(pattern: Pattern) -> Self {
        let dist = build::build(pattern);
        debug_assert_eq!(dist.len() as u64, pattern.num_projected_states());
        Self {
            pattern,
            storage: Storage::Owned(dist),
        }
    }

    /// Build via multi-threaded BFS (`parallel` feature). Byte-identical output.
    #[cfg(feature = "parallel")]
    pub fn build_parallel(pattern: Pattern) -> Self {
        let dist = build::build_parallel(pattern);
        debug_assert_eq!(dist.len() as u64, pattern.num_projected_states());
        Self {
            pattern,
            storage: Storage::Owned(dist),
        }
    }

    pub fn from_dist(pattern: Pattern, dist: Vec<u8>) -> Self {
        assert_eq!(dist.len() as u64, pattern.num_projected_states());
        Self {
            pattern,
            storage: Storage::Owned(dist),
        }
    }

    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    pub fn bytes_stored(&self) -> usize {
        self.storage.dist().len()
    }

    /// PDB heuristic value for `s` (admissible: `≤ dist(s, GOAL)`).
    #[inline]
    pub fn h(&self, s: &State) -> u8 {
        let proj = ProjectedState::from_state(s, self.pattern);
        let d = self.storage.dist()[proj.rank(self.pattern) as usize];
        debug_assert!(
            d != build::UNVISITED,
            "PDB queried at unreachable projection"
        );
        d
    }

    /// PDB value for an already-projected state (the incremental hot path).
    #[inline]
    pub fn value(&self, proj: &ProjectedState) -> u8 {
        self.storage.dist()[proj.rank(self.pattern) as usize]
    }

    pub fn raw(&self) -> &[u8] {
        self.storage.dist()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&self.pattern.0.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?;
        f.write_all(self.storage.dist())?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let mut f = File::open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        f.read_exact(&mut header)?;
        let pattern = parse_header(&header)?;
        let n = pattern.num_projected_states() as usize;
        let mut dist = vec![0u8; n];
        f.read_exact(&mut dist)?;
        let mut tail = [0u8; 1];
        match f.read(&mut tail)? {
            0 => Ok(Self {
                pattern,
                storage: Storage::Owned(dist),
            }),
            _ => Err(LoadError::TrailingBytes),
        }
    }

    #[cfg(feature = "mmap")]
    pub fn load_mmap(path: &Path) -> Result<Self, LoadError> {
        let f = File::open(path)?;
        // SAFETY: PDB files are write-once-then-immutable build artifacts.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() < HEADER_BYTES {
            return Err(LoadError::ShortFile { got: map.len() });
        }
        let header: [u8; HEADER_BYTES] = map[..HEADER_BYTES].try_into().unwrap();
        let pattern = parse_header(&header)?;
        let expected = HEADER_BYTES as u64 + pattern.num_projected_states();
        if (map.len() as u64) != expected {
            return Err(LoadError::SizeMismatch {
                got: map.len() as u64,
                expected,
            });
        }
        Ok(Self {
            pattern,
            storage: Storage::Mmapped(map),
        })
    }
}

fn parse_header(header: &[u8; HEADER_BYTES]) -> Result<Pattern, LoadError> {
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
    Ok(Pattern(bits))
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
    use crate::puzzle24::state::{Move, GOAL};

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
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let path = tmp_path("p24_pdb_roundtrip.bin");
        pdb.save(&path).unwrap();
        let loaded = PatternDb::load(&path).unwrap();
        assert_eq!(loaded.pattern().0, pdb.pattern().0);
        assert_eq!(loaded.raw(), pdb.raw());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_size_matches_format() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2]));
        let path = tmp_path("p24_pdb_filesize.bin");
        pdb.save(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            file_size_for(pdb.pattern())
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("p24_pdb_bad_magic.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern().0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(pdb.raw()).unwrap();
        drop(f);
        assert!(matches!(
            PatternDb::load(&path),
            Err(LoadError::BadMagic(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn value_matches_h() {
        let pdb = PatternDb::build(Pattern::new(&[1, 7, 13]));
        let s = GOAL.apply(Move::Up).apply(Move::Left);
        let proj = ProjectedState::from_state(&s, pdb.pattern());
        assert_eq!(pdb.value(&proj), pdb.h(&s));
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_load_matches_owned() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
        let path = tmp_path("p24_pdb_mmap.bin");
        pdb.save(&path).unwrap();
        let owned = PatternDb::load(&path).unwrap();
        let mapped = PatternDb::load_mmap(&path).unwrap();
        assert_eq!(owned.raw(), mapped.raw());
        assert_eq!(owned.h(&GOAL), mapped.h(&GOAL));
        std::fs::remove_file(&path).ok();
    }
}
