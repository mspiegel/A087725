//! [`PatternDb`]: a built pattern database plus its on-disk format.
//!
//! Binary format (little-endian):
//! ```text
//!   offset  bytes   description
//!   -------------------------------------------------------
//!   0       4       magic "P8PD"
//!   4       4       version (u32)
//!   8       2       pattern bitmask (u16)
//!   10      2       reserved (must be 0)
//!   12      N       distance bytes (N = num_projected_states)
//! ```
//!
//! `N` depends on the pattern size. The size of the file uniquely determines
//! the pattern at load time *given the bitmask*; we still read the bitmask
//! explicitly so the file is self-describing.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use super::build;
use super::pattern::{Pattern, ProjectedState};
use crate::puzzle8::state::State;

pub const MAGIC: &[u8; 4] = b"P8PD";
pub const VERSION: u32 = 1;
const HEADER_BYTES: usize = 12;

/// A built pattern database. Stores the pattern definition plus a distance
/// table indexed by projected state rank.
#[derive(Clone)]
pub struct PatternDb {
    pattern: Pattern,
    dist: Vec<u8>,
}

impl PatternDb {
    /// Build a PDB for the given pattern via backward BFS in projected space.
    pub fn build(pattern: Pattern) -> Self {
        let dist = build::build(pattern);
        debug_assert_eq!(dist.len() as u32, pattern.num_projected_states());
        Self { pattern, dist }
    }

    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    /// Number of bytes the distance table occupies in memory (matches the
    /// on-disk payload size, excluding the 12-byte header).
    pub fn bytes_stored(&self) -> usize {
        self.dist.len()
    }

    /// PDB heuristic value for `s`: the projected distance from `s` to the
    /// projected goal. Always `≤ dist(s, GOAL)` (admissible).
    pub fn h(&self, s: &State) -> u8 {
        let proj = ProjectedState::from_state(s, self.pattern);
        let r = proj.rank(self.pattern) as usize;
        let d = self.dist[r];
        debug_assert!(
            d != build::UNVISITED,
            "PDB queried at unreachable projection for solvable state {:?}",
            s.0
        );
        d
    }

    /// Raw distance entry at a projected rank. Mostly useful for tests.
    pub fn raw_distance(&self, rank: u32) -> u8 {
        self.dist[rank as usize]
    }

    pub fn raw(&self) -> &[u8] {
        &self.dist
    }

    /// Write to `path`, overwriting any existing file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&self.pattern.0.to_le_bytes())?;
        f.write_all(&0u16.to_le_bytes())?; // reserved
        f.write_all(&self.dist)?;
        f.flush()?;
        Ok(())
    }

    /// Load from `path`, verifying header and length.
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(LoadError::BadMagic(magic));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let v = u32::from_le_bytes(buf4);
        if v != VERSION {
            return Err(LoadError::UnsupportedVersion(v));
        }
        let mut buf2 = [0u8; 2];
        f.read_exact(&mut buf2)?;
        let bits = u16::from_le_bytes(buf2);
        let pattern = Pattern(bits);
        let mut reserved = [0u8; 2];
        f.read_exact(&mut reserved)?;
        if reserved != [0, 0] {
            return Err(LoadError::ReservedNonZero);
        }

        let n = pattern.num_projected_states() as usize;
        let mut dist = vec![0u8; n];
        f.read_exact(&mut dist)?;

        // Verify no trailing bytes.
        let mut tail = [0u8; 1];
        match f.read(&mut tail)? {
            0 => Ok(Self { pattern, dist }),
            _ => Err(LoadError::TrailingBytes),
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    ReservedNonZero,
    TrailingBytes,
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {}", e),
            LoadError::BadMagic(m) => write!(f, "bad magic: {:?}, expected {:?}", m, MAGIC),
            LoadError::UnsupportedVersion(v) => {
                write!(f, "unsupported version: {} (this build expects {})", v, VERSION)
            }
            LoadError::ReservedNonZero => write!(f, "reserved bytes must be zero"),
            LoadError::TrailingBytes => write!(f, "file has trailing bytes after expected payload"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Convenience for callers: file size for a built PDB given its pattern size.
pub fn file_size_for(pattern: Pattern) -> usize {
    HEADER_BYTES + pattern.num_projected_states() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::state::GOAL;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn build_then_query_goal_is_zero() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
        assert_eq!(pdb.h(&GOAL), 0);
    }

    #[test]
    fn save_load_roundtrip() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let path = tmp_path("p8_pdb_roundtrip.bin");
        pdb.save(&path).unwrap();
        let loaded = PatternDb::load(&path).unwrap();
        assert_eq!(loaded.pattern.0, pdb.pattern.0);
        assert_eq!(loaded.dist, pdb.dist);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_size_matches_format() {
        let pdb = PatternDb::build(Pattern::new(&[1, 2]));
        let path = tmp_path("p8_pdb_filesize.bin");
        pdb.save(&path).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len() as usize, file_size_for(pdb.pattern));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("p8_pdb_bad_magic.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern.0.to_le_bytes()).unwrap();
        f.write_all(&[0u8, 0]).unwrap();
        f.write_all(&pdb.dist).unwrap();
        drop(f);
        match PatternDb::load(&path) {
            Err(LoadError::BadMagic(_)) => {}
            Err(e) => panic!("expected BadMagic, got {}", e),
            Ok(_) => panic!("expected BadMagic, got Ok"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_version() {
        let path = tmp_path("p8_pdb_bad_version.bin");
        let pdb = PatternDb::build(Pattern::new(&[1]));
        let mut f = File::create(&path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&999u32.to_le_bytes()).unwrap();
        f.write_all(&pdb.pattern.0.to_le_bytes()).unwrap();
        f.write_all(&[0u8, 0]).unwrap();
        f.write_all(&pdb.dist).unwrap();
        drop(f);
        match PatternDb::load(&path) {
            Err(LoadError::UnsupportedVersion(999)) => {}
            Err(e) => panic!("expected UnsupportedVersion(999), got {}", e),
            Ok(_) => panic!("expected UnsupportedVersion(999), got Ok"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn build_is_deterministic() {
        let a = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let b = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        assert_eq!(a.dist, b.dist);
    }
}
