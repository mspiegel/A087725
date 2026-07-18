//! Binary save/load for the [`DistanceTable`].
//!
//! Format:
//! ```text
//!   offset  bytes   description
//!   ----------------------------
//!   0       4       magic "P8DT"
//!   4       4       version (u32, little-endian)
//!   8       181440  distance bytes (one per rank)
//! ```
//! Total file size: 181_448 bytes.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::puzzle8::bfs::DistanceTable;
use crate::puzzle8::state::N_STATES;

pub const MAGIC: &[u8; 4] = b"P8DT";
pub const VERSION: u32 = 1;
pub const FILE_SIZE: usize = 8 + N_STATES as usize;

/// Errors when loading a distance table from disk.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
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
            LoadError::Io(e) => write!(f, "I/O error: {e}"),
            LoadError::BadMagic(m) => {
                write!(f, "bad magic: {m:?}, expected {MAGIC:?}")
            }
            LoadError::UnsupportedVersion(v) => {
                write!(f, "unsupported version: {v} (this build expects {VERSION})")
            }
            LoadError::TrailingBytes => write!(f, "file has trailing bytes after expected payload"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Write `table` to `path`, overwriting any existing file.
pub fn save(table: &DistanceTable, path: &Path) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(table.raw())?;
    f.flush()?;
    Ok(())
}

/// Read a distance table from `path`, verifying the magic and version.
pub fn load(path: &Path) -> Result<DistanceTable, LoadError> {
    let mut f = File::open(path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LoadError::BadMagic(magic));
    }

    let mut version_bytes = [0u8; 4];
    f.read_exact(&mut version_bytes)?;
    let v = u32::from_le_bytes(version_bytes);
    if v != VERSION {
        return Err(LoadError::UnsupportedVersion(v));
    }

    let mut table = DistanceTable::new_empty();
    f.read_exact(table.raw_mut())?;

    // Verify no trailing bytes.
    let mut trailing = [0u8; 1];
    match f.read(&mut trailing)? {
        0 => Ok(table),
        _ => Err(LoadError::TrailingBytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::state::GOAL;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn save_load_roundtrip() {
        let t1 = DistanceTable::build();
        let path = tmp_path("p8_save_load_roundtrip.bin");
        save(&t1, &path).expect("save");
        let t2 = load(&path).expect("load");
        assert_eq!(t1.raw(), t2.raw());
        assert_eq!(t2.dist(&GOAL), 0);
        assert_eq!(t2.diameter(), crate::puzzle8::state::DIAMETER);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("p8_bad_magic.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&vec![0u8; N_STATES as usize]).unwrap();
        drop(f);
        match load(&path) {
            Err(LoadError::BadMagic(_)) => {}
            Err(e) => panic!("expected BadMagic, got {e}"),
            Ok(_) => panic!("expected BadMagic, got Ok"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_version() {
        let path = tmp_path("p8_bad_version.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&999u32.to_le_bytes()).unwrap();
        f.write_all(&vec![0u8; N_STATES as usize]).unwrap();
        drop(f);
        match load(&path) {
            Err(LoadError::UnsupportedVersion(999)) => {}
            Err(e) => panic!("expected UnsupportedVersion(999), got {e}"),
            Ok(_) => panic!("expected UnsupportedVersion(999), got Ok"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_trailing_bytes() {
        let path = tmp_path("p8_trailing.bin");
        let t = DistanceTable::build();
        save(&t, &path).unwrap();
        // Append a byte.
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0u8]).unwrap();
        drop(f);
        match load(&path) {
            Err(LoadError::TrailingBytes) => {}
            Err(e) => panic!("expected TrailingBytes, got {e}"),
            Ok(_) => panic!("expected TrailingBytes, got Ok"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_size_constant_matches_format() {
        let t = DistanceTable::build();
        let path = tmp_path("p8_file_size.bin");
        save(&t, &path).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len() as usize, FILE_SIZE);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn build_is_deterministic() {
        // Two independent builds must produce byte-identical tables.
        let a = DistanceTable::build();
        let b = DistanceTable::build();
        assert_eq!(a.raw(), b.raw());
    }
}
