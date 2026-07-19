//! Persistent solve cache: `rank → exact depth` for every board ever verified
//! by idastar (kept *and* rejected). Lets a run resume — or a lower-floor run
//! reuse a higher-floor run's work — without re-solving.
//!
//! Depths are absolute (independent of the band floor), so one cache serves all
//! runs. On-disk format: 7 bytes/entry — 6-byte little-endian rank + 1-byte
//! depth. Saved atomically (temp file + rename) so a kill mid-save can't corrupt
//! it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub type Cache = HashMap<u64, u8>;

/// Load the cache, or an empty one if the file does not exist.
pub fn load(path: &Path) -> io::Result<Cache> {
    if !path.exists() {
        return Ok(Cache::new());
    }
    let mut buf = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buf)?;
    if buf.len() % 7 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cache {} length {} not a multiple of 7",
                path.display(),
                buf.len()
            ),
        ));
    }
    let mut cache = Cache::with_capacity(buf.len() / 7);
    for chunk in buf.chunks_exact(7) {
        let mut r = [0u8; 8];
        r[..6].copy_from_slice(&chunk[..6]);
        cache.insert(u64::from_le_bytes(r), chunk[6]);
    }
    Ok(cache)
}

/// Write the cache atomically (temp file + rename).
pub fn save(path: &Path, cache: &Cache) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        for (&r, &d) in cache {
            w.write_all(&r.to_le_bytes()[..6])?;
            w.write_all(&[d])?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}
