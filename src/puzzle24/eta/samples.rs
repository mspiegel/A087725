//! On-disk sphere samples, one set of files per stratum `k` in a campaign
//! directory:
//!
//! - `kNN.meta` — `key=value` lines naming the walker and verifier settings the
//!   samples were drawn under. Extending a stratum with different settings is
//!   refused, because reach probabilities depend on them.
//! - `kNN.chunks.tsv` — one line per completed chunk of attempts: seed, chunk
//!   index, attempts, accepted, thread nanoseconds. Attempts that were rejected
//!   exist only here, as the zeros of the Horvitz–Thompson mean.
//! - `kNN.samples` — accepted boards, 28-byte little-endian records: packed
//!   board (`u128`, [`pack`](crate::puzzle24::eta::layers::pack)), reach
//!   probability (`f64`), chunk index (`u32`).
//!
//! A chunk's samples are appended before its chunk line, so after an
//! interruption the samples of chunks without a line are ignored by readers
//! that filter on the chunk records.
//!
//! The samples do not depend on any heuristic: scoring a heuristic reads the
//! boards and evaluates it.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Bytes per record in a `.samples` file.
pub const RECORD_BYTES: usize = 28;

/// One accepted board.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRecord {
    /// Packed board.
    pub board: u128,
    /// Reach probability `P(v)`.
    pub prob: f64,
    /// Chunk the sample was drawn in.
    pub chunk: u32,
}

/// One completed chunk of attempts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkRecord {
    pub seed: u64,
    pub chunk: u64,
    pub attempts: u64,
    pub accepted: u64,
    pub thread_ns: u64,
}

pub fn meta_path(dir: &Path, k: u32) -> PathBuf {
    dir.join(format!("k{k:02}.meta"))
}

pub fn chunks_path(dir: &Path, k: u32) -> PathBuf {
    dir.join(format!("k{k:02}.chunks.tsv"))
}

pub fn samples_path(dir: &Path, k: u32) -> PathBuf {
    dir.join(format!("k{k:02}.samples"))
}

/// Append sample records.
pub fn append_samples(path: &Path, records: &[SampleRecord]) -> io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut w = BufWriter::new(file);
    for r in records {
        w.write_all(&r.board.to_le_bytes())?;
        w.write_all(&r.prob.to_le_bytes())?;
        w.write_all(&r.chunk.to_le_bytes())?;
    }
    w.flush()
}

/// Read all sample records; a missing file is empty. A trailing partial
/// record (an interrupted append) is ignored.
pub fn read_samples(path: &Path) -> io::Result<Vec<SampleRecord>> {
    let mut bytes = Vec::new();
    match File::open(path) {
        Ok(mut f) => {
            f.read_to_end(&mut bytes)?;
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    }
    Ok(bytes
        .chunks_exact(RECORD_BYTES)
        .map(|b| SampleRecord {
            board: u128::from_le_bytes(b[0..16].try_into().unwrap()),
            prob: f64::from_le_bytes(b[16..24].try_into().unwrap()),
            chunk: u32::from_le_bytes(b[24..28].try_into().unwrap()),
        })
        .collect())
}

/// Append chunk records, writing a header line if the file is new.
pub fn append_chunks(path: &Path, records: &[ChunkRecord]) -> io::Result<()> {
    let fresh = !path.exists();
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut w = BufWriter::new(file);
    if fresh {
        writeln!(w, "# seed\tchunk\tattempts\taccepted\tthread_ns")?;
    }
    for r in records {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}",
            r.seed, r.chunk, r.attempts, r.accepted, r.thread_ns
        )?;
    }
    w.flush()
}

/// Read chunk records; a missing file is empty. A line that does not parse
/// (an interrupted append) ends the read.
pub fn read_chunks(path: &Path) -> io::Result<Vec<ChunkRecord>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<u64> = match line.split('\t').map(str::parse).collect() {
            Ok(v) => v,
            Err(_) => break,
        };
        if f.len() != 5 {
            break;
        }
        out.push(ChunkRecord {
            seed: f[0],
            chunk: f[1],
            attempts: f[2],
            accepted: f[3],
            thread_ns: f[4],
        });
    }
    Ok(out)
}

/// Write `settings` as the stratum's meta file, or check an existing one
/// matches exactly. Returns an error naming the first differing key.
pub fn write_or_check_meta(path: &Path, settings: &BTreeMap<String, String>) -> io::Result<()> {
    if path.exists() {
        let existing = read_meta(path)?;
        if &existing != settings {
            let key = settings
                .keys()
                .chain(existing.keys())
                .find(|k| settings.get(*k) != existing.get(*k))
                .cloned()
                .unwrap_or_default();
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{}: setting {key:?} is {:?} on disk but {:?} now",
                    path.display(),
                    existing.get(&key),
                    settings.get(&key)
                ),
            ));
        }
        return Ok(());
    }
    let mut w = BufWriter::new(File::create(path)?);
    for (k, v) in settings {
        writeln!(w, "{k}={v}")?;
    }
    w.flush()
}

pub fn read_meta(path: &Path) -> io::Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if let Some((k, v)) = line.split_once('=') {
            out.insert(k.to_string(), v.to_string());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("eta-samples-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn samples_roundtrip_and_ignore_a_partial_record() {
        let dir = tempdir("samples");
        let path = samples_path(&dir, 7);
        let a = [
            SampleRecord {
                board: u128::MAX - 3,
                prob: 1.5e-30,
                chunk: 9,
            },
            SampleRecord {
                board: 12345,
                prob: 0.25,
                chunk: 0,
            },
        ];
        append_samples(&path, &a[..1]).unwrap();
        append_samples(&path, &a[1..]).unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[1, 2, 3]).unwrap();
        assert_eq!(read_samples(&path).unwrap(), a.to_vec());
        assert!(read_samples(&dir.join("missing")).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn chunks_roundtrip_and_stop_at_a_partial_line() {
        let dir = tempdir("chunks");
        let path = chunks_path(&dir, 12);
        let c = [
            ChunkRecord {
                seed: 1,
                chunk: 0,
                attempts: 256,
                accepted: 17,
                thread_ns: 99,
            },
            ChunkRecord {
                seed: 1,
                chunk: 1,
                attempts: 100,
                accepted: 0,
                thread_ns: 7,
            },
        ];
        append_chunks(&path, &c).unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"1\t2\t25").unwrap();
        assert_eq!(read_chunks(&path).unwrap(), c.to_vec());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn meta_is_written_once_and_checked() {
        let dir = tempdir("meta");
        let path = meta_path(&dir, 30);
        let mut s = BTreeMap::new();
        s.insert("window".to_string(), "14".to_string());
        s.insert("choice".to_string(), "Lookahead".to_string());
        write_or_check_meta(&path, &s).unwrap();
        write_or_check_meta(&path, &s).unwrap();
        s.insert("window".to_string(), "11".to_string());
        let err = write_or_check_meta(&path, &s).unwrap_err();
        assert!(err.to_string().contains("window"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
