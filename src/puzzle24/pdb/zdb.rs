//! A built 1-bit zero-aware PDB, plus its on-disk format and the cold
//! absolute-h lookup the IDA\* root needs.
//!
//! Binary format (little-endian): 4-byte magic `Z24D`, `u32` version, `u32`
//! pattern bitmask (bits 1..=24), `u32` reserved (0), `u64` entry total
//! (`ZpdbLayout::total()`), then `ceil(total / 8)` bytes of packed bit-1
//! values per [`crate::puzzle24::pdb::zcodec::pack_bits`]. The pattern and
//! total in the header are self-describing and cross-checked at load.
//!
//! Use:
//! - **Hot path** (incremental IDA\*): keep `(idx, h)` per move and use
//!   [`ZPatternDb::diff_lookup`] for each step — O(1).
//! - **Cold path** (IDA\* root, or a one-off query): use
//!   [`ZPatternDb::cold_lookup`], which descends the abstract ZPDB graph by
//!   "happy moves" until no neighbor lowers `h` — O(h) per call. Walking
//!   `gen_moves` in the abstract graph means each step is a unit-cost
//!   pattern move; the descent length is the true `h`.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use super::pattern::{Pattern, ProjectedState};
use super::zbuild;
use super::zcodec::{diff_lookup, lookup_bit, pack_bits, parity_of_h};
use super::zpdb::ZpdbLayout;
use crate::puzzle24::state::State;

pub const MAGIC: &[u8; 4] = b"Z24D";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 24;

/// Header size + packed-bit length in bytes for a built ZPDB of the given
/// pattern.
pub fn file_size_for(pattern: Pattern) -> u64 {
    let total = ZpdbLayout::new(pattern).total();
    HEADER_BYTES as u64 + (total + 7) / 8
}

enum Storage {
    Owned(Vec<u8>),
    #[cfg(feature = "mmap")]
    Mmapped(memmap2::Mmap),
}

impl Storage {
    #[inline]
    fn packed(&self) -> &[u8] {
        match self {
            Storage::Owned(v) => v,
            #[cfg(feature = "mmap")]
            Storage::Mmapped(m) => &m[HEADER_BYTES..],
        }
    }
}

/// A built 1-bit zero-aware PDB, indexed by `(m, p, r)` via [`ZpdbLayout`].
pub struct ZPatternDb {
    pattern: Pattern,
    layout: ZpdbLayout,
    storage: Storage,
}

impl ZPatternDb {
    /// Wrap a freshly-built uncompressed byte distance array and pack it.
    pub fn from_dist(pattern: Pattern, dist: &[u8]) -> Self {
        let layout = ZpdbLayout::new(pattern);
        assert_eq!(
            dist.len() as u64,
            layout.total(),
            "dist length {} != layout total {}",
            dist.len(),
            layout.total()
        );
        let packed = pack_bits(dist);
        Self { pattern, layout, storage: Storage::Owned(packed) }
    }

    /// Wrap an already-packed 1-bit table (e.g. from the frontier-free
    /// [`zbuild::build_zpdb_2bit_packed`], which never materializes a byte
    /// `dist`). `packed` must be exactly `(total + 7) / 8` bytes.
    pub fn from_packed(pattern: Pattern, packed: Vec<u8>) -> Self {
        let layout = ZpdbLayout::new(pattern);
        assert_eq!(
            packed.len() as u64,
            (layout.total() + 7) / 8,
            "packed length {} != (total + 7)/8 = {}",
            packed.len(),
            (layout.total() + 7) / 8
        );
        Self { pattern, layout, storage: Storage::Owned(packed) }
    }

    /// Build the ZPDB end-to-end (single-threaded BFS, then pack).
    pub fn build(pattern: Pattern) -> Self {
        let (dist, _) = zbuild::build_zpdb(pattern);
        Self::from_dist(pattern, &dist)
    }

    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    pub fn layout(&self) -> &ZpdbLayout {
        &self.layout
    }

    /// The raw 1-bit packed payload (`HEADER_BYTES`-stripped). Stable on
    /// disk; used by SHA-256 pinning.
    pub fn packed(&self) -> &[u8] {
        self.storage.packed()
    }

    /// Stored bit at `idx`, returned at value position (0 or 2).
    #[inline]
    pub fn lookup_bit(&self, idx: u64) -> u8 {
        lookup_bit(self.storage.packed(), idx)
    }

    /// O(1) differential update: given current absolute `old_h` and the
    /// neighbor's `(m, p, r)` index, returns the neighbor's absolute `h`.
    #[inline]
    pub fn diff_lookup(&self, neighbor_idx: u64, old_h: u8) -> u8 {
        diff_lookup(self.storage.packed(), neighbor_idx, old_h)
    }

    /// Parity (bit 0 of `h`) at index `idx`, derived from the permutation.
    #[inline]
    pub fn parity(&self, idx: u64) -> u8 {
        parity_of_h(&self.layout, idx)
    }

    /// Cold absolute lookup of `h(state)`. Descends the abstract ZPDB graph
    /// from `state`'s projection, taking any neighbor whose `diff_lookup`
    /// yields `h_cur - 1`, until none does (= reached a goal-region entry).
    /// Number of descent steps = `h(state)`. O(h) per call.
    ///
    /// The high anchor is picked with `mod 4` matching the start state's
    /// `(parity, stored_bit_1)` so the differential reconstruction stays
    /// aligned to the true distances throughout the descent. The descent
    /// terminates because each step strictly decreases `h_cur`, and at any
    /// goal-region entry `dist = 0` so every successor stores
    /// `h = 1 ≡ h_goal + 1`, blocking further descent.
    pub fn cold_lookup(&self, state: &State) -> u8 {
        let proj = ProjectedState::from_state(state, self.pattern);
        self.cold_lookup_proj(&proj)
    }

    pub(crate) fn cold_lookup_proj(&self, proj: &ProjectedState) -> u8 {
        let idx0 = self.layout.rank(proj, self.pattern);
        let par = self.parity(idx0);
        let bit1 = self.lookup_bit(idx0); // 0 or 2
                                           // SEED ≡ 0 mod 4, large enough that descent never underflows for
                                           // the 6-tile ZPDB (max h ≈ 75). 200 leaves ~125 headroom.
        const SEED: u8 = 200;
        let h0: u8 = SEED + (bit1 | par);
        let mut h_cur = h0;
        let mut cur = *proj;
        let mut succ: Vec<ProjectedState> = Vec::with_capacity(24);
        loop {
            succ.clear();
            zbuild::gen_moves(&self.layout, &cur, &mut succ);
            let mut descended = false;
            for ns in &succ {
                let nidx = self.layout.rank(ns, self.pattern);
                let nh = self.diff_lookup(nidx, h_cur);
                if nh + 1 == h_cur {
                    cur = *ns;
                    h_cur = nh;
                    descended = true;
                    break;
                }
            }
            if !descended {
                break;
            }
        }
        h0 - h_cur
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&self.pattern.0.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?;
        f.write_all(&self.layout.total().to_le_bytes())?;
        f.write_all(self.storage.packed())?;
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
        let nbytes = ((total + 7) / 8) as usize;
        let mut packed = vec![0u8; nbytes];
        f.read_exact(&mut packed)?;
        let mut tail = [0u8; 1];
        match f.read(&mut tail)? {
            0 => Ok(Self { pattern, layout, storage: Storage::Owned(packed) }),
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
        let expected = HEADER_BYTES as u64 + (total + 7) / 8;
        if (map.len() as u64) != expected {
            return Err(LoadError::SizeMismatch {
                got: map.len() as u64,
                expected,
            });
        }
        Ok(Self { pattern, layout, storage: Storage::Mmapped(map) })
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
            LoadError::Io(e) => write!(f, "I/O error: {}", e),
            LoadError::BadMagic(m) => write!(f, "bad magic: {:?}, expected {:?}", m, MAGIC),
            LoadError::UnsupportedVersion(v) => write!(f, "unsupported version: {}", v),
            LoadError::ReservedNonZero => write!(f, "reserved bytes must be zero"),
            LoadError::TrailingBytes => write!(f, "file has trailing bytes after payload"),
            LoadError::ShortFile { got } => write!(f, "file too short: {} bytes", got),
            LoadError::SizeMismatch { got, expected } => {
                write!(f, "file size {} != expected {}", got, expected)
            }
            LoadError::TotalMismatch { in_file, expected } => {
                write!(f, "entry total mismatch: file {} vs layout {}", in_file, expected)
            }
        }
    }
}

impl std::error::Error for LoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::pdb::build as additive_build;
    use crate::puzzle24::pdb::pattern::ANON;
    use crate::puzzle24::pdb::zbuild::{build_zpdb, gen_moves};
    use crate::puzzle24::pdb::zcodec::pack_bits;
    use crate::puzzle24::pdb::zpdb::{regions, OCCUPIED};
    use crate::puzzle24::state::GOAL;

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
        let path = tmp_path("p24_zpdb_roundtrip.bin");
        zdb.save(&path).unwrap();
        let loaded = ZPatternDb::load(&path).unwrap();
        assert_eq!(loaded.pattern().0, zdb.pattern().0);
        assert_eq!(loaded.packed(), zdb.packed());
        assert_eq!(loaded.layout().total(), zdb.layout().total());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_size_matches_format() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        let path = tmp_path("p24_zpdb_filesize.bin");
        zdb.save(&path).unwrap();
        let actual = std::fs::metadata(&path).unwrap().len();
        assert_eq!(actual, file_size_for(zdb.pattern()));
        assert_eq!(
            actual,
            HEADER_BYTES as u64 + (zdb.layout().total() + 7) / 8
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("p24_zpdb_bad_magic.bin");
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"XXXX").unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&zdb.pattern().0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&zdb.layout().total().to_le_bytes()).unwrap();
        f.write_all(zdb.packed()).unwrap();
        drop(f);
        assert!(matches!(ZPatternDb::load(&path), Err(LoadError::BadMagic(_))));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_short_file_truncation() {
        let path = tmp_path("p24_zpdb_short.bin");
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        zdb.save(&path).unwrap();
        // Truncate to half-payload — the read_exact on the payload must fail.
        let full = std::fs::metadata(&path).unwrap().len();
        let trunc = HEADER_BYTES as u64 + (full - HEADER_BYTES as u64) / 2;
        let opt = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        opt.set_len(trunc).unwrap();
        drop(opt);
        match ZPatternDb::load(&path) {
            Err(LoadError::Io(_)) => {}
            Err(e) => panic!("expected Io error on truncated file, got {:?}", e),
            Ok(_) => panic!("expected Io error on truncated file, got Ok"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_total_mismatch_in_header() {
        let path = tmp_path("p24_zpdb_total_mismatch.bin");
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        // Same pattern, but lie about `total` so the layout-recomputed total
        // disagrees.
        let mut f = File::create(&path).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&zdb.pattern().0.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(zdb.layout().total() + 8).to_le_bytes()).unwrap();
        f.write_all(zdb.packed()).unwrap();
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
        let path = tmp_path("p24_zpdb_mmap.bin");
        zdb.save(&path).unwrap();
        let owned = ZPatternDb::load(&path).unwrap();
        let mapped = ZPatternDb::load_mmap(&path).unwrap();
        assert_eq!(owned.packed(), mapped.packed());
        assert_eq!(owned.layout().total(), mapped.layout().total());
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_load_rejects_size_mismatch() {
        let path = tmp_path("p24_zpdb_mmap_size.bin");
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2]));
        zdb.save(&path).unwrap();
        // Append a stray byte so mmap'd size != header-implied size.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&[0u8]).unwrap();
        drop(f);
        assert!(matches!(
            ZPatternDb::load_mmap(&path),
            Err(LoadError::SizeMismatch { .. })
        ));
        std::fs::remove_file(&path).ok();
    }

    /// Cold lookup at every reachable entry must equal `dist[idx]`. This
    /// ties the packed bits + parity-from-index + descent loop together.
    #[test]
    fn cold_lookup_matches_dist_at_every_entry_k2() {
        assert_cold_lookup_matches_dist(Pattern::new(&[1, 2]));
    }

    #[test]
    fn cold_lookup_matches_dist_at_every_entry_k3() {
        assert_cold_lookup_matches_dist(Pattern::new(&[1, 7, 13]));
    }

    /// Admissibility oracle through the codec: for every reachable (k=3) ZPDB
    /// entry, the codec-decoded value is `≥` the verified additive PDB. The
    /// codec round-trips losslessly, so any regression here means a bug in
    /// pack/unpack or in `from_dist` (not in the underlying ZPDB invariant —
    /// `zbuild::tests::zpdb_geq_additive_pointwise_k3` covers that).
    #[test]
    fn codec_path_is_geq_additive_k3() {
        let pattern = Pattern::new(&[1, 7, 13]);
        let (dist, layout) = build_zpdb(pattern);
        let zdb = ZPatternDb::from_dist(pattern, &dist);
        let additive = additive_build::build(pattern);

        // Walk every reachable entry via the BFS used elsewhere in this file.
        for_each_reachable(pattern, &layout, |proj, _idx| {
            // Cold lookup (only public absolute-h API).
            let z = zdb.cold_lookup_proj(proj);
            let a_idx = proj.rank(pattern) as usize;
            assert!(
                z >= additive[a_idx],
                "ZPDB (codec) {} < additive {} at {:?}",
                z,
                additive[a_idx],
                proj.cells
            );
            // Sanity: cold lookup matches the raw dist too.
            let zr = dist[layout.rank(proj, pattern) as usize];
            assert_eq!(z, zr, "cold lookup {} != dist {} at {:?}", z, zr, proj.cells);
        });
    }

    // --- helpers --------------------------------------------------------

    fn assert_cold_lookup_matches_dist(pattern: Pattern) {
        let (dist, layout) = build_zpdb(pattern);
        let _packed = pack_bits(&dist); // ensure pack succeeds (debug_assert)
        let zdb = ZPatternDb::from_dist(pattern, &dist);
        for_each_reachable(pattern, &layout, |proj, idx| {
            let want = dist[idx as usize];
            let got = zdb.cold_lookup_proj(proj);
            assert_eq!(
                got, want,
                "cold_lookup at idx={} returned {} but dist={} (cells={:?})",
                idx, got, want, proj.cells
            );
        });
    }

    /// Walk the abstract ZPDB graph from the goal-region seeds and call
    /// `f(proj, idx)` exactly once per reachable `(m,p,r)`. Mirrors the
    /// codec-side enumerator in `zcodec.rs` tests; kept here so the two
    /// modules don't need an internal test API.
    fn for_each_reachable(
        pattern: Pattern,
        layout: &ZpdbLayout,
        mut f: impl FnMut(&ProjectedState, u64),
    ) {
        let goal = ProjectedState::goal(pattern);
        let mut occ = 0u32;
        for (c, &v) in goal.cells.iter().enumerate() {
            if v != 0 && v != ANON {
                occ |= 1u32 << c;
            }
        }
        let (count, labels) = regions(occ);
        let mut base = [ANON; 25];
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
        let mut succ: Vec<ProjectedState> = Vec::with_capacity(24);
        while let Some(s) = frontier.pop() {
            let idx = layout.rank(&s, pattern);
            f(&s, idx);
            succ.clear();
            gen_moves(layout, &s, &mut succ);
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
