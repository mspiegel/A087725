//! Walking Distance for the 24-puzzle (Ken'ichiro Takahashi). Ported from
//! `crate::puzzle15::search::walking_distance` to the 5×5 board.
//!
//! The row-WD state is a 5×5 matrix `M[r][g]` = "non-blank tiles in row r whose
//! goal row is g," plus the blank's row index ∈ {0..4}. Column-WD is the same
//! shape with rows and columns swapped. By symmetry of the 5×5 goal, row-WD and
//! column-WD share one BFS distance table.
//!
//! **Differences from the 15-puzzle port:**
//!
//! - *State count.* The reachable 5×5 row-WD space is **65,650,495** states
//!   (measured), vs the 4×4's 24,964 — five orders of magnitude bigger. The BFS
//!   that fills the table is a one-off but takes ~25–30 s and the resident
//!   `HashMap` is ~1.2 GiB, so [`WalkingDistanceHeuristic::warm_up`] should be
//!   called once at startup and the heuristic reused across many solves (the
//!   ladder/Phase-2 harnesses do exactly this).
//! - *Key packing.* A naïve pack of all 25 cells (3 bits each) + blank axis is
//!   78 bits and would need a `u128`. We instead drop **each row's last column**:
//!   given the blank axis-index, row `r`'s margin is known (5, except the blank's
//!   row which is 4), so `m[r][W-1] = margin(r) − Σ_{c<W-1} m[r][c]` is derivable.
//!   That leaves `5·4 = 20` cells (3 bits each) + the 3-bit blank index = **63
//!   bits → fits a `u64`**, halving both the key width and the table footprint.

use super::{Heuristic, IncHeuristic, SearchStats};
use crate::puzzle24::state::{Move, State, N_CELLS, W};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A 5×5 row-distribution matrix.
type WdMatrix = [[u8; W]; W];

/// Fast, well-distributing hasher for the WD table's `u64` `pack` keys: one
/// **widening** multiply by 2^64/φ, then xor the two halves together.
///
/// The keys are structured (3-bit fields, each 0–5), and a naive single multiply
/// really does fail on them — a multiply propagates entropy *upward*, so low
/// input bits reach high product bits but never the reverse, while SwissTable
/// picks its bucket index from the **low** bits. Taking the high half of a
/// widening multiply and folding it down is what fixes that, and it is the whole
/// difference between this and the FxHash-style hash that was rejected here
/// earlier.
///
/// Measured over all 65,650,495 merged-table keys, modelling hashbrown's
/// placement (`cwd.rs::probe_length_distribution_by_hash`), at 49% load:
///
/// | hash                | mean group hops | p99 | max | displaced |
/// |---------------------|-----------------|-----|-----|-----------|
/// | Murmur3 `fmix64`    | 0.0003          | 0   | 2   | 0.03%     |
/// | **this**            | 0.0003          | 0   | 3   | 0.03%     |
/// | single multiply     | 79.17           | 325 | 581 | 98.72%    |
///
/// So this places as well as the `fmix64` it replaces (and spreads the 7-bit
/// control byte slightly better, 1.001 vs 1.005 max/mean) for roughly half the
/// work — `mul`/`umulh` issue in parallel, against `fmix64`'s strictly serial
/// 2 multiplies and 3 xor-shifts. That matters because this sits on the node's
/// critical dependency chain, between the packed key and the probe address.
///
/// The "single multiply" row is not hypothetical: it reproduces the blow-up that
/// the earlier FxHash rejection recorded, which is what validates the model.
#[derive(Default)]
pub struct WdHasher(u64);
impl Hasher for WdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Not exercised for u64 keys, but keep a correct fallback.
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        let p = ((i ^ self.0) as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15_u128);
        self.0 = (p >> 64) as u64 ^ (p as u64);
    }
}
pub type WdBuild = BuildHasherDefault<WdHasher>;

/// A Walking-Distance-style distance table: `pack`ed `u64` key → distance-to-goal
/// (`u8`), hashed by the fully-avalanching [`WdHasher`]. This is the concrete
/// in-memory representation shared by the full row/column WD table and any future
/// WD-style abstraction table (e.g. the wildcard-complement `hB` heuristic), and
/// the type persisted by [`save_dist_table`] / [`load_dist_table`].
pub type WdTable = HashMap<u64, u8, WdBuild>;

/// Pack `(matrix, blank-axis-index)` into a u64 key. Each stored cell holds a
/// value ≤ 5 (3 bits); we omit the last column of every row (derivable from the
/// blank-aware row margin), leaving `5·4 = 20` cells + the 3-bit blank index =
/// 63 bits — within u64. Injective on reachable WD-states because the blank
/// index (carried in the key) pins down every omitted cell.
#[inline]
fn pack(m: &WdMatrix, blank_idx: u8) -> u64 {
    let mut k: u64 = blank_idx as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

/// Goal row-distribution matrix for the 24-puzzle (also the column-distribution
/// goal by symmetry). Tiles 1–5 have goal row 0; 6–10 row 1; 11–15 row 2; 16–20
/// row 3; 21–24 row 4 (the blank, also row 4, is excluded → only 4 there).
fn goal_matrix() -> WdMatrix {
    goal_matrix_for(GOAL_BLANK_IDX)
}

/// Goal contingency for an ARBITRARY target whose blank sits on axis-index
/// `blank_idx` (its row for the row-WD table, its column for the col-WD table).
/// The solved contingency is always the identity — every non-blank tile sits in
/// its own target row/col — so the diagonal carries each axis's margin: 5 tiles
/// per axis, except the target's blank axis which carries 4 (the blank occupies
/// one of its cells). Generalizes [`goal_matrix`] (the GOAL case `blank_idx = 4`)
/// for the retargetable [`WalkingDistanceTo`] heuristic. The BFS transition graph
/// is *target-agnostic*; only this seed matrix depends on the target.
fn goal_matrix_for(blank_idx: u8) -> WdMatrix {
    let mut m = [[0u8; W]; W];
    for d in 0..W {
        m[d][d] = 5;
    }
    m[blank_idx as usize][blank_idx as usize] = 4;
    m
}

const GOAL_BLANK_IDX: u8 = 4;

/// BFS from the GOAL WD-state (`blank_idx = 4`), collecting every reachable
/// matrix-and-blank combination with its distance to goal.
fn build_table() -> WdTable {
    build_table_from(&goal_matrix(), GOAL_BLANK_IDX)
}

/// BFS from an arbitrary seed WD-state `(goal_m, blank_idx)`, collecting every
/// reachable matrix-and-blank combination with its distance back to that seed.
/// The transition rule (a tile crossing the axis the blank steps across) is
/// *target-agnostic*, so this same routine builds the GOAL table and every
/// retargeted [`WalkingDistanceTo`] table — only the seed differs. The reachable
/// set has the same 65,650,495 states for any `blank_idx` (axis-reflection
/// symmetry), so the capacity reservation is shared.
fn build_table_from(goal_m: &WdMatrix, blank_idx: u8) -> WdTable {
    let goal_key = pack(goal_m, blank_idx);

    // Reserve for the full reachable set (65,650,495) up front so the BFS never
    // pays for an incremental rehash.
    let mut table: WdTable = HashMap::with_capacity_and_hasher(66_000_000, WdBuild::default());
    table.insert(goal_key, 0);

    let mut frontier: Vec<(WdMatrix, u8)> = vec![(*goal_m, blank_idx)];
    let mut depth: u8 = 0;

    while !frontier.is_empty() {
        let mut next: Vec<(WdMatrix, u8)> = Vec::new();
        let next_depth = depth + 1;
        for (m, br) in &frontier {
            // Blank axis-index decreases: a tile from axis (br - 1) crosses
            // into axis br, picking any non-empty goal-axis slot.
            if *br > 0 {
                let from = (*br - 1) as usize;
                let to = *br as usize;
                for g in 0..W {
                    if m[from][g] > 0 {
                        let mut m2 = *m;
                        m2[from][g] -= 1;
                        m2[to][g] += 1;
                        let new_br = *br - 1;
                        let key = pack(&m2, new_br);
                        if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key) {
                            e.insert(next_depth);
                            next.push((m2, new_br));
                        }
                    }
                }
            }
            // Blank axis-index increases: symmetric.
            if (*br as usize) < W - 1 {
                let from = (*br + 1) as usize;
                let to = *br as usize;
                for g in 0..W {
                    if m[from][g] > 0 {
                        let mut m2 = *m;
                        m2[from][g] -= 1;
                        m2[to][g] += 1;
                        let new_br = *br + 1;
                        let key = pack(&m2, new_br);
                        if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key) {
                            e.insert(next_depth);
                            next.push((m2, new_br));
                        }
                    }
                }
            }
        }
        depth = next_depth;
        frontier = next;
    }
    table
}

// ---------------------------------------------------------------------------
// On-disk persistence: precompute the WD table once, reuse across processes.
// ---------------------------------------------------------------------------
//
// The BFS in `build_table` is a one-off ~17–25 s cost paid per *process*. For
// harnesses that spawn many short-lived processes (or just to avoid the startup
// stall), the table can be built once by the `build_wd24` tool, pinned by SHA
// (like the ZPDB artifacts), and loaded here instead of rebuilt. The in-memory
// structure loaded is the *same* `WdTable` HashMap proven fastest for lookups
// (cf. the rejected flat/rank encoding) — only the *construction* is replaced.
//
// The file format is deliberately generic (a `u64 → u8` distance table tagged by
// a `kind`) so the future wildcard-complement `hB` heuristic can reuse the exact
// same builder/loader path.

/// Magic bytes at the head of a persisted WD-style distance table.
pub const WD_TABLE_MAGIC: &[u8; 4] = b"WD24";
/// On-disk format version.
pub const WD_TABLE_VERSION: u32 = 1;
/// `kind` tag for the full row/column Walking-Distance table (all 24 tiles
/// tracked, shared by both axes). Future WD-style tables (e.g. `hB`) use other
/// tags so a mismatched artifact is rejected at load rather than silently used.
pub const WD_KIND_FULL: u32 = 0;
/// The exact number of reachable states in the full 5×5 row-WD space, also pinned
/// by `tests::table_size_matches_measured`. Used as a cheap load-time integrity
/// check for `WD_KIND_FULL` artifacts.
pub const FULL_WD_ENTRIES: u64 = 65_650_495;

const WD_TABLE_HEADER_BYTES: usize = 24; // magic(4) ver(4) kind(4) reserved(4) entries(8)

/// Where the full-WD table was obtained from, for honest startup logging.
#[derive(Clone, Debug)]
pub enum WdTableSource {
    /// Loaded from a pre-built artifact.
    Loaded { path: PathBuf, entries: usize },
    /// Constructed in-process by the BFS.
    Built { entries: usize },
}

/// Errors from [`load_dist_table`].
#[derive(Debug)]
pub enum WdLoadError {
    Io(io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    KindMismatch {
        in_file: u32,
        expected: u32,
    },
    ReservedNonZero,
    EntryMismatch {
        in_file: u64,
        expected: u64,
    },
    SizeMismatch {
        got: u64,
        expected: u64,
    },
    /// [`warm_up_from`] was called after the table was already initialized.
    AlreadyInitialized,
}

impl From<io::Error> for WdLoadError {
    fn from(e: io::Error) -> Self {
        WdLoadError::Io(e)
    }
}

impl std::fmt::Display for WdLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WdLoadError::Io(e) => write!(f, "I/O error: {e}"),
            WdLoadError::BadMagic(m) => {
                write!(f, "bad magic: {m:?}, expected {WD_TABLE_MAGIC:?}")
            }
            WdLoadError::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            WdLoadError::KindMismatch { in_file, expected } => {
                write!(f, "kind mismatch: file {in_file} vs expected {expected}")
            }
            WdLoadError::ReservedNonZero => write!(f, "reserved bytes must be zero"),
            WdLoadError::EntryMismatch { in_file, expected } => {
                write!(
                    f,
                    "entry count mismatch: file {in_file} vs expected {expected}"
                )
            }
            WdLoadError::SizeMismatch { got, expected } => {
                write!(f, "file size {got} != expected {expected}")
            }
            WdLoadError::AlreadyInitialized => {
                write!(
                    f,
                    "WD table already initialized; warm_up_from must be called first"
                )
            }
        }
    }
}

impl std::error::Error for WdLoadError {}

/// Serialize a WD-style distance table to `path` in a byte-deterministic format
/// (entries sorted by key), so repeated builds hash identically and the artifact
/// can be SHA-pinned. Layout: a 24-byte header (`magic`, `version`, `kind`,
/// `reserved=0`, `entries`) followed by all keys as little-endian `u64`s (sorted
/// ascending) and then all matching distances as `u8`s (a struct-of-arrays that
/// keeps the key block 8-aligned).
pub fn save_dist_table(path: &Path, kind: u32, table: &WdTable) -> io::Result<()> {
    let mut entries: Vec<(u64, u8)> = table.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|&(k, _)| k);

    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(WD_TABLE_MAGIC)?;
    w.write_all(&WD_TABLE_VERSION.to_le_bytes())?;
    w.write_all(&kind.to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?; // reserved
    w.write_all(&(entries.len() as u64).to_le_bytes())?;
    for &(k, _) in &entries {
        w.write_all(&k.to_le_bytes())?;
    }
    for &(_, v) in &entries {
        w.write_all(&[v])?;
    }
    w.flush()?;
    Ok(())
}

/// Load a WD-style distance table written by [`save_dist_table`]. Validates the
/// magic, version, `kind` (must equal `expected_kind`), reserved bytes, exact
/// file length, and — when `expected_entries` is `Some` — the entry count. These
/// cheap checks reject a corrupt or stale artifact; the external `.sha256` pin is
/// the full-integrity guarantee. Returns the same `WdTable` HashMap `build_table`
/// would have produced.
pub fn load_dist_table(
    path: &Path,
    expected_kind: u32,
    expected_entries: Option<u64>,
) -> Result<WdTable, WdLoadError> {
    let mut f = std::fs::File::open(path)?;
    let mut header = [0u8; WD_TABLE_HEADER_BYTES];
    f.read_exact(&mut header)?;

    let magic: [u8; 4] = header[0..4].try_into().unwrap();
    if &magic != WD_TABLE_MAGIC {
        return Err(WdLoadError::BadMagic(magic));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != WD_TABLE_VERSION {
        return Err(WdLoadError::UnsupportedVersion(version));
    }
    let kind = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if kind != expected_kind {
        return Err(WdLoadError::KindMismatch {
            in_file: kind,
            expected: expected_kind,
        });
    }
    let reserved = u32::from_le_bytes(header[12..16].try_into().unwrap());
    if reserved != 0 {
        return Err(WdLoadError::ReservedNonZero);
    }
    let n = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if let Some(exp) = expected_entries {
        if n != exp {
            return Err(WdLoadError::EntryMismatch {
                in_file: n,
                expected: exp,
            });
        }
    }

    // Exact-size check: header + n keys (8 B) + n values (1 B). Guards against a
    // truncated or padded file before we allocate the map.
    let file_len = f.metadata()?.len();
    let expected_len = WD_TABLE_HEADER_BYTES as u64 + n * 8 + n;
    if file_len != expected_len {
        return Err(WdLoadError::SizeMismatch {
            got: file_len,
            expected: expected_len,
        });
    }

    let n = n as usize;
    let mut key_bytes = vec![0u8; n * 8];
    f.read_exact(&mut key_bytes)?;
    let mut vals = vec![0u8; n];
    f.read_exact(&mut vals)?;

    let mut table: WdTable = HashMap::with_capacity_and_hasher(n, WdBuild::default());
    for (chunk, &v) in key_bytes.chunks_exact(8).zip(vals.iter()) {
        let k = u64::from_le_bytes(chunk.try_into().unwrap());
        table.insert(k, v);
    }
    Ok(table)
}

/// Build the full row/column WD table via BFS (the authoritative constructor).
/// Exposed for the `build_wd24` persistence tool; normal solving goes through the
/// lazily-initialized [`table`].
pub fn build_full_table() -> WdTable {
    build_table()
}

static T: OnceLock<WdTable> = OnceLock::new();
static SOURCE: OnceLock<WdTableSource> = OnceLock::new();

/// Load-or-build cascade for the full WD table:
/// 1. If `data/wd24.bin` exists, load it — on failure *warn and fall back*
///    to BFS (an opportunistic default; a failed load can't corrupt a result,
///    only a successfully-loaded-but-wrong table could, which the header/count
///    checks reject).
/// 2. Else build via BFS (the original behavior).
///
/// An explicit artifact path bypasses the cascade via [`warm_up_from`]
/// (`WalkingDistanceHeuristic::warm_up_from`), which fails loudly instead of
/// silently rebuilding.
fn load_or_build() -> (WdTable, WdTableSource) {
    let default = PathBuf::from("data/wd24.bin");
    if default.exists() {
        match load_dist_table(&default, WD_KIND_FULL, Some(FULL_WD_ENTRIES)) {
            Ok(m) => {
                let entries = m.len();
                return (
                    m,
                    WdTableSource::Loaded {
                        path: default,
                        entries,
                    },
                );
            }
            Err(e) => eprintln!(
                "warning: {} exists but failed to load ({}); rebuilding via BFS. \
                 Remove or rebuild it with `build_wd24`.",
                default.display(),
                e
            ),
        }
    }

    let m = build_table();
    let entries = m.len();
    (m, WdTableSource::Built { entries })
}

fn table() -> &'static WdTable {
    T.get_or_init(|| {
        let (m, src) = load_or_build();
        let _ = SOURCE.set(src);
        m
    })
}

/// `WD_row(s) + WD_col(s)`. Admissible.
pub struct WalkingDistanceHeuristic;

impl WalkingDistanceHeuristic {
    /// Force the lookup table to be initialized. Optional — `h` will initialize
    /// on first call regardless. Prefers the pre-built `data/wd24.bin` artifact
    /// and falls back to the ~17–25 s BFS; warming up at startup keeps the
    /// first solve from paying for whichever path is taken.
    pub fn warm_up() {
        let _ = table();
    }

    /// [`warm_up`](Self::warm_up), then print one honest line to stderr saying
    /// whether the table was loaded (and from where) or built via BFS, with the
    /// elapsed time. A convenience so the CLIs don't each reimplement — and don't
    /// print a misleading "building…" when the table was actually loaded.
    pub fn warm_up_verbose() {
        let t = std::time::Instant::now();
        Self::warm_up();
        let elapsed = t.elapsed();
        match Self::table_source() {
            Some(WdTableSource::Loaded { path, entries }) => {
                eprintln!(
                    "WD table: loaded {} entries from {} in {:?}",
                    entries,
                    path.display(),
                    elapsed
                );
            }
            Some(WdTableSource::Built { entries }) => {
                eprintln!("WD table: built {entries} entries via BFS in {elapsed:?}");
            }
            None => {}
        }
    }

    /// Force-initialize the table from a specific artifact `path`, bypassing the
    /// env-var/default/BFS cascade. Must be called before any other WD lookup
    /// (returns [`WdLoadError::AlreadyInitialized`] otherwise).
    pub fn warm_up_from(path: &Path) -> Result<(), WdLoadError> {
        let m = load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES))?;
        let entries = m.len();
        T.set(m).map_err(|_| WdLoadError::AlreadyInitialized)?;
        let _ = SOURCE.set(WdTableSource::Loaded {
            path: path.to_path_buf(),
            entries,
        });
        Ok(())
    }

    /// How the (already-initialized) table was obtained, or `None` if the table
    /// has not been initialized yet (no lookup or `warm_up*` has run).
    pub fn table_source() -> Option<WdTableSource> {
        SOURCE.get().cloned()
    }

    /// Size of the reachable WD-state space, for diagnostics.
    pub fn table_size() -> usize {
        table().len()
    }
}

impl Heuristic for WalkingDistanceHeuristic {
    fn h(&self, s: &State) -> u8 {
        let mut m_row = [[0u8; W]; W];
        let mut m_col = [[0u8; W]; W];
        let mut br: u8 = 0;
        let mut bc: u8 = 0;
        for pos in 0..(W * W) {
            let tile = s.0[pos];
            let r = (pos / W) as u8;
            let c = (pos % W) as u8;
            if tile == 0 {
                br = r;
                bc = c;
                continue;
            }
            let goal_pos = (tile - 1) as usize;
            let gr = goal_pos / W;
            let gc = goal_pos % W;
            m_row[r as usize][gr] += 1;
            m_col[c as usize][gc] += 1;
        }
        let t = table();
        let h_row = *t
            .get(&pack(&m_row, br))
            .expect("row-WD state must be reachable from goal");
        let h_col = *t
            .get(&pack(&m_col, bc))
            .expect("col-WD state must be reachable from goal");
        h_row + h_col
    }
}

/// Retargetable Walking Distance: an admissible lower bound on the number of
/// moves from a board `s` to an **arbitrary** target board `T` (not just GOAL).
/// This is the backward heuristic for a bidirectional/meet-in-the-middle solver.
///
/// The construction mirrors [`WalkingDistanceHeuristic`] but with two things
/// retargeted to `T`:
///
/// - *The projection.* Each tile's target row/col is read from `T`'s layout
///   (`target_row`/`target_col`) rather than GOAL's.
/// - *The seed.* The BFS distance table is built from `T`'s solved contingency:
///   the identity with the `4`-margin on `T`'s blank axis (see [`goal_matrix_for`]).
///   Because the transition graph is target-agnostic, only that seed changes.
///
/// Row-WD needs a table seeded at `blank_idx = t_blank_row`; col-WD needs one at
/// `blank_idx = t_blank_col`. When those coincide (e.g. a corner blank like the R
/// board's cell 0 → row 0, col 0) a single table serves both axes, so `col_table`
/// is left `None` and the lookups fall back to `row_table`.
///
/// Unlike [`WalkingDistanceHeuristic`], each `WalkingDistanceTo::new` builds its
/// own in-memory table(s) via the same BFS — it does **not** touch the shared
/// GOAL table or the `data/wd24.bin` artifact.
pub struct WalkingDistanceTo {
    /// Row-WD distance table, seeded at `t_blank_row`.
    row_table: WdTable,
    /// Column-WD distance table, seeded at `t_blank_col`. `None` when
    /// `t_blank_col == t_blank_row`, in which case `row_table` serves both axes.
    col_table: Option<WdTable>,
    /// Per-tile target row in `T`, indexed by tile value `1..=24`. Index `0`
    /// (the blank) is unused.
    target_row: [u8; N_CELLS],
    /// Per-tile target column in `T`, indexed by tile value `1..=24`. Index `0`
    /// (the blank) is unused.
    target_col: [u8; N_CELLS],
    /// `T`'s blank row (the row-WD seed's blank axis-index).
    t_blank_row: u8,
    /// `T`'s blank column (the col-WD seed's blank axis-index).
    t_blank_col: u8,
}

impl WalkingDistanceTo {
    /// Build the retargetable WD heuristic for `target`. Computes each non-blank
    /// tile's target row/col and the target's blank row/col, then BFS-builds the
    /// row table (seeded at `t_blank_row`) and — only if the blank's column
    /// differs from its row — the col table (seeded at `t_blank_col`). Each build
    /// is the same ~15–25 s BFS as the GOAL table.
    pub fn new(target: &State) -> Self {
        let mut target_row = [0u8; N_CELLS];
        let mut target_col = [0u8; N_CELLS];
        let mut t_blank_row = 0u8;
        let mut t_blank_col = 0u8;
        for pos in 0..(W * W) {
            let tile = target.0[pos];
            let r = (pos / W) as u8;
            let c = (pos % W) as u8;
            if tile == 0 {
                t_blank_row = r;
                t_blank_col = c;
                continue;
            }
            target_row[tile as usize] = r;
            target_col[tile as usize] = c;
        }

        let row_table = build_table_from(&goal_matrix_for(t_blank_row), t_blank_row);
        let col_table = if t_blank_col == t_blank_row {
            None
        } else {
            Some(build_table_from(&goal_matrix_for(t_blank_col), t_blank_col))
        };

        WalkingDistanceTo {
            row_table,
            col_table,
            target_row,
            target_col,
            t_blank_row,
            t_blank_col,
        }
    }

    /// `WD_row(s→T) + WD_col(s→T)`. Admissible lower bound on moves from `s` to
    /// `T`. Projects `s` with `T`'s per-tile target rows/cols, packs each axis
    /// with `s`'s *current* blank row/col, and sums the two table lookups.
    pub fn h(&self, s: &State) -> u8 {
        let mut m_row = [[0u8; W]; W];
        let mut m_col = [[0u8; W]; W];
        let mut br: u8 = 0;
        let mut bc: u8 = 0;
        for pos in 0..(W * W) {
            let tile = s.0[pos];
            let r = (pos / W) as u8;
            let c = (pos % W) as u8;
            if tile == 0 {
                br = r;
                bc = c;
                continue;
            }
            let gr = self.target_row[tile as usize] as usize;
            let gc = self.target_col[tile as usize] as usize;
            m_row[r as usize][gr] += 1;
            m_col[c as usize][gc] += 1;
        }
        // The col table falls back to the row table when the target's blank
        // shares an axis-index across rows and columns.
        let col_t = self.col_table.as_ref().unwrap_or(&self.row_table);
        let h_row = *self
            .row_table
            .get(&pack(&m_row, br))
            .expect("row-WD state must be reachable from target");
        let h_col = *col_t
            .get(&pack(&m_col, bc))
            .expect("col-WD state must be reachable from target");
        h_row + h_col
    }

    /// The target's blank row and column, for diagnostics / test assertions.
    pub fn target_blank(&self) -> (u8, u8) {
        (self.t_blank_row, self.t_blank_col)
    }
}

/// Incremental Walking Distance: same admissible heuristic as
/// [`WalkingDistanceHeuristic`], advanced in O(1) per node.
///
/// Key invariant: a slide moves one tile by one cell, vertically OR
/// horizontally. The row-WD axis only sees changes when a tile crosses a row
/// boundary (vertical slide); the column-WD axis only on horizontal slides.
/// So per node, **exactly one** of the two matrices, blanks, and `h` halves
/// changes — the other half is carried forward verbatim.
///
/// For the affected axis we do two `+/-1` updates on the matrix entries, a fresh
/// `pack`, and a single hashmap lookup. `WalkingDistanceHeuristic::warm_up()`
/// must have been called first (same as for the non-incremental version).
pub struct WalkingDistanceInc;

/// Per-node state for [`WalkingDistanceInc`]. The two matrices, the blank's row
/// and column, and both axis-`h` values pre-computed. `Copy` so the IDA*
/// recursion can thread it by value.
#[derive(Clone, Copy, Debug)]
pub struct WdCtx {
    m_row: WdMatrix,
    m_col: WdMatrix,
    br: u8,
    bc: u8,
    h_row: u8,
    h_col: u8,
}

impl IncHeuristic for WalkingDistanceInc {
    type Ctx = WdCtx;

    fn root(&self, s: &State, _stats: &mut SearchStats) -> (u8, WdCtx) {
        let mut m_row = [[0u8; W]; W];
        let mut m_col = [[0u8; W]; W];
        let mut br: u8 = 0;
        let mut bc: u8 = 0;
        for pos in 0..(W * W) {
            let tile = s.0[pos];
            let r = (pos / W) as u8;
            let c = (pos % W) as u8;
            if tile == 0 {
                br = r;
                bc = c;
                continue;
            }
            let goal_pos = (tile - 1) as usize;
            m_row[r as usize][goal_pos / W] += 1;
            m_col[c as usize][goal_pos % W] += 1;
        }
        let t = table();
        let h_row = *t
            .get(&pack(&m_row, br))
            .expect("row-WD state must be reachable from goal");
        let h_col = *t
            .get(&pack(&m_col, bc))
            .expect("col-WD state must be reachable from goal");
        (
            h_row + h_col,
            WdCtx {
                m_row,
                m_col,
                br,
                bc,
                h_row,
                h_col,
            },
        )
    }

    fn advance(
        &self,
        parent: &WdCtx,
        child: &State,
        m: Move,
        _stats: &mut SearchStats,
    ) -> (u8, WdCtx) {
        // The moved tile's (from, to) cells (see LinearConflictInc for the
        // identity): from = blank_new, to = blank_old. The parent carries the
        // blank's old (br, bc), so blank_old is reconstructed without rescanning.
        let parent_blank = (parent.br as usize) * W + parent.bc as usize;
        let delta: i32 = match m {
            Move::Up => -(W as i32),
            Move::Down => W as i32,
            Move::Left => -1,
            Move::Right => 1,
        };
        let from_cell = (parent_blank as i32 + delta) as usize;
        let to_cell = parent_blank;
        let tile = child.0[to_cell];
        debug_assert_ne!(tile, 0, "moved tile cannot be the blank");
        let goal_pos = (tile - 1) as usize;
        let gr = goal_pos / W;
        let gc = goal_pos % W;

        let rf = from_cell / W;
        let rt = to_cell / W;
        let cf = from_cell % W;
        let ct = to_cell % W;
        let new_br = rf as u8;
        let new_bc = cf as u8;
        let t = table();

        if rf != rt {
            // Vertical slide: row-WD matrix changes, blank-row changes; column
            // axis (m_col, bc, h_col) is byte-identical to parent.
            let mut m_row = parent.m_row;
            m_row[rf][gr] -= 1;
            m_row[rt][gr] += 1;
            let h_row = *t
                .get(&pack(&m_row, new_br))
                .expect("row-WD state must be reachable from goal");
            let ctx = WdCtx {
                m_row,
                m_col: parent.m_col,
                br: new_br,
                bc: new_bc,
                h_row,
                h_col: parent.h_col,
            };
            (h_row + parent.h_col, ctx)
        } else {
            // Horizontal slide: column-WD matrix changes, blank-col changes;
            // row axis is byte-identical.
            let mut m_col = parent.m_col;
            m_col[cf][gc] -= 1;
            m_col[ct][gc] += 1;
            let h_col = *t
                .get(&pack(&m_col, new_bc))
                .expect("col-WD state must be reachable from goal");
            let ctx = WdCtx {
                m_row: parent.m_row,
                m_col,
                br: new_br,
                bc: new_bc,
                h_row: parent.h_row,
                h_col,
            };
            (parent.h_row + h_col, ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::tests_util::bfs_distances;
    use crate::puzzle24::search::ManhattanHeuristic;
    use crate::puzzle24::state::{Move, State, GOAL};

    #[test]
    fn wd_of_goal_is_zero() {
        assert_eq!(WalkingDistanceHeuristic.h(&GOAL), 0);
    }

    #[test]
    fn wd_after_single_vertical_move() {
        // Blank Up from goal: one tile crosses a row → row-WD = 1, col-WD = 0.
        let s = GOAL.apply(Move::Up);
        assert_eq!(WalkingDistanceHeuristic.h(&s), 1);
    }

    #[test]
    fn wd_after_single_horizontal_move() {
        // Blank Left from goal: one tile crosses a column → col-WD = 1.
        let s = GOAL.apply(Move::Left);
        assert_eq!(WalkingDistanceHeuristic.h(&s), 1);
    }

    #[test]
    fn wd_admissible_on_shallow_bfs() {
        let truth = bfs_distances(8);
        for (raw, &true_dist) in &truth {
            let est = WalkingDistanceHeuristic.h(&State(*raw));
            assert!(est <= true_dist, "WD {est} > truth {true_dist} for {raw:?}");
        }
    }

    #[test]
    fn wd_can_exceed_manhattan() {
        // WD is supposed to be tighter than Manhattan on average. Look for *any*
        // state in shallow BFS where WD strictly exceeds MD — proves the table
        // isn't degenerate.
        let truth = bfs_distances(8);
        let mut found_strict = false;
        for raw in truth.keys() {
            let wd = WalkingDistanceHeuristic.h(&State(*raw));
            let md = ManhattanHeuristic.h(&State(*raw));
            if wd > md {
                found_strict = true;
                break;
            }
        }
        assert!(found_strict, "WD never exceeded Manhattan on shallow BFS");
    }

    /// `pack`/`unpack` round-trip is implicit in lookups; here we check the key
    /// is injective on a few hand matrices (no 3-bit overflow at value 5).
    #[test]
    fn pack_is_injective_on_full_rows() {
        let goal = goal_matrix();
        let mut other = goal;
        other[0][0] = 4;
        other[1][0] = 1;
        assert_ne!(pack(&goal, GOAL_BLANK_IDX), pack(&other, GOAL_BLANK_IDX));
        // A cell value of 5 must not bleed into the neighbouring field.
        assert_eq!(goal[0][0], 5);
        assert_ne!(pack(&goal, 0), pack(&goal, 1));
    }

    /// Incremental WD `root` must match the from-scratch WD on every state.
    #[test]
    fn wd_inc_root_matches_scratch_on_shallow_bfs() {
        WalkingDistanceHeuristic::warm_up();
        let truth = bfs_distances(9);
        let mut stats = SearchStats::default();
        for raw in truth.keys() {
            let s = State(*raw);
            let scratch = WalkingDistanceHeuristic.h(&s);
            let (inc, _) = IncHeuristic::root(&WalkingDistanceInc, &s, &mut stats);
            assert_eq!(inc, scratch, "root mismatch for {raw:?}");
        }
    }

    /// `advance` must agree with a fresh `root` at every step of a long random
    /// walk. Also verifies it matches the from-scratch WD at every step.
    #[test]
    fn wd_inc_advance_matches_fresh_root_random_walk() {
        WalkingDistanceHeuristic::warm_up();
        let mut rng: u64 = 0xBADD_C0FE_F1F0_BEEF;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut stats = SearchStats::default();
        let (_, mut ctx) = IncHeuristic::root(&WalkingDistanceInc, &s, &mut stats);
        for step in 0..2000 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = WalkingDistanceInc.advance(&ctx, &ns, m, &mut stats);
            let (h_fresh, _) = IncHeuristic::root(&WalkingDistanceInc, &ns, &mut stats);
            assert_eq!(
                h_adv, h_fresh,
                "advance vs root diverged at step {} (move {:?}, state {:?})",
                step, m, ns.0
            );
            assert_eq!(
                h_adv,
                WalkingDistanceHeuristic.h(&ns),
                "advance vs scratch WD diverged at step {step}"
            );
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// The 5×5 row-WD reachable state space has a fixed, measured size. Pin it
    /// exactly: any change means the BFS transition rule or goal margins drifted.
    /// (4×4 reference: 24,964.)
    #[test]
    fn table_size_matches_measured() {
        const WD24_STATES: usize = 65_650_495;
        let n = WalkingDistanceHeuristic::table_size();
        assert_eq!(
            n, WD24_STATES,
            "WD table size drifted from the measured value"
        );
        println!("24-puzzle WD table size: {n} states");
    }

    // --- persistence (save_dist_table / load_dist_table) ---

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}_{}", std::process::id(), name))
    }

    /// A small synthetic table so the persistence tests don't pay the full BFS.
    /// Uses a non-`FULL` kind and no entry-count expectation on load.
    const TEST_KIND: u32 = 0xABCD;
    fn synthetic_table() -> WdTable {
        let mut t: WdTable = HashMap::with_capacity_and_hasher(8, WdBuild::default());
        for i in 0u64..37 {
            // Keys spanning both u32 halves; values in the u8 distance range.
            let key = (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & ((1 << 63) - 1);
            t.insert(key, (i % 200) as u8);
        }
        t
    }

    #[test]
    fn dist_table_round_trip_synthetic() {
        let path = tmp_path("wd_roundtrip.bin");
        let table = synthetic_table();
        save_dist_table(&path, TEST_KIND, &table).unwrap();
        let loaded = load_dist_table(&path, TEST_KIND, Some(table.len() as u64)).unwrap();
        assert_eq!(loaded, table, "round-trip changed the table");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dist_table_serialization_is_deterministic() {
        // Byte-identical output across writes ⇒ the SHA-256 pin is stable. The
        // HashMap iteration order is not deterministic, so this exercises the
        // key-sort in `save_dist_table`.
        let table = synthetic_table();
        let p1 = tmp_path("wd_det1.bin");
        let p2 = tmp_path("wd_det2.bin");
        save_dist_table(&p1, TEST_KIND, &table).unwrap();
        save_dist_table(&p2, TEST_KIND, &table).unwrap();
        let b1 = std::fs::read(&p1).unwrap();
        let b2 = std::fs::read(&p2).unwrap();
        assert_eq!(b1, b2, "serialization is not byte-deterministic");
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn load_rejects_kind_mismatch() {
        let path = tmp_path("wd_kind.bin");
        save_dist_table(&path, TEST_KIND, &synthetic_table()).unwrap();
        let err = load_dist_table(&path, WD_KIND_FULL, None).unwrap_err();
        assert!(
            matches!(err, WdLoadError::KindMismatch { .. }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_entry_count_mismatch() {
        let path = tmp_path("wd_count.bin");
        let table = synthetic_table();
        save_dist_table(&path, TEST_KIND, &table).unwrap();
        let wrong = table.len() as u64 + 1;
        let err = load_dist_table(&path, TEST_KIND, Some(wrong)).unwrap_err();
        assert!(
            matches!(err, WdLoadError::EntryMismatch { .. }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = tmp_path("wd_magic.bin");
        save_dist_table(&path, TEST_KIND, &synthetic_table()).unwrap();
        // Corrupt the first magic byte.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let err = load_dist_table(&path, TEST_KIND, None).unwrap_err();
        assert!(matches!(err, WdLoadError::BadMagic(_)), "got {err:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_truncation() {
        let path = tmp_path("wd_trunc.bin");
        save_dist_table(&path, TEST_KIND, &synthetic_table()).unwrap();
        // Drop the final byte → declared entry count no longer matches the size.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        std::fs::write(&path, &bytes).unwrap();
        let err = load_dist_table(&path, TEST_KIND, None).unwrap_err();
        assert!(
            matches!(err, WdLoadError::SizeMismatch { .. }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// End-to-end on the *real* full WD table: build via BFS, persist, reload, and
    /// confirm the reloaded table is identical — the exact path the `build_wd24`
    /// tool and the `data/wd24.bin` load cascade rely on. Ignored by default
    /// because the BFS is the ~17–25 s cost this whole feature exists to avoid.
    #[test]
    #[ignore = "full WD BFS (~17-25s); run with --ignored"]
    fn full_wd_table_build_save_load_round_trip() {
        let table = build_full_table();
        assert_eq!(table.len() as u64, FULL_WD_ENTRIES);
        let path = tmp_path("wd24_full_roundtrip.bin");
        save_dist_table(&path, WD_KIND_FULL, &table).unwrap();
        let loaded = load_dist_table(&path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).unwrap();
        assert_eq!(loaded, table, "full-table round-trip changed the table");
        let _ = std::fs::remove_file(&path);
    }

    // --- retargetable WD (WalkingDistanceTo) ---

    /// The "R" board (double antipode): blank at cell 0, tile `25 - i` at cell
    /// `i` for `i ∈ 1..25`. Its blank sits at row 0, column 0.
    fn r_board() -> State {
        let mut cells = [0u8; N_CELLS];
        for i in 1..N_CELLS {
            cells[i] = (25 - i) as u8;
        }
        State(cells)
    }

    /// **Regression (critical):** `WalkingDistanceTo::new(&GOAL)` must reproduce
    /// the GOAL-only [`WalkingDistanceHeuristic`] exactly. If the projection or
    /// the seed/table were wrong this would diverge. Needs the shared GOAL table
    /// for the reference, so WD-gate on `data/wd24.bin`. Uses the `ml` scramble
    /// generator, so it is gated to the `ml` feature (keeps non-`ml` builds clean).
    #[cfg(feature = "ml")]
    #[ignore = "builds a fresh full 24-puzzle WD table via BFS (~15-25s, uncached) + loads the \
                shared table; run with --ignored. The WD algorithm is smoke-tested fast on the \
                15-puzzle (puzzle15::search::walking_distance)"]
    #[test]
    fn wd_to_goal_matches_goal_heuristic_on_random_boards() {
        if !std::path::Path::new("data/wd24.bin").exists() {
            return;
        }
        WalkingDistanceHeuristic::warm_up();
        let wd_to_goal = WalkingDistanceTo::new(&GOAL);
        let mut rng = crate::puzzle24::ml::scramble::Rng::new(0xD1CE_F00D_1234_5678);
        for i in 0..500 {
            let (s, _k) = crate::puzzle24::ml::scramble::scramble(&mut rng, 200);
            let reference = WalkingDistanceHeuristic.h(&s);
            let got = wd_to_goal.h(&s);
            assert_eq!(
                got, reference,
                "WD-to-GOAL != GOAL WD on board {} {:?}",
                i, s.0
            );
        }
    }

    /// A target is at distance 0 from itself: `WD-to-T(T) == 0` for T = GOAL and
    /// T = R. Builds its own tables, so no `data/wd24.bin` needed.
    #[ignore = "builds fresh full 24-puzzle WD tables via BFS (~15-25s each, uncached); run with \
                --ignored. The WD algorithm is smoke-tested fast on the 15-puzzle \
                (puzzle15::search::walking_distance)"]
    #[test]
    fn wd_to_target_of_target_is_zero() {
        assert_eq!(
            WalkingDistanceTo::new(&GOAL).h(&GOAL),
            0,
            "WD-to-GOAL(GOAL)"
        );
        let r = r_board();
        let wd_to_r = WalkingDistanceTo::new(&r);
        assert_eq!(
            wd_to_r.target_blank(),
            (0, 0),
            "R blank should be row 0, col 0"
        );
        assert_eq!(wd_to_r.h(&r), 0, "WD-to-R(R)");
    }

    /// Every one-move neighbour of R is at WD-to-R exactly 1 (one tile crosses
    /// one axis boundary; the other axis is unchanged).
    #[ignore = "builds a fresh full 24-puzzle WD table via BFS (~15-25s, uncached); run with \
                --ignored. The WD algorithm is smoke-tested fast on the 15-puzzle \
                (puzzle15::search::walking_distance)"]
    #[test]
    fn wd_to_r_of_r_neighbors_is_one() {
        let r = r_board();
        let wd_to_r = WalkingDistanceTo::new(&r);
        for m in r.legal_moves().iter() {
            let n = r.apply(m);
            assert_eq!(wd_to_r.h(&n), 1, "WD-to-R(R.apply({m:?})) should be 1");
        }
    }

    /// Build-time sanity: `WalkingDistanceTo::new(&R)` completes well under a
    /// minute (same BFS as the GOAL table). Single table, since R's blank shares
    /// row/col index 0.
    #[ignore = "times a fresh full 24-puzzle WD BFS build (~15-25s) — this test IS the slow build; \
                run with --ignored. The WD algorithm is smoke-tested fast on the 15-puzzle \
                (puzzle15::search::walking_distance)"]
    #[test]
    fn wd_to_r_builds_quickly() {
        let t = std::time::Instant::now();
        let wd_to_r = WalkingDistanceTo::new(&r_board());
        let elapsed = t.elapsed();
        // The col table must be shared (blank at row 0 == col 0).
        assert!(
            wd_to_r.col_table.is_none(),
            "R should reuse one table for both axes"
        );
        println!("WalkingDistanceTo::new(&R) built in {elapsed:?}");
        assert!(
            elapsed.as_secs() < 60,
            "R WD build took too long: {elapsed:?}"
        );
    }
}
