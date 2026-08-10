//! cWD — escape-constrained Walking Distance, as a per-node admissible heuristic.
//!
//! WD projects the board onto one axis and takes the exact optimum of the
//! row/column-sorting relaxation. cWD keeps that abstraction but adds a **side
//! constraint**: for each goal line `g`, an admissible plan must make at least
//! `x_g` "escape" moves — a type-`g` tile leaving physical line `g` — where
//! `x_g = residents_g − LIS(their goal-cross order)` is the n-tile linear-conflict
//! bound. The escapes enter as a constraint on WD's own move budget (never an
//! addend), so `cWD ≥ WD` and cWD stays admissible; the row and column halves sum
//! admissibly (disjoint move classes). See `docs`/`PUZZLE24.md` and the machine-
//! checked soundness of the escape bound in `proofs/puzzle15-wd`.
//!
//! This evaluator is **table-free**: it reuses the shared WD distance table
//! (`data/wd24.bin`) and runs a small vector-constrained A\* over the product
//! graph (WD-state × per-line escape counters) at each node. That A\* is far more
//! expensive than a WD lookup, so cWD here is a reference/verification heuristic
//! and a search heuristic for modest instances — a fast production form would
//! precompute the escape-constrained distances into a table.

use super::heuristic::Heuristic;
use super::recursive::IncHeuristic;
use super::walking_distance::{
    build_full_table, load_dist_table, WdBuild, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use super::SearchStats;
use crate::puzzle24::state::{Move, State, W};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub(crate) type Matrix = [[u8; W]; W];

/// The precomputed single-line escape-constrained surcharge overlay
/// (`data/cwd_single.bin`, built by `examples/build_cwd_table.rs`). Maps each WD
/// contingency key to a packed surcharge curve per goal line: nibble `d-1` of
/// `curves[g]` is `Δ(σ, g, d)/2` for demand `d = 1..=4`. Lookup is O(1), so the
/// table-based heuristic costs ~2 lookups per axis instead of a constrained A\*.
pub type CwdOverlay = HashMap<u64, [u16; W], WdBuild>;

/// One merged contingency record: the WD value, the per-line surcharge curves,
/// **and** the WD of every axis-neighbor, so the hot path resolves both with a
/// single probe per axis *and* can bound a child before probing (see `nbr_wd`).
#[derive(Clone, Copy)]
pub struct CwdCell {
    pub wd: u8,
    pub curves: [u16; W],
    /// WD of the ≤`2·W` axis-neighbors. Slot `dir·W + g` (`dir` 0 = blank-index
    /// −1, 1 = +1) is the WD of the state reached when a goal-type-`g` tile
    /// crosses the axis the blank steps over; `255` where that step is off-board
    /// or no such tile is present. Enables "prune the child before probing it".
    pub nbr_wd: [u8; 2 * W],
}

/// Merged WD + surcharge table (built once at load; `σ → (WD, curves)`).
pub type CwdMerged = HashMap<u64, CwdCell, WdBuild>;

// ------------------- mmap'd merged-table artifact -------------------

/// Geometry of `data/cwd_mm.bin` (`magic "CWDN"`): 32 B slots (key 8 +
/// curves 10 at offset 8 so the u16s stay aligned + wd 1 + nbr_wd 10 +
/// pad 3), linear probing at 0.70 load (slot count in header field \[4..8\]),
/// key 0 = empty (the zero key decodes to an impossible token matrix).
/// Snapshot of the merged table with `nbr_wd` FILLED, so loading it replaces
/// both the merge and the neighbour pass. Probes are essentially always
/// successful (only reachable contingency keys are queried), so the relevant
/// chain cost is the successful-search one — mean ~2.1 slots at 0.70 load,
/// within a cache line or two of the home slot.
const CWDM_SLOT: usize = 32;
const CWDM_HEADER: usize = 16;

/// Home slot by fastrange over the Fibonacci-mixed key: the top 64 bits of
/// `hash × slots`. One multiply mixes the packed key's structured low bits
/// upward, exactly where fastrange reads.
#[inline]
fn cwdm_home(key: u64, slots: usize) -> usize {
    let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (((h as u128) * (slots as u128)) >> 64) as usize
}

/// Zero-parse backing for the merged-table fast path: "loading" is a
/// page-table operation plus one sequential pre-touch pass.
pub struct CwdMm {
    map: memmap2::Mmap,
    /// Slot count of this artifact's geometry (header field \[4..8\]).
    slots: usize,
}

impl CwdMm {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let map = crate::puzzle24::hugemap::map_table(path)?;
        assert_eq!(
            &map[..4],
            b"CWDN",
            "bad cwd_mm magic (legacy CWDM artifacts are no longer supported; \
             rebuild with build_cwd_mm_artifact)"
        );
        let slots = u32::from_le_bytes(map[4..8].try_into().unwrap()) as usize;
        assert_eq!(
            map.len(),
            CWDM_HEADER + slots * CWDM_SLOT,
            "cwd_mm artifact has the wrong size for its header geometry"
        );
        // Eager pre-touch (one byte per 16 KiB page, sequential): keeps
        // demand-paging faults out of the search phase.
        let mut sum = 0u64;
        for off in (0..map.len()).step_by(16384) {
            sum = sum.wrapping_add(map[off] as u64);
        }
        std::hint::black_box(sum);
        Ok(CwdMm { map, slots })
    }

    /// What `merged.get(&key).copied()` would return.
    #[inline]
    pub(crate) fn probe_cell(&self, key: u64) -> Option<CwdCell> {
        let mut i = cwdm_home(key, self.slots);
        loop {
            let off = CWDM_HEADER + i * CWDM_SLOT;
            let k = u64::from_le_bytes(self.map[off..off + 8].try_into().unwrap());
            if k == key {
                let mut curves = [0u16; W];
                for (g, c) in curves.iter_mut().enumerate() {
                    *c = u16::from_le_bytes(
                        self.map[off + 8 + 2 * g..off + 10 + 2 * g]
                            .try_into()
                            .unwrap(),
                    );
                }
                let wd = self.map[off + 18];
                let mut nbr_wd = [0u8; 2 * W];
                nbr_wd.copy_from_slice(&self.map[off + 19..off + 29]);
                return Some(CwdCell { wd, curves, nbr_wd });
            }
            if k == 0 {
                return None;
            }
            i += 1;
            if i == self.slots {
                i = 0;
            }
        }
    }
}

/// Either backing for the merged-table fast path. `Copy`; the branch is
/// taken only on front-cache misses, the hit path never sees it.
#[derive(Clone, Copy)]
pub(crate) enum MergedBacking<'a> {
    Map(&'a CwdMerged),
    Mm(&'a CwdMm),
}

impl MergedBacking<'_> {
    #[inline(always)]
    pub(crate) fn cell(&self, key: u64) -> Option<CwdCell> {
        match self {
            MergedBacking::Map(m) => m.get(&key).copied(),
            MergedBacking::Mm(mm) => mm.probe_cell(key),
        }
    }
}

/// Slot count for a freshly built dense artifact: 0.70 load. Measured on R
/// (exhaust-146 A/B, 2026-08-05): search time within noise of the 0.489
/// legacy geometry (mean chain 2.07 vs 1.5), resident footprint −30%.
/// 0.875 saved another 1.2 GB but cost a repeatable ~2% in search.
#[inline]
pub(crate) fn dense_slots(count: usize) -> usize {
    count * 10 / 7
}

/// Build the mmap artifact from a fully-prepared [`Cwd`] (merged table
/// present, `nbr_wd` filled). Writes the dense geometry (`magic "CWDN"`,
/// 0.70 load) directly — building and compaction are one step.
/// Build the merged-table mmap artifact at `out` (from `data/wd24.bin` +
/// `data/cwd_single.bin`, ~4 GiB written) and verify every key round-trips.
/// Panics on any mismatch — this is the artifact's validation gate, run by
/// the `build_cwd_artifacts mm` binary.
pub fn build_cwd_mm_artifact(out: &Path) {
    let t0 = std::time::Instant::now();
    let cwd = Cwd::new().with_neighbor_prune(true);
    assert!(cwd.has_overlay(), "needs data/cwd_single.bin");
    eprintln!("merged table ready in {:.0?}", t0.elapsed());
    build_cwd_mm(&cwd, out).expect("build");
    let mm = CwdMm::load(out).expect("load");
    let merged = cwd.merged_table().unwrap();
    let mut n = 0u64;
    for (&k, cell) in merged {
        let got = mm.probe_cell(k).expect("every key must probe");
        assert_eq!(got.wd, cell.wd, "wd mismatch at {k:#x}");
        assert_eq!(got.curves, cell.curves, "curves mismatch at {k:#x}");
        assert_eq!(got.nbr_wd, cell.nbr_wd, "nbr mismatch at {k:#x}");
        n += 1;
    }
    assert!(!merged.contains_key(&1));
    assert!(mm.probe_cell(1).is_none(), "absent key must return None");
    eprintln!(
        "verified {n} cells round-trip in {:.0?} total",
        t0.elapsed()
    );
}

pub fn build_cwd_mm(cwd: &Cwd, path: &Path) -> std::io::Result<()> {
    let merged = cwd
        .merged_table()
        .expect("build_cwd_mm needs the merged table");
    assert!(
        cwd.neighbor_prune_enabled(),
        "fill nbr_wd first: Cwd::new().with_neighbor_prune(true)"
    );
    let slots = dense_slots(merged.len());
    let mut buf = vec![0u8; CWDM_HEADER + slots * CWDM_SLOT];
    buf[..4].copy_from_slice(b"CWDN");
    buf[4..8].copy_from_slice(
        &u32::try_from(slots)
            .expect("slot count fits u32")
            .to_le_bytes(),
    );
    buf[8..16].copy_from_slice(&(merged.len() as u64).to_le_bytes());
    let (mut total_chain, mut max_chain) = (0u64, 0u32);
    for (&k, cell) in merged {
        assert_ne!(k, 0, "key 0 is the empty sentinel");
        let mut i = cwdm_home(k, slots);
        let mut chain = 1u32;
        loop {
            let off = CWDM_HEADER + i * CWDM_SLOT;
            let cur = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            if cur == 0 {
                buf[off..off + 8].copy_from_slice(&k.to_le_bytes());
                for (g, c) in cell.curves.iter().enumerate() {
                    buf[off + 8 + 2 * g..off + 10 + 2 * g].copy_from_slice(&c.to_le_bytes());
                }
                buf[off + 18] = cell.wd;
                buf[off + 19..off + 29].copy_from_slice(&cell.nbr_wd);
                break;
            }
            assert_ne!(cur, k, "duplicate key");
            i += 1;
            if i == slots {
                i = 0;
            }
            chain += 1;
        }
        total_chain += chain as u64;
        max_chain = max_chain.max(chain);
    }
    eprintln!(
        "  cwd_mm: {} keys into {slots} slots (load {:.3}), mean chain {:.2}, longest {max_chain}",
        merged.len(),
        merged.len() as f64 / slots as f64,
        total_chain as f64 / merged.len() as f64
    );
    std::fs::write(path, &buf)
}

/// Surcharge (in moves) for one axis: single-line-max — the strongest single-line
/// escape bound over the demanded lines. `Δ = 2 · nibble`.
#[inline]
pub(crate) fn surcharge_from_curves(curves: &[u16; W], dem: &[u8; W]) -> u8 {
    let mut best = 0u8;
    for g in 0..W {
        let d = dem[g];
        if (1..=4).contains(&d) {
            let s = 2 * ((curves[g] >> (4 * (d as u16 - 1))) & 0xF) as u8;
            best = best.max(s);
        }
    }
    best
}

/// One process-wide merged cWD table for the table-gated tests.
///
/// Each `Cwd::new()` builds a ~4.4 GB merged map (65,650,495 entries in
/// 134,217,728 buckets at 32 B). `cargo test` runs tests in parallel, so N
/// table-gated tests meant N × 4.4 GB — on this 34 GB machine that produced
/// **116 GB of swapouts and 907 free pages**, turning a compute-bound suite into
/// a thrashing one. Sharing one immutable instance makes the cost constant
/// instead of per-test, and lets the suite run in parallel again.
///
/// `Cwd` is plain data with no interior mutability, so a shared `&'static` is
/// sound and needs no lock.
#[cfg(all(test, feature = "cwd-table-tests"))]
pub(crate) fn shared_merged_cwd() -> Option<&'static Cwd> {
    use std::sync::OnceLock;
    static SHARED: OnceLock<Option<Cwd>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            if Path::new("data/wd24.bin").exists() && Path::new("data/cwd_single.bin").exists() {
                let c = Cwd::new().with_neighbor_prune(true);
                if c.has_overlay() {
                    return Some(c);
                }
            }
            eprintln!("cWD tables absent — skipping table-gated test");
            None
        })
        .as_ref()
}

/// Load the surcharge overlay from `data/cwd_single.bin`.
pub fn load_cwd_overlay(path: &Path) -> std::io::Result<CwdOverlay> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"CWDS" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad cwd_single magic",
        ));
    }
    let mut u4 = [0u8; 4];
    f.read_exact(&mut u4)?; // version
    let mut u8b = [0u8; 8];
    f.read_exact(&mut u8b)?;
    let count = u64::from_le_bytes(u8b) as usize;
    let mut overlay: CwdOverlay = HashMap::with_capacity_and_hasher(count, WdBuild::default());
    let mut rec = [0u8; 18];
    for _ in 0..count {
        f.read_exact(&mut rec)?;
        let key = u64::from_le_bytes(rec[0..8].try_into().unwrap());
        let mut curves = [0u16; W];
        for g in 0..W {
            curves[g] = u16::from_le_bytes(rec[8 + 2 * g..10 + 2 * g].try_into().unwrap());
        }
        overlay.insert(key, curves);
    }
    Ok(overlay)
}

// ---- WD-table key codec (identical layout to walking_distance::pack) --------

pub(crate) fn pack(m: &Matrix, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

/// Bit offset of the packed 3-bit field for matrix cell `(r, c)` (`c < W-1`).
/// [`pack`] shifts cell `(r,c)` — linear index `r*(W-1)+c` — left the most for
/// `(0,0)` and least for the last cell, so field `i` sits at `((W*(W-1)-1)-i)*3`.
#[inline]
pub(crate) const fn key_bit(r: usize, c: usize) -> u64 {
    ((W * (W - 1) - 1 - (r * (W - 1) + c)) * 3) as u64
}
/// The blank field's bit offset (the top 3 bits, above the `W*(W-1)` cells).
pub(crate) const KEY_BLANK_BIT: u64 = (W * (W - 1) * 3) as u64;

pub(crate) fn unpack(key: u64) -> (Matrix, u8) {
    let blank = ((key >> 60) & 0x7) as u8;
    let mut m = [[0u8; W]; W];
    let mut k = key;
    for r in (0..W).rev() {
        for c in (0..(W - 1)).rev() {
            m[r][c] = (k & 0x7) as u8;
            k >>= 3;
        }
    }
    for (r, row) in m.iter_mut().enumerate() {
        let margin: u8 = if r as u8 == blank { 4 } else { 5 };
        let partial: u8 = row[..W - 1].iter().sum();
        row[W - 1] = margin - partial;
    }
    (m, blank)
}

/// The ≤`2·W` axis-neighbor keys of a contingency `key` (shared codec for both
/// axes). Slot `dir·W + g` (`dir` 0 = blank-index −1, 1 = +1) holds the key of
/// the state reached when a goal-type-`g` tile crosses the axis the blank steps
/// over — i.e. the exact transition `make` applies. `None` where that step is
/// off-board or no such tile is present. Mirrors the update in `make`
/// (`m[nb][g] -= 1; m[b][g] += 1;` new blank `nb`).
fn neighbor_keys(key: u64) -> [Option<u64>; 2 * W] {
    let (m, blank) = unpack(key);
    let b = blank as usize;
    let mut out = [None; 2 * W];
    for (dir, nb) in [b.wrapping_sub(1), b + 1].into_iter().enumerate() {
        if nb >= W {
            continue; // off-board (b == 0 ⇒ wrapping_sub(1) overflows past W)
        }
        for g in 0..W {
            if m[nb][g] > 0 {
                let mut m2 = m;
                m2[nb][g] -= 1;
                m2[b][g] += 1;
                out[dir * W + g] = Some(pack(&m2, nb as u8));
            }
        }
    }
    out
}

/// The solved contingency (identity with the 4-margin in the blank's axis).
/// Serves both axes: the goal blank sits at row 4 and column 4.
pub(crate) fn goal_key() -> u64 {
    let mut m = [[0u8; W]; W];
    for d in 0..W {
        m[d][d] = 5;
    }
    m[W - 1][W - 1] = 4;
    pack(&m, (W - 1) as u8)
}

/// Longest strictly-increasing subsequence length (goal-cross indices within one
/// goal line are distinct ⇒ strict). `k ≤ 5` ⇒ trivial O(k²).
fn lis_strict(seq: &[u8]) -> usize {
    let n = seq.len();
    let mut dp = [0usize; W];
    let mut best = 0usize;
    for i in 0..n {
        dp[i] = 1;
        for j in 0..i {
            if seq[j] < seq[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
            }
        }
        best = best.max(dp[i]);
    }
    best
}

/// Project `s` onto both axes. Returns
/// `(m_row, blank_row, demand_row, m_col, blank_col, demand_col)` where
/// `demand_row[g] = x_g` = (goal-row-`g` residents physically in row `g`)
/// − LIS(their goal columns): the forced type-`g` escapes. Symmetric for columns.
pub(crate) fn project(s: &State) -> (Matrix, u8, [u8; W], Matrix, u8, [u8; W]) {
    let mut m_row = [[0u8; W]; W];
    let mut m_col = [[0u8; W]; W];
    let mut br = 0u8;
    let mut bc = 0u8;
    let mut row_res: [Vec<u8>; W] = Default::default();
    let mut col_res: [Vec<u8>; W] = Default::default();
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
        let gr = (goal_pos / W) as u8;
        let gc = (goal_pos % W) as u8;
        m_row[r as usize][gr as usize] += 1;
        m_col[c as usize][gc as usize] += 1;
        if gr == r {
            row_res[r as usize].push(gc); // physical-column order
        }
        if gc == c {
            col_res[c as usize].push(gr); // physical-row order
        }
    }
    let mut dem_row = [0u8; W];
    let mut dem_col = [0u8; W];
    for g in 0..W {
        dem_row[g] = (row_res[g].len() - lis_strict(&row_res[g])) as u8;
        dem_col[g] = (col_res[g].len() - lis_strict(&col_res[g])) as u8;
    }
    (m_row, br, dem_row, m_col, bc, dem_col)
}

// ---- vector-constrained A* on the product graph -----------------------------

/// Reusable A\* scratch, so the per-node evaluation allocates nothing.
pub struct CwdScratch {
    best: HashMap<u128, u8>,
    buckets: Vec<Vec<(u64, u32)>>,
}

impl CwdScratch {
    pub fn new() -> Self {
        CwdScratch {
            best: HashMap::with_capacity(1 << 14),
            buckets: (0..210).map(|_| Vec::new()).collect(),
        }
    }
}

impl Default for CwdScratch {
    fn default() -> Self {
        Self::new()
    }
}

const UNSEEN: u8 = 0xFF;
const CLOSED: u8 = 0x80;
const POP_BUDGET: u64 = 40_000_000;

/// Min cost of an abstract row/column plan `(m, blank) → goal` making ≥ `dem[g]`
/// type-`g` escapes for every line `g`. `h` = unconstrained WD (consistent), so
/// the first pop of the saturated goal is optimal. `None` iff the pop budget is
/// exhausted first (caller falls back to the admissible WD half).
fn cwd_axis(
    table: &WdTable,
    m: &Matrix,
    blank: u8,
    goal: u64,
    dem: &[u8; W],
    scratch: &mut CwdScratch,
) -> Option<u8> {
    // demanded lines + mixed-radix escape-counter layout
    let mut lines: [usize; W] = [0; W];
    let mut n_lines = 0usize;
    for g in 0..W {
        if dem[g] > 0 {
            lines[n_lines] = g;
            n_lines += 1;
        }
    }
    let start_key = pack(m, blank);
    if n_lines == 0 {
        return Some(*table.get(&start_key).expect("start reachable"));
    }
    let mut radix = [1u32; W];
    let mut full: u32 = 1;
    for i in 0..n_lines {
        radix[i] = full;
        full *= dem[lines[i]] as u32 + 1;
    }
    let full_index = full - 1;
    let counter_of = |g: usize| -> Option<usize> { (0..n_lines).find(|&i| lines[i] == g) };

    let h0 = *table.get(&start_key).expect("start reachable") as usize;
    let best = &mut scratch.best;
    best.clear();
    for b in scratch.buckets.iter_mut() {
        b.clear();
    }
    let statekey = |wd: u64, ci: u32| -> u128 { ((wd as u128) << 16) | ci as u128 };

    best.insert(statekey(start_key, 0), 0);
    scratch.buckets[h0].push((start_key, 0));

    let mut pops: u64 = 0;
    for f in h0..scratch.buckets.len() {
        let mut i = 0;
        while i < scratch.buckets[f].len() {
            let (key, ci) = scratch.buckets[f][i];
            i += 1;
            let sk = statekey(key, ci);
            let g = match best.get(&sk) {
                Some(&v) if v & CLOSED == 0 => v,
                _ => continue,
            };
            best.insert(sk, g | CLOSED);
            pops += 1;
            if pops > POP_BUDGET {
                return None;
            }
            if key == goal && ci == full_index {
                return Some(g);
            }
            let (mm, br) = unpack(key);
            let g2 = g + 1;
            for from in [br.wrapping_sub(1), br + 1] {
                let from = from as usize;
                if from >= W {
                    continue;
                }
                for t in 0..W {
                    if mm[from][t] == 0 {
                        continue;
                    }
                    let mut m2 = mm;
                    m2[from][t] -= 1;
                    m2[br as usize][t] += 1;
                    let child_key = pack(&m2, from as u8);
                    // escape of line t iff a type-t tile leaves physical line t
                    let mut ci2 = ci;
                    if from == t {
                        if let Some(idx) = counter_of(t) {
                            let cur = (ci / radix[idx]) % (dem[t] as u32 + 1);
                            if cur < dem[t] as u32 {
                                ci2 += radix[idx];
                            }
                        }
                    }
                    let h = *table.get(&child_key).expect("child reachable") as usize;
                    let csk = statekey(child_key, ci2);
                    let slot = best.entry(csk).or_insert(UNSEEN);
                    if *slot == UNSEEN || (*slot & CLOSED == 0 && g2 < *slot) {
                        *slot = g2;
                        scratch.buckets[g2 as usize + h].push((child_key, ci2));
                    }
                }
            }
        }
    }
    None
}

// ---- the heuristic ----------------------------------------------------------

/// Escape-constrained Walking Distance heuristic. Owns a WD distance table, and
/// optionally the precomputed single-line surcharge overlay — when present, each
/// node costs ~2 lookups per axis (the fast path); otherwise it runs the
/// per-node constrained A\* (the reference path).
pub struct Cwd {
    /// WD table for the reference A\* path. Emptied once `merged` is built (the
    /// fast path never touches it), to avoid holding WD twice.
    table: WdTable,
    goal: u64,
    merged: Option<CwdMerged>,
    /// When set (and the merged table is present), the search may bound a child
    /// from its parent's cached `nbr_wd` and prune it before probing. Off by
    /// default; toggled per-run for A/B. See [`Cwd::child_h_lb`].
    neighbor_prune: bool,
    /// Precomputed per-line escape-demand table (32 KiB); replaces the per-node
    /// LIS in `make`. See [`build_demand_lut`].
    demand_lut: Box<[u8; DEMAND_LUT_LEN]>,
    /// mmap'd merged-table artifact; preferred over `merged` when present.
    mm: Option<CwdMm>,
}

impl Cwd {
    /// Load the WD table from `data/wd24.bin` (build if absent). If the surcharge
    /// overlay `data/cwd_single.bin` is present, merge WD+surcharge into one table
    /// for the single-probe-per-axis fast path and drop the standalone WD table.
    pub fn new() -> Self {
        let mut c = Self::with_table_path(Path::new("data/wd24.bin"));
        let ov = Path::new("data/cwd_single.bin");
        if ov.exists() {
            if let Ok(overlay) = load_cwd_overlay(ov) {
                let mut merged: CwdMerged =
                    HashMap::with_capacity_and_hasher(c.table.len(), WdBuild::default());
                for (&k, &wd) in c.table.iter() {
                    let curves = overlay.get(&k).copied().unwrap_or([0u16; W]);
                    // `nbr_wd` is filled lazily by `with_neighbor_prune(true)` — an
                    // opt-in, so a plain cWD run pays neither the pass nor the RAM.
                    merged.insert(
                        k,
                        CwdCell {
                            wd,
                            curves,
                            nbr_wd: [255; 2 * W],
                        },
                    );
                }
                c.merged = Some(merged);
                c.table = HashMap::with_capacity_and_hasher(0, WdBuild::default());
                // free WD copy
            }
        }
        c
    }

    /// Load the WD table from `path`, falling back to a fresh BFS build. No merge.
    pub fn with_table_path(path: &Path) -> Self {
        let table = if path.exists() {
            load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES))
                .unwrap_or_else(|_| build_full_table())
        } else {
            build_full_table()
        };
        Cwd {
            table,
            goal: goal_key(),
            merged: None,
            neighbor_prune: false,
            demand_lut: build_demand_lut(),
            mm: None,
        }
    }

    /// Build from an already-loaded WD table (shares the codec). No merge.
    pub fn from_table(table: WdTable) -> Self {
        Cwd {
            table,
            goal: goal_key(),
            merged: None,
            neighbor_prune: false,
            demand_lut: build_demand_lut(),
            mm: None,
        }
    }

    /// The zero-parse constructor: mmap `data/cwd_mm.bin` (pre-touched) and
    /// skip the WD load, the overlay merge and the neighbour pass entirely.
    /// The artifact snapshots `nbr_wd` filled, so the pre-prune is live.
    pub fn mm_only(path: &Path) -> std::io::Result<Self> {
        let mm = CwdMm::load(path)?;
        Ok(Cwd {
            table: HashMap::with_capacity_and_hasher(0, WdBuild::default()),
            goal: goal_key(),
            merged: None,
            neighbor_prune: true,
            demand_lut: build_demand_lut(),
            mm: Some(mm),
        })
    }

    /// Enable/disable the prune-child-before-probe optimization (needs the merged
    /// table; a no-op on the A\* reference path). When enabling, fill each merged
    /// cell's neighbor-WD from the table itself (one pass, ~seconds, ~+1 GB).
    /// Returns `self` for chaining.
    pub fn with_neighbor_prune(mut self, on: bool) -> Self {
        self.neighbor_prune = on;
        if on {
            self.fill_neighbor_wd();
        }
        self
    }

    /// Populate every merged cell's `nbr_wd` with the WD of its axis-neighbors,
    /// derived purely from the (now complete) merged table — no on-disk artifact.
    /// Idempotent; a no-op when the merged table is absent.
    fn fill_neighbor_wd(&mut self) {
        let Some(merged) = self.merged.as_mut() else {
            return;
        };
        let keys: Vec<u64> = merged.keys().copied().collect();
        for k in keys {
            let mut nbr = [255u8; 2 * W];
            for (slot, opt) in neighbor_keys(k).iter().enumerate() {
                if let Some(nkey) = opt {
                    if let Some(cell) = merged.get(nkey) {
                        nbr[slot] = cell.wd;
                    }
                }
            }
            merged.get_mut(&k).unwrap().nbr_wd = nbr;
        }
    }

    /// Whether the fast (merged-table) path is active.
    pub fn has_overlay(&self) -> bool {
        self.merged.is_some() || self.mm.is_some()
    }

    /// The fast path's backing store: the mmap artifact when present,
    /// else the in-memory merged table.
    #[inline]
    pub(crate) fn backing(&self) -> Option<MergedBacking<'_>> {
        if let Some(mm) = &self.mm {
            return Some(MergedBacking::Mm(mm));
        }
        self.merged.as_ref().map(MergedBacking::Map)
    }

    /// The merged WD+surcharge table, for engines that drive the fast path
    /// themselves rather than through [`IncHeuristicMut`] (see `flat.rs`).
    #[inline]
    pub(crate) fn merged_table(&self) -> Option<&CwdMerged> {
        self.merged.as_ref()
    }

    /// The per-line escape-demand LUT, shared with such engines.
    #[inline]
    pub(crate) fn demand_lut(&self) -> &[u8; DEMAND_LUT_LEN] {
        &self.demand_lut
    }

    /// Whether the neighbour-WD child pre-prune is enabled (and hence whether
    /// each merged cell's `nbr_wd` has been filled).
    #[inline]
    pub(crate) fn neighbor_prune_enabled(&self) -> bool {
        self.neighbor_prune
    }

    /// `cWD(s) = cWD_row + cWD_col`, each ≥ its WD half. With the merged table
    /// this is the single-line-max lookup — **one probe per axis**; otherwise the
    /// constrained A\* (falling back to WD on any axis whose A\* exhausts its pop
    /// budget — still admissible).
    pub fn eval(&self, s: &State, scratch: &mut CwdScratch) -> u8 {
        let (m_row, br, dem_row, m_col, bc, dem_col) = project(s);
        let kr = pack(&m_row, br);
        let kc = pack(&m_col, bc);
        if let Some(mg) = &self.merged {
            let r = mg.get(&kr).expect("row reachable");
            let c = mg.get(&kc).expect("col reachable");
            return r.wd
                + surcharge_from_curves(&r.curves, &dem_row)
                + c.wd
                + surcharge_from_curves(&c.curves, &dem_col);
        }
        let wd_row = *self.table.get(&kr).expect("row reachable");
        let wd_col = *self.table.get(&kc).expect("col reachable");
        let c_row =
            cwd_axis(&self.table, &m_row, br, self.goal, &dem_row, scratch).unwrap_or(wd_row);
        let c_col =
            cwd_axis(&self.table, &m_col, bc, self.goal, &dem_col, scratch).unwrap_or(wd_col);
        c_row.max(wd_row) + c_col.max(wd_col)
    }
}

impl Default for Cwd {
    fn default() -> Self {
        Self::new()
    }
}

impl Heuristic for Cwd {
    fn h(&self, s: &State) -> u8 {
        let mut scratch = CwdScratch::new();
        self.eval(s, &mut scratch)
    }
}

/// `IncHeuristic` recomputes cWD from scratch per node (a per-call scratch alloc).
/// Only the parallel driver's shallow split uses this path; the main search runs
/// through `IncHeuristicMut`, which reuses a scratch buffer.
impl IncHeuristic for Cwd {
    type Ctx = ();

    fn root(&self, s: &State, stats: &mut SearchStats) -> (u8, ()) {
        let _ = stats;
        (self.h(s), ())
    }

    fn advance(&self, _parent: &(), child: &State, _m: Move, stats: &mut SearchStats) -> (u8, ()) {
        let _ = stats;
        (self.h(child), ())
    }
}

// ---- incremental evaluator (fast path) --------------------------------------

/// Physical-line demand LUT. `demand = n − LIS(goal-value sequence)` depends only
/// on the line's residency pattern, so it precomputes to a table instead of an
/// O(n²) LIS per node. Key: `W` cells × 3 bits — cell code `0` = blank/non-resident
/// of this line, `1..=W` = a resident carrying goal-value `code−1` (goal-column for
/// a row line, goal-row for a column line). 32 KiB, built once per [`Cwd`]. Removes
/// the top cWD hotspot (`demand_*_line` ≈ 24% of cWD in the `cwd-zpdb8-lazy` sample
/// profile). Node-identical to the LIS reference (`demand_*_line_ref`).
pub(crate) const DEMAND_LUT_LEN: usize = 1 << (3 * W); // W cells × 3 bits

pub(crate) fn build_demand_lut() -> Box<[u8; DEMAND_LUT_LEN]> {
    let mut lut = Box::new([0u8; DEMAND_LUT_LEN]);
    for (key, slot) in lut.iter_mut().enumerate() {
        let mut seq = [0u8; W];
        let mut n = 0usize;
        for p in 0..W {
            let code = (key >> (3 * p)) & 0x7;
            if (1..=W).contains(&code) {
                seq[n] = (code - 1) as u8;
                n += 1;
            }
        }
        *slot = (n - lis_strict(&seq[..n])) as u8;
    }
    lut
}

/// Pack physical row `g` of `s` into a demand-LUT key (residents' goal-columns).
#[inline]
fn demand_row_key(s: &State, g: usize) -> usize {
    let mut key = 0usize;
    for c in 0..W {
        let tile = s.0[g * W + c];
        if tile != 0 {
            let gp = (tile - 1) as usize;
            if gp / W == g {
                key |= (gp % W + 1) << (3 * c);
            }
        }
    }
    key
}

/// Pack physical column `g` of `s` into a demand-LUT key (residents' goal-rows).
#[inline]
fn demand_col_key(s: &State, g: usize) -> usize {
    let mut key = 0usize;
    for r in 0..W {
        let tile = s.0[r * W + g];
        if tile != 0 {
            let gp = (tile - 1) as usize;
            if gp % W == g {
                key |= (gp / W + 1) << (3 * r);
            }
        }
    }
    key
}

/// Forced-escape demand of physical row `g` (LUT fast path; see [`build_demand_lut`]).
#[inline]
pub(crate) fn demand_row_line(lut: &[u8; DEMAND_LUT_LEN], s: &State, g: usize) -> u8 {
    let d = lut[demand_row_key(s, g)];
    // Gated so the `_ref` reference is absent in release (`debug_assert_eq!` still
    // *compiles* its args under `if false`, which would need `_ref` to exist).
    #[cfg(debug_assertions)]
    debug_assert_eq!(
        d,
        demand_row_line_ref(s, g),
        "demand-row LUT mismatch (g={g})"
    );
    d
}

/// Forced-escape demand of physical column `g` (symmetric: goal-row order).
#[inline]
pub(crate) fn demand_col_line(lut: &[u8; DEMAND_LUT_LEN], s: &State, g: usize) -> u8 {
    let d = lut[demand_col_key(s, g)];
    #[cfg(debug_assertions)]
    debug_assert_eq!(
        d,
        demand_col_line_ref(s, g),
        "demand-col LUT mismatch (g={g})"
    );
    d
}

/// Reference (LIS) demand — residents (goal-row-`g` tiles in row `g`) minus the LIS
/// of their goal columns (physical-column order). The LUT is built from and checked
/// against this; retained for `debug_assert` and tests.
#[cfg(any(test, debug_assertions))]
fn demand_row_line_ref(s: &State, g: usize) -> u8 {
    let mut seq = [0u8; W];
    let mut n = 0usize;
    for c in 0..W {
        let tile = s.0[g * W + c];
        if tile == 0 {
            continue;
        }
        let gp = (tile - 1) as usize;
        if gp / W == g {
            seq[n] = (gp % W) as u8;
            n += 1;
        }
    }
    (n - lis_strict(&seq[..n])) as u8
}

/// Reference (LIS) demand for a physical column (symmetric: goal-row order).
#[cfg(any(test, debug_assertions))]
fn demand_col_line_ref(s: &State, g: usize) -> u8 {
    let mut seq = [0u8; W];
    let mut n = 0usize;
    for r in 0..W {
        let tile = s.0[r * W + g];
        if tile == 0 {
            continue;
        }
        let gp = (tile - 1) as usize;
        if gp % W == g {
            seq[n] = (gp / W) as u8;
            n += 1;
        }
    }
    (n - lis_strict(&seq[..n])) as u8
}

impl Cwd {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::WalkingDistanceHeuristic;

    /// The demand LUT + key-packing must reproduce the LIS reference on every line
    /// of arbitrary boards. Table-free, so it runs fast (no `#[ignore]`).
    #[test]
    fn demand_lut_matches_lis_reference() {
        let lut = build_demand_lut();
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..2000 {
            // Fisher–Yates shuffle of 0..=24 via a xorshift stream.
            let mut a = [0u8; 25];
            for (i, cell) in a.iter_mut().enumerate() {
                *cell = i as u8;
            }
            for i in (1..25usize).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                a.swap(i, (x as usize) % (i + 1));
            }
            let s = State(a);
            for g in 0..W {
                assert_eq!(
                    demand_row_line(&lut, &s, g),
                    demand_row_line_ref(&s, g),
                    "row demand mismatch at line {g}"
                );
                assert_eq!(
                    demand_col_line(&lut, &s, g),
                    demand_col_line_ref(&s, g),
                    "col demand mismatch at line {g}"
                );
            }
        }
    }

    /// The 180° rotation R. cWD(R) must be 144 (= WD 140 + 4).
    fn r_board() -> State {
        let mut a = [0u8; 25];
        for i in 1..25u8 {
            a[i as usize] = 25 - i;
        }
        State(a)
    }

    fn table_or_skip() -> Option<Cwd> {
        let p = Path::new("data/wd24.bin");
        if p.exists() {
            Some(Cwd::with_table_path(p))
        } else {
            eprintln!("data/wd24.bin absent — skipping cWD table test");
            None
        }
    }

    #[ignore = "loads the 563 MB 24-puzzle WD table (~15-25s); run with --ignored. cWD is \
                24-puzzle-only; the underlying WD algorithm is smoke-tested fast on the \
                15-puzzle (puzzle15::search::walking_distance)"]
    #[test]
    fn cwd_of_r_is_144() {
        let Some(cwd) = table_or_skip() else { return };
        assert_eq!(cwd.h(&r_board()), 144);
    }

    #[ignore = "loads the 563 MB 24-puzzle WD table + expands a BFS ball (~30s); run with --ignored. \
                cWD ≥ WD / admissibility is 24-puzzle-only; WD admissibility (WD ≥ Manhattan, admissible \
                on a BFS ball) is smoke-tested fast on the 15-puzzle (puzzle15::search::walking_distance)"]
    #[test]
    fn cwd_ge_wd_and_admissible_on_bfs_ball() {
        let Some(cwd) = table_or_skip() else { return };
        WalkingDistanceHeuristic::warm_up();
        let wd = WalkingDistanceHeuristic;
        // Exact ground truth to depth 11 (a few million states).
        let truth = crate::puzzle24::search::tests_util::bfs_distances(11);
        let mut scratch = CwdScratch::new();
        for (state, &d) in &truth {
            let s = State(*state);
            let h = cwd.eval(&s, &mut scratch);
            assert!(h <= d, "cWD {h} over-estimates true {d} at {state:?}");
            assert!(
                h >= wd.h(&s),
                "cWD {} below WD {} at {:?}",
                h,
                wd.h(&s),
                state
            );
        }
    }

    // ------------------------- probe-length instrumentation --------------------

    /// Candidate hash functions for the merged-table probe.
    ///
    /// `fmix64` is what ships. The others exist to answer one question: the
    /// rejection recorded at `walking_distance.rs:36` says a *single multiply*
    /// (FxHash) "clusters them in hashbrown's low-bit buckets and probes blow
    /// up". That failure is directional — a multiply propagates entropy
    /// **upward**, so low input bits reach high product bits but never the
    /// reverse, and SwissTable picks its bucket index from the **low** bits.
    /// Fibonacci-style hashing classically reads the *high* bits instead, which
    /// is exactly where a bare multiply mixes well; as a `Hasher` we cannot
    /// choose the bits hashbrown reads, so the variants below fold the high half
    /// down by hand. `mul_only` is the control that should reproduce the
    /// documented blow-up.
    #[cfg(feature = "cwd-table-tests")]
    type HashFn = fn(u64) -> u64;

    #[cfg(feature = "cwd-table-tests")]
    const HASHES: &[(&str, HashFn)] = &[
        ("fmix64 (current)", |i| {
            let mut x = i;
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            x ^= x >> 33;
            x
        }),
        ("mul_only (control)", |i| {
            i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        }),
        ("fib + xorshift32", |i| {
            let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 32;
            x
        }),
        ("fib + rot32", |i| {
            i.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(32)
        }),
        ("widemul fold", |i| {
            let p = (i as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15_u128);
            (p >> 64) as u64 ^ (p as u64)
        }),
    ];

    /// hashbrown's SIMD group width.
    #[cfg(feature = "cwd-table-tests")]
    const GROUP: usize = 16;

    /// [`place`] for an arbitrary (possibly non-power-of-two) bucket count.
    #[cfg(feature = "cwd-table-tests")]
    fn place_mod(slots: &mut [bool], buckets: usize, h1: usize) -> usize {
        let mut pos = h1 % buckets;
        let mut stride = 0usize;
        let mut hops = 0usize;
        loop {
            for i in 0..GROUP {
                let idx = (pos + i) % buckets;
                if !slots[idx] {
                    slots[idx] = true;
                    return hops;
                }
            }
            stride += GROUP;
            pos = (pos + stride) % buckets;
            hops += 1;
            assert!(hops < 1 << 22, "probe sequence failed to terminate");
        }
    }

    /// Insert `h1` into a SwissTable-shaped slot array using hashbrown's
    /// triangular probe sequence, returning the number of **group hops** past
    /// the home group. 0 hops means the key landed in the group its hash chose.
    #[cfg(feature = "cwd-table-tests")]
    fn place(slots: &mut [bool], mask: usize, h1: usize) -> usize {
        let mut pos = h1 & mask;
        let mut stride = 0usize;
        let mut hops = 0usize;
        loop {
            for i in 0..GROUP {
                let idx = (pos + i) & mask;
                if !slots[idx] {
                    slots[idx] = true;
                    return hops;
                }
            }
            stride += GROUP;
            pos = (pos + stride) & mask;
            hops += 1;
            assert!(hops < 1 << 20, "probe sequence failed to terminate");
        }
    }

    /// **Sizes the "shrink the table by raising the load factor" idea.**
    ///
    /// The merged map holds 65,650,495 entries in 134,217,728 buckets — 48.9%
    /// load — because hashbrown rounds to a power of two and 65.65M x 8/7 just
    /// crosses 2^26. Over half the allocation is empty slots, and the depth
    /// slowdown is footprint-driven (TLB misses +52.7% per node at exhaust-146,
    /// `puzzle24::pmu`), so a denser table is the obvious lever.
    ///
    /// The catch is that density costs probe length. This models hashbrown's
    /// placement over the real key set at several bucket counts, so the
    /// trade-off is measured rather than assumed: footprint saved on the left,
    /// extra group hops per lookup on the right.
    #[cfg(feature = "cwd-table-tests")]
    #[test]
    fn probe_length_vs_load_factor() {
        let Some(cwd) = shared_merged_cwd() else {
            return;
        };
        let Some(merged) = cwd.merged_table() else {
            eprintln!("no merged cWD table — skipping");
            return;
        };
        let mut keys: Vec<u64> = merged.keys().copied().collect();
        keys.sort_unstable();
        let n = keys.len();
        // 32 B per (u64, CwdCell) entry + 1 control byte per bucket.
        let bytes = |b: usize| (b * 33) as f64 / 1e9;
        let cur = (n * 8 / 7 + 1).next_power_of_two();
        println!("\n{n} keys; current {cur} buckets = {:.2} GB\n", bytes(cur));
        println!(
            "{:>12} {:>7} {:>9} {:>7} {:>7} {:>9} {:>9}",
            "buckets", "load", "mean hops", "p99", "max", "size GB", "vs now"
        );
        for &b in &[
            cur,
            100_000_000,
            90_000_000,
            80_000_000,
            75_030_000,
            70_000_000,
        ] {
            if b < n {
                continue;
            }
            let mut slots = vec![false; b];
            let mask_pow2 = b.is_power_of_two();
            let mut hops = vec![0u32; n];
            for (i, &k) in keys.iter().enumerate() {
                let h = {
                    let mut x = k;
                    x ^= x >> 33;
                    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
                    x ^= x >> 33;
                    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
                    x ^= x >> 33;
                    x
                };
                // Power-of-two tables mask; a non-power-of-two table needs a
                // multiply-shift (fastrange) instead, which is what a custom
                // table would use.
                let home = if mask_pow2 {
                    (h as usize) & (b - 1)
                } else {
                    (((h as u128) * (b as u128)) >> 64) as usize
                };
                hops[i] = place_mod(&mut slots, b, home) as u32;
            }
            let total: u64 = hops.iter().map(|&h| h as u64).sum();
            let mut srt = hops.clone();
            srt.sort_unstable();
            println!(
                "{b:>12} {:>6.1}% {:>9.4} {:>7} {:>7} {:>9.2} {:>8.0}%",
                n as f64 / b as f64 * 100.0,
                total as f64 / n as f64,
                srt[(n as f64 * 0.99) as usize],
                srt.last().unwrap(),
                bytes(b),
                bytes(b) / bytes(cur) * 100.0
            );
        }
        println!();
    }

    /// **Decides whether a cheaper hash is even a candidate.** Models hashbrown's
    /// placement over the *real* merged-table key set and reports the probe-length
    /// distribution per hash, plus the hash's own cost.
    ///
    /// This is deliberately a distribution measurement, not a timing A/B. A timing
    /// A/B conflates two opposite effects — the hash got cheaper, the probes may
    /// have got longer — so a null result would not say which, and the failure
    /// mode here is a silent throughput cliff rather than anything that breaks.
    /// If a cheap hash shows displacement comparable to `fmix64` here, a timing
    /// A/B becomes trustworthy; if it blows up, the question is closed cheaply.
    #[cfg(feature = "cwd-table-tests")]
    #[test]
    fn probe_length_distribution_by_hash() {
        let Some(cwd) = shared_merged_cwd() else {
            return;
        };
        let Some(merged) = cwd.merged_table() else {
            eprintln!("no merged cWD table — skipping");
            return;
        };

        let mut keys: Vec<u64> = merged.keys().copied().collect();
        keys.sort_unstable();
        let n = keys.len();

        // hashbrown grows so that len <= 7/8 * buckets.
        let buckets = (n * 8 / 7 + 1).next_power_of_two();
        let mask = buckets - 1;
        println!(
            "\nmerged table: {n} keys, modelled at {buckets} buckets (load {:.3})",
            n as f64 / buckets as f64
        );
        println!(
            "{:<20} {:>10} {:>9} {:>7} {:>7} {:>9} {:>12}",
            "hash", "mean hops", "p99", "max", "%>0", "ctrl max/mean", "ns/hash"
        );

        for (name, f) in HASHES {
            let mut slots = vec![false; buckets];
            let mut hops = vec![0u32; n];
            let mut ctrl = [0u64; 128];
            for (i, &k) in keys.iter().enumerate() {
                let h = f(k);
                ctrl[(h >> 57) as usize & 127] += 1;
                hops[i] = place(&mut slots, mask, h as usize) as u32;
            }
            let total: u64 = hops.iter().map(|&h| h as u64).sum();
            let mean = total as f64 / n as f64;
            let mut sorted = hops.clone();
            sorted.sort_unstable();
            let p99 = sorted[(n as f64 * 0.99) as usize];
            let max = *sorted.last().unwrap();
            let nonzero = hops.iter().filter(|&&h| h > 0).count() as f64 / n as f64 * 100.0;
            let cmax = *ctrl.iter().max().unwrap() as f64;
            let cmean = n as f64 / 128.0;

            // Hash cost, measured on the same keys so the loads are warm.
            let t = std::time::Instant::now();
            let mut acc = 0u64;
            for &k in keys.iter() {
                acc = acc.wrapping_add(f(k));
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            std::hint::black_box(acc);

            println!(
                "{name:<20} {mean:>10.4} {p99:>9} {max:>7} {nonzero:>6.2}% {:>9.3} {ns:>12.2}",
                cmax / cmean
            );
        }
        println!();
    }
}
