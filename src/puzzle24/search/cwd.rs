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
use super::idastar::{IncHeuristic, IncHeuristicMut};
use super::walking_distance::{
    build_full_table, load_dist_table, WdBuild, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use super::SearchStats;
use crate::puzzle24::state::{Move, State, W};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

type Matrix = [[u8; W]; W];

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

/// Surcharge (in moves) for one axis: single-line-max — the strongest single-line
/// escape bound over the demanded lines. `Δ = 2 · nibble`.
#[inline]
fn surcharge_from_curves(curves: &[u16; W], dem: &[u8; W]) -> u8 {
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

/// Load the surcharge overlay from `data/cwd_single.bin`.
pub fn load_cwd_overlay(path: &Path) -> std::io::Result<CwdOverlay> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"CWDS" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad cwd_single magic"));
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

fn pack(m: &Matrix, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

fn unpack(key: u64) -> (Matrix, u8) {
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
fn goal_key() -> u64 {
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
fn project(s: &State) -> (Matrix, u8, [u8; W], Matrix, u8, [u8; W]) {
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
                    merged.insert(k, CwdCell { wd, curves, nbr_wd: [255; 2 * W] });
                }
                c.merged = Some(merged);
                c.table = HashMap::with_capacity_and_hasher(0, WdBuild::default()); // free WD copy
            }
        }
        c
    }

    /// Load the WD table from `path`, falling back to a fresh BFS build. No merge.
    pub fn with_table_path(path: &Path) -> Self {
        let table = if path.exists() {
            load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).unwrap_or_else(|_| build_full_table())
        } else {
            build_full_table()
        };
        Cwd { table, goal: goal_key(), merged: None, neighbor_prune: false }
    }

    /// Build from an already-loaded WD table (shares the codec). No merge.
    pub fn from_table(table: WdTable) -> Self {
        Cwd { table, goal: goal_key(), merged: None, neighbor_prune: false }
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
        let Some(merged) = self.merged.as_mut() else { return };
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
        self.merged.is_some()
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
            return r.wd + surcharge_from_curves(&r.curves, &dem_row)
                + c.wd + surcharge_from_curves(&c.curves, &dem_col);
        }
        let wd_row = *self.table.get(&kr).expect("row reachable");
        let wd_col = *self.table.get(&kc).expect("col reachable");
        let c_row = cwd_axis(&self.table, &m_row, br, self.goal, &dem_row, scratch).unwrap_or(wd_row);
        let c_col = cwd_axis(&self.table, &m_col, bc, self.goal, &dem_col, scratch).unwrap_or(wd_col);
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
        #[cfg(feature = "verifier-stats")]
        {
            stats.wd_advances += 1;
        }
        let _ = stats;
        (self.h(s), ())
    }

    fn advance(&self, _parent: &(), child: &State, _m: Move, stats: &mut SearchStats) -> (u8, ()) {
        #[cfg(feature = "verifier-stats")]
        {
            stats.wd_advances += 1;
        }
        let _ = stats;
        (self.h(child), ())
    }
}

// ---- incremental evaluator (fast path) --------------------------------------

/// Forced-escape demand of physical row `g`: residents (goal-row-`g` tiles in
/// row `g`) minus the LIS of their goal columns (read in physical-column order).
#[inline]
fn demand_row_line(s: &State, g: usize) -> u8 {
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

/// Forced-escape demand of physical column `g` (symmetric: goal-row order).
#[inline]
fn demand_col_line(s: &State, g: usize) -> u8 {
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

/// Incrementally-maintained per-node state for the fast (merged-table) path.
#[derive(Clone, Copy)]
pub struct CwdState {
    m_row: Matrix,
    m_col: Matrix,
    br: u8,
    bc: u8,
    dem_row: [u8; W],
    dem_col: [u8; W],
    wd_row: u8,
    wd_col: u8,
    curves_row: [u16; W],
    curves_col: [u16; W],
    surch_row: u8,
    surch_col: u8,
    nbr_wd_row: [u8; 2 * W],
    nbr_wd_col: [u8; 2 * W],
}

impl CwdState {
    #[inline]
    fn h(&self) -> u8 {
        self.wd_row + self.surch_row + self.wd_col + self.surch_col
    }
}

/// One undo frame: the changed axis's matrix cell-pair, the pre-move blank, the
/// (at most one) changed demand, and the pre-move cached WD/surcharge/curves for
/// that axis. Reversing is O(1). `demline == 255` ⇒ no demand changed.
struct CwdUndo {
    vertical: bool,
    i: u8,
    j: u8,
    g: u8,
    br: u8,
    bc: u8,
    demline: u8,
    demold: u8,
    wd: u8,
    surch: u8,
    curves: [u16; W],
    nbr_wd: [u8; 2 * W],
}

/// The hot-path context: the incremental state, an undo stack, and a scratch for
/// the A\* fallback (only used when no merged table is loaded).
pub struct CwdMutCtx {
    state: CwdState,
    undo: Vec<CwdUndo>,
    scratch: CwdScratch,
}

impl Cwd {
    /// Build the full incremental state for the root (used by the fast path).
    fn root_state(&self, s: &State) -> CwdState {
        let (m_row, br, dem_row, m_col, bc, dem_col) = project(s);
        let mg = self.merged.as_ref().expect("root_state needs merged table");
        let r = mg.get(&pack(&m_row, br)).expect("row reachable");
        let c = mg.get(&pack(&m_col, bc)).expect("col reachable");
        CwdState {
            m_row,
            m_col,
            br,
            bc,
            dem_row,
            dem_col,
            wd_row: r.wd,
            wd_col: c.wd,
            curves_row: r.curves,
            curves_col: c.curves,
            surch_row: surcharge_from_curves(&r.curves, &dem_row),
            surch_col: surcharge_from_curves(&c.curves, &dem_col),
            nbr_wd_row: r.nbr_wd,
            nbr_wd_col: c.nbr_wd,
        }
    }
}

impl IncHeuristicMut for Cwd {
    type Ctx = CwdMutCtx;

    fn root(&self, s: &State) -> (u8, CwdMutCtx) {
        let mut scratch = CwdScratch::new();
        if self.merged.is_some() {
            let state = self.root_state(s);
            (state.h(), CwdMutCtx { state, undo: Vec::with_capacity(220), scratch })
        } else {
            let h = self.eval(s, &mut scratch);
            // dummy state (unused on the A* path)
            let state = CwdState {
                m_row: [[0; W]; W],
                m_col: [[0; W]; W],
                br: 0,
                bc: 0,
                dem_row: [0; W],
                dem_col: [0; W],
                wd_row: 0,
                wd_col: 0,
                curves_row: [0; W],
                curves_col: [0; W],
                surch_row: 0,
                surch_col: 0,
                nbr_wd_row: [255; 2 * W],
                nbr_wd_col: [255; 2 * W],
            };
            (h, CwdMutCtx { state, undo: Vec::new(), scratch })
        }
    }

    fn make(&self, ctx: &mut CwdMutCtx, child: &State, m: Move) -> u8 {
        let mg = match &self.merged {
            Some(mg) => mg,
            None => return self.eval(child, &mut ctx.scratch), // A* fallback
        };
        let st = &mut ctx.state;
        let parent_blank = st.br as usize * W + st.bc as usize;
        let delta: i32 = match m {
            Move::Up => -(W as i32),
            Move::Down => W as i32,
            Move::Left => -1,
            Move::Right => 1,
        };
        let from_cell = (parent_blank as i32 + delta) as usize;
        let to_cell = parent_blank;
        let tile = child.0[to_cell];
        let gp = (tile - 1) as usize;
        let gr = gp / W;
        let gc = gp % W;
        let rf = from_cell / W;
        let rt = to_cell / W;
        let cf = from_cell % W;
        let ct = to_cell % W;
        let mut undo = CwdUndo {
            vertical: rf != rt,
            i: 0,
            j: 0,
            g: 0,
            br: st.br,
            bc: st.bc,
            demline: 255,
            demold: 0,
            wd: 0,
            surch: 0,
            curves: [0; W],
            nbr_wd: [255; 2 * W],
        };
        if rf != rt {
            // vertical slide: only the row axis (contingency, blank-row) changes
            undo.wd = st.wd_row;
            undo.surch = st.surch_row;
            undo.curves = st.curves_row;
            undo.nbr_wd = st.nbr_wd_row;
            st.m_row[rf][gr] -= 1;
            st.m_row[rt][gr] += 1;
            undo.i = rf as u8;
            undo.j = rt as u8;
            undo.g = gr as u8;
            // the moved tile's own goal-line is the only demand that can change,
            // and only when it enters or leaves its own physical line
            if gr == rf || gr == rt {
                undo.demline = gr as u8;
                undo.demold = st.dem_row[gr];
                st.dem_row[gr] = demand_row_line(child, gr);
            }
            st.br = rf as u8;
            st.bc = cf as u8;
            let cell = mg.get(&pack(&st.m_row, st.br)).expect("row reachable");
            st.wd_row = cell.wd;
            st.curves_row = cell.curves;
            st.nbr_wd_row = cell.nbr_wd;
            st.surch_row = surcharge_from_curves(&st.curves_row, &st.dem_row);
        } else {
            // horizontal slide: only the column axis changes
            undo.wd = st.wd_col;
            undo.surch = st.surch_col;
            undo.curves = st.curves_col;
            undo.nbr_wd = st.nbr_wd_col;
            st.m_col[cf][gc] -= 1;
            st.m_col[ct][gc] += 1;
            undo.i = cf as u8;
            undo.j = ct as u8;
            undo.g = gc as u8;
            if gc == cf || gc == ct {
                undo.demline = gc as u8;
                undo.demold = st.dem_col[gc];
                st.dem_col[gc] = demand_col_line(child, gc);
            }
            st.br = rf as u8;
            st.bc = cf as u8;
            let cell = mg.get(&pack(&st.m_col, st.bc)).expect("col reachable");
            st.wd_col = cell.wd;
            st.curves_col = cell.curves;
            st.nbr_wd_col = cell.nbr_wd;
            st.surch_col = surcharge_from_curves(&st.curves_col, &st.dem_col);
        }
        ctx.undo.push(undo);
        st.h()
    }

    fn unmake(&self, ctx: &mut CwdMutCtx, _m: Move) {
        if self.merged.is_none() {
            return;
        }
        let u = ctx.undo.pop().expect("unmake without matching make");
        let st = &mut ctx.state;
        if u.vertical {
            st.m_row[u.i as usize][u.g as usize] += 1;
            st.m_row[u.j as usize][u.g as usize] -= 1;
            st.wd_row = u.wd;
            st.surch_row = u.surch;
            st.curves_row = u.curves;
            st.nbr_wd_row = u.nbr_wd;
            if u.demline != 255 {
                st.dem_row[u.demline as usize] = u.demold;
            }
        } else {
            st.m_col[u.i as usize][u.g as usize] += 1;
            st.m_col[u.j as usize][u.g as usize] -= 1;
            st.wd_col = u.wd;
            st.surch_col = u.surch;
            st.curves_col = u.curves;
            st.nbr_wd_col = u.nbr_wd;
            if u.demline != 255 {
                st.dem_col[u.demline as usize] = u.demold;
            }
        }
        st.br = u.br;
        st.bc = u.bc;
    }

    /// Lower-bound the child's `h` from the parent's cached `nbr_wd` — no probe.
    /// A vertical move changes only the row axis, so the child's WD is the cached
    /// row-neighbor WD; the column axis (WD + surcharge) carries over unchanged.
    /// The changed axis's surcharge (≥ 0) is dropped, keeping the bound admissible.
    #[inline]
    fn child_h_lb(&self, ctx: &CwdMutCtx, s: &State, blank: u8, m: Move) -> Option<u8> {
        if !self.neighbor_prune || self.merged.is_none() {
            return None;
        }
        let st = &ctx.state;
        let b = blank as usize;
        // (neighbor-WD array, slot direction, goal-line index, carried other-axis WD+surch)
        let (nbr, dir, g, other_wd, other_surch) = match m {
            Move::Up | Move::Down => {
                let from = if matches!(m, Move::Up) { b - W } else { b + W };
                let gr = (s.0[from] as usize - 1) / W; // moved tile's goal row
                let dir = if matches!(m, Move::Up) { 0 } else { 1 };
                (&st.nbr_wd_row, dir, gr, st.wd_col, st.surch_col)
            }
            Move::Left | Move::Right => {
                let from = if matches!(m, Move::Left) { b - 1 } else { b + 1 };
                let gc = (s.0[from] as usize - 1) % W; // moved tile's goal column
                let dir = if matches!(m, Move::Left) { 0 } else { 1 };
                (&st.nbr_wd_col, dir, gc, st.wd_row, st.surch_row)
            }
        };
        let w = nbr[dir * W + g];
        if w == 255 {
            return None; // no cached neighbor (unreachable for a legal move)
        }
        Some(w.saturating_add(other_wd).saturating_add(other_surch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::WalkingDistanceHeuristic;

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

    /// Needs both tables (WD + merged surcharge overlay) for the fast path.
    fn cwd_merged_or_skip() -> Option<Cwd> {
        if Path::new("data/wd24.bin").exists() && Path::new("data/cwd_single.bin").exists() {
            let c = Cwd::new();
            if c.has_overlay() {
                return Some(c);
            }
        }
        eprintln!("data tables absent — skipping incremental cWD test");
        None
    }

    /// The incremental evaluator must return exactly the same `h` as a fresh full
    /// evaluation, at every node of a random walk with backtracking (so both
    /// `make` and `unmake` are exercised).
    #[ignore = "loads the 563 MB + 1.1 GB 24-puzzle WD/cWD tables (~20-30s); run with --ignored. \
                Incremental-heuristic consistency (advance == fresh root) is smoke-tested fast on \
                the 15-puzzle (puzzle15::search::walking_distance::wd_inc_advance_matches_fresh_root_random_walk)"]
    #[test]
    fn incremental_matches_full_on_random_walk() {
        use crate::puzzle24::search::idastar::IncHeuristicMut;
        let Some(cwd) = cwd_merged_or_skip() else { return };
        let mut scratch = CwdScratch::new();
        let start = r_board();
        let (h0, mut ctx) = IncHeuristicMut::root(&cwd, &start);
        assert_eq!(h0, cwd.eval(&start, &mut scratch), "root h mismatch");
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut nextr = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = start;
        let mut path: Vec<Move> = Vec::new();
        for _ in 0..30000 {
            let descend = path.is_empty() || (path.len() < 60 && nextr() & 1 == 0);
            if descend {
                let moves: Vec<Move> = s.legal_moves().iter().collect();
                let m = moves[(nextr() as usize) % moves.len()];
                let child = s.apply(m);
                let h_inc = IncHeuristicMut::make(&cwd, &mut ctx, &child, m);
                let h_full = cwd.eval(&child, &mut scratch);
                assert_eq!(h_inc, h_full, "incremental {h_inc} != full {h_full} on move {m:?}");
                s = child;
                path.push(m);
            } else {
                let m = path.pop().unwrap();
                IncHeuristicMut::unmake(&cwd, &mut ctx, m);
                s = s.apply(m.inverse());
            }
        }
    }

    /// `child_h_lb` must (a) be available for every legal move on the merged
    /// path, (b) never exceed the child's true `h` (admissible), and (c) equal
    /// exactly `h_child − changed-axis surcharge` — i.e. its neighbor-WD term is
    /// the very value `make`'s probe produces. Exercised over a random walk.
    #[ignore = "loads the 563 MB + 1.1 GB 24-puzzle WD/cWD tables (~20-30s); run with --ignored. \
                cWD neighbor-prune is 24-puzzle-only; the underlying WD algorithm is smoke-tested \
                fast on the 15-puzzle (puzzle15::search::walking_distance)"]
    #[test]
    fn child_h_lb_matches_probe_on_random_walk() {
        use crate::puzzle24::search::idastar::IncHeuristicMut;
        let Some(cwd) = cwd_merged_or_skip() else { return };
        let cwd = cwd.with_neighbor_prune(true);
        let start = r_board();
        let (_h0, mut ctx) = IncHeuristicMut::root(&cwd, &start);
        let mut rng: u64 = 0x0bad_c0de_dead_beef;
        let mut nextr = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = start;
        let mut path: Vec<Move> = Vec::new();
        let mut checked = 0u64;
        for _ in 0..30000 {
            let descend = path.is_empty() || (path.len() < 60 && nextr() & 1 == 0);
            if descend {
                let moves: Vec<Move> = s.legal_moves().iter().collect();
                let m = moves[(nextr() as usize) % moves.len()];
                let blank = s.0.iter().position(|&x| x == 0).unwrap() as u8;
                // ctx tracks the parent `s` at this point.
                let lb = cwd
                    .child_h_lb(&ctx, &s, blank, m)
                    .expect("neighbor-WD lb available for a legal move on the merged path");
                let child = s.apply(m);
                let h_child = IncHeuristicMut::make(&cwd, &mut ctx, &child, m);
                // After `make`, the changed axis's surcharge is the only term the lb drops.
                let changed_surch = if matches!(m, Move::Up | Move::Down) {
                    ctx.state.surch_row
                } else {
                    ctx.state.surch_col
                };
                assert!(lb <= h_child, "lb {lb} > child h {h_child} (inadmissible)");
                assert_eq!(lb, h_child - changed_surch, "lb term != probed WD on move {m:?}");
                s = child;
                path.push(m);
                checked += 1;
            } else {
                let m = path.pop().unwrap();
                IncHeuristicMut::unmake(&cwd, &mut ctx, m);
                s = s.apply(m.inverse());
            }
        }
        assert!(checked > 1000, "walk too shallow to be meaningful ({checked})");
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
            assert!(h >= wd.h(&s), "cWD {} below WD {} at {:?}", h, wd.h(&s), state);
        }
    }
}
