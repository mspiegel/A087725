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

/// One merged contingency record: the WD value **and** the per-line surcharge
/// curves, so the hot path resolves both with a single probe per axis.
#[derive(Clone, Copy)]
pub struct CwdCell {
    pub wd: u8,
    pub curves: [u16; W],
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
                    merged.insert(k, CwdCell { wd, curves });
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
        Cwd { table, goal: goal_key(), merged: None }
    }

    /// Build from an already-loaded WD table (shares the codec). No merge.
    pub fn from_table(table: WdTable) -> Self {
        Cwd { table, goal: goal_key(), merged: None }
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

/// The hot path: the scratch buffer lives in the context and is reused across the
/// whole search, so per-node evaluation allocates nothing.
impl IncHeuristicMut for Cwd {
    type Ctx = CwdScratch;

    fn root(&self, s: &State) -> (u8, CwdScratch) {
        let mut scratch = CwdScratch::new();
        let h = self.eval(s, &mut scratch);
        (h, scratch)
    }

    fn make(&self, ctx: &mut CwdScratch, child: &State, _m: Move) -> u8 {
        self.eval(child, ctx)
    }

    fn unmake(&self, _ctx: &mut CwdScratch, _m: Move) {}
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

    #[test]
    fn cwd_of_r_is_144() {
        let Some(cwd) = table_or_skip() else { return };
        assert_eq!(cwd.h(&r_board()), 144);
    }

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
            assert!(h <= d, "cWD {} over-estimates true {} at {:?}", h, d, state);
            assert!(h >= wd.h(&s), "cWD {} below WD {} at {:?}", h, wd.h(&s), state);
        }
    }
}
