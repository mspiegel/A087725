//! Positional-index layout for the k6 zPDB family.
//!
//! The ranked layout pays a `ZpdbLayout::rank` walk per consult — measured at
//! **65% of the k6 tier's cost** (23.6% of whole-engine samples, sequential
//! exhaust-146 profile), with the 8-`ProjectedState` slot copy another 16%.
//! This module re-indexes the *same* 1-bit differential codec by tile
//! positions, so the address changes but the values do not:
//!
//! ```text
//! P = blank · STRIDE_BITS + Σ_j p_j · 25^(5-j)
//! ```
//!
//! where `p_j` is the cell of the pattern's `j`-th tile in ascending tile
//! order. `STRIDE_BITS` is the 25⁶ tile space rounded up to a byte boundary so
//! each blank's slab starts byte-aligned. 763 MB per pattern, **3.05 GB** for
//! the family — against 24.4 GB had the values been stored as `u8` rather than
//! as differential bits, which is what makes this layout affordable at all.
//!
//! What it buys: a move changes exactly one tile's cell, so the tile part of
//! the index updates with a single multiply-add
//! (`idx += (dst − src) · COEFF[tile]`), the blank digit is applied at consult
//! time from the frame's blank, and the per-depth slot shrinks from eight
//! `ProjectedState`s to eight `u32` + eight `u8`. `rank` disappears entirely.
//!
//! What it risks: the map is 8× the compressed family's footprint, so consults
//! touch 186 K pages instead of ~5 K — TLB and page pressure replace ALU work.
//! Which side wins is an empirical question, and the reason the ranked tier is
//! kept intact for a canary-bracketed A/B.

use std::path::Path;

use crate::puzzle24::pdb::{Pattern, ZPatternDb};
use crate::puzzle24::state::{State, N_CELLS};

/// 25⁶ — the six-tile position space for one pattern.
pub const TILE_SPACE: u64 = 244_140_625;
/// `TILE_SPACE` rounded up to a multiple of 8, so per-blank slabs are
/// byte-aligned (costs 7 bits per slab).
pub const STRIDE_BITS: u64 = 244_140_632;
pub const STRIDE_BYTES: u64 = STRIDE_BITS / 8;
/// One pattern's map: 25 blank slabs.
pub const MAP_BYTES: usize = (25 * STRIDE_BYTES) as usize;

/// The four k6 patterns, in `pdb24_{a,b,c,d}.zbin` order.
pub const K6_NAMES: [&str; 4] = [
    "pdb24_a.zbin",
    "pdb24_b.zbin",
    "pdb24_c.zbin",
    "pdb24_d.zbin",
];

/// Radix coefficients by slot: `25^(5-j)`.
const COEFF_SLOT: [u64; 6] = [
    390_625 * 25, // 25^5
    390_625,      // 25^4
    15_625,
    625,
    25,
    1,
];

/// `coeff[tile]` for every tile of `pattern` (ascending tile order → slot).
fn coeffs_of(pattern: Pattern) -> [u32; N_CELLS] {
    let mut c = [0u32; N_CELLS];
    for (j, t) in pattern.iter().enumerate() {
        c[t as usize] = COEFF_SLOT[j] as u32;
    }
    c
}

/// Tile-part of the positional index for `board` under `pattern`.
fn tiles_index(board: &State, pattern: Pattern) -> u32 {
    let mut idx = 0u64;
    for (j, t) in pattern.iter().enumerate() {
        let p = board.0.iter().position(|&x| x == t).unwrap() as u64;
        idx += p * COEFF_SLOT[j];
    }
    idx as u32
}

// ------------------------------- the builder ---------------------------------

/// Dist-array stride: one blank slab of 25^6 tile placements (unlike the
/// output bitmap, the build array needs no byte alignment).
const DSTRIDE: u64 = TILE_SPACE;

/// 4-neighbourhood of every cell, `255`-terminated.
const NBRS: [[u8; 5]; N_CELLS] = {
    let mut n = [[255u8; 5]; N_CELLS];
    let mut c = 0;
    while c < N_CELLS {
        let (r, k) = (c / 5, c % 5);
        let mut w = 0;
        if r > 0 {
            n[c][w] = (c - 5) as u8;
            w += 1;
        }
        if r < 4 {
            n[c][w] = (c + 5) as u8;
            w += 1;
        }
        if k > 0 {
            n[c][w] = (c - 1) as u8;
            w += 1;
        }
        if k < 4 {
            n[c][w] = (c + 1) as u8;
        }
        c += 1;
    }
    n
};

/// Shared mutable byte array for the BFS distance table.
///
/// Threads race only on "write my layer's number where 0xFF stands"; within a
/// phase every writer stores the *same* value, and phases are separated by a
/// join, so the races are benign and no ordering is required.
struct DistBuf {
    cell: std::cell::UnsafeCell<Vec<u8>>,
}
// SAFETY: see `DistBuf` — same-value races inside a phase, joins between phases.
unsafe impl Sync for DistBuf {}

impl DistBuf {
    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn get(&self) -> &mut Vec<u8> {
        // SAFETY: callers are the phase workers described above.
        unsafe { &mut *self.cell.get() }
    }
}

/// Pack a state as `blank | p_0..p_5`, 5 bits each (35 bits total).
#[inline]
fn pack_state(blank: u8, p: &[u8; 6]) -> u64 {
    let mut v = (blank as u64) << 30;
    for (j, &x) in p.iter().enumerate() {
        v |= (x as u64) << (5 * j);
    }
    v
}

#[inline]
fn unpack_state(v: u64) -> (u8, [u8; 6]) {
    let blank = ((v >> 30) & 31) as u8;
    let mut p = [0u8; 6];
    for (j, x) in p.iter_mut().enumerate() {
        *x = ((v >> (5 * j)) & 31) as u8;
    }
    (blank, p)
}

/// Dist-array index of a packed state.
#[inline]
fn dist_index(blank: u8, p: &[u8; 6]) -> u64 {
    let mut idx = blank as u64 * DSTRIDE;
    for (j, &x) in p.iter().enumerate() {
        idx += x as u64 * COEFF_SLOT[j];
    }
    idx
}

/// Build one pattern's positional map by BFS **from scratch** over the
/// zero-aware abstract space: a state is (blank cell, the six pattern tiles'
/// cells); moving a pattern tile costs 1, moving the blank past an anonymous
/// tile costs 0. Distances are exact for that abstraction, and the stored bit
/// is `(dist >> 1) & 1` — the same differential encoding the ranked tables
/// use, so [`K6PosCtx::probe`] reconstructs absolute values from a parent's
/// `h` across any cost-1 edge.
///
/// Layer structure: each layer is closed under cost-0 blank walks before its
/// cost-1 edges are expanded, which is what makes a plain layered BFS correct
/// in the presence of zero-cost edges.
pub fn build_from_scratch(pattern: Pattern, threads: usize) -> Vec<u8> {
    let tiles: Vec<u8> = pattern.iter().collect();
    assert_eq!(
        tiles.len(),
        6,
        "k6 positional layout expects 6-tile patterns"
    );
    let goal: [u8; 6] = std::array::from_fn(|j| tiles[j] - 1);

    let dist = DistBuf {
        cell: std::cell::UnsafeCell::new(vec![0xFFu8; (25 * DSTRIDE) as usize]),
    };

    // Seed: the single goal state — pattern tiles home AND the blank at its
    // own goal cell (24). Seeding the blank in every free cell would hand
    // distance 0 to configurations whose blank sits in a *different* region
    // than the goal's, collapsing the region distinction the zero-aware
    // abstraction is built on (`zbuild::build_zpdb` documents the same trap).
    // Pattern d's goal isolates cell 24 between tiles 20 and 24, so this is
    // not hypothetical: it is where the first cross-check disagreed.
    let mut cur: Vec<u64> = Vec::new();
    const BLANK_GOAL: u8 = 24;
    assert!(!goal.contains(&BLANK_GOAL), "blank goal cell must be free");
    dist.get()[dist_index(BLANK_GOAL, &goal) as usize] = 0;
    cur.push(pack_state(BLANK_GOAL, &goal));

    let mut d: u8 = 0;
    loop {
        // ---- close this layer under cost-0 blank moves ----
        let mut wave = cur.clone();
        while !wave.is_empty() {
            let next_wave: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
            let idx = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut local = Vec::new();
                        loop {
                            let i = idx.fetch_add(4096, std::sync::atomic::Ordering::Relaxed);
                            if i >= wave.len() {
                                break;
                            }
                            for &st in &wave[i..(i + 4096).min(wave.len())] {
                                let (b, p) = unpack_state(st);
                                for &c in NBRS[b as usize].iter().take_while(|&&x| x != 255) {
                                    if p.contains(&c) {
                                        continue; // cost-1 edge, handled below
                                    }
                                    let ni = dist_index(c, &p) as usize;
                                    let slot = &mut dist.get()[ni];
                                    if *slot == 0xFF {
                                        *slot = d;
                                        local.push(pack_state(c, &p));
                                    }
                                }
                            }
                        }
                        next_wave.lock().unwrap().extend_from_slice(&local);
                    });
                }
            });
            let nw = next_wave.into_inner().unwrap();
            cur.extend_from_slice(&nw);
            wave = nw;
        }

        // ---- expand cost-1 edges into the next layer ----
        let next: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
        let idx = std::sync::atomic::AtomicUsize::new(0);
        let nd = d + 1;
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = idx.fetch_add(4096, std::sync::atomic::Ordering::Relaxed);
                        if i >= cur.len() {
                            break;
                        }
                        for &st in &cur[i..(i + 4096).min(cur.len())] {
                            let (b, p) = unpack_state(st);
                            for &c in NBRS[b as usize].iter().take_while(|&&x| x != 255) {
                                let Some(j) = p.iter().position(|&x| x == c) else {
                                    continue; // cost-0, already closed
                                };
                                let mut np = p;
                                np[j] = b;
                                let ni = dist_index(c, &np) as usize;
                                let slot = &mut dist.get()[ni];
                                if *slot == 0xFF {
                                    *slot = nd;
                                    local.push(pack_state(c, &np));
                                }
                            }
                        }
                    }
                    next.lock().unwrap().extend_from_slice(&local);
                });
            }
        });
        let nx = next.into_inner().unwrap();
        if nx.is_empty() {
            break;
        }
        cur = nx;
        d = nd;
    }

    // ---- extract bit 1 into the byte-aligned output bitmap ----
    let dv = dist.get();
    let mut out = vec![0u8; MAP_BYTES];
    for b in 0..25u64 {
        let base = (b * DSTRIDE) as usize;
        let obase = (b * STRIDE_BYTES) as usize;
        for t in 0..DSTRIDE as usize {
            let h = dv[base + t];
            if h != 0xFF && (h >> 1) & 1 == 1 {
                out[obase + (t >> 3)] |= 1 << (t & 7);
            }
        }
    }
    eprintln!("    depth {d}");
    out
}

// ------------------------------- the runtime ---------------------------------

/// The k6 family in positional layout: the ranked zPDBs (for absolute seed
/// values, which are rare) plus one mmap'd positional bitmap per pattern.
pub struct K6PosCtx {
    dbs: [ZPatternDb; 4],
    /// Positional maps; empty when running map-free (ranked miss path only).
    maps: Vec<memmap2::Mmap>,
    /// `group_of[t]` = which pattern holds tile `t`.
    pub group_of: [u8; N_CELLS],
    /// `coeff[t]` = the tile's radix weight inside its own pattern's index.
    pub coeff: [u32; N_CELLS],
    /// Shared front cache; when present the per-worker caches go unused.
    shared: Option<crate::puzzle24::search::flat::K6SharedCache>,
}

impl K6PosCtx {
    /// Load the four zPDBs from `dir` and their positional maps
    /// (`k6pos_{a..d}.bin`, built by [`build_positional`]).
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_opt(dir, true)
    }

    /// `want_maps = false` skips the 2.9 GB positional maps entirely: the tier
    /// then serves cache misses from the ranked zPDBs.
    pub fn load_opt(dir: &Path, want_maps: bool) -> Result<Self, String> {
        let mut dbs = Vec::with_capacity(4);
        let mut maps = Vec::with_capacity(4);
        let mut group_of = [0u8; N_CELLS];
        let mut coeff = [0u32; N_CELLS];
        let mut cover = 0u32;
        for (i, name) in K6_NAMES.iter().enumerate() {
            let db =
                ZPatternDb::load_mmap(&dir.join(name)).map_err(|e| format!("{name}: {e:?}"))?;
            if cover & db.pattern().0 != 0 {
                return Err("k6 patterns overlap".into());
            }
            cover |= db.pattern().0;
            for t in db.pattern().iter() {
                group_of[t as usize] = i as u8;
            }
            let c = coeffs_of(db.pattern());
            for t in db.pattern().iter() {
                coeff[t as usize] = c[t as usize];
            }
            if want_maps {
                let pos_name = format!("k6pos_{}.bin", (b'a' + i as u8) as char);
                let f = std::fs::File::open(dir.join(&pos_name)).map_err(|e| {
                    format!("{pos_name}: {e} (build it with the k6pos builder test)")
                })?;
                // SAFETY: write-once-then-immutable build artifact.
                let m =
                    unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("{pos_name}: {e}"))?;
                if m.len() != MAP_BYTES {
                    return Err(format!("{pos_name}: wrong size {}", m.len()));
                }
                maps.push(m);
            }
            dbs.push(db);
        }
        if cover != 0x01FF_FFFE {
            return Err("k6 patterns must cover tiles 1..=24".into());
        }
        // Pre-touch (16 KiB pages): keep demand-paging faults out of search.
        let mut sum = 0u64;
        for m in &maps {
            for off in (0..m.len()).step_by(16384) {
                sum = sum.wrapping_add(m[off] as u64);
            }
        }
        std::hint::black_box(sum);
        Ok(K6PosCtx {
            dbs: dbs
                .try_into()
                .map_err(|_| "expected four dbs".to_string())?,
            maps,
            group_of,
            coeff,
            shared: None,
        })
    }

    /// Absolute values for a seed board: `(tile indices, h)` per group, and the
    /// σ-max sum. Cold — seeds happen once per subtree.
    pub fn seed(&self, board: &State) -> ([[u32; 4]; 2], [[u8; 4]; 2], u8) {
        let rs = crate::puzzle24::symmetry::reflect(board);
        let (mut idx, mut h) = ([[0u32; 4]; 2], [[0u8; 4]; 2]);
        let (mut s0, mut s1) = (0u16, 0u16);
        for (i, db) in self.dbs.iter().enumerate() {
            idx[0][i] = tiles_index(board, db.pattern());
            idx[1][i] = tiles_index(&rs, db.pattern());
            h[0][i] = db.cold_lookup(board);
            h[1][i] = db.cold_lookup(&rs);
            s0 += h[0][i] as u16;
            s1 += h[1][i] as u16;
        }
        (idx, h, s0.max(s1).min(255) as u8)
    }

    /// One positional probe: the codec bit at `(blank, tiles_idx)` combined
    /// with the parent's `old_h`. Same reconstruction as
    /// [`crate::puzzle24::pdb::zcodec::diff_lookup`], different address.
    #[inline(always)]
    pub fn probe(&self, group: usize, tiles_idx: u32, blank: u8, old_h: u8) -> u8 {
        let bit = blank as u64 * STRIDE_BITS + tiles_idx as u64;
        let byte = self.maps[group][(bit >> 3) as usize];
        let entry = (((byte >> (bit & 7)) & 1) << 1) as u32;
        let oh = old_h as u32;
        (oh + 1 - ((entry ^ oh ^ (oh << 1)) & 2)) as u8
    }

    /// Miss path without the positional map: rebuild the group's projection
    /// from `board` and read the ranked zPDB (88 MB — a 33x smaller working
    /// set than the positional maps, which is what k6's miss rate needs).
    #[inline]
    pub fn probe_ranked(&self, group: usize, board: &State, old_h: u8) -> u8 {
        let db = &self.dbs[group];
        let proj = crate::puzzle24::pdb::ProjectedState::from_state(board, db.pattern());
        let idx = db.layout().rank(&proj, db.pattern());
        db.diff_lookup(idx, old_h)
    }

    /// Whether the positional maps were loaded (optional once the ranked miss
    /// path exists).
    #[inline]
    pub fn has_maps(&self) -> bool {
        !self.maps.is_empty()
    }

    /// The shared front cache, when the tier was built with one.
    #[inline]
    pub fn shared_cache(&self) -> Option<&crate::puzzle24::search::flat::K6SharedCache> {
        self.shared.as_ref()
    }

    /// Attach a shared lock-free cache in place of the per-worker caches.
    pub fn with_shared_cache(mut self) -> Self {
        self.shared = Some(crate::puzzle24::search::flat::K6SharedCache::new());
        self
    }

    /// Cold σ-max sum, for the drift assert.
    pub fn cold_h(&self, board: &State) -> u8 {
        let rs = crate::puzzle24::symmetry::reflect(board);
        let (mut s0, mut s1) = (0u16, 0u16);
        for db in &self.dbs {
            s0 += db.cold_lookup(board) as u16;
            s1 += db.cold_lookup(&rs) as u16;
        }
        s0.max(s1).min(255) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the four positional maps from scratch (BFS per pattern).
    ///
    ///   cargo test --release build_k6_positional_maps -- --ignored --nocapture
    #[test]
    #[ignore = "BFS over 2.42 G abstract states per pattern; writes data/k6pos_{a..d}.bin (763 MB each)"]
    fn build_k6_positional_maps() {
        for (i, name) in K6_NAMES.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let db = ZPatternDb::load_mmap(Path::new("data").join(name).as_path())
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let map = build_from_scratch(db.pattern(), 8);
            let out = format!("data/k6pos_{}.bin", (b'a' + i as u8) as char);
            std::fs::write(&out, &map).expect("write");
            eprintln!("{out}: {} bytes in {:.0?}", map.len(), t0.elapsed());
        }
    }

    /// Positional probes must reproduce the ranked path's values exactly:
    /// for every legal move of a sampled board, the child's group value read
    /// through the positional map (from the parent's absolute `h`) equals the
    /// child's own cold lookup.
    ///
    ///   cargo test --release k6pos_matches_ranked -- --ignored --nocapture
    #[test]
    #[ignore = "needs data/k6pos_{a..d}.bin and the k6 zPDBs"]
    fn k6pos_matches_ranked() {
        use crate::puzzle24::search::tests_util::bfs_distances;
        let ctx = K6PosCtx::load(Path::new("data")).expect("k6pos");
        let mut checked = 0u64;
        for (cells, _) in bfs_distances(9).iter().take(3000) {
            let b = &State(*cells);
            let (_, ph, _) = ctx.seed(b);
            for m in b.legal_moves().iter() {
                let child = b.apply(m);
                let (cidx, ch, _) = ctx.seed(&child);
                // Only the moved tile's group sees a cost-1 edge; the other
                // three are cost-0 (h unchanged), where the differential
                // reconstruction does not apply and the engine never probes.
                let moved = b.0[child.blank_pos() as usize];
                let g = ctx.group_of[moved as usize] as usize;
                let got = ctx.probe(g, cidx[0][g], child.blank_pos(), ph[0][g]);
                assert_eq!(got, ch[0][g], "positional != ranked, group {g}");
                for other in 0..4 {
                    if other != g {
                        assert_eq!(ch[0][other], ph[0][other], "cost-0 group changed");
                    }
                }
                checked += 1;
            }
        }
        eprintln!("k6pos: {checked} probes match the ranked path");
    }
}
