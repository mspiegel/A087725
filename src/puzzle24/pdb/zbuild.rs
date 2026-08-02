//! Zero-aware PDB construction via a region-aware `(m,p,r)` BFS (Slice B).
//!
//! Instead of the standard 0/1 BFS over `(pattern_config, blank_cell)` — whose
//! serial 0-cost closure dominates at m=5 (≈8 min, ~1 core for a 6-tile PDB) —
//! we BFS over `(pattern_config, blank_region)`. Collapsing the blank's free
//! wandering into its zero-tile region (`zpdb::regions`) removes the 0-cost
//! closure entirely: **every edge is a unit-cost pattern move**, so the graph is
//! the bipartite quotient the 1-bit codec needs, the state space shrinks ~13×
//! (181M vs 2.42B for k=6), and the BFS is fully parallelizable.
//!
//! The result `dist[(m,p,r)]` is the zero-aware value `ZPDB(C,r)`. The standard
//! additive PDB falls out as `additive(C) = min_r ZPDB(C,r)` — which we assert
//! byte-identical to the verified [`super::build::build`] output, the oracle for
//! this whole construction.

use super::pattern::{Pattern, ProjectedState, ANON};
use super::zpdb::ZpdbLayout;
use crate::puzzle24::state::{N_CELLS, W};

/// "Not yet visited" sentinel during construction.
pub const UNVISITED: u8 = u8::MAX;

/// Pattern-occupied-cell mask of a projection.
fn occupied_mask(proj: &ProjectedState) -> u32 {
    let mut m = 0u32;
    for (c, &v) in proj.cells.iter().enumerate() {
        if v != 0 && v != ANON {
            m |= 1u32 << c;
        }
    }
    m
}

/// 4-neighbours of cell `c`, written to `out`; returns the count.
#[inline]
fn neighbours(c: usize, out: &mut [usize; 4]) -> usize {
    let (r, col) = (c / W, c % W);
    let mut n = 0;
    if r > 0 {
        out[n] = c - W;
        n += 1;
    }
    if r < W - 1 {
        out[n] = c + W;
        n += 1;
    }
    if col > 0 {
        out[n] = c - 1;
        n += 1;
    }
    if col < W - 1 {
        out[n] = c + 1;
        n += 1;
    }
    n
}

/// Append every abstract unit-cost successor of `proj` to `out`: for each
/// pattern tile with a free neighbour cell in the blank's region, slide that
/// tile one cell into the region (the blank ends where the tile was).
///
/// Visible to the rest of the `pdb` module — the 1-bit codec's cold-lookup
/// descent (`zdb::ZPatternDb::cold_lookup`) walks the same abstract graph.
pub(crate) fn gen_moves(layout: &ZpdbLayout, proj: &ProjectedState, out: &mut Vec<ProjectedState>) {
    let occ = occupied_mask(proj);
    let (_, labels) = layout.regions_for(occ);
    let r = labels[proj.blank_pos() as usize];

    // Pattern tiles only (no blank), copied once as the successor base.
    let mut base = [ANON; N_CELLS];
    for (c, &v) in proj.cells.iter().enumerate() {
        if v != 0 && v != ANON {
            base[c] = v;
        }
    }

    let mut nb = [0usize; 4];
    for (tc, &v) in proj.cells.iter().enumerate() {
        if v == 0 || v == ANON {
            continue; // only pattern tiles move
        }
        let k = neighbours(tc, &mut nb);
        for &fc in &nb[..k] {
            // fc must be a free cell in the blank's region (so the blank can
            // travel to fc and swap with this tile).
            if labels[fc] == r {
                let mut nc = base;
                nc[tc] = 0; // blank lands where the tile was
                nc[fc] = v; // tile slides tc -> fc
                out.push(ProjectedState::from_projection(nc));
            }
        }
    }
}

/// Build the zero-aware PDB for `pattern`: `dist[(m,p,r)] = ZPDB(C,r)`, indexed
/// by [`ZpdbLayout::rank`], length [`ZpdbLayout::total`]. Returns the layout too
/// (it carries the cohort table needed to interpret the indices).
pub fn build_zpdb(pattern: Pattern) -> (Vec<u8>, ZpdbLayout) {
    let layout = ZpdbLayout::new(pattern);
    let mut dist = vec![UNVISITED; layout.total() as usize];

    // Seed: ONLY the single goal state (pattern tiles home, blank at its goal
    // cell) is distance 0 — exactly the additive build's seed, so that
    // `min_r ZPDB(C,r) == additive(C)`. Seeding the blank in *every* goal region
    // makes the blank free and collapses every region of every config to the
    // blank-agnostic value, breaking the invariant on multi-region goals.
    // (24-puzzle k=6 goals are single-region, so this matches the old behaviour
    // here; the fix matters for multi-region goals — see puzzle15 k=8.)
    let goal = ProjectedState::goal(pattern);
    dist[layout.rank(&goal, pattern) as usize] = 0;
    let mut frontier: Vec<ProjectedState> = vec![goal];

    // Layer-synchronous BFS; every edge has cost 1 (no 0-cost closure).
    let mut depth: u8 = 0;
    let mut succ: Vec<ProjectedState> = Vec::new();
    while !frontier.is_empty() {
        let next_depth = depth.checked_add(1).expect("ZPDB depth overflowed u8");
        let mut next: Vec<ProjectedState> = Vec::new();
        for s in &frontier {
            succ.clear();
            gen_moves(&layout, s, &mut succ);
            for ns in succ.drain(..) {
                let idx = layout.rank(&ns, pattern) as usize;
                if dist[idx] == UNVISITED {
                    dist[idx] = next_depth;
                    next.push(ns);
                }
            }
        }
        depth = next_depth;
        frontier = next;
    }

    (dist, layout)
}

/// Multi-threaded region BFS (rayon). Byte-identical output to [`build_zpdb`].
/// Each frontier state's successor generation is independent; the layer barrier
/// keeps the BFS deterministic (every entry's depth is layer-determined).
///
/// **Memory:** the frontier holds 4-byte entry **ranks** (`Vec<u32>`), not
/// 50-byte `ProjectedState`s — at production scale the frontier, not the `dist`
/// array, dominates build memory (k=7 measured: ~46 GiB of 50 GiB peak). Each
/// expanded node is reconstructed to a representative state via
/// [`ZpdbLayout::unrank_representative`] (valid because a region is connected, so
/// any cell in it yields the same successors). This keeps the k=7 build in RAM
/// (≈8 GiB) instead of spilling to compressed swap. The 1-bit codec's entry
/// count fits `u32` through k=7 (4.07e9 < 4.29e9); k≥8 would need a wider
/// frontier element (asserted below).
#[cfg(feature = "parallel")]
pub fn build_zpdb_parallel(pattern: Pattern) -> (Vec<u8>, ZpdbLayout) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    const CHUNK: usize = 2048;

    let layout = ZpdbLayout::new(pattern);
    assert!(
        layout.total() <= u32::MAX as u64,
        "rank-frontier build needs total ({}) <= u32::MAX; widen the frontier element to u64 for k>=8",
        layout.total()
    );
    let dist: Vec<AtomicU8> = (0..layout.total() as usize)
        .map(|_| AtomicU8::new(UNVISITED))
        .collect();

    // Seed ONLY the single goal state at distance 0 (see `build_zpdb`).
    let goal = ProjectedState::goal(pattern);
    let goal_rank = layout.rank(&goal, pattern);
    dist[goal_rank as usize].store(0, Ordering::Relaxed);
    let mut frontier: Vec<u32> = vec![goal_rank as u32];

    let mut depth: u8 = 0;
    while !frontier.is_empty() {
        let next_depth = depth.checked_add(1).expect("ZPDB depth overflowed u8");
        let next: Vec<u32> = frontier
            .par_chunks(CHUNK)
            .flat_map_iter(|chunk| {
                let mut local: Vec<u32> = Vec::new();
                let mut succ: Vec<ProjectedState> = Vec::new();
                for &rk in chunk {
                    let s = layout.unrank_representative(rk as u64);
                    succ.clear();
                    gen_moves(&layout, &s, &mut succ);
                    for ns in succ.drain(..) {
                        let idx = layout.rank(&ns, pattern);
                        if dist[idx as usize].fetch_min(next_depth, Ordering::Relaxed) == UNVISITED
                        {
                            local.push(idx as u32);
                        }
                    }
                }
                local.into_iter()
            })
            .collect();
        depth = next_depth;
        frontier = next;
    }

    let out = dist.into_iter().map(|a| a.into_inner()).collect();
    (out, layout)
}

/// Load the 2-bit code of state `idx` from the packed atomic working array.
#[cfg(feature = "parallel")]
#[inline]
fn load_2bit(view: &[std::sync::atomic::AtomicU8], idx: u64) -> u8 {
    use std::sync::atomic::Ordering;
    let byte = view[(idx / 4) as usize].load(Ordering::Relaxed);
    (byte >> (2 * (idx % 4))) & 0b11
}

/// Mark state `idx` with 2-bit `code` **iff it is currently unvisited (`00`)**.
/// Returns true iff this call performed the mark (i.e. `idx` was newly visited).
/// CAS-guarded — a blind `fetch_or` would flip an already-visited neighbor's
/// bit1 (the codec bit) and silently corrupt the table.
#[cfg(feature = "parallel")]
#[inline]
fn mark_if_unvisited(view: &[std::sync::atomic::AtomicU8], idx: u64, code: u8) -> bool {
    use std::sync::atomic::Ordering;
    let a = &view[(idx / 4) as usize];
    let shift = 2 * (idx % 4) as u8;
    let mut cur = a.load(Ordering::Relaxed);
    loop {
        if (cur >> shift) & 0b11 != 0 {
            return false; // already visited
        }
        let new = cur | (code << shift);
        match a.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(actual) => cur = actual,
        }
    }
}

/// Memory-frugal zero-aware PDB build: a **frontier-free 2-bit BFS** that returns
/// the already-packed 1-bit codec table (identical to `pack_bits(&build_zpdb(p))`).
///
/// Instead of the byte `dist` array + explicit frontier of [`build_zpdb_parallel`]
/// (which needs ~80 GiB of `dist` alone at k=8), this holds a single **2-bit per
/// state** working array — `00` = unvisited, else `0b1<bit1>` where `bit1 =
/// (h>>1)&1` is the codec bit — and no frontier. Peak memory is `(total+3)/4`
/// bytes (~20 GiB for k=8, vs infeasible), compacted in place to `(total+7)/8`.
///
/// The build is layer-synchronous (layer index = distance `h`). Each layer `d`
/// sweeps the array: a state is the current frontier iff it is visited, its shape
/// parity ([`ZpdbLayout::shape_parity`]) equals `d&1`, and its stored `bit1`
/// equals `(d>>1)&1` — together `h ≡ d (mod 4)`, so the sweep also re-hits depths
/// `d-4, d-8, …` whose re-expansion marks nothing new (harmless). Newly-visited
/// successors are opposite-parity, so they are never re-selected within the same
/// layer — intra-layer races are benign; the parallel `.sum()` is the barrier.
///
/// Trades build time (per-state re-expansion, ~2–5×) for memory; intended for
/// k≥8, where the frontier build is infeasible. `build_zpdb_parallel` remains the
/// fast path for k≤7.
///
/// Returns `(packed, layout, eccentricity)`, where `eccentricity` is the deepest
/// BFS layer reached — i.e. the maximum PDB value / abstract-graph radius. This is
/// free here (the build is layer-synchronous), and cannot be recovered from the
/// 1-bit `packed` table afterwards (it stores only distance parity), so it is
/// surfaced now for e.g. `Σ eccentricity` additive-family ceiling calculations.
#[cfg(feature = "parallel")]
pub fn build_zpdb_2bit_packed(pattern: Pattern) -> (Vec<u8>, ZpdbLayout, u8) {
    use rayon::prelude::*;
    use std::sync::atomic::AtomicU8;

    let layout = ZpdbLayout::new(pattern);
    let total = layout.total();
    let n_bytes = (total as usize).div_ceil(4);
    let mut buf: Vec<u8> = vec![0u8; n_bytes]; // 00 = unvisited everywhere

    let cb = layout.cohort_base();
    let counts = layout.region_counts();
    let sp = layout.shape_parity();
    let num_shapes = counts.len();
    debug_assert_eq!(cb.len(), num_shapes);

    // SAFETY: AtomicU8 has the same layout as u8; `buf` outlives `view` and is
    // only touched through `view` (all-atomic) until `view` is dropped below.
    let view: &[AtomicU8] =
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const AtomicU8, n_bytes) };

    // Seed the single goal state at depth 0 (bit1 = 0 ⇒ code 0b10).
    let goal = ProjectedState::goal(pattern);
    let goal_rank = layout.rank(&goal, pattern);
    mark_if_unvisited(view, goal_rank, 0b10);

    let mut cumulative: u64 = 1;
    let mut d: usize = 0;
    loop {
        let parity = (d & 1) as u8;
        let want_bit1 = ((d >> 1) & 1) as u8;
        let next_code = 0b10u8 | (((d + 1) >> 1) & 1) as u8;

        let marked: u64 = (0..num_shapes)
            .into_par_iter()
            .map(|sr| {
                if sp[sr] != parity {
                    return 0u64; // wrong-parity cohort: no depth-d frontier here
                }
                let start = cb[sr];
                let end = if sr + 1 < num_shapes {
                    cb[sr + 1]
                } else {
                    total
                };
                let count = counts[sr] as u64;
                let mut local: u64 = 0;
                let mut succ: Vec<ProjectedState> = Vec::new();
                for i in start..end {
                    let code = load_2bit(view, i);
                    if code == 0 || (code & 1) != want_bit1 {
                        continue; // unvisited, or not h ≡ d (mod 4)
                    }
                    let offset = i - start;
                    let s = layout.unrank_in_cohort(sr, offset / count, (offset % count) as u8);
                    succ.clear();
                    gen_moves(&layout, &s, &mut succ);
                    for ns in succ.drain(..) {
                        let idx = layout.rank(&ns, pattern);
                        if mark_if_unvisited(view, idx, next_code) {
                            local += 1;
                        }
                    }
                }
                local
            })
            .sum();

        if marked == 0 {
            break;
        }
        cumulative += marked;
        d += 1;
    }

    // `view` is no longer used; safe to mutate `buf` directly now.
    assert_eq!(
        cumulative, total,
        "frontier-free BFS covered {cumulative} of {total} states — graph not fully reached"
    );
    super::zcodec::pack1_from_2bit_inplace(&mut buf, total as usize);
    (buf, layout, d as u8)
}

#[cfg(test)]
mod tests {
    /// Probe: does a blank-centric pattern (goal cells hugging the blank's home
    /// corner) carry more zero-aware surplus over its own group-Manhattan than a
    /// distant or scattered one? Full-space exact measurement at k=6 (~181 M
    /// entries per pattern, built in memory). Global-space geometry probe — the
    /// workload weighting is a separate question.
    ///
    ///   cargo test --release -p puzzle8 probe_blank_centric -- --ignored --nocapture
    #[test]
    #[ignore = "builds five k=6 ZPDBs in memory (~minutes each); run explicitly"]
    #[cfg(feature = "parallel")]
    fn probe_blank_centric_partitions() {
        use rayon::prelude::*;
        let cases: [(&str, [u8; 6]); 13] = [
            // quadrant partition (existing pdb24_a..d); total 5.773
            ("QUAD-A/FAR {1,2,3,6,7,8}", [1, 2, 3, 6, 7, 8]),
            ("QUAD-B {4,5,9,10,14,15}", [4, 5, 9, 10, 14, 15]),
            ("QUAD-C {11,12,16,17,21,22}", [11, 12, 16, 17, 21, 22]),
            ("QUAD-D {13,18,19,20,23,24}", [13, 18, 19, 20, 23, 24]),
            // row-band partition (existing pdb24_rb_a..d); total 4.776
            ("BAND-A {1..6}", [1, 2, 3, 4, 5, 6]),
            ("BAND-B {7..12}", [7, 8, 9, 10, 11, 12]),
            ("BAND-C {13..18}", [13, 14, 15, 16, 17, 18]),
            ("BAND-D {19..24}", [19, 20, 21, 22, 23, 24]),
            // distance-ring partition around the blank's corner; total 4.886
            (
                "RING-1/CORNER {14,15,19,20,23,24}",
                [14, 15, 19, 20, 23, 24],
            ),
            ("RING-2 {9,10,13,17,18,22}", [9, 10, 13, 17, 18, 22]),
            ("RING-3 {4,5,8,12,16,21}", [4, 5, 8, 12, 16, 21]),
            ("RING-4 {1,2,3,6,7,11}", [1, 2, 3, 6, 7, 11]),
            // hybrid = CORNER + QUAD-A + QUAD-C + this leftover; total 5.976 (best)
            ("HYBRID-LEFTOVER {4,5,9,10,13,18}", [4, 5, 9, 10, 13, 18]),
        ];
        for (name, tiles) in cases {
            let mut mask = 0u32;
            for &t in &tiles {
                mask |= 1 << t;
            }
            let pattern = Pattern(mask);
            let t0 = std::time::Instant::now();
            let (dist, layout) = build_zpdb_parallel(pattern);
            let built = t0.elapsed();
            let total = layout.total();
            // Exact surplus histogram over every entry.
            const CHUNK: u64 = 1 << 20;
            let hist = (0..total.div_ceil(CHUNK))
                .into_par_iter()
                .map(|c| {
                    let mut h = [0u64; 32];
                    let lo = c * CHUNK;
                    let hi = (lo + CHUNK).min(total);
                    for idx in lo..hi {
                        let d = dist[idx as usize];
                        if d == UNVISITED {
                            continue;
                        }
                        let proj = layout.unrank_representative(idx);
                        let mut md = 0u16;
                        for &t in &tiles {
                            let p = proj.pos_of(t) as u16;
                            let g = (t - 1) as u16;
                            md += (p / 5).abs_diff(g / 5) + (p % 5).abs_diff(g % 5);
                        }
                        let s = (d as u16).saturating_sub(md).min(31);
                        h[s as usize] += 1;
                    }
                    h
                })
                .reduce(
                    || [0u64; 32],
                    |mut a, b| {
                        for i in 0..32 {
                            a[i] += b[i];
                        }
                        a
                    },
                );
            let n: u64 = hist.iter().sum();
            let tail =
                |s: usize| -> f64 { 100.0 * hist[s..].iter().sum::<u64>() as f64 / n as f64 };
            let mean: f64 = hist
                .iter()
                .enumerate()
                .map(|(s, &c)| s as f64 * c as f64)
                .sum::<f64>()
                / n as f64;
            println!(
                "{name}: {n} entries (built {:.0?})  surplus>=2: {:6.2}%  >=4: {:5.2}%  >=6: {:5.2}%  >=8: {:5.2}%  mean {:.3}",
                built,
                tail(2),
                tail(4),
                tail(6),
                tail(8),
                mean
            );
        }
    }

    use super::super::zpdb::{regions, OCCUPIED};
    use super::*;
    use crate::puzzle24::pdb::build;

    /// Enumerate every placement of the `k` pattern tiles (ascending value) onto
    /// `k` distinct cells, calling `f(pattern_cells)` with the projection array
    /// (tiles placed, all other cells [`ANON`], no blank yet).
    fn for_each_placement(pattern: Pattern, mut f: impl FnMut([u8; N_CELLS])) {
        let tiles: Vec<u8> = pattern.iter().collect();
        let k = tiles.len();
        let mut chosen = [0usize; 8];
        fn rec(
            i: usize,
            k: usize,
            tiles: &[u8],
            chosen: &mut [usize; 8],
            used: u32,
            f: &mut impl FnMut([u8; N_CELLS]),
        ) {
            if i == k {
                let mut cells = [ANON; N_CELLS];
                for j in 0..k {
                    cells[chosen[j]] = tiles[j];
                }
                f(cells);
                return;
            }
            for c in 0..N_CELLS {
                if used & (1u32 << c) == 0 {
                    chosen[i] = c;
                    rec(i + 1, k, tiles, chosen, used | (1u32 << c), f);
                }
            }
        }
        rec(0, k, &tiles, &mut chosen, 0, &mut f);
    }

    /// Project the zero-aware PDB to the standard additive PDB: for each pattern
    /// configuration, `min` over its zero-tile regions. Indexed by
    /// [`ProjectedState::rank`], matching [`build::build`].
    fn project_additive(pattern: Pattern, dist: &[u8], layout: &ZpdbLayout) -> Vec<u8> {
        let mut add = vec![UNVISITED; pattern.num_projected_states() as usize];
        for_each_placement(pattern, |pattern_cells| {
            let mut occ = 0u32;
            for (c, &v) in pattern_cells.iter().enumerate() {
                if v != ANON {
                    occ |= 1u32 << c;
                }
            }
            let (count, labels) = regions(occ);
            // representative free cell per region
            let mut rep = vec![usize::MAX; count as usize];
            for (c, &l) in labels.iter().enumerate() {
                if l != OCCUPIED && rep[l as usize] == usize::MAX {
                    rep[l as usize] = c;
                }
            }
            let mut best = UNVISITED;
            let mut arank = 0u64;
            for (rr, &rc) in rep.iter().enumerate() {
                let mut nc = pattern_cells;
                nc[rc] = 0;
                let ps = ProjectedState::from_projection(nc);
                if rr == 0 {
                    arank = ps.rank(pattern); // blank-independent
                }
                let d = dist[layout.rank(&ps, pattern) as usize];
                if d < best {
                    best = d;
                }
            }
            add[arank as usize] = best;
        });
        add
    }

    fn assert_matches_additive(pattern: Pattern) {
        let (dist, layout) = build_zpdb(pattern);
        // No gaps: every reachable region of every config is filled.
        let projected = project_additive(pattern, &dist, &layout);
        let reference = build::build(pattern);
        assert_eq!(
            projected.len(),
            reference.len(),
            "additive sizes differ for {pattern:?}"
        );
        assert_eq!(
            projected, reference,
            "min-over-regions ZPDB != verified additive build for {pattern:?}"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn zpdb_parallel_matches_sequential_k3() {
        let p = Pattern::new(&[1, 7, 13]);
        assert_eq!(build_zpdb(p).0, build_zpdb_parallel(p).0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn zpdb_parallel_matches_sequential_k4() {
        let p = Pattern::new(&[2, 5, 8, 11]);
        assert_eq!(build_zpdb(p).0, build_zpdb_parallel(p).0);
    }

    /// The frontier-free 2-bit builder must emit byte-identical packed tables to
    /// `pack_bits(build_zpdb(..))` (the verified sequential oracle). This single
    /// gate catches every corruption mode: CAS marking, the parity+bit1 frontier
    /// predicate, the in-place 2→1 compaction, and the AtomicU8 reinterpret.
    #[cfg(feature = "parallel")]
    #[test]
    fn zpdb_2bit_packed_matches_pack_bits_k2_k3_k4() {
        use super::super::zcodec::pack_bits;
        for tiles in [&[1u8, 2][..], &[1, 7, 13], &[2, 5, 8, 11]] {
            let p = Pattern::new(tiles);
            let (packed, _, ecc) = build_zpdb_2bit_packed(p);
            let dist = build_zpdb(p).0;
            let reference = pack_bits(&dist);
            assert_eq!(
                packed, reference,
                "2-bit packed != pack_bits(build_zpdb) for {tiles:?}"
            );
            // The exposed eccentricity must equal the true max distance (oracle).
            let true_ecc = *dist.iter().max().unwrap();
            assert_eq!(
                ecc, true_ecc,
                "eccentricity {ecc} != oracle max {true_ecc} for {tiles:?}"
            );
        }
    }

    /// The packed table from the 2-bit builder must decode (through `from_packed` +
    /// the codec's cold-lookup) back to the true distances — ties the new
    /// builder to the query path, not just the packing.
    #[cfg(feature = "parallel")]
    #[test]
    fn zpdb_2bit_packed_decodes_to_true_dist_k3() {
        let p = Pattern::new(&[1, 7, 13]);
        let (dist, layout) = build_zpdb(p);
        let (packed, _, _) = build_zpdb_2bit_packed(p);
        let zdb = super::super::zdb::ZPatternDb::from_packed(p, packed);
        for idx in 0..layout.total() {
            let s = layout.unrank_representative(idx);
            assert_eq!(
                zdb.cold_lookup_proj(&s),
                dist[idx as usize],
                "decode mismatch at idx {idx}"
            );
        }
    }

    /// Full k=7 correctness + performance probe: build one k7 part BOTH ways and
    /// assert byte-identical packed output (the frontier path is itself SHA-pinned
    /// to the committed k7 artifacts, so this transitively proves the frontier-free
    /// builder matches them), and report the re-expansion time multiplier.
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "full k7 build both ways (~30-60 min); correctness + perf probe, --nocapture"]
    fn zpdb_2bit_packed_matches_parallel_k7() {
        use super::super::zcodec::pack_bits;
        use std::time::Instant;
        let p = Pattern::new(&[1, 2, 3, 6, 7, 8, 11]); // K7 partition, part a
        let t = Instant::now();
        let frontier = pack_bits(&build_zpdb_parallel(p).0);
        let t_frontier = t.elapsed();
        let t = Instant::now();
        let frugal = build_zpdb_2bit_packed(p).0;
        let t_frugal = t.elapsed();
        println!(
            "k7 frontier build {:.1?}, frontier-free {:.1?} ({:.2}x slower)",
            t_frontier,
            t_frugal,
            t_frugal.as_secs_f64() / t_frontier.as_secs_f64()
        );
        assert_eq!(frugal, frontier, "k7 frontier-free != frontier build");
    }

    /// k=8 sizing pre-flight: build only the layout (cheap) and report the
    /// working-array (2-bit) and packed (1-bit) byte sizes, to confirm the
    /// frontier-free build fits RAM before launching a multi-hour k8 build.
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "k8 layout sizing pre-flight; run with --ignored --nocapture"]
    fn zpdb_k8_layout_size_preflight() {
        let p = Pattern::new(&[1, 2, 3, 6, 7, 8, 11, 12]); // total() depends only on k=8
        let layout = ZpdbLayout::new(p);
        let total = layout.total();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        println!("k8 total()          = {total}");
        println!(
            "2-bit working bytes = {} ({:.2} GiB)",
            total.div_ceil(4),
            gib(total.div_ceil(4))
        );
        println!(
            "1-bit packed bytes  = {} ({:.2} GiB)",
            total.div_ceil(8),
            gib(total.div_ceil(8))
        );
    }

    /// Full k=6 equivalence to the fast frontier builder (the k≤7 shipped path).
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "full k6 frontier-free builds (~minutes); run with --ignored"]
    fn zpdb_2bit_packed_matches_parallel_k6() {
        use super::super::zcodec::pack_bits;
        for tiles in [
            &[1u8, 2, 3, 6, 7, 8][..],
            &[4, 5, 9, 10, 14, 15],
            &[11, 12, 16, 17, 21, 22],
            &[13, 18, 19, 20, 23, 24],
        ] {
            let p = Pattern::new(tiles);
            assert_eq!(
                build_zpdb_2bit_packed(p).0,
                pack_bits(&build_zpdb_parallel(p).0),
                "k6 mismatch {tiles:?}"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "builds the full 6-tile ZPDB; run with `--release --features parallel,mmap -- --ignored --nocapture`"]
    fn time_full_6tile_build() {
        use std::time::Instant;
        let p = Pattern::new(&[1, 2, 3, 6, 7, 8]);
        let t = Instant::now();
        let (dist, layout) = build_zpdb_parallel(p);
        let el = t.elapsed();
        assert_eq!(layout.total(), 181_008_000);
        assert_eq!(dist.len(), 181_008_000);
        let unvisited = dist.iter().filter(|&&d| d == UNVISITED).count();
        let maxd = dist
            .iter()
            .filter(|&&d| d != UNVISITED)
            .copied()
            .max()
            .unwrap();
        println!(
            "6-tile ZPDB: {} entries, {} unvisited, max depth {}, built in {:?}",
            layout.total(),
            unvisited,
            maxd,
            el
        );
    }

    #[test]
    fn zpdb_goal_entries_are_zero() {
        let (dist, _) = build_zpdb(Pattern::new(&[1, 7, 13]));
        // The goal seeds were set to 0; at least one zero must exist.
        assert!(dist.contains(&0));
    }

    #[test]
    fn zpdb_additive_projection_matches_build_k2() {
        assert_matches_additive(Pattern::new(&[1, 2]));
    }

    #[test]
    fn zpdb_additive_projection_matches_build_k3() {
        assert_matches_additive(Pattern::new(&[1, 7, 13]));
    }

    #[test]
    #[ignore = "runs the OLD blank-cell build::build oracle over P(25,5)=6.4M states (~13s); \
                run with `cargo test -- --ignored`. The k=2/k=3 gates verify the new build on every run."]
    fn zpdb_additive_projection_matches_build_k4() {
        assert_matches_additive(Pattern::new(&[2, 5, 8, 11]));
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "k=6 production-partition dominance check (~15-25 min, ~3 GB); \
                run with `cargo test --release --features parallel,mmap -- --ignored --nocapture zpdb_matches_additive_k6`"]
    fn zpdb_matches_additive_k6_part_a() {
        // Mirrors assert_matches_additive at the production k=6 size, using the
        // parallel builds, and REPORTS the violation count/magnitude instead of just
        // panicking — to detect the gen_moves over-permissiveness bug found on the
        // 15-puzzle at k=8 (admissible but min_r ZPDB < additive on multi-region configs).
        let pattern = Pattern::new(&[1, 2, 3, 6, 7, 8]);
        let (dist, layout) = build_zpdb_parallel(pattern);
        let projected = project_additive(pattern, &dist, &layout);
        let reference = build::build_parallel(pattern);
        assert_eq!(projected.len(), reference.len(), "additive sizes differ");
        let mut violations = 0usize;
        let mut max_deficit = 0i32;
        let mut shown = 0;
        for (i, (&p, &r)) in projected.iter().zip(reference.iter()).enumerate() {
            if p != r {
                violations += 1;
                let deficit = r as i32 - p as i32;
                max_deficit = max_deficit.max(deficit);
                if shown < 8 {
                    eprintln!("  config rank {i}: min_r ZPDB = {p}, additive = {r} (ZPDB too low by {deficit})");
                    shown += 1;
                }
            }
        }
        eprintln!(
            "k=6 part a [1,2,3,6,7,8]: {} / {} configs violate (min_r ZPDB == additive); max deficit {}",
            violations, projected.len(), max_deficit,
        );
        assert_eq!(
            violations, 0,
            "ZPDB build is NOT dominance-correct at k=6 ({violations} violations)"
        );
    }

    #[test]
    fn zpdb_geq_additive_pointwise_k3() {
        // Every ZPDB entry is >= the additive value for its config (it's a min
        // over regions), and the strongest region equals additive.
        let pattern = Pattern::new(&[1, 7, 13]);
        let (dist, layout) = build_zpdb(pattern);
        let additive = build::build(pattern);
        for_each_placement(pattern, |pattern_cells| {
            let mut occ = 0u32;
            for (c, &v) in pattern_cells.iter().enumerate() {
                if v != ANON {
                    occ |= 1u32 << c;
                }
            }
            let (count, labels) = regions(occ);
            let mut rep = vec![usize::MAX; count as usize];
            for (c, &l) in labels.iter().enumerate() {
                if l != OCCUPIED && rep[l as usize] == usize::MAX {
                    rep[l as usize] = c;
                }
            }
            let mut best = UNVISITED;
            let mut arank = 0u64;
            for (rr, &rc) in rep.iter().enumerate() {
                let mut nc = pattern_cells;
                nc[rc] = 0;
                let ps = ProjectedState::from_projection(nc);
                if rr == 0 {
                    arank = ps.rank(pattern);
                }
                let z = dist[layout.rank(&ps, pattern) as usize];
                assert_ne!(z, UNVISITED, "reachable ZPDB entry left unvisited");
                assert!(z >= additive[arank as usize], "ZPDB < additive");
                best = best.min(z);
            }
            assert_eq!(
                best, additive[arank as usize],
                "min-region ZPDB != additive"
            );
        });
    }
}
