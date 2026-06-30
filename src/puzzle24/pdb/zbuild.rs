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
                        if dist[idx as usize].fetch_min(next_depth, Ordering::Relaxed) == UNVISITED {
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::zpdb::{regions, OCCUPIED};
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
            "additive sizes differ for {:?}",
            pattern
        );
        assert_eq!(
            projected, reference,
            "min-over-regions ZPDB != verified additive build for {:?}",
            pattern
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
        let maxd = dist.iter().filter(|&&d| d != UNVISITED).copied().max().unwrap();
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
        assert!(dist.iter().any(|&d| d == 0));
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
                    eprintln!("  config rank {}: min_r ZPDB = {}, additive = {} (ZPDB too low by {})", i, p, r, deficit);
                    shown += 1;
                }
            }
        }
        eprintln!(
            "k=6 part a [1,2,3,6,7,8]: {} / {} configs violate (min_r ZPDB == additive); max deficit {}",
            violations, projected.len(), max_deficit,
        );
        assert_eq!(violations, 0, "ZPDB build is NOT dominance-correct at k=6 ({} violations)", violations);
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
            assert_eq!(best, additive[arank as usize], "min-region ZPDB != additive");
        });
    }
}
