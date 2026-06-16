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
use super::zpdb::{regions, ZpdbLayout, OCCUPIED};
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

    // Seed: pattern tiles home with the blank in *every* region is distance 0.
    let goal = ProjectedState::goal(pattern);
    let occ = occupied_mask(&goal);
    let (count, labels) = regions(occ);
    let mut base = [ANON; N_CELLS];
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
    let mut frontier: Vec<ProjectedState> = Vec::new();
    for &rc in &rep {
        let mut nc = base;
        nc[rc] = 0;
        let ps = ProjectedState::from_projection(nc);
        let idx = layout.rank(&ps, pattern) as usize;
        if dist[idx] == UNVISITED {
            dist[idx] = 0;
            frontier.push(ps);
        }
    }

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
#[cfg(feature = "parallel")]
pub fn build_zpdb_parallel(pattern: Pattern) -> (Vec<u8>, ZpdbLayout) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    const CHUNK: usize = 2048;

    let layout = ZpdbLayout::new(pattern);
    let dist: Vec<AtomicU8> = (0..layout.total() as usize)
        .map(|_| AtomicU8::new(UNVISITED))
        .collect();

    // Seed all regions of the goal config at distance 0 (single-threaded; ≤5).
    let goal = ProjectedState::goal(pattern);
    let occ = occupied_mask(&goal);
    let (count, labels) = layout.regions_for(occ);
    let mut base = [ANON; N_CELLS];
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
    let mut frontier: Vec<ProjectedState> = Vec::new();
    for &rc in &rep {
        let mut nc = base;
        nc[rc] = 0;
        let ps = ProjectedState::from_projection(nc);
        let idx = layout.rank(&ps, pattern) as usize;
        if dist[idx].swap(0, Ordering::Relaxed) == UNVISITED {
            frontier.push(ps);
        }
    }

    let mut depth: u8 = 0;
    while !frontier.is_empty() {
        let next_depth = depth.checked_add(1).expect("ZPDB depth overflowed u8");
        let next: Vec<ProjectedState> = frontier
            .par_chunks(CHUNK)
            .flat_map_iter(|chunk| {
                let mut local: Vec<ProjectedState> = Vec::new();
                let mut succ: Vec<ProjectedState> = Vec::new();
                for s in chunk {
                    succ.clear();
                    gen_moves(&layout, s, &mut succ);
                    for ns in succ.drain(..) {
                        let idx = layout.rank(&ns, pattern) as usize;
                        if dist[idx].fetch_min(next_depth, Ordering::Relaxed) == UNVISITED {
                            local.push(ns);
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
