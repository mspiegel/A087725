//! Zero-aware PDB construction for the 15-puzzle via a region-aware `(m,p,r)`
//! BFS. Ported from `puzzle24::pdb::zbuild`.
//!
//! We BFS over `(pattern_config, blank_region)`. Collapsing the blank's free
//! wandering into its zero-tile region (`zpdb::regions`) removes the 0-cost
//! closure entirely: **every edge is a unit-cost pattern move**. The result
//! `dist[(m,p,r)]` is the zero-aware value `ZPDB(C,r)`. The standard additive
//! PDB falls out as `additive(C) = min_r ZPDB(C,r)`, which we assert
//! byte-identical to the verified [`super::build::build`] output.

use super::pattern::{Pattern, ProjectedState, ANON};
use super::zpdb::ZpdbLayout;
use crate::puzzle15::state::{N_CELLS, W};

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
/// by [`ZpdbLayout::rank`], length [`ZpdbLayout::total`]. Returns the layout too.
pub fn build_zpdb(pattern: Pattern) -> (Vec<u8>, ZpdbLayout) {
    let layout = ZpdbLayout::new(pattern);
    let mut dist = vec![UNVISITED; layout.total() as usize];

    // Seed: ONLY the single goal state (pattern tiles home, blank at its goal
    // cell) is distance 0 — exactly the additive build's seed, so that
    // `min_r ZPDB(C,r) == additive(C)`. Seeding the blank in *every* goal region
    // (the previous behaviour) makes the blank free, which collapses every region
    // of every config to the same blank-agnostic value — breaking the invariant
    // on multi-region goals (e.g. the 8-tile pattern, whose goal isolates cell 15)
    // and defeating the whole point of being zero-aware.
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
#[cfg(feature = "parallel")]
pub fn build_zpdb_parallel(pattern: Pattern) -> (Vec<u8>, ZpdbLayout) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    const CHUNK: usize = 2048;

    let layout = ZpdbLayout::new(pattern);
    let dist: Vec<AtomicU8> = (0..layout.total() as usize)
        .map(|_| AtomicU8::new(UNVISITED))
        .collect();

    // Seed ONLY the single goal state (see `build_zpdb` for why seeding every
    // goal region is wrong on multi-region goals).
    let goal = ProjectedState::goal(pattern);
    dist[layout.rank(&goal, pattern) as usize].store(0, Ordering::Relaxed);
    let mut frontier: Vec<ProjectedState> = vec![goal];

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
    use super::super::zpdb::{regions, OCCUPIED};
    use crate::puzzle15::pdb::build;

    /// Enumerate every placement of the `k` pattern tiles (ascending value) onto
    /// `k` distinct cells, calling `f(pattern_cells)`.
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
    /// configuration, `min` over its zero-tile regions.
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
        let projected = project_additive(pattern, &dist, &layout);
        let reference = build::build(pattern);
        assert_eq!(projected.len(), reference.len(), "additive sizes differ for {pattern:?}");
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

    #[test]
    fn zpdb_goal_entries_are_zero() {
        let (dist, _) = build_zpdb(Pattern::new(&[1, 7, 13]));
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
    fn zpdb_additive_projection_matches_build_k4() {
        assert_matches_additive(Pattern::new(&[1, 2, 3, 4]));
    }

    #[test]
    fn zpdb_geq_additive_pointwise_k3() {
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

    #[test]
    fn zpdb_matches_additive_on_multiregion_goal() {
        // These patterns' GOALS isolate cell 15 (tiles at 11 and/or 14 wall it
        // off), so the goal splits into regions — the case the seeding fix
        // addresses. Before the fix, min_r ZPDB (blank-free) < additive here.
        assert_matches_additive(Pattern::new(&[12, 15]));
        assert_matches_additive(Pattern::new(&[11, 14, 15]));
        assert_matches_additive(Pattern::new(&[10, 11, 13, 14, 15]));
    }

    /// Physical reference move generator, independent of `gen_moves`: BFS the
    /// blank's reachable free cells, then for each reachable cell slide any
    /// adjacent pattern tile into it.
    fn reference_moves(proj: &ProjectedState) -> std::collections::HashSet<[u8; N_CELLS]> {
        use std::collections::HashSet;
        let nbrs = |c: usize| -> Vec<usize> {
            let (r, col) = (c / W, c % W);
            let mut v = Vec::new();
            if r > 0 { v.push(c - W); }
            if r < W - 1 { v.push(c + W); }
            if col > 0 { v.push(c - 1); }
            if col < W - 1 { v.push(c + 1); }
            v
        };
        let is_free = |c: usize| proj.cells[c] == 0 || proj.cells[c] == ANON;
        let blank = proj.blank_pos() as usize;
        let mut reach = [false; N_CELLS];
        let mut stack = vec![blank];
        reach[blank] = true;
        while let Some(c) = stack.pop() {
            for n in nbrs(c) {
                if is_free(n) && !reach[n] {
                    reach[n] = true;
                    stack.push(n);
                }
            }
        }
        let mut base = [ANON; N_CELLS];
        for (c, &v) in proj.cells.iter().enumerate() {
            if v != 0 && v != ANON { base[c] = v; }
        }
        let mut out = HashSet::new();
        for b in 0..N_CELLS {
            if !reach[b] { continue; }
            for c in nbrs(b) {
                let t = proj.cells[c];
                if t != 0 && t != ANON {
                    let mut nc = base;
                    nc[c] = 0; // blank lands where the tile was
                    nc[b] = t; // tile slides c -> b (into the blank's reachable cell)
                    out.insert(nc);
                }
            }
        }
        out
    }

    #[test]
    fn gen_moves_matches_physical_reference_k8() {
        use crate::puzzle15::state::State;
        use std::collections::HashSet;
        let pattern = Pattern::new(&[8, 9, 10, 11, 12, 13, 14, 15]);
        let layout = ZpdbLayout::new(pattern);
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..500_000 {
            let mut perm = [0u8; N_CELLS];
            for (i, p) in perm.iter_mut().enumerate() { *p = i as u8; }
            for i in (1..N_CELLS).rev() {
                let j = (rng() % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }
            let proj = ProjectedState::from_state(&State(perm), pattern);
            let mut succ = Vec::new();
            gen_moves(&layout, &proj, &mut succ);
            let a: HashSet<[u8; N_CELLS]> = succ.iter().map(|s| s.cells).collect();
            let b = reference_moves(&proj);
            if a != b {
                let extra: Vec<_> = a.difference(&b).collect();
                let missing: Vec<_> = b.difference(&a).collect();
                panic!(
                    "gen_moves != reference\n  proj    = {:?}\n  blank   = {}\n  EXTRA (gen_moves has, invalid): {:?}\n  MISSING (gen_moves lacks): {:?}",
                    proj.cells, proj.blank_pos(), extra, missing,
                );
            }
        }
    }
}
