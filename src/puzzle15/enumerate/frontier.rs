//! Top-down frontier: descent + solve-free local-maxima recovery, gated by the
//! exact `N(d)` counts. See `ENUMERATION.md`.

use std::collections::HashSet;
use std::path::Path;

use crate::puzzle15::rank::{rank, unrank};
use crate::puzzle15::state::State;
use crate::puzzle15::symmetry::reflect;

use super::cache::{self, Cache};
use super::histogram::Histogram;
use super::store::Store;

/// Append the (2–4) neighbor ranks of `r` to `out` (cleared first).
#[inline]
fn neighbors(r: u64, out: &mut Vec<u64>) {
    out.clear();
    let s = unrank(r);
    let blank = s.blank_pos();
    for m in State::legal_moves_at(blank).iter() {
        let (ns, _) = s.apply_at(m, blank);
        out.push(rank(&ns));
    }
}

/// True iff every neighbor of `w` is present in the store at depth `target`.
/// By the Bellman relation this certifies `depth(w) == target + 1` with no
/// search.
#[inline]
fn all_neighbors_at_depth(store: &Store, w: u64, target: u8) -> bool {
    let s = unrank(w);
    let blank = s.blank_pos();
    for m in State::legal_moves_at(blank).iter() {
        let (ns, _) = s.apply_at(m, blank);
        match store.depth_of(rank(&ns)) {
            Some(dd) if dd == target => {}
            _ => return false,
        }
    }
    true
}

/// Per-layer outcome, for the run report.
#[derive(Clone, Debug)]
pub struct LayerReport {
    pub depth: u8,
    pub descent: usize,
    pub maxima_bellman: usize,
    pub maxima_fallback: usize,
    pub total: usize,
    pub expected: u64,
    pub complete: bool,
}

/// Descent: expand the complete layer `d+1`; every hash-miss neighbor is
/// necessarily depth `d`. Returns the number of newly inserted boards.
fn descent(store: &mut Store, d: u8) -> usize {
    let above: Vec<u64> = store.layer(d + 1).to_vec();
    let mut nb = Vec::with_capacity(4);
    let mut added = 0usize;
    for x in above {
        neighbors(x, &mut nb);
        for &n in &nb {
            if store.insert(n, d) {
                added += 1;
            }
        }
    }
    added
}

/// Build the depth-`d-1` shell: hash-miss neighbors of the current depth-`d`
/// boards (necessarily depth `d-1`, since the depth-`d+1` neighbors are already
/// in the complete layer above). Returns the newly inserted shell ranks.
fn build_shell(store: &mut Store, d: u8) -> Vec<u64> {
    let layer_d: Vec<u64> = store.layer(d).to_vec();
    let mut nb = Vec::with_capacity(4);
    let mut shell = Vec::new();
    for x in layer_d {
        neighbors(x, &mut nb);
        for &n in &nb {
            if store.insert(n, d - 1) {
                shell.push(n);
            }
        }
    }
    shell
}

/// Process one layer `d` (the complete layers `d+1`, `d+2` must already be in
/// `store`). `verifier`, if given, maps a rank to its exact optimal depth and is
/// used only when the solve-free recovery leaves the count short.
pub fn process_layer(
    store: &mut Store,
    d: u8,
    expected: u64,
    verifier: Option<&dyn Fn(u64) -> u8>,
) -> LayerReport {
    let descent_added = descent(store, d);
    let shell = build_shell(store, d);

    // Solve-free local-maxima recovery: an up-neighbor of the shell, not yet
    // known, all of whose neighbors sit at depth d-1, is a depth-d maximum.
    let mut bellman = 0usize;
    let mut failed: Vec<u64> = Vec::new();
    let mut tested: HashSet<u64> = HashSet::new();
    let mut nb = Vec::with_capacity(4);
    for &z in &shell {
        neighbors(z, &mut nb);
        let cand = nb.clone();
        for &w in &cand {
            if store.contains(w) || !tested.insert(w) {
                continue;
            }
            if all_neighbors_at_depth(store, w, d - 1) {
                store.insert(w, d);
                bellman += 1;
            } else {
                failed.push(w);
            }
        }
    }

    // Fallback: if still short, the residue is local maxima in pockets the shell
    // didn't anchor. Solve the failed candidates and keep the depth-d ones.
    let mut fallback = 0usize;
    if (store.count(d) as u64) < expected {
        if let Some(verify) = verifier {
            for &w in &failed {
                if store.contains(w) {
                    continue;
                }
                if verify(w) == d {
                    store.insert(w, d);
                    fallback += 1;
                }
                if (store.count(d) as u64) == expected {
                    break;
                }
            }
        }
    }

    let total = store.count(d);
    LayerReport {
        depth: d,
        descent: descent_added,
        maxima_bellman: bellman,
        maxima_fallback: fallback,
        total,
        expected,
        complete: total as u64 == expected,
    }
}

/// Distinct neighbor ranks of every board in `frontier` that are not already in
/// `store`.
fn fresh_neighbors(store: &Store, frontier: &[u64]) -> Vec<u64> {
    let mut set: HashSet<u64> = HashSet::new();
    let mut nb = Vec::with_capacity(4);
    for &x in frontier {
        neighbors(x, &mut nb);
        for &n in &nb {
            if !store.contains(n) {
                set.insert(n);
            }
        }
    }
    set.into_iter().collect()
}

/// Band BFS: enumerate every board with optimal depth ≥ `floor` that is
/// connected to the antipodes through the band, computing each board's exact
/// depth with `verify` (idastar) — parallel over each BFS round. Unlike the
/// descent, this reaches local maxima (their down-neighbors at `floor` keep the
/// band connected). Writes the complete layers `[down_to..=80]`.
///
/// `floor` should be at least `down_to - 1` so the layers we keep are fully
/// connected; the work scales with the cumulative `N(≥floor)`, so it is only
/// practical for the top of the distribution.
#[cfg(feature = "parallel")]
pub fn run_band<F>(
    store: &mut Store,
    hist: &Histogram,
    down_to: u8,
    floor: u8,
    budget: u64,
    cache: &mut Cache,
    cache_path: Option<&Path>,
    out_dir: &Path,
    verify: F,
    mut log: impl FnMut(&str),
) -> Result<Vec<LayerReport>, String>
where
    F: Fn(u64) -> u8 + Sync,
{
    use rayon::prelude::*;

    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir.display(), e))?;
    let seeded = store.count(80) as u64;
    if seeded != hist[80] {
        return Err(format!("seeded {} antipodes, expected N(80)={}", seeded, hist[80]));
    }

    use std::io::Write;
    use std::time::Instant;
    // Time-based heartbeat: rounds can run many minutes, so check the clock
    // between sub-batches. Default 10 min; override with ENUM_STATUS_SECS.
    let status_secs = std::env::var("ENUM_STATUS_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(600u64);
    const CHUNK: usize = 2048;
    let start = Instant::now();
    let mut last_status = Instant::now();
    let status = |round: u32, solves: u64, store: &Store| {
        let secs = start.elapsed().as_secs_f64().max(1.0);
        let mut line = format!(
            "[STATUS {:>5.1}m] round {}, {} solves ({:.1}/s), stored {} |",
            secs / 60.0, round, solves, solves as f64 / secs, store.total(),
        );
        for d in (floor..=80).rev() {
            line.push_str(&format!(" d{} {}/{}", d, store.count(d), hist[d as usize]));
        }
        line
    };

    // Initialize the frontier from every board already in the Store at depth
    // ≥ floor — not just the depth-80 antipodes. This matters whenever the
    // Store was pre-populated beyond the antipode shell (e.g. by
    // auto-seed-from-cache or --seed-ranks). Without this, all pre-seeded
    // boards just sit there: round 1 expands only the antipodes, finds their
    // d79 neighbors already in the Store (filtered by `fresh_neighbors`), and
    // the BFS exhausts immediately without ever touching the pre-seeded
    // layers' neighbors or solving anything new.
    let mut frontier: Vec<u64> = (floor..=80)
        .flat_map(|d| store.layer(d).iter().copied())
        .collect();
    let mut round = 0u32;
    let mut solves: u64 = 0;
    let mut capped = false;
    let mut done = false;
    while !frontier.is_empty() {
        round += 1;
        let cands = fresh_neighbors(store, &frontier);
        let mut next = Vec::new();
        // Solve the round in sub-batches so the 10-min heartbeat fires mid-round.
        for chunk in cands.chunks(CHUNK) {
            // Symmetry-aware triage: depth is reflection-invariant
            // (`symmetry::reflect`), so for each candidate `r` we probe both `r`
            // and `r_refl = rank(reflect(unrank(r)))` in the cache. When both
            // miss, we queue the lex-smaller of the pair for an actual IDA*
            // solve and the partner is deferred to be filled from the cache
            // after solves complete (we mirror `(r, d)` to `(r_refl, d)`). In a
            // fully symmetry-closed cache this halves the IDA* calls — every
            // partner becomes a free cache hit.
            let mut misses: Vec<u64> = Vec::new();
            let mut deferred: Vec<u64> = Vec::new();
            let mut results: Vec<(u64, u8)> = Vec::with_capacity(chunk.len());
            let mut queued: HashSet<u64> = HashSet::new();
            for &r in chunk {
                let r_refl = rank(&reflect(&unrank(r)));
                if let Some(&d) = cache.get(&r) {
                    results.push((r, d));
                } else if let Some(&d) = cache.get(&r_refl) {
                    // Mirror the partner's depth so future probes hit `r` directly.
                    cache.insert(r, d);
                    results.push((r, d));
                } else {
                    // Both miss. Queue the lex-canonical of the pair; the
                    // partner is filled in from the cache below.
                    let canonical = r.min(r_refl);
                    if queued.insert(canonical) {
                        misses.push(canonical);
                    }
                    deferred.push(r);
                }
            }
            let solved: Vec<(u64, u8)> = misses.par_iter().map(|&r| (r, verify(r))).collect();
            solves += misses.len() as u64;
            for &(r, d) in &solved {
                let r_refl = rank(&reflect(&unrank(r)));
                cache.insert(r, d);
                if r_refl != r {
                    cache.insert(r_refl, d);
                }
            }
            for r in deferred {
                let d = *cache.get(&r).expect("canonical solve should have populated r");
                results.push((r, d));
            }
            for (r, d) in results {
                if d >= floor && store.insert(r, d) {
                    next.push(r);
                }
            }
            if last_status.elapsed().as_secs() >= status_secs {
                log(&status(round, solves, store));
                if let Some(p) = cache_path {
                    if let Err(e) = cache::save(p, cache) {
                        log(&format!("cache save failed: {e}"));
                    }
                }
                let _ = std::io::stdout().flush();
                last_status = Instant::now();
            }
        }
        log(&format!(
            "round {:>3}: {:>9} cands, kept {:>9} (≥{}), stored {:>9}, solves {:>10}",
            round, cands.len(), next.len(), floor, store.total(), solves,
        ));
        let _ = std::io::stdout().flush();
        frontier = next;
        // Early success: the shallowest target layer is fully enumerated.
        if store.count(down_to) as u64 == hist[down_to as usize] {
            log(&format!(
                "target layer {} COMPLETE (N={}) after {} solves — stopping",
                down_to, hist[down_to as usize], solves,
            ));
            done = true;
            break;
        }
        if budget > 0 && solves >= budget {
            capped = true;
            log(&format!("solve budget {budget} reached after round {round} — stopping (component not exhausted)"));
            break;
        }
    }
    if let Some(p) = cache_path {
        if let Err(e) = cache::save(p, cache) {
            log(&format!("final cache save failed: {e}"));
        }
    }
    log(&format!(
        "{} after {} solves, {:.1?} ({} cached)",
        if capped { "BUDGET-CAPPED" } else if done { "TARGET COMPLETE" } else { "component EXHAUSTED" },
        solves, start.elapsed(), cache.len(),
    ));

    // Validate and write the requested layers.
    let mut reports = Vec::new();
    let mut all_ok = true;
    for d in (down_to..=80).rev() {
        let total = store.count(d);
        let complete = total as u64 == hist[d as usize];
        all_ok &= complete;
        log(&format!(
            "depth {:>2}: {:>15} boards  N={}  {}",
            d, total, hist[d as usize], if complete { "OK" } else { "MISMATCH" },
        ));
        reports.push(LayerReport {
            depth: d,
            descent: 0,
            maxima_bellman: 0,
            maxima_fallback: 0,
            total,
            expected: hist[d as usize],
            complete,
        });
        if complete {
            let path = out_dir.join(format!("depth{d}.ranks"));
            store.write_layer(d, &path).map_err(|e| format!("{}: {}", path.display(), e))?;
        }
    }
    if !all_ok {
        log("note: a layer above down_to is still short — lower --floor to connect its maxima");
    }
    Ok(reports)
}

/// Run the frontier from the seeded antipodes (already in `store` at depth 80)
/// down to `down_to`, writing each complete layer to `out_dir/depthNN.ranks`.
/// Stops at the first incomplete layer (returns the reports gathered so far).
pub fn run(
    store: &mut Store,
    hist: &Histogram,
    down_to: u8,
    out_dir: &Path,
    verifier: Option<&dyn Fn(u64) -> u8>,
    mut log: impl FnMut(&str),
) -> Result<Vec<LayerReport>, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir.display(), e))?;

    // Seed layer 80 must match N(80).
    let seeded = store.count(80) as u64;
    if seeded != hist[80] {
        return Err(format!("seeded {} antipodes, expected N(80)={}", seeded, hist[80]));
    }
    let p80 = out_dir.join("depth80.ranks");
    store.write_layer(80, &p80).map_err(|e| format!("{}: {}", p80.display(), e))?;
    log(&format!("depth 80: {seeded:>15} boards (seeded, = N)"));

    let mut reports = Vec::new();
    for d in (down_to..=79).rev() {
        let rep = process_layer(store, d, hist[d as usize], verifier);
        log(&format!(
            "depth {:>2}: {:>15} boards  (descent {}, bellman-max {}, fallback {})  N={}  {}",
            rep.depth,
            rep.total,
            rep.descent,
            rep.maxima_bellman,
            rep.maxima_fallback,
            rep.expected,
            if rep.complete { "OK" } else { "INCOMPLETE" },
        ));
        let complete = rep.complete;
        reports.push(rep);
        if !complete {
            log(&format!(
                "stopping: layer {} incomplete ({} short of N={}); not writing it or descending further",
                d,
                hist[d as usize] - store.count(d) as u64,
                hist[d as usize],
            ));
            break;
        }
        let path = out_dir.join(format!("depth{d}.ranks"));
        store.write_layer(d, &path).map_err(|e| format!("{}: {}", path.display(), e))?;
    }
    Ok(reports)
}
