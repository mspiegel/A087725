//! cwd24 — escape-constrained Walking Distance (cWD) probe on the R board.
//!
//! Phase-2 experiment: can the row/column WD abstraction be *constrained* by
//! linear-conflict-style escape demands without double-counting — and does it
//! move the admissible bound on R above WD's 140?
//!
//! **Theory.** `WD_row(s)` is the exact optimum of the row-projected puzzle, so
//! it already lower-bounds ALL vertical moves of any real solution; adding LC's
//! +2-per-conflict on top of it is unsound (the abstract plan's slack moves may
//! be the very escapes LC charges). The sound composition is a SIDE CONSTRAINT:
//! any real solution's projection is an abstract row-plan containing, for each
//! goal row `g`, at least
//!
//!   x_g(s) = (# tiles of goal-row g resident in row g)
//!            − (longest subsequence of them in goal column order)
//!
//! "escape edges" — moves taking a type-`g` tile OUT of physical row `g`. Tiles
//! that never leave their goal row can never pass one another, so their relative
//! order is invariant; at least `x_g` residents must therefore exit (and, since
//! their goal is row `g`, later re-enter — the re-entry cost is internalized by
//! the abstract plan's occupancy). Hence
//!
//!   V ≥ cWD_row(s, Σ_g x_g),   cWD(σ, k) = min cost of an abstract row-plan
//!                                          σ → goal using ≥ k escape edges.
//!
//! `cWD ≥ WD` by construction, is admissible per move class, and row+col sum
//! stays admissible (disjoint move classes) — the "cycle detector" enters WD as
//! a constraint, not as an addend, so nothing is double-charged.
//!
//! **On R** (blank cell 0, tile `25−i` at cell `i`): the middle row is the five
//! goal-row-2 tiles fully reversed → `x_2 = 5 − 1 = 4`; no other row has
//! residents → row demand 4. The column projection is the identical abstract
//! problem (middle column reversed; same anti-diagonal contingency), so the
//! static sound bound is `2 · cWD(R_row, 4)` vs WD's `70 + 70 = 140`.
//!
//! This probe answers: **is cWD(R_row, 4) > 70**, i.e. do R's 70-optimal
//! abstract row-plans absorb 4 forced escapes for free?
//!
//! Method: A* over the product graph (WD-state, escapes-so-far saturating at
//! `cap`), `h` = exact unconstrained WD table lookup (consistent, so first pop
//! is optimal), bucket queue by `f`. Runs:
//!   (A) sharp escapes (`g = 2` exits only), cap 4 — the sound demand on R;
//!   (B) aggregate escapes (any `g`), cap 4 — the cheap per-node generalization;
//!   (C) sharp, cap 8 — the marginal-cost curve past the sound demand
//!       (diagnostic for future per-node use; only k ≤ 4 is sound on R itself).
//!
//! Output is written to stdout; run from the repo root so `data/wd24.bin`
//! resolves (falls back to the ~25 s BFS build).

use puzzle8::puzzle24::search::walking_distance::{
    build_full_table, load_dist_table, WdBuild, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const W: usize = 5;
type Matrix = [[u8; W]; W];

/// Same key layout as `walking_distance::pack` (private there): 3-bit blank
/// axis-index, then rows 0..5 × goal-cols 0..4, 3 bits each (last goal column
/// derivable from the blank-aware row margin).
fn pack(m: &Matrix, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

/// Inverse of [`pack`]: field (r,c) sits above field (r,c+1); blank in the top
/// 3 used bits. The omitted last goal-column is reconstructed from the margin
/// (5 tiles per row, 4 in the blank's row).
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

/// R's row contingency: row r holds five tiles of goal-row 4−r (four in row 0,
/// where the blank sits). R's *column* contingency is the identical matrix with
/// blank column 0 — asserted in `main`, so one probe serves both axes.
fn r_state() -> (Matrix, u8) {
    let mut m = [[0u8; W]; W];
    for r in 0..W {
        m[r][W - 1 - r] = 5;
    }
    m[0][W - 1] = 4;
    (m, 0)
}

/// GOAL contingency: identity with the 4-margin in the blank's row (row 4).
fn goal_state() -> (Matrix, u8) {
    let mut m = [[0u8; W]; W];
    for d in 0..W {
        m[d][d] = 5;
    }
    m[W - 1][W - 1] = 4;
    (m, (W - 1) as u8)
}

const UNSEEN: u8 = 0xFF;
const CLOSED: u8 = 0x80;
const MAX_CAP: usize = 8;

struct ProbeResult {
    /// dist[c] = min cost R→goal whose escape count saturates at exactly c
    /// (c < cap: exactly c escapes; c = cap: ≥ cap). `None` = not reached
    /// within the explored f-range.
    dist: Vec<Option<u8>>,
    /// If the pop budget ran out, all unreached targets are ≥ this f.
    truncated_at: Option<u8>,
    pops: u64,
    states: usize,
}

/// A* on the product graph (WD-state, saturating escape count). `special`
/// receives (physical row the tile leaves, its goal row) in real move
/// direction; `h` is the exact unconstrained WD — admissible for the
/// constrained problem and consistent, so the first pop of a node is optimal.
fn probe(
    table: &WdTable,
    start: (Matrix, u8),
    goal_key: u64,
    special: impl Fn(usize, usize) -> bool,
    cap: usize,
    pop_budget: u64,
    name: &str,
) -> ProbeResult {
    assert!(cap <= MAX_CAP);
    let t0 = Instant::now();
    let start_key = pack(&start.0, start.1);
    let h0 = *table.get(&start_key).expect("start must be reachable") as usize;

    // best[key][c] = g (bit 7 = closed). One slot array per WD-state.
    let mut best: HashMap<u64, [u8; MAX_CAP + 1], WdBuild> =
        HashMap::with_capacity_and_hasher(1 << 22, WdBuild::default());
    let mut buckets: Vec<Vec<(u64, u8)>> = vec![Vec::new(); 200];

    best.entry(start_key).or_insert([UNSEEN; MAX_CAP + 1])[0] = 0;
    buckets[h0].push((start_key, 0));

    let mut dist: Vec<Option<u8>> = vec![None; cap + 1];
    let mut pops: u64 = 0;
    let mut finish_f: Option<usize> = None;
    let mut truncated_at: Option<u8> = None;

    'outer: for f in h0..buckets.len() {
        if let Some(ff) = finish_f {
            if f > ff {
                break;
            }
        }
        if !buckets[f].is_empty() {
            eprintln!(
                "  [{name}] f={f}: bucket {} entries, {pops} pops, {} states, {:.1}s",
                buckets[f].len(),
                best.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        let mut i = 0;
        // Same-f pushes (h consistent ⇒ f' ≥ f) are appended and processed.
        while i < buckets[f].len() {
            let (key, c) = buckets[f][i];
            i += 1;
            let slot = &mut best.get_mut(&key).expect("queued node must have a slot")[c as usize];
            if *slot & CLOSED != 0 {
                continue; // stale duplicate
            }
            let g = *slot;
            *slot |= CLOSED;
            pops += 1;
            if pops > pop_budget {
                truncated_at = Some(f as u8);
                break 'outer;
            }

            if key == goal_key {
                if dist[c as usize].is_none() {
                    dist[c as usize] = Some(g);
                    eprintln!("  [{name}] goal @ escapes={c}: cost {g}");
                }
                if c as usize == cap {
                    // Finish this f-bucket so every dist ≤ f is finalized.
                    finish_f = Some(f);
                    continue;
                }
                // c < cap: still expand — a plan may pass through the goal
                // contingency and leave again to collect more escapes.
            }

            let (m, br) = unpack(key);
            let g2 = g + 1;
            for from in [br.wrapping_sub(1), br + 1] {
                let from = from as usize;
                if from >= W {
                    continue;
                }
                for t in 0..W {
                    if m[from][t] == 0 {
                        continue;
                    }
                    let mut m2 = m;
                    m2[from][t] -= 1;
                    m2[br as usize][t] += 1;
                    let child_key = pack(&m2, from as u8);
                    let c2 = (c as usize + special(from, t) as usize).min(cap) as u8;
                    let h = *table.get(&child_key).expect("child must be reachable") as usize;
                    let slot =
                        &mut best.entry(child_key).or_insert([UNSEEN; MAX_CAP + 1])[c2 as usize];
                    // UNSEEN has the CLOSED bit set, so one comparison covers
                    // "never seen" and "strictly better than open g".
                    if g2 < *slot & !CLOSED && *slot & CLOSED == 0 || *slot == UNSEEN {
                        *slot = g2;
                        buckets[g2 as usize + h].push((child_key, c2));
                    }
                }
            }
        }
    }

    ProbeResult {
        dist,
        truncated_at,
        pops,
        states: best.len(),
    }
}

/// D_k = min cost with ≥ k escapes = suffix-min of `dist` (a plan with more
/// escapes also meets demand k). Unreached slots count as "≥ truncation f".
fn demand_curve(r: &ProbeResult, cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    for k in 0..=cap {
        let mut best: Option<u8> = None;
        let mut all_known = true;
        for c in k..=cap {
            match r.dist[c] {
                Some(d) => best = Some(best.map_or(d, |b: u8| b.min(d))),
                None => all_known = false,
            }
        }
        let s = match (best, all_known, r.truncated_at) {
            (Some(d), _, _) => format!("D_{k} = {d}"),
            (None, false, Some(f)) => format!("D_{k} ≥ {f} (budget hit)"),
            (None, false, None) => format!("D_{k} = unreached (bug?)"),
            (None, true, _) => format!("D_{k} = ∞"),
        };
        out.push(s);
    }
    out
}

fn main() {
    let t0 = Instant::now();
    let path = Path::new("data/wd24.bin");
    let table = if path.exists() {
        eprintln!("loading WD table from {} …", path.display());
        load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).expect("wd24.bin load")
    } else {
        eprintln!("data/wd24.bin missing — building WD table via BFS (~25 s) …");
        build_full_table()
    };
    eprintln!(
        "WD table ready: {} entries in {:.1}s",
        table.len(),
        t0.elapsed().as_secs_f64()
    );

    let (gm, gb) = goal_state();
    let goal_key = pack(&gm, gb);
    let (rm, rb) = r_state();
    let r_key = pack(&rm, rb);

    assert_eq!(
        table.get(&goal_key),
        Some(&0),
        "goal contingency must be at distance 0"
    );
    let wd_r = *table.get(&r_key).expect("R row-state must be reachable");
    println!(
        "WD_row(R) = {wd_r}  (expect 70; WD(R) = {} total)",
        2 * wd_r
    );
    assert_eq!(wd_r, 70, "R row-WD drifted from the known 70");

    // pack/unpack round-trip sanity on both endpoint states.
    for &(m, b) in &[(rm, rb), (gm, gb)] {
        let (m2, b2) = unpack(pack(&m, b));
        assert_eq!((m, b), (m2, b2), "pack/unpack round-trip failed");
    }

    // Verify the column-side claim from first principles: build R's column
    // contingency tile-by-tile and check it equals the row contingency, so one
    // probe serves both axes. Tile t sits at cell 25−t: column (25−t) mod 5,
    // goal column (t−1) mod 5 — the indices sum to 4 (anti-diagonal), blank in
    // column 0.
    let mut m_col = [[0u8; W]; W];
    for t in 1..25usize {
        m_col[(25 - t) % W][(t - 1) % W] += 1;
    }
    assert_eq!(
        (m_col, 0u8),
        (rm, rb),
        "R column contingency must equal its row contingency"
    );

    // Run A — the sound demand on R: four type-2 tiles must exit row 2.
    println!("\n== A: sharp escapes (type-2 exits from row 2), cap 4 ==");
    let a = probe(
        &table,
        (rm, rb),
        goal_key,
        |from, t| from == 2 && t == 2,
        4,
        2_000_000_000,
        "A",
    );
    for line in demand_curve(&a, 4) {
        println!("  {line}");
    }
    println!("  ({} pops, {} states)", a.pops, a.states);

    // Run B — aggregate escapes (any goal row): the cheap generalization a
    // per-node table would use. Weaker constraint ⇒ D_k ≤ run A's D_k.
    println!("\n== B: aggregate escapes (any type-g exit from row g), cap 4 ==");
    let b = probe(
        &table,
        (rm, rb),
        goal_key,
        |from, t| from == t,
        4,
        2_000_000_000,
        "B",
    );
    for line in demand_curve(&b, 4) {
        println!("  {line}");
    }
    println!("  ({} pops, {} states)", b.pops, b.states);

    // Run C — marginal-cost curve past the sound demand (diagnostic only).
    println!("\n== C: sharp escapes, cap 8 (diagnostic curve; only k ≤ 4 sound on R) ==");
    let c = probe(
        &table,
        (rm, rb),
        goal_key,
        |from, t| from == 2 && t == 2,
        8,
        3_000_000_000,
        "C",
    );
    for line in demand_curve(&c, 8) {
        println!("  {line}");
    }
    println!("  ({} pops, {} states)", c.pops, c.states);

    // The static sound bound: row and column demands are on disjoint move
    // classes; R's column contingency equals its row contingency (both
    // anti-diagonal, blank axis 0), so the column probe is the same number.
    match (a.dist[4], a.truncated_at) {
        (Some(d4), _) => println!(
            "\nstatic sound bound on R:  cWD_row(R,4) + cWD_col(R,4) = {} + {} = {}   (WD gives 140)",
            d4, d4, 2 * d4 as u32
        ),
        (None, Some(f)) => println!("\nstatic sound bound on R:  ≥ 2·{f} (probe truncated)"),
        (None, None) => println!("\nstatic sound bound: probe failed to reach (goal, 4)"),
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
}
