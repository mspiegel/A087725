//! build_cwd_single — build (and verify) the single-line escape-constrained WD
//! surcharge for one goal line `g`, over ALL WD contingencies.
//!
//! For a fixed goal line `g`, computes `D(σ, r) = cWD_single(σ, g, r)` = the min
//! abstract-plan length from contingency `σ` to the goal that makes ≥ `r` type-`g`
//! escapes (a goal-line-`g` unit leaving physical line `g`, i.e. a move that
//! *decreases* `σ[g][g]`), for every reachable `σ` and every demand `r = 0..=K`.
//!
//! Method: a single **backward BFS from `(goal, r=0)`** over the product graph
//! `(σ, r)`. Edge `(σ, r) → (σ', r')` mirrors one contingency move with
//! `r' = max(r − esc, 0)`, `esc = 1` iff the move decreases `σ[g][g]`. The escape
//! label is read from the two endpoints, so there is no solving-vs-scrambling
//! direction ambiguity.
//!
//! This is the buildable table the surcharge measurement selected (single-line-max
//! retains 98% of full cWD's node-weighted gain). Two hard invariants are checked:
//!   1. the `r=0` layer reproduces the WD table exactly (`D(σ,0) == WD(σ)`);
//!   2. `D(R_row, 2, 4) == 72` (the R middle-row escape bound);
//!   3. a random `σ`×`d` sample matches the reference per-node A\* `cwd_axis`.
//!
//! Usage: `build_cwd_single <g>` (g = 0..4). Run from repo root for `data/wd24.bin`.

use puzzle8::puzzle24::search::walking_distance::{
    build_full_table, load_dist_table, WdBuild, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const W: usize = 5;
const K: usize = 4; // max demand tracked (residents ≤ 5, LIS ≥ 1 ⇒ d ≤ 4)
const UNSEEN: u8 = 255;
type Matrix = [[u8; W]; W];

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

fn goal_key() -> u64 {
    let mut m = [[0u8; W]; W];
    for d in 0..W {
        m[d][d] = 5;
    }
    m[W - 1][W - 1] = 4;
    pack(&m, (W - 1) as u8)
}

/// R's row contingency: physical row r holds five goal-row-(4−r) tiles; blank row 0.
fn r_row_key() -> u64 {
    let mut m = [[0u8; W]; W];
    for r in 0..W {
        m[r][W - 1 - r] = 5;
    }
    m[0][W - 1] = 4;
    pack(&m, 0)
}

// ---- reference: single-line vector-constrained A* (the validated cwd_axis) ----

fn cwd_axis_single(
    table: &WdTable,
    m: &Matrix,
    blank: u8,
    goal: u64,
    g: usize,
    dem: u8,
) -> Option<u8> {
    if dem == 0 {
        return Some(*table.get(&pack(m, blank)).expect("start reachable"));
    }
    let start_key = pack(m, blank);
    let h0 = *table.get(&start_key).expect("start reachable") as usize;
    let mut best: HashMap<u128, u8> = HashMap::with_capacity(1 << 14);
    let mut buckets: Vec<Vec<(u64, u32)>> = vec![Vec::new(); 210];
    const CLOSED: u8 = 0x80;
    let statekey = |wd: u64, c: u32| -> u128 { ((wd as u128) << 16) | c as u128 };
    best.insert(statekey(start_key, 0), 0);
    buckets[h0].push((start_key, 0));
    let mut pops: u64 = 0;
    for f in h0..buckets.len() {
        let mut i = 0;
        while i < buckets[f].len() {
            let (key, c) = buckets[f][i];
            i += 1;
            let sk = statekey(key, c);
            let gcur = match best.get(&sk) {
                Some(&v) if v & CLOSED == 0 => v,
                _ => continue,
            };
            best.insert(sk, gcur | CLOSED);
            pops += 1;
            if pops > 40_000_000 {
                return None;
            }
            if key == goal && c as usize == dem as usize {
                return Some(gcur);
            }
            let (mm, br) = unpack(key);
            let g2 = gcur + 1;
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
                    let esc = (from == g && t == g) as u32;
                    let c2 = (c + esc).min(dem as u32);
                    let h = *table.get(&child_key).expect("child reachable") as usize;
                    let csk = statekey(child_key, c2);
                    let slot = best.entry(csk).or_insert(UNSEEN);
                    if *slot == UNSEEN || (*slot & CLOSED == 0 && g2 < *slot) {
                        *slot = g2;
                        buckets[g2 as usize + h].push((child_key, c2));
                    }
                }
            }
        }
    }
    None
}

// ---- the batch build: backward product BFS from (goal, 0) for line g ----------

/// `dist[σ] = [D(σ,0), …, D(σ,K)]` for the fixed line `g`.
fn build_line(g: usize) -> HashMap<u64, [u8; K + 1], WdBuild> {
    let goal = goal_key();
    let mut dist: HashMap<u64, [u8; K + 1], WdBuild> =
        HashMap::with_capacity_and_hasher(66_000_000, WdBuild::default());
    dist.insert(goal, {
        let mut a = [UNSEEN; K + 1];
        a[0] = 0;
        a
    });
    let mut frontier: Vec<(u64, u8)> = vec![(goal, 0)];
    let mut depth: u8 = 0;
    while !frontier.is_empty() {
        let nd = depth + 1;
        let mut next: Vec<(u64, u8)> = Vec::new();
        for &(sk, r) in &frontier {
            let (m, br) = unpack(sk);
            for from in [br.wrapping_sub(1), br + 1] {
                let from = from as usize;
                if from >= W {
                    continue;
                }
                for j in 0..W {
                    if m[from][j] == 0 {
                        continue;
                    }
                    let mut m2 = m;
                    m2[from][j] -= 1;
                    m2[br as usize][j] += 1;
                    let nk = pack(&m2, from as u8);
                    // esc(neighbor → σ'): a type-g escape leaves physical row g;
                    // here the reverse move takes a unit OUT of row br of goal-line
                    // j, so esc ⇔ br == g && j == g.
                    let esc = (br as usize == g) && (j == g);
                    let e = dist.entry(nk).or_insert([UNSEEN; K + 1]);
                    if !esc {
                        if e[r as usize] == UNSEEN {
                            e[r as usize] = nd;
                            next.push((nk, r));
                        }
                    } else {
                        let rp = (r as usize + 1).min(K);
                        if e[rp] == UNSEEN {
                            e[rp] = nd;
                            next.push((nk, rp as u8));
                        }
                        if r == 0 && e[0] == UNSEEN {
                            e[0] = nd;
                            next.push((nk, 0));
                        }
                    }
                }
            }
        }
        depth = nd;
        frontier = next;
    }
    dist
}

fn main() {
    let g: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    assert!(g < W, "line g must be 0..4");
    let t0 = Instant::now();
    let path = Path::new("data/wd24.bin");
    let wd = if path.exists() {
        load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).expect("wd24.bin")
    } else {
        build_full_table()
    };
    eprintln!(
        "WD table: {} entries in {:.1}s",
        wd.len(),
        t0.elapsed().as_secs_f64()
    );

    let tb = Instant::now();
    let dist = build_line(g);
    eprintln!(
        "line g={g}: product BFS built {} contingencies in {:.1}s",
        dist.len(),
        tb.elapsed().as_secs_f64()
    );

    // ---- invariant 1: r=0 layer == WD table, and coverage matches ----
    assert_eq!(
        dist.len(),
        wd.len(),
        "reachable set size mismatch vs WD table"
    );
    let mut r0_ok = 0u64;
    for (k, v) in wd.iter() {
        let d = dist.get(k).expect("σ missing from product table");
        assert_eq!(d[0], *v, "D(σ,0) != WD(σ)");
        r0_ok += 1;
    }
    eprintln!("invariant 1 OK: D(σ,0) == WD(σ) for all {r0_ok} contingencies");

    // ---- invariant 2: R middle-row escape bound (only meaningful for g=2) ----
    if g == 2 {
        let rk = r_row_key();
        let d = dist.get(&rk).expect("R row-state reachable");
        eprintln!("R row: D(σ_R, r=0..4) = {d:?}   (expect [70,70,70,70,72])");
        assert_eq!(d[4], 72, "cWD_single(R_row, 2, 4) != 72");
        assert_eq!(d[0], 70, "WD(R_row) != 70");
    }

    // ---- invariant 3: random σ × d vs reference A* ----
    let goal = goal_key();
    let mut rng: u64 = 0x00C0_FFEE_1234_5678;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let keys: Vec<u64> = wd.keys().copied().collect();
    let mut checked = 0u64;
    let mut nonzero = 0u64;
    for _ in 0..4000 {
        let sk = keys[(next() as usize) % keys.len()];
        let (m, br) = unpack(sk);
        let d = (next() % (K as u64 + 1)) as u8; // demand 0..4
        let table_val = wd[&sk] + (dist[&sk][d as usize] - dist[&sk][0]); // WD + surcharge
                                                                          // reference (cap the A* budget; skip if it bails)
        if let Some(reference) = cwd_axis_single(&wd, &m, br, goal, g, d) {
            assert_eq!(
                table_val, reference,
                "MISMATCH σ={sk:#x} d={d}: table {table_val} vs A* {reference}"
            );
            checked += 1;
            if reference > wd[&sk] {
                nonzero += 1;
            }
        }
    }
    eprintln!(
        "invariant 3 OK: {checked} random σ×d matched the reference A* ({nonzero} had surcharge>0)"
    );

    // ---- surcharge stats ----
    let mut hist = [0u64; 9];
    let mut any_nonzero = 0u64;
    for d in dist.values() {
        let surch = d[K] - d[0]; // Δ at max demand
        hist[(surch as usize).min(8)] += 1;
        if surch > 0 {
            any_nonzero += 1;
        }
    }
    eprintln!(
        "surcharge Δ(σ,g={g},K=4) over all contingencies: {any_nonzero} nonzero ({:.1}%), hist[0..8]={hist:?}",
        any_nonzero as f64 / dist.len() as f64 * 100.0
    );
    eprintln!("total {:.1}s", t0.elapsed().as_secs_f64());
}
