//! bandwd24 — coupled band Walking Distance, an admissible heuristic that is
//! genuinely 2-D without being a tile-subset pattern database.
//!
//! ## Why this can beat WD without a PDB
//!
//! `WD_row(s)` is exactly optimal for the "horizontal moves are free" relaxation
//! (free horizontal ⇒ only the tiles-per-row multiset + blank row carry cost ⇒
//! the WD_row transition system). So you cannot beat WD_row while keeping
//! horizontal free — any gain must come from charging that horizontal is *not*
//! free, i.e. from row↔column coupling. Admissibility of `V_LB + H_LB` needs
//! only that vertical ⊥ horizontal are disjoint, exhaustive move classes; it is
//! indifferent to *how* each bound is computed. So a 2-D heuristic = the
//! strongest bound you can compute on each move class from the full board.
//!
//! **Band-WD** does this with a *coarse* column coordinate. Type a tile by
//! `(goal_row, goal_col_BUCKET)` and track its `(row, col_bucket)` super-cell.
//! A single combined distance table then couples row-sorting and (coarse)
//! column-sorting: a move crossing a row boundary OR a bucket boundary costs 1;
//! a horizontal move *within* a bucket is free. This is a homomorphic
//! abstraction of the real puzzle (every real move ↦ an abstract move of cost
//! ≤ 1, real goal ↦ abstract goal), so `band-WD ≤ real distance` — admissible.
//!
//! Two provable anchors used below:
//!   * `band-WD ≥ WD_row + WD_bucket`  (project the abstract solution onto rows
//!     and onto buckets separately; every abstract move is a row-crossing XOR a
//!     bucket-crossing, so the two projected costs add). This makes
//!     `h = WD_row + WD_bucket` an admissible, consistent guide for the abstract
//!     A* — and its value at the start is a free lower bound on band-WD.
//!   * `band-WD ≤ real optimal` (admissibility).
//!
//! The open question band-WD answers empirically: is the coupling gain
//! (band-WD − (WD_row+WD_bucket)) bigger than the resolution lost by coarsening
//! columns (WD_col − WD_bucket)? I.e. **is band-WD > WD_full = WD_row+WD_col?**
//! If yes anywhere, we have a strictly better admissible heuristic than WD with
//! no PDB. R is the adversarial case (max-WD board); the random deep boards are
//! the population where WD is loose and coupling should pay.
//!
//! Buckets: B=3, columns {0,1}→0, {2}→1, {3,4}→2 (reflection-symmetric, sizes
//! 2,1,2). Super-cells sc = row*3 + bucket (0..15); types ty = goal_row*3 +
//! goal_bucket (0..15).
//!
//! Method: A* in the abstract space, bucket-queue by f = g + h, closed-on-pop.
//! `h` consistent ⇒ first goal pop is optimal band-WD. On hitting the state cap
//! we return the current pop-f as a proven lower bound (all f < that are
//! exhausted, goal not among them ⇒ band-WD ≥ f). Run from the repo root so
//! `data/wd24.bin` resolves.

use puzzle8::puzzle24::search::walking_distance::{
    build_full_table, load_dist_table, WdBuild, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use puzzle8::puzzle24::state::{State, GOAL, N_CELLS};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const W: usize = 5;
const NB: usize = 3; // number of column buckets
const NSC: usize = W * NB; // super-cells (row, bucket)
const NT: usize = W * NB; // tile types (goal_row, goal_bucket)
/// column -> bucket. {0,1}->0, {2}->1, {3,4}->2.
const BKT: [usize; W] = [0, 0, 1, 2, 2];
/// bucket -> number of columns in it.
const BSIZE: [usize; NB] = [2, 1, 2];

// ---------------------------------------------------------------------------
// Row-WD table reuse (same 63-bit pack as walking_distance.rs, which is private
// there). Used for both the row projection and the WD_full=WD_row+WD_col ref.
// ---------------------------------------------------------------------------

type Mat5 = [[u8; W]; W];

fn pack_row(m: &Mat5, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

/// Row-WD of a 5×5 contingency with the given blank axis-index.
fn wd_row(t: &WdTable, m: &Mat5, blank: u8) -> u8 {
    *t.get(&pack_row(m, blank)).expect("row/col-WD state must be reachable")
}

// ---------------------------------------------------------------------------
// Bucket-WD table: 1-D WD over NB unequal lanes (bucket b spans 5*BSIZE[b]
// cells). Tiles typed by goal-bucket. Admissible lower bound on the number of
// real horizontal moves that cross a bucket boundary (≤ WD_col). Tiny space.
// ---------------------------------------------------------------------------

type MatB = [[u8; NB]; NB];

fn pack_bkt(m: &MatB, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for b in 0..NB {
        for gb in 0..NB {
            k = (k << 4) | (m[b][gb] as u64);
        }
    }
    k
}

fn bkt_goal() -> (MatB, u8) {
    let mut m = [[0u8; NB]; NB];
    for b in 0..NB {
        m[b][b] = (5 * BSIZE[b]) as u8; // all goal-bucket-b tiles home in lane b
    }
    let blank_b = BKT[W - 1]; // GOAL blank at col 4 -> bucket 2
    m[blank_b][blank_b] -= 1; // blank occupies one cell of its lane
    (m, blank_b as u8)
}

fn build_bkt_table() -> WdTable {
    let (gm, gb) = bkt_goal();
    let mut table: WdTable = HashMap::with_capacity_and_hasher(1 << 16, WdBuild::default());
    let gkey = pack_bkt(&gm, gb);
    table.insert(gkey, 0);
    let mut frontier = vec![(gm, gb)];
    let mut depth = 0u8;
    while !frontier.is_empty() {
        let nd = depth + 1;
        let mut next = Vec::new();
        for (m, bb) in &frontier {
            let bb = *bb as usize;
            for nb in [bb.wrapping_sub(1), bb + 1] {
                if nb >= NB {
                    continue;
                }
                for gb in 0..NB {
                    if m[nb][gb] == 0 {
                        continue;
                    }
                    let mut m2 = *m;
                    m2[nb][gb] -= 1;
                    m2[bb][gb] += 1;
                    let key = pack_bkt(&m2, nb as u8);
                    if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key) {
                        e.insert(nd);
                        next.push((m2, nb as u8));
                    }
                }
            }
        }
        depth = nd;
        frontier = next;
    }
    table
}

fn wd_bkt(t: &WdTable, m: &MatB, blank: u8) -> u8 {
    *t.get(&pack_bkt(m, blank)).expect("bucket-WD state must be reachable")
}

// ---------------------------------------------------------------------------
// Abstract band state: contingency M[super-cell][type] + blank super-cell.
// Canonical u128 key: 24 tile-types (4 bits each, laid out super-cell by
// super-cell with known per-cell counts) + blank super-cell (4 bits) = 100 bits.
// ---------------------------------------------------------------------------

type BandM = [[u8; NT]; NSC];

fn sc_count(sc: usize, blank_sc: usize) -> usize {
    BSIZE[sc % NB] - if sc == blank_sc { 1 } else { 0 }
}

fn pack_band(m: &BandM, blank_sc: usize) -> u128 {
    let mut k: u128 = 0;
    for sc in 0..NSC {
        for ty in 0..NT {
            for _ in 0..m[sc][ty] {
                k = (k << 4) | (ty as u128);
            }
        }
    }
    (k << 4) | (blank_sc as u128)
}

fn unpack_band(key: u128) -> (BandM, usize) {
    let blank_sc = (key & 0xF) as usize;
    let rem = key >> 4;
    let mut tiles = [0usize; 24];
    for (i, tile) in tiles.iter_mut().enumerate() {
        *tile = ((rem >> (4 * (23 - i))) & 0xF) as usize;
    }
    let mut m = [[0u8; NT]; NSC];
    let mut idx = 0;
    for sc in 0..NSC {
        for _ in 0..sc_count(sc, blank_sc) {
            m[sc][tiles[idx]] += 1;
            idx += 1;
        }
    }
    (m, blank_sc)
}

/// Real board -> abstract band state.
fn board_to_band(s: &State) -> (BandM, usize) {
    let mut m = [[0u8; NT]; NSC];
    let mut blank_sc = 0;
    for pos in 0..N_CELLS {
        let tile = s.0[pos];
        let r = pos / W;
        let c = pos % W;
        let b = BKT[c];
        let sc = r * NB + b;
        if tile == 0 {
            blank_sc = sc;
            continue;
        }
        let gpos = (tile - 1) as usize;
        let gr = gpos / W;
        let gb = BKT[gpos % W];
        m[sc][gr * NB + gb] += 1;
    }
    (m, blank_sc)
}

fn band_goal() -> (BandM, usize) {
    board_to_band(&GOAL)
}

/// Project a band state onto rows (sum out buckets) -> 5×5 row contingency.
fn project_rows(m: &BandM) -> Mat5 {
    let mut mr = [[0u8; W]; W];
    for r in 0..W {
        for b in 0..NB {
            for gr in 0..W {
                for gb in 0..NB {
                    mr[r][gr] += m[r * NB + b][gr * NB + gb];
                }
            }
        }
    }
    mr
}

/// Project a band state onto buckets (sum out rows) -> NB×NB bucket contingency.
fn project_bkts(m: &BandM) -> MatB {
    let mut mb = [[0u8; NB]; NB];
    for r in 0..W {
        for b in 0..NB {
            for gr in 0..W {
                for gb in 0..NB {
                    mb[b][gb] += m[r * NB + b][gr * NB + gb];
                }
            }
        }
    }
    mb
}

/// h = WD_row(rows) + WD_bucket(buckets). Admissible & consistent for band-WD.
fn band_h(rowt: &WdTable, bktt: &WdTable, m: &BandM, blank_sc: usize) -> u8 {
    let mr = project_rows(m);
    let mb = project_bkts(m);
    wd_row(rowt, &mr, (blank_sc / NB) as u8) + wd_bkt(bktt, &mb, (blank_sc % NB) as u8)
}

struct BandResult {
    /// Some(v) = exact band-WD; None = capped before reaching goal.
    exact: Option<u8>,
    /// Proven lower bound (== exact when found; == pop-f at cap otherwise).
    lb: u8,
    closed: usize,
    pops: u64,
}

/// A* for band-WD from `start` to the abstract goal. Bucket-queue by f, closed
/// map key->best g (bit 7 = closed). Returns exact value or a proven LB at cap.
fn band_wd(
    rowt: &WdTable,
    bktt: &WdTable,
    start: &State,
    goal_key: u128,
    cap: usize,
) -> BandResult {
    const CLOSED: u8 = 0x80;
    let (m0, b0) = board_to_band(start);
    let h0 = band_h(rowt, bktt, &m0, b0);
    let start_key = pack_band(&m0, b0);

    let mut best: HashMap<u128, u8, WdBuild> =
        HashMap::with_capacity_and_hasher(1 << 20, WdBuild::default());
    best.insert(start_key, 0);
    let mut buckets: Vec<Vec<(u128, u8)>> = vec![Vec::new(); 200];
    buckets[h0 as usize].push((start_key, 0));

    let mut pops: u64 = 0;
    for f in (h0 as usize)..buckets.len() {
        let mut i = 0;
        while i < buckets[f].len() {
            let (key, g) = buckets[f][i];
            i += 1;
            let slot = best.get_mut(&key).expect("queued node has a slot");
            if *slot & CLOSED != 0 || (*slot & !CLOSED) != g {
                continue; // stale / superseded
            }
            *slot |= CLOSED;
            pops += 1;

            if key == goal_key {
                return BandResult { exact: Some(g), lb: g, closed: best.len(), pops };
            }
            if best.len() > cap {
                // All f' < f exhausted, goal not among them ⇒ band-WD ≥ f.
                return BandResult { exact: None, lb: f as u8, closed: best.len(), pops };
            }

            let (m, blank_sc) = unpack_band(key);
            let br = blank_sc / NB;
            let bb = blank_sc % NB;
            let mut neigh: [Option<usize>; 4] = [None; 4];
            if br > 0 { neigh[0] = Some((br - 1) * NB + bb); }
            if br < W - 1 { neigh[1] = Some((br + 1) * NB + bb); }
            if bb > 0 { neigh[2] = Some(br * NB + (bb - 1)); }
            if bb < NB - 1 { neigh[3] = Some(br * NB + (bb + 1)); }

            let g2 = g + 1;
            for nsc in neigh.into_iter().flatten() {
                for ty in 0..NT {
                    if m[nsc][ty] == 0 {
                        continue;
                    }
                    let mut m2 = m;
                    m2[nsc][ty] -= 1;
                    m2[blank_sc][ty] += 1;
                    let key2 = pack_band(&m2, nsc);
                    let slot = best.entry(key2).or_insert(0xFF); // 0xFF = unseen (has CLOSED bit)
                    if *slot == 0xFF || (*slot & CLOSED == 0 && g2 < *slot) {
                        *slot = g2;
                        let h2 = band_h(rowt, bktt, &m2, nsc);
                        buckets[g2 as usize + h2 as usize].push((key2, g2));
                    }
                }
            }
        }
    }
    // f-range exhausted without goal (should not happen: goal within 160).
    BandResult { exact: None, lb: buckets.len() as u8, closed: best.len(), pops }
}

/// WD_full = WD_row + WD_col, the current best admissible baseline.
fn wd_full(rowt: &WdTable, s: &State) -> (u8, u8) {
    let mut mr = [[0u8; W]; W];
    let mut mc = [[0u8; W]; W];
    let mut br = 0u8;
    let mut bc = 0u8;
    for pos in 0..N_CELLS {
        let tile = s.0[pos];
        let r = (pos / W) as u8;
        let c = (pos % W) as u8;
        if tile == 0 {
            br = r;
            bc = c;
            continue;
        }
        let gpos = (tile - 1) as usize;
        mr[r as usize][gpos / W] += 1;
        mc[c as usize][gpos % W] += 1;
    }
    (wd_row(rowt, &mr, br), wd_row(rowt, &mc, bc))
}

fn r_board() -> State {
    let mut cells = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        cells[i] = (25 - i) as u8;
    }
    State(cells)
}

/// A depth-`len` board from a xorshift random walk out of GOAL (no ml feature).
/// Avoids immediately undoing the previous move so the walk drifts outward
/// instead of dithering near GOAL.
fn random_board(seed: &mut u64, len: usize) -> State {
    let mut s = GOAL;
    let mut prev_blank = usize::MAX; // blank position before the last move
    for _ in 0..len {
        let blank = s.0.iter().position(|&t| t == 0).unwrap();
        let moves: Vec<_> = s.legal_moves().iter().collect();
        // Candidate moves that don't send the blank back where it just came from.
        let fwd: Vec<_> = moves
            .iter()
            .copied()
            .filter(|&m| {
                let ns = s.apply(m);
                let nb = ns.0.iter().position(|&t| t == 0).unwrap();
                nb != prev_blank
            })
            .collect();
        let pick = if fwd.is_empty() { &moves } else { &fwd };
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let m = pick[(*seed as usize) % pick.len()];
        prev_blank = blank;
        s = s.apply(m);
    }
    s
}

fn main() {
    let t0 = Instant::now();
    let path = Path::new("data/wd24.bin");
    let rowt = if path.exists() {
        eprintln!("loading row-WD table from {} …", path.display());
        load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).expect("wd24.bin load")
    } else {
        eprintln!("data/wd24.bin missing — building row-WD via BFS (~25 s) …");
        build_full_table()
    };
    let bktt = build_bkt_table();
    eprintln!(
        "row-WD {} entries; bucket-WD {} entries; {:.1}s",
        rowt.len(),
        bktt.len(),
        t0.elapsed().as_secs_f64()
    );

    let (gm, gb) = band_goal();
    let goal_key = pack_band(&gm, gb);

    // --- sanity ---
    assert_eq!(band_h(&rowt, &bktt, &gm, gb), 0, "h(goal) must be 0");
    {
        let r = band_wd(&rowt, &bktt, &GOAL, goal_key, 1_000_000);
        assert_eq!(r.exact, Some(0), "band-WD(goal) must be 0");
    }
    // pack/unpack round-trip on R and goal
    for s in [&GOAL, &r_board()] {
        let (m, b) = board_to_band(s);
        let (m2, b2) = unpack_band(pack_band(&m, b));
        assert_eq!((m, b), (m2, b2), "pack/unpack round-trip");
    }
    println!("sanity OK (h(goal)=0, band-WD(goal)=0, pack round-trips)");

    // Admissibility spot-check: band-WD ≤ walk length on random boards.
    {
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..40 {
            let len = 30;
            let s = random_board(&mut seed, len);
            let r = band_wd(&rowt, &bktt, &s, goal_key, 5_000_000);
            let v = r.exact.expect("shallow board should solve under cap");
            assert!(v as usize <= len, "band-WD {v} > walk length {len} — INADMISSIBLE");
            let (wr, _) = wd_full(&rowt, &s);
            let mb = project_bkts(&board_to_band(&s).0);
            let wbk = wd_bkt(&bktt, &mb, (board_to_band(&s).1 % NB) as u8);
            assert!(v >= wr + wbk, "band-WD {} < WD_row+WD_bucket {} — bound violated", v, wr + wbk);
        }
        println!("admissibility spot-check OK (40 shallow boards: band-WD ≤ dist, ≥ WD_row+WD_bkt)");
    }

    // --- R ---
    let r = r_board();
    let (wr, wc) = wd_full(&rowt, &r);
    println!("\n== R (double antipode) ==");
    println!("  WD_full(R)      = WD_row {} + WD_col {} = {}", wr, wc, wr + wc);
    let mb = project_bkts(&board_to_band(&r).0);
    let wbk = wd_bkt(&bktt, &mb, 0);
    println!("  WD_row+WD_bkt   = {} + {} = {}  (band-WD floor)", wr, wbk, wr + wbk);
    eprintln!("  running band-WD A* on R (cap 120M states)…");
    let tr = Instant::now();
    let res = band_wd(&rowt, &bktt, &r, goal_key, 120_000_000);
    match res.exact {
        Some(v) => println!(
            "  band-WD(R)      = {}   ({} vs WD_full {}: {:+})   [{} closed, {} pops, {:.1}s]",
            v, v, wr + wc, v as i32 - (wr + wc) as i32, res.closed, res.pops, tr.elapsed().as_secs_f64()
        ),
        None => println!(
            "  band-WD(R)     ≥ {}   (capped; {} vs WD_full {}: {:+})   [{} closed, {} pops, {:.1}s]",
            res.lb, res.lb, wr + wc, res.lb as i32 - (wr + wc) as i32, res.closed, res.pops, tr.elapsed().as_secs_f64()
        ),
    }

    // --- random deep boards: where WD is loose and coupling should pay ---
    println!("\n== random deep boards (walk length 120) ==");
    let mut seed = 0xDEAD_BEEF_CAFE_1234u64;
    let n = 50usize;
    let mut wins = 0usize; // band-WD > WD_full
    let mut ties = 0usize;
    let mut done = 0usize;
    let mut sum_band: i64 = 0;
    let mut sum_full: i64 = 0;
    let mut best_margin = i32::MIN;
    for i in 0..n {
        let s = random_board(&mut seed, 120);
        let (wr, wc) = wd_full(&rowt, &s);
        let full = wr + wc;
        let res = band_wd(&rowt, &bktt, &s, goal_key, 25_000_000);
        match res.exact {
            Some(v) => {
                done += 1;
                sum_band += v as i64;
                sum_full += full as i64;
                let margin = v as i32 - full as i32;
                if margin > 0 { wins += 1; }
                if margin == 0 { ties += 1; }
                if margin > best_margin { best_margin = margin; }
                if i < 8 || margin > 0 {
                    println!(
                        "  #{i:02}: band-WD {v}  WD_full {full} ({:+})   [{} closed]",
                        margin, res.closed
                    );
                }
            }
            None => {
                println!("  #{i:02}: band-WD ≥ {} (capped)  WD_full {full}   [{} closed]", res.lb, res.closed);
            }
        }
    }
    println!(
        "\nsummary: {}/{} solved under cap; band-WD > WD_full in {} ({:.0}%), tie {}, best margin {:+}",
        done, n, wins, 100.0 * wins as f64 / done.max(1) as f64, ties, best_margin
    );
    if done > 0 {
        println!(
            "  mean band-WD {:.1} vs mean WD_full {:.1} (Δ {:+.2}) over solved boards",
            sum_band as f64 / done as f64,
            sum_full as f64 / done as f64,
            (sum_band - sum_full) as f64 / done as f64
        );
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
}
