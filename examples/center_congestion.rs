//! center_congestion — test the hypothesis that congestion near the board
//! CENTER is more expensive (in true distance) than the same congestion at the
//! edges, i.e. that a location-aware second-order term on top of cWD has
//! headroom.
//!
//! Two experiments:
//!
//! **`motifs`** (controlled): place an identical within-line congestion motif
//! (3-cycle rotation, or a double transposition) in each row r=0..4 and each
//! column c=0..4, blank at its goal corner (4,4), everything else solved. By
//! construction MD / WD / cWD are *placement-invariant* across the row (resp.
//! column) placements: the tiles stay in their goal line, the line-multiset
//! projections and the per-line LIS escape demands are identical. So any
//! variation in the exact distance d\* across placements is pure location
//! economics. Discriminator: with the blank at the corner, pure blank-travel
//! predicts d\* monotone in the line's distance from row/col 4; the
//! center-congestion hypothesis predicts a bump at line 2 on top of that.
//!
//! **`walks N L SEED`** (observational, deep boards): N random-walk boards of
//! length L, each solved exactly with cWD; emits one TSV row per board with
//! d\*, cWD, slack = d\* − cWD, and congestion-location features (per-line
//! escape demands weighted by line centrality, misplaced-tile center mass), for
//! offline regression of slack against center-congestion share.
//!
//! Run from repo root (needs `data/wd24.bin` for a fast cWD load):
//!   cargo run --release --features mmap --example center_congestion -- motifs
//!   cargo run --release --features mmap --example center_congestion -- walks 150 70 1

use std::time::Instant;

use std::collections::HashMap;

use puzzle8::puzzle24::search::walking_distance::{
    load_dist_table, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use puzzle8::puzzle24::search::{idastar_inc_mut_with_stats, Cwd, Heuristic};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS, W};

type Matrix = [[u8; W]; W];

// ---- features ----------------------------------------------------------------

fn manhattan(s: &State) -> u32 {
    let mut md = 0u32;
    for pos in 0..N_CELLS {
        let t = s.0[pos];
        if t == 0 {
            continue;
        }
        let g = (t - 1) as usize;
        md += ((pos / W).abs_diff(g / W) + (pos % W).abs_diff(g % W)) as u32;
    }
    md
}

/// Longest strictly-increasing subsequence (k ≤ 5). Mirrors cwd.rs.
fn lis_strict(seq: &[u8]) -> usize {
    let n = seq.len();
    let mut dp = [0usize; W];
    let mut best = 0;
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

/// Per-line forced-escape demands, both axes — the cWD constraint vector.
/// `dem_row[g]` = (goal-row-`g` residents physically in row `g`) − LIS of their
/// goal columns in physical order. Mirrors cwd.rs `project`.
fn escape_demands(s: &State) -> ([u8; W], [u8; W]) {
    let mut row_res: [Vec<u8>; W] = Default::default();
    let mut col_res: [Vec<u8>; W] = Default::default();
    for pos in 0..N_CELLS {
        let t = s.0[pos];
        if t == 0 {
            continue;
        }
        let (r, c) = (pos / W, pos % W);
        let g = (t - 1) as usize;
        let (gr, gc) = (g / W, g % W);
        if gr == r {
            row_res[r].push(gc as u8);
        }
        if gc == c {
            col_res[c].push(gr as u8);
        }
    }
    let mut dr = [0u8; W];
    let mut dc = [0u8; W];
    for g in 0..W {
        dr[g] = (row_res[g].len() - lis_strict(&row_res[g])) as u8;
        dc[g] = (col_res[g].len() - lis_strict(&col_res[g])) as u8;
    }
    (dr, dc)
}

/// Vertical/horizontal WD flow per line boundary: `flow[b]` = number of tiles
/// whose current→goal line interval spans the boundary between lines `b` and
/// `b+1` (each such tile must cross it at least once — the WD flow bound).
/// Boundaries 1 and 2 are the "center traffic" ones.
fn boundary_flow(s: &State) -> ([u32; W - 1], [u32; W - 1]) {
    let mut fr = [0u32; W - 1];
    let mut fc = [0u32; W - 1];
    for pos in 0..N_CELLS {
        let t = s.0[pos];
        if t == 0 {
            continue;
        }
        let (r, c) = (pos / W, pos % W);
        let g = (t - 1) as usize;
        let (gr, gc) = (g / W, g % W);
        for b in r.min(gr)..r.max(gr) {
            fr[b] += 1;
        }
        for b in c.min(gc)..c.max(gc) {
            fc[b] += 1;
        }
    }
    (fr, fc)
}

/// Line centrality weight: 2 at the center line, 1 at the middle ring, 0 at edges.
fn line_wt(g: usize) -> u32 {
    2 - (g as i32 - 2).unsigned_abs()
}

/// Misplaced-tile center mass: Σ over out-of-place tiles of the cell's
/// centrality (row weight + col weight, 0..=4), plus the plain misplaced count.
fn misplaced_center_mass(s: &State) -> (u32, u32) {
    let mut mass = 0u32;
    let mut cnt = 0u32;
    for pos in 0..N_CELLS {
        let t = s.0[pos];
        if t == 0 || (t - 1) as usize == pos {
            continue;
        }
        cnt += 1;
        mass += line_wt(pos / W) + line_wt(pos % W);
    }
    (mass, cnt)
}

// ---- exact solve ---------------------------------------------------------------

fn solve_exact(cwd: &Cwd, s: &State) -> (u32, u8, f64) {
    let h0 = cwd.h(s);
    let t = Instant::now();
    let (sol, _st) = idastar_inc_mut_with_stats(s, cwd);
    let d = sol.expect("solvable by construction").len() as u32;
    (d, h0, t.elapsed().as_secs_f64())
}

// ---- experiment 1: controlled motifs -------------------------------------------

/// Apply a permutation cycle to GOAL cells: cells[i] receives the tile that
/// GOAL has at cells[(i+1) % len] — a single k-cycle (even iff k is odd).
fn cycled(cells: &[usize]) -> State {
    let mut s = GOAL;
    let n = cells.len();
    for i in 0..n {
        s.0[cells[i]] = GOAL.0[cells[(i + 1) % n]];
    }
    s
}

/// Double transposition (even): swap tiles at (a,b) and at (c,d).
fn swapped2(a: usize, b: usize, c: usize, d: usize) -> State {
    let mut s = GOAL;
    s.0.swap(a, b);
    s.0.swap(c, d);
    s
}

fn run_motifs(cwd: &Cwd) {
    println!("== motif scan: identical motif, varying line; blank at (4,4) ==");
    println!("{:<28}{:>5}{:>5}{:>5}{:>7}{:>9}", "board", "d*", "MD", "cWD", "slack", "sec");

    let show = |label: &str, s: &State| {
        let (d, h0, secs) = solve_exact(cwd, s);
        let md = manhattan(s);
        println!(
            "{:<28}{:>5}{:>5}{:>5}{:>7}{:>9.2}",
            label,
            d,
            md,
            h0,
            d as i64 - h0 as i64,
            secs
        );
    };

    // rot3 in row r, columns 1..=3 (fits every row; row 4 cells 21..23, no blank).
    for r in 0..W {
        let base = r * W;
        let cells = [base + 1, base + 2, base + 3];
        show(&format!("rot3 row {r} (cols 1-3)"), &cycled(&cells));
    }
    // rot3 in col c, rows 1..=3 (col 4 rows 1..3 = cells 9,14,19, no blank).
    for c in 0..W {
        let cells = [W + c, 2 * W + c, 3 * W + c];
        show(&format!("rot3 col {c} (rows 1-3)"), &cycled(&cells));
    }
    // swap2 in row r: swap (cols 0,1) and (cols 2,3) — fits every row.
    for r in 0..W {
        let b = r * W;
        show(&format!("swap2 row {r} (01)(23)"), &swapped2(b, b + 1, b + 2, b + 3));
    }
    // swap2 in col c: swap (rows 0,1) and (rows 2,3).
    for c in 0..W {
        show(&format!("swap2 col {c} (01)(23)"), &swapped2(c, W + c, 2 * W + c, 3 * W + c));
    }
    // Composite: rot3(cols 1-3) in TWO rows — center-adjacent pair vs edge pair.
    // Travel-cost predicts {1,3} cheaper than {0,4} (nearer the corner blank on
    // average is {1,3}... rows 1,3 vs 0,4: mean distance from row 4 is 2.5 vs 3);
    // center-congestion predicts {1,3} dearer. Opposing predictions.
    for pair in [[0usize, 4], [1, 3], [0, 2], [2, 4], [0, 1], [3, 4], [1, 2], [2, 3]] {
        let [a, b] = pair;
        let mut s = cycled(&[a * W + 1, a * W + 2, a * W + 3]);
        let t = cycled(&[b * W + 1, b * W + 2, b * W + 3]);
        for i in 0..W {
            s.0[b * W + i] = t.0[b * W + i];
        }
        show(&format!("rot3 rows {{{a},{b}}}"), &s);
    }
}

// ---- experiment 1b: joint (A*-vector) cWD vs production single-line-max ---------

/// Re-evaluate the composite pair boards with BOTH cWD paths: the production
/// merged table (surcharge = max over per-line curves) and the reference
/// constrained A\* (joint per-line demand VECTOR). If the joint path scores
/// higher on multi-site boards, the fast path's single-line-max is leaving
/// admissible surcharge on the table.
fn run_pairjoint() {
    eprintln!("loading production (merged single-line-max) cWD…");
    let fast = Cwd::new();
    eprintln!("loading reference (joint-vector A*) cWD…");
    let joint = Cwd::with_table_path(std::path::Path::new("data/wd24.bin"));
    assert!(fast.has_overlay(), "need data/cwd_single.bin for the comparison");
    assert!(!joint.has_overlay());
    println!("{:<24}{:>10}{:>10}{:>6}", "board", "cWD_fast", "cWD_joint", "diff");
    let show = |label: &str, s: &State| {
        let hf = fast.h(s);
        let hj = joint.h(s);
        println!("{:<24}{:>10}{:>10}{:>6}", label, hf, hj, hj as i32 - hf as i32);
    };
    for r in 0..W {
        let base = r * W;
        show(&format!("rot3 row {r}"), &cycled(&[base + 1, base + 2, base + 3]));
    }
    for pair in [[0usize, 4], [1, 3], [0, 2], [2, 4], [0, 1], [3, 4], [1, 2], [2, 3]] {
        let [a, b] = pair;
        let mut s = cycled(&[a * W + 1, a * W + 2, a * W + 3]);
        let t = cycled(&[b * W + 1, b * W + 2, b * W + 3]);
        for i in 0..W {
            s.0[b * W + i] = t.0[b * W + i];
        }
        show(&format!("rot3 rows {{{a},{b}}}"), &s);
    }
    for r in 0..W {
        let b = r * W;
        show(&format!("swap2 row {r}"), &swapped2(b, b + 1, b + 2, b + 3));
    }
    // Triple: rot3 in rows 0, 2, 4 — three separated demand sites.
    {
        let mut s = GOAL;
        for a in [0usize, 2, 4] {
            let t = cycled(&[a * W + 1, a * W + 2, a * W + 3]);
            for i in 0..W {
                s.0[a * W + i] = t.0[a * W + i];
            }
        }
        show("rot3 rows {0,2,4}", &s);
    }
    // Deep boards: R (the 180° rotation) and the three §8e catalog boards.
    {
        let mut r = [0u8; N_CELLS];
        for i in 1..N_CELLS {
            r[i] = (25 - i) as u8;
        }
        show("R", &State(r));
    }
    const DEEP: [(&str, &str); 3] = [
        ("B1 (g2)", "19 24 23 22 21 20 18 13 17 16 14 15 0 12 11 10 9 8 7 6 5 4 3 2 1"),
        ("B2 (g3)", "20 24 23 22 21 19 0 18 17 16 15 13 14 12 11 10 9 8 7 6 5 3 4 2 1"),
        ("B3 (g4)", "0 24 23 22 21 15 20 18 17 11 14 19 13 12 16 10 9 8 7 6 5 4 3 2 1"),
    ];
    for (label, txt) in DEEP {
        let mut cells = [0u8; N_CELLS];
        for (i, tok) in txt.split_whitespace().enumerate() {
            cells[i] = tok.parse().unwrap();
        }
        show(label, &State(cells));
    }
}

// ---- WD-key codec + vector-constrained A* (mirrors cwd_gap.rs) -------------------

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

/// Project both axes: contingency matrices + per-line escape demand vectors.
fn project(s: &State) -> (Matrix, u8, [u8; W], Matrix, u8, [u8; W]) {
    let mut m_row = [[0u8; W]; W];
    let mut m_col = [[0u8; W]; W];
    let (mut br, mut bc) = (0u8, 0u8);
    let mut row_res: [Vec<u8>; W] = Default::default();
    let mut col_res: [Vec<u8>; W] = Default::default();
    for pos in 0..N_CELLS {
        let tile = s.0[pos];
        let (r, c) = ((pos / W) as u8, (pos % W) as u8);
        if tile == 0 {
            br = r;
            bc = c;
            continue;
        }
        let g = (tile - 1) as usize;
        let (gr, gc) = ((g / W) as u8, (g % W) as u8);
        m_row[r as usize][gr as usize] += 1;
        m_col[c as usize][gc as usize] += 1;
        if gr == r {
            row_res[r as usize].push(gc);
        }
        if gc == c {
            col_res[c as usize].push(gr);
        }
    }
    let (mut dr, mut dc) = ([0u8; W], [0u8; W]);
    for g in 0..W {
        dr[g] = (row_res[g].len() - lis_strict(&row_res[g])) as u8;
        dc[g] = (col_res[g].len() - lis_strict(&col_res[g])) as u8;
    }
    (m_row, br, dr, m_col, bc, dc)
}

/// Min cost of an abstract plan `start → goal` making ≥ dem[g] escapes of each
/// goal-line g. Product state = (WD key, saturating per-line counters). h =
/// plain WD (consistent). `None` = pop budget exhausted (caller falls back).
fn cwd_axis(table: &WdTable, m: &Matrix, blank: u8, goal: u64, dem: &[u8; W]) -> Option<u8> {
    let lines: Vec<usize> = (0..W).filter(|&g| dem[g] > 0).collect();
    if lines.is_empty() {
        return Some(*table.get(&pack(m, blank)).expect("start reachable"));
    }
    let mut radix = [1u32; W];
    let mut full: u32 = 1;
    for (i, &g) in lines.iter().enumerate() {
        radix[i] = full;
        full *= dem[g] as u32 + 1;
    }
    let full_index = full - 1;
    let counter_of = |g: usize| -> Option<usize> { lines.iter().position(|&x| x == g) };

    let start_key = pack(m, blank);
    let h0 = *table.get(&start_key).expect("start reachable") as usize;
    let mut best: HashMap<u128, u8> = HashMap::with_capacity(1 << 14);
    let mut buckets: Vec<Vec<(u64, u32)>> = vec![Vec::new(); 210];
    const UNSEEN: u8 = 0xFF;
    const CLOSED: u8 = 0x80;
    let statekey = |wd: u64, ci: u32| -> u128 { ((wd as u128) << 16) | ci as u128 };
    best.insert(statekey(start_key, 0), 0);
    buckets[h0].push((start_key, 0));
    let mut pops: u64 = 0;
    const POP_BUDGET: u64 = 40_000_000;

    for f in h0..buckets.len() {
        let mut i = 0;
        while i < buckets[f].len() {
            let (key, ci) = buckets[f][i];
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
                        buckets[g2 as usize + h].push((child_key, ci2));
                    }
                }
            }
        }
    }
    None
}

/// One axis, three tiers: (single-line-max, pair-max, full-joint), each ≥ WD.
fn axis_tiers(table: &WdTable, m: &Matrix, blank: u8, goal: u64, dem: &[u8; W]) -> (u8, u8, u8) {
    let wd0 = *table.get(&pack(m, blank)).expect("start reachable");
    let lines: Vec<usize> = (0..W).filter(|&g| dem[g] > 0).collect();
    if lines.is_empty() {
        return (wd0, wd0, wd0);
    }
    let mut single = wd0;
    for &g in &lines {
        let mut d = [0u8; W];
        d[g] = dem[g];
        single = single.max(cwd_axis(table, m, blank, goal, &d).unwrap_or(wd0));
    }
    let mut pair = single;
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            let mut d = [0u8; W];
            d[lines[i]] = dem[lines[i]];
            d[lines[j]] = dem[lines[j]];
            pair = pair.max(cwd_axis(table, m, blank, goal, &d).unwrap_or(wd0));
        }
    }
    let full = if lines.len() <= 2 {
        pair
    } else {
        pair.max(cwd_axis(table, m, blank, goal, dem).unwrap_or(wd0))
    };
    (single, pair, full)
}

// ---- experiment 1c: joint-vs-fast gap over R's search interior -------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn r_board() -> State {
    let mut a = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        a[i] = (25 - i) as u8;
    }
    State(a)
}

struct InteriorSampler<'a> {
    fast: &'a Cwd,
    threshold: u16,
    cap: u64,
    expansions: u64,
    rng: Rng,
    seen: u64,
    reservoir: Vec<(State, u16, u8)>, // (state, depth, fast h)
    size: usize,
}

impl InteriorSampler<'_> {
    fn dfs(&mut self, s: &State, g: u16, prev_inv: Option<Move>, h_fast: u8) {
        if self.expansions >= self.cap {
            return;
        }
        self.expansions += 1;
        self.seen += 1;
        if self.reservoir.len() < self.size {
            self.reservoir.push((*s, g, h_fast));
        } else {
            let j = (self.rng.next() % self.seen) as usize;
            if j < self.size {
                self.reservoir[j] = (*s, g, h_fast);
            }
        }
        let mut cands: Vec<Move> =
            s.legal_moves().iter().filter(|&m| Some(m) != prev_inv).collect();
        for k in (1..cands.len()).rev() {
            let j = (self.rng.next() as usize) % (k + 1);
            cands.swap(k, j);
        }
        for m in cands {
            let child = s.apply(m);
            let ch = self.fast.h(&child);
            if g + 1 + ch as u16 <= self.threshold {
                self.dfs(&child, g + 1, Some(m.inverse()), ch);
                if self.expansions >= self.cap {
                    return;
                }
            }
        }
    }
}

/// Sample R's threshold-`thr` production search tree (pruned by the FAST cWD —
/// the tree the real solver walks) and measure, per sampled node, the surcharge
/// the joint-vector A\* path would add: diff = cWD_joint − cWD_fast ≥ 0.
fn run_interior(thr: u16, cap: u64, size: usize, seed: u64) {
    eprintln!("loading production (merged) cWD…");
    let fast = Cwd::new();
    assert!(fast.has_overlay(), "need data/cwd_single.bin");
    eprintln!("loading raw WD table for the vector A*…");
    let table = load_dist_table(
        std::path::Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("data/wd24.bin");
    let goal = goal_key();
    let r = r_board();
    let h0 = fast.h(&r);
    eprintln!("R: cWD_fast root = {h0}; sampling thr={thr} cap={cap} reservoir={size} seed={seed}");
    let mut smp = InteriorSampler {
        fast: &fast,
        threshold: thr,
        cap,
        expansions: 0,
        rng: Rng(seed | 1),
        seen: 0,
        reservoir: Vec::with_capacity(size),
        size,
    };
    let t = Instant::now();
    smp.dfs(&r, 0, None, h0);
    eprintln!(
        "sampled {} of {} expansions in {:.1}s; evaluating joint A* on reservoir…",
        smp.reservoir.len(),
        smp.expansions,
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    // per node: (single, pair, full) summed over the two axes, diffed vs fast h.
    // by-depth buckets (width 8): (n, Σ single−fast, Σ pair−fast, Σ full−fast)
    let mut by_depth: std::collections::BTreeMap<u16, (u64, i64, i64, i64)> = Default::default();
    let mut pair_hist = [0u64; 8];
    let mut full_hist = [0u64; 8];
    let (mut sum_s, mut sum_p, mut sum_f) = (0i64, 0i64, 0i64);
    for &(s, g, hf) in &smp.reservoir {
        let (mr, br, dr, mc, bc, dc) = project(&s);
        let (rs, rp, rf) = axis_tiers(&table, &mr, br, goal, &dr);
        let (cs, cp, cf) = axis_tiers(&table, &mc, bc, goal, &dc);
        let ds = (rs + cs) as i64 - hf as i64;
        let dp = (rp + cp) as i64 - hf as i64;
        let df = (rf + cf) as i64 - hf as i64;
        sum_s += ds;
        sum_p += dp;
        sum_f += df;
        pair_hist[(dp.max(0) as usize).min(7)] += 1;
        full_hist[(df.max(0) as usize).min(7)] += 1;
        let e = by_depth.entry(g / 8).or_insert((0, 0, 0, 0));
        e.0 += 1;
        e.1 += ds;
        e.2 += dp;
        e.3 += df;
    }
    let n = smp.reservoir.len() as f64;
    println!(
        "tiers−fast over R thr-{thr} interior ({} samples, A* eval {:.1}s):",
        smp.reservoir.len(),
        t.elapsed().as_secs_f64()
    );
    println!(
        "  mean single−fast = {:.4} (sanity: ≈0)\n  mean pair−fast   = {:.4}\n  mean full−fast   = {:.4}",
        sum_s as f64 / n,
        sum_p as f64 / n,
        sum_f as f64 / n
    );
    for (label, hist) in [("pair", &pair_hist), ("full", &full_hist)] {
        let line: Vec<String> = hist
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(d, &c)| format!("{}:{:.2}%", d, c as f64 / n * 100.0))
            .collect();
        println!("  {label}−fast hist: {}", line.join("  "));
    }
    println!("  by depth bucket (×8): n / single / pair / full");
    for (b, (cnt, ss, sp, sf)) in &by_depth {
        println!(
            "    depth {:>3}-{:<3} n={:<6} {:+.4} {:+.4} {:+.4}",
            b * 8,
            b * 8 + 7,
            cnt,
            *ss as f64 / *cnt as f64,
            *sp as f64 / *cnt as f64,
            *sf as f64 / *cnt as f64
        );
    }
}

// ---- experiment 1d: real A/B at thr — fast cWD vs fast + memoized pair bonus -----

/// Pack a demand vector (entries ≤ 4) into 15 bits for the memo key.
fn pack_dem(dem: &[u8; W]) -> u64 {
    let mut k = 0u64;
    for g in 0..W {
        k = (k << 3) | dem[g] as u64;
    }
    k
}

struct PairBonus<'a> {
    table: &'a WdTable,
    goal: u64,
    /// (axis key << 15 | packed demands) → pair-max − single-max (the bonus).
    memo: HashMap<u64, u8>,
    misses: u64,
    hits: u64,
}

impl PairBonus<'_> {
    /// Bonus for one axis; 0 unless ≥ 2 lines are demanded.
    fn bonus(&mut self, m: &Matrix, blank: u8, dem: &[u8; W]) -> u8 {
        let lines: Vec<usize> = (0..W).filter(|&g| dem[g] > 0).collect();
        if lines.len() < 2 {
            return 0;
        }
        let key = pack(m, blank);
        let mk = (key << 15) | pack_dem(dem);
        if let Some(&b) = self.memo.get(&mk) {
            self.hits += 1;
            return b;
        }
        self.misses += 1;
        let wd0 = *self.table.get(&key).expect("reachable");
        let mut single = wd0;
        for &g in &lines {
            let mut d = [0u8; W];
            d[g] = dem[g];
            single = single.max(cwd_axis(self.table, m, blank, self.goal, &d).unwrap_or(wd0));
        }
        let mut pair = single;
        for i in 0..lines.len() {
            for j in (i + 1)..lines.len() {
                let mut d = [0u8; W];
                d[lines[i]] = dem[lines[i]];
                d[lines[j]] = dem[lines[j]];
                pair = pair.max(cwd_axis(self.table, m, blank, self.goal, &d).unwrap_or(wd0));
            }
        }
        let b = pair - single;
        self.memo.insert(mk, b);
        b
    }
}

struct AbRun<'a> {
    fast: &'a Cwd,
    pair: Option<PairBonus<'a>>,
    threshold: u16,
    nodes: u64,
}

impl AbRun<'_> {
    fn h(&mut self, s: &State) -> u16 {
        let hf = self.fast.h(s) as u16;
        match &mut self.pair {
            None => hf,
            Some(pb) => {
                let (mr, br, dr, mc, bc, dc) = project(s);
                hf + pb.bonus(&mr, br, &dr) as u16 + pb.bonus(&mc, bc, &dc) as u16
            }
        }
    }

    fn dfs(&mut self, s: &State, g: u16, prev_inv: Option<Move>) {
        self.nodes += 1;
        for m in s.legal_moves().iter() {
            if Some(m) == prev_inv {
                continue;
            }
            let child = s.apply(m);
            let ch = self.h(&child);
            if g + 1 + ch <= self.threshold {
                self.dfs(&child, g + 1, Some(m.inverse()));
            }
        }
    }
}

/// Exhaust R at `thr` twice — fast cWD vs fast + memoized pair bonus — and report
/// the REAL node reduction plus the memo-cache economics (distinct (key, dem)
/// pairs touched, hit rate, wall).
fn run_ab(thr: u16) {
    eprintln!("loading production (merged) cWD…");
    let fast = Cwd::new();
    assert!(fast.has_overlay(), "need data/cwd_single.bin");
    eprintln!("loading raw WD table…");
    let table = load_dist_table(
        std::path::Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("data/wd24.bin");
    let goal = goal_key();
    let r = r_board();
    let h0 = fast.h(&r) as u16;
    assert!(thr >= h0, "thr {thr} < root h {h0} — empty tree");

    let mut a = AbRun { fast: &fast, pair: None, threshold: thr, nodes: 0 };
    let t = Instant::now();
    a.dfs(&r, 0, None);
    let ta = t.elapsed().as_secs_f64();
    println!("A (fast cWD)      thr {thr}: {} nodes in {:.1}s", a.nodes, ta);

    let mut b = AbRun {
        fast: &fast,
        pair: Some(PairBonus { table: &table, goal, memo: HashMap::new(), misses: 0, hits: 0 }),
        threshold: thr,
        nodes: 0,
    };
    let t = Instant::now();
    b.dfs(&r, 0, None);
    let tb = t.elapsed().as_secs_f64();
    let pb = b.pair.as_ref().unwrap();
    println!("B (fast+pairmemo) thr {thr}: {} nodes in {:.1}s", b.nodes, tb);
    println!(
        "node reduction {:.3}x | wall {:.2}x | memo: {} distinct (key,dem), {} misses, {:.2}% hit rate",
        a.nodes as f64 / b.nodes as f64,
        tb / ta,
        pb.memo.len(),
        pb.misses,
        pb.hits as f64 / (pb.hits + pb.misses).max(1) as f64 * 100.0
    );
}

// ---- experiment 2: random-walk deep boards --------------------------------------

/// No-immediate-undo xorshift random walk of `len` steps from GOAL.
fn random_walk(seed: u64, len: u32) -> State {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = GOAL;
    let mut last: Option<Move> = None;
    for _ in 0..len {
        let opts: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| last.map_or(true, |p: Move| m != p.inverse()))
            .collect();
        let m = opts[(next() % opts.len() as u64) as usize];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

fn run_walks(cwd: &Cwd, n: u64, len: u32, seed0: u64) {
    println!(
        "board\tseed\tlen\td\tcwd\tslack\tmd\tdem_tot\tdem_ctr_w\tdem_row\tdem_col\tmis_cnt\tmis_mass\tflow_row\tflow_col\tblank_r\tblank_c\tsec"
    );
    for i in 0..n {
        let seed = seed0.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let s = random_walk(seed, len);
        let (d, h0, secs) = solve_exact(cwd, &s);
        let (dr, dc) = escape_demands(&s);
        let dem_tot: u32 = dr.iter().chain(dc.iter()).map(|&x| x as u32).sum();
        let dem_ctr_w: u32 =
            (0..W).map(|g| line_wt(g) * (dr[g] as u32 + dc[g] as u32)).sum();
        let (mass, cnt) = misplaced_center_mass(&s);
        let (fr, fc) = boundary_flow(&s);
        let blank = s.0.iter().position(|&t| t == 0).unwrap();
        let fmt = |v: &[u8]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let fmtu = |v: &[u32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}",
            i,
            seed,
            len,
            d,
            h0,
            d as i64 - h0 as i64,
            manhattan(&s),
            dem_tot,
            dem_ctr_w,
            fmt(&dr),
            fmt(&dc),
            cnt,
            mass,
            fmtu(&fr),
            fmtu(&fc),
            blank / W,
            blank % W,
            secs
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("motifs");
    if mode == "pairjoint" {
        return run_pairjoint();
    }
    if mode == "ab" {
        let thr: u16 = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(144);
        return run_ab(thr);
    }
    if mode == "interior" {
        let thr: u16 = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(146);
        let cap: u64 = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(2_000_000);
        let size: usize = args.get(4).and_then(|x| x.parse().ok()).unwrap_or(8_000);
        let seed: u64 = args.get(5).and_then(|x| x.parse().ok()).unwrap_or(1);
        return run_interior(thr, cap, size, seed);
    }
    eprintln!("loading cWD tables…");
    let t = Instant::now();
    let cwd = Cwd::new();
    eprintln!("cWD ready in {:.1}s (overlay: {})", t.elapsed().as_secs_f64(), cwd.has_overlay());
    match mode {
        "motifs" => run_motifs(&cwd),
        "walks" => {
            let n: u64 = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(50);
            let len: u32 = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(70);
            let seed: u64 = args.get(4).and_then(|x| x.parse().ok()).unwrap_or(1);
            run_walks(&cwd, n, len, seed);
        }
        other => {
            eprintln!("unknown mode {other}; use `motifs` or `walks N LEN SEED`");
            std::process::exit(2);
        }
    }
}
