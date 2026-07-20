//! Slack anatomy: replay optimal (or best-known) paths and account for where
//! every move physically goes, versus what WD/cWD charge.
//!
//! Identity used throughout (per axis): every vertical move crosses exactly one
//! row boundary, so V = F_v + 2·RT_v where F_v = Σ tile flow requirements
//! (crossings forced by start→goal line intervals) and RT_v = excess round-trip
//! pairs (churn). Hence
//!     slack_v = V − WD_row = 2·RT_v − (WD_row − F_v)
//! i.e. observed slack = churn minus WD's over-flow charge (blank ferry +
//! ordering). This decomposition localizes the 12–16 band slack by axis,
//! boundary, tile, goal-line adjacency, and path phase — constraint mining for
//! an operator-counting/LP bound.
//!
//! Modes:
//!   walks N LEN SEED0   re-generate the cc_walks boards (same seed schedule),
//!                       re-solve exactly, analyze each optimal path (TSV to
//!                       stdout, aggregate to stderr)
//!   rpath [FILE]        analyze R's best-known 156-move path
//!                       (default data/r156_ours_solution.txt)
//!
//! Run from repo root (needs data/wd24.bin + data/cwd_single.bin).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use puzzle8::puzzle24::search::walking_distance::{
    load_dist_table, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use puzzle8::puzzle24::search::{Cwd, Heuristic, Search};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS, W};

type Matrix = [[u8; W]; W];

// ---- WD-key codec (mirrors cwd_gap.rs / center_congestion.rs) -------------------

fn pack(m: &Matrix, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

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

// ---- path anatomy ----------------------------------------------------------------

const NB: usize = W - 1; // boundaries per axis

#[derive(Default, Clone)]
struct Anat {
    d: u32,
    v: u32,
    h: u32,
    wd_r: u32,
    wd_c: u32,
    cwd: u32,
    fv: u32,
    fc: u32,
    rt_v: u32, // non-blank round-trip pairs, row boundaries
    rt_c: u32,
    rt_home: u32, // RT pairs where the crossed boundary is adjacent to the tile's goal line
    rt_far: u32,
    rt_b_v: [u32; NB], // RT pairs by boundary index (location)
    rt_b_c: [u32; NB],
    blank_rt_v: u32, // blank round-trip pairs beyond its own displacement need
    blank_rt_c: u32,
    dem_tot: u32,
    phase: [u32; 3], // excess (non-blank) crossing events by path third
}

/// tile 0 = blank; axis 0 = row boundaries (vertical moves), 1 = col.
fn analyze(start: &State, path: &[Move], wd: &WdTable, cwd: &Cwd) -> Anat {
    let mut a = Anat {
        d: path.len() as u32,
        ..Default::default()
    };
    let (mr, br, dr, mc, bc, dc) = project(start);
    a.wd_r = *wd.get(&pack(&mr, br)).expect("row key") as u32;
    a.wd_c = *wd.get(&pack(&mc, bc)).expect("col key") as u32;
    a.cwd = cwd.h(start) as u32;
    a.dem_tot = dr.iter().chain(dc.iter()).map(|&x| x as u32).sum();

    // crossing event indices per (tile, axis, boundary)
    let mut ev: Vec<Vec<u32>> = vec![Vec::new(); N_CELLS * 2 * NB];
    let idx = |t: usize, ax: usize, b: usize| (t * 2 + ax) * NB + b;

    let mut s = *start;
    let mut blank = s.0.iter().position(|&t| t == 0).unwrap();
    for (i, &m) in path.iter().enumerate() {
        let s2 = s.apply(m);
        let b2 = s2.0.iter().position(|&t| t == 0).unwrap();
        let tile = s2.0[blank] as usize; // tile that just moved into old blank cell
        let (r0, c0) = (blank / W, blank % W);
        let (r1, c1) = (b2 / W, b2 % W);
        if r0 != r1 {
            a.v += 1;
            let b = r0.min(r1);
            ev[idx(tile, 0, b)].push(i as u32);
            ev[idx(0, 0, b)].push(i as u32);
        } else {
            a.h += 1;
            let b = c0.min(c1);
            ev[idx(tile, 1, b)].push(i as u32);
            ev[idx(0, 1, b)].push(i as u32);
        }
        s = s2;
        blank = b2;
    }
    assert_eq!(s, GOAL, "path must end at GOAL");

    // required crossings r_tb from start→goal line intervals
    let mut req = vec![0u32; N_CELLS * 2 * NB];
    for pos in 0..N_CELLS {
        let tile = start.0[pos] as usize;
        let (r, c) = (pos / W, pos % W);
        let (gr, gc) = if tile == 0 {
            (W - 1, W - 1)
        } else {
            ((tile - 1) / W, (tile - 1) % W)
        };
        for b in r.min(gr)..r.max(gr) {
            req[idx(tile, 0, b)] = 1;
        }
        for b in c.min(gc)..c.max(gc) {
            req[idx(tile, 1, b)] = 1;
        }
    }

    let d = a.d.max(1);
    for tile in 0..N_CELLS {
        for ax in 0..2 {
            for b in 0..NB {
                let x = ev[idx(tile, ax, b)].len() as u32;
                let r = req[idx(tile, ax, b)];
                assert!(
                    x >= r && (x - r) % 2 == 0,
                    "parity violated tile={tile} ax={ax} b={b} x={x} r={r}"
                );
                let pairs = (x - r) / 2;
                if tile == 0 {
                    if ax == 0 {
                        a.blank_rt_v += pairs;
                    } else {
                        a.blank_rt_c += pairs;
                    }
                    continue;
                }
                if ax == 0 {
                    a.fv += r;
                    a.rt_v += pairs;
                    a.rt_b_v[b] += pairs;
                } else {
                    a.fc += r;
                    a.rt_c += pairs;
                    a.rt_b_c[b] += pairs;
                }
                if pairs > 0 {
                    let g = tile - 1;
                    let gl = if ax == 0 { g / W } else { g % W };
                    if gl == b || gl == b + 1 {
                        a.rt_home += pairs;
                    } else {
                        a.rt_far += pairs;
                    }
                    // excess events = all crossings beyond the first r direct ones
                    for &e in ev[idx(tile, ax, b)].iter().skip(r as usize) {
                        a.phase[(e * 3 / d).min(2) as usize] += 1;
                    }
                }
            }
        }
    }
    // identities: every vertical move is one non-blank tile crossing
    assert_eq!(a.v, a.fv + 2 * a.rt_v, "row-axis crossing identity");
    assert_eq!(a.h, a.fc + 2 * a.rt_c, "col-axis crossing identity");
    a
}

// ---- transit-yield gap measurement -------------------------------------------------
//
// A "yield" is the same abstract event cWD's escape counters already track: a
// goal-g tile exiting line g. So candidate transit-yield charges are just
// bigger demand vectors fed to the vector-constrained A* (cwd_axis, copied
// verbatim from cwd_gap.rs / center_congestion.rs). Rules R1–R3 compute
// demands from the state (R1 is provable, R2/R3 are progressively aggressive
// and get empirical soundness checks against exact d*). R4 sets demands to
// the OBSERVED per-line exits of the board's own optimal path — sound for
// this path by construction, and an upper envelope for the entire per-line
// escape-demand family. d* − h(R4) is slack no such rule can ever certify.

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

/// Per-axis static yield features: k[g] = residents home in line g,
/// thru[g] = tiles (plus blank) whose cur→goal interval strictly spans line g.
fn yield_features(s: &State) -> ([u8; W], [u32; W], [u8; W], [u32; W]) {
    let mut k_r = [0u8; W];
    let mut k_c = [0u8; W];
    let mut t_r = [0u32; W];
    let mut t_c = [0u32; W];
    for pos in 0..N_CELLS {
        let tile = s.0[pos];
        let (r, c) = (pos / W, pos % W);
        let (gr, gc) = if tile == 0 {
            (W - 1, W - 1)
        } else {
            (((tile - 1) as usize) / W, ((tile - 1) as usize) % W)
        };
        if tile != 0 {
            if gr == r {
                k_r[r] += 1;
            }
            if gc == c {
                k_c[c] += 1;
            }
        }
        for g in (r.min(gr) + 1)..r.max(gr) {
            t_r[g] += 1;
        }
        for g in (c.min(gc) + 1)..c.max(gc) {
            t_c[g] += 1;
        }
    }
    (k_r, t_r, k_c, t_c)
}

/// Candidate yield demands per line for rule `rule` (1..=3). Sound-by-proof
/// only for rule 1; rules 2–3 are gap probes whose soundness is checked
/// empirically against exact d*.
fn yield_dem(k: &[u8; W], thru: &[u32; W], rule: u8) -> [u8; W] {
    let mut y = [0u8; W];
    for g in 0..W {
        y[g] = match rule {
            1 => u8::from(k[g] == 5 && thru[g] > 0),
            2 => u8::from(k[g] >= 4 && thru[g] > 0),
            3 => u8::from(k[g] >= 3 && thru[g] > 0) + u8::from(k[g] >= 4 && thru[g] >= 2),
            _ => 0,
        };
    }
    y
}

/// Gap-blocked demand (Phase-2 first cut). Reads the ACTUAL line layout, not
/// just counts. For a line with `k` goal-residents, the (5-k) non-resident cells
/// are "gaps" a transiter can slip through without forcing a yield (§8p). This
/// rule demands 1 exit only when either the line is fully saturated (k=5, the
/// provable R1 core — no gap at all) or every gap is INTERIOR (columns 1..3,
/// flanked by residents), the hypothesis being that an interior gap is not
/// reachable by a transiter. `r1_only` restricts to the k=5 core for comparison.
fn gap_blocked_line(resident: &[bool; W], thru: u32, r1_only: bool) -> u8 {
    if thru == 0 {
        return 0;
    }
    let k = resident.iter().filter(|&&b| b).count();
    if k == 5 {
        return 1; // R1: fully saturated => a transiter must force a yield
    }
    if r1_only {
        return 0;
    }
    // gap-blocked hypothesis: fire iff every gap is interior (flanked)
    let mut any_gap = false;
    let mut all_interior = true;
    for (c, &res) in resident.iter().enumerate() {
        if !res {
            any_gap = true;
            if c == 0 || c == W - 1 {
                all_interior = false;
            }
        }
    }
    u8::from(any_gap && all_interior && k >= 3)
}

/// (row, col) gap-blocked demand vectors for a state.
fn gap_blocked_dem(s: &State, r1_only: bool) -> ([u8; W], [u8; W]) {
    let (_, t_r, _, t_c) = yield_features(s);
    let mut y_r = [0u8; W];
    let mut y_c = [0u8; W];
    for g in 0..W {
        let mut res_row = [false; W];
        let mut res_col = [false; W];
        for i in 0..W {
            let tr = s.0[g * W + i]; // row g, col i
            res_row[i] = tr != 0 && ((tr as usize - 1) / W) == g;
            let tc = s.0[i * W + g]; // row i, col g
            res_col[i] = tc != 0 && ((tc as usize - 1) % W) == g;
        }
        y_r[g] = gap_blocked_line(&res_row, t_r[g], r1_only);
        y_c[g] = gap_blocked_line(&res_col, t_c[g], r1_only);
    }
    (y_r, y_c)
}

/// Observed per-line exits along a path: goal-g tile leaving line g (per axis).
fn observed_exits(start: &State, path: &[Move]) -> ([u8; W], [u8; W]) {
    let mut e_r = [0u32; W];
    let mut e_c = [0u32; W];
    let mut s = *start;
    let mut blank = s.0.iter().position(|&t| t == 0).unwrap();
    for &m in path {
        let s2 = s.apply(m);
        let b2 = s2.0.iter().position(|&t| t == 0).unwrap();
        let tile = s2.0[blank] as usize; // moved tile, now at old blank cell
        let g = tile - 1; // never blank==tile here; blank swaps the other way
        let (r_from, c_from) = (b2 / W, b2 % W); // cell the tile left
        let (r_to, c_to) = (blank / W, blank % W);
        if r_from != r_to && g / W == r_from {
            e_r[r_from] += 1;
        }
        if c_from != c_to && g % W == c_from {
            e_c[c_from] += 1;
        }
        s = s2;
        blank = b2;
    }
    const CAP: u32 = 5; // counter-product safety cap; log if ever hit
    let clamp = |v: &[u32; W]| {
        let mut o = [0u8; W];
        for g in 0..W {
            o[g] = v[g].min(CAP) as u8;
        }
        o
    };
    (clamp(&e_r), clamp(&e_c))
}

fn max_dem(a: &[u8; W], b: &[u8; W]) -> [u8; W] {
    let mut o = [0u8; W];
    for g in 0..W {
        o[g] = a[g].max(b[g]);
    }
    o
}

#[derive(Default, Clone)]
struct YieldRow {
    d: u32,
    h0: u32, // reference cWD (LIS demands, both axes summed)
    h_rule: [u32; 3],
    h_ceil: u32,
    fired: [u32; 3], // lines with y>0 across both axes
    exits_tot: u32,
    fallback: bool, // any cwd_axis budget exhaustion (value fell back to h0 part)
}

fn yield_eval(s: &State, path: &[Move], table: &WdTable) -> YieldRow {
    let goal = goal_key();
    let (mr, br, dr, mc, bc, dc) = project(s);
    let (k_r, t_r, k_c, t_c) = yield_features(s);
    let (e_r, e_c) = observed_exits(s, path);
    let mut row = YieldRow {
        d: path.len() as u32,
        exits_tot: e_r.iter().chain(e_c.iter()).map(|&x| x as u32).sum(),
        ..Default::default()
    };
    let mut fb = false;
    let mut axis = |m: &Matrix, blank: u8, dem: &[u8; W], base: &mut u32| {
        let v = cwd_axis(table, m, blank, goal, dem);
        if v.is_none() {
            fb = true;
        }
        *base += v.unwrap_or_else(|| *table.get(&pack(m, blank)).unwrap()) as u32;
    };
    axis(&mr, br, &dr, &mut row.h0);
    axis(&mc, bc, &dc, &mut row.h0);
    for r in 1..=3u8 {
        let y_r = yield_dem(&k_r, &t_r, r);
        let y_c = yield_dem(&k_c, &t_c, r);
        row.fired[(r - 1) as usize] =
            y_r.iter().chain(y_c.iter()).filter(|&&y| y > 0).count() as u32;
        let mut h = 0u32;
        axis(&mr, br, &max_dem(&dr, &y_r), &mut h);
        axis(&mc, bc, &max_dem(&dc, &y_c), &mut h);
        row.h_rule[(r - 1) as usize] = h;
    }
    let mut hc = 0u32;
    axis(&mr, br, &max_dem(&dr, &e_r), &mut hc);
    axis(&mc, bc, &max_dem(&dc, &e_c), &mut hc);
    row.h_ceil = hc;
    row.fallback = fb;
    row
}

fn run_yieldgap(n: u64, len: u32, seed0: u64, threads: usize) {
    eprintln!("loading cWD (solver) + raw WD table… ({threads} worker thread(s))");
    let cwd = Cwd::new();
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");
    let next: std::sync::atomic::AtomicU64 = 0.into();
    let done: std::sync::atomic::AtomicU64 = 0.into();
    let results: std::sync::Mutex<Vec<(u64, YieldRow)>> =
        std::sync::Mutex::new(Vec::with_capacity(n as usize));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let seed = seed0.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let s = random_walk(seed, len);
                let (sol, _st) = Search::new(&s, &cwd).solve_with_stats();
                let path = sol.expect("solvable by construction");
                let row = yield_eval(&s, &path, &table);
                results.lock().unwrap().push((i, row));
                let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if k % 10 == 0 {
                    eprintln!("  {k}/{n} done");
                }
            });
        }
    });
    let mut rows = results.into_inner().unwrap();
    rows.sort_by_key(|r| r.0);
    println!("board\td\th0\thR1\thR2\thR3\thCEIL\tfired1\tfired2\tfired3\texits\tfb");
    let mut gain = [0f64; 3];
    let mut viol = [0u32; 3];
    let mut fired = [0f64; 3];
    let (mut gain_c, mut resid, mut exits, mut nfb) = (0f64, 0f64, 0f64, 0u32);
    for (i, r) in &rows {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            i,
            r.d,
            r.h0,
            r.h_rule[0],
            r.h_rule[1],
            r.h_rule[2],
            r.h_ceil,
            r.fired[0],
            r.fired[1],
            r.fired[2],
            r.exits_tot,
            r.fallback as u8
        );
        for k in 0..3 {
            gain[k] += (r.h_rule[k] - r.h0) as f64;
            viol[k] += u32::from(r.h_rule[k] > r.d);
            fired[k] += r.fired[k] as f64;
        }
        gain_c += (r.h_ceil - r.h0) as f64;
        resid += r.d as f64 - r.h_ceil as f64;
        exits += r.exits_tot as f64;
        nfb += r.fallback as u32;
        assert!(r.h_ceil <= r.d, "ceiling exceeded d* on board {i} — bug");
    }
    let nn = rows.len() as f64;
    eprintln!(
        "=== transit-yield gap [len={len}] (n={}, fallbacks={nfb}) ===",
        rows.len()
    );
    for k in 0..3 {
        eprintln!(
            "  R{}: mean gain {:.3}  fired-lines/board {:.2}  UNSOUND on {} boards",
            k + 1,
            gain[k] / nn,
            fired[k] / nn,
            viol[k]
        );
    }
    eprintln!(
        "  CEILING (observed-exit demands): mean gain {:.2} of mean slack {:.2}; mean residual d*−hCEIL {:.2}; mean observed exits {:.2}",
        gain_c / nn,
        rows.iter().map(|(_, r)| r.d as f64 - r.h0 as f64).sum::<f64>() / nn,
        resid / nn,
        exits / nn
    );
}

fn run_yieldr(file: &str) {
    eprintln!("loading raw WD table…");
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");
    let text = std::fs::read_to_string(file).expect("read move file");
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let moves = parse_moves(&body);
    let start = r_board();
    let row = yield_eval(&start, &moves, &table);
    eprintln!("=== transit-yield on R (156-move path; d* unknown, LB 150 / UB 156) ===");
    eprintln!(
        "  h0(ref cWD) {}  R1 {}  R2 {}  R3 {}  CEIL {}  (observed exits {}, fallback {})",
        row.h0,
        row.h_rule[0],
        row.h_rule[1],
        row.h_rule[2],
        row.h_ceil,
        row.exits_tot,
        row.fallback
    );
}

// ---- board sources ----------------------------------------------------------------

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
            .filter(|&m| last.is_none_or(|p: Move| m != p.inverse()))
            .collect();
        let m = opts[(next() % opts.len() as u64) as usize];
        s = s.apply(m);
        last = Some(m);
    }
    s
}

fn r_board() -> State {
    let mut a = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        a[i] = (25 - i) as u8;
    }
    State(a)
}

fn parse_moves(s: &str) -> Vec<Move> {
    s.split_whitespace()
        .filter_map(|t| match t {
            "U" => Some(Move::Up),
            "D" => Some(Move::Down),
            "L" => Some(Move::Left),
            "R" => Some(Move::Right),
            _ => None,
        })
        .collect()
}

// ---- aggregation -------------------------------------------------------------------

#[derive(Default)]
struct Agg {
    n: u32,
    slack: f64,
    slack_v: f64,
    slack_h: f64,
    rt2: f64,     // 2(RT_v + RT_c)
    wd_over: f64, // (wd_r − fv) + (wd_c − fc)
    surch: f64,   // cwd − wd_r − wd_c
    rt_home: f64,
    rt_far: f64,
    blank_rt: f64,
    dem: f64,
    rt_b_v: [f64; NB],
    rt_b_c: [f64; NB],
    phase: [f64; 3],
    // for corr(slack, 2RT)
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl Agg {
    fn push(&mut self, a: &Anat) {
        let slack = a.d as f64 - a.cwd as f64;
        let rt2 = 2.0 * (a.rt_v + a.rt_c) as f64;
        self.n += 1;
        self.slack += slack;
        self.slack_v += a.v as f64 - a.wd_r as f64;
        self.slack_h += a.h as f64 - a.wd_c as f64;
        self.rt2 += rt2;
        self.wd_over += (a.wd_r - a.fv) as f64 + (a.wd_c - a.fc) as f64;
        self.surch += a.cwd as f64 - a.wd_r as f64 - a.wd_c as f64;
        self.rt_home += a.rt_home as f64;
        self.rt_far += a.rt_far as f64;
        self.blank_rt += (a.blank_rt_v + a.blank_rt_c) as f64;
        self.dem += a.dem_tot as f64;
        for b in 0..NB {
            self.rt_b_v[b] += a.rt_b_v[b] as f64;
            self.rt_b_c[b] += a.rt_b_c[b] as f64;
        }
        for k in 0..3 {
            self.phase[k] += a.phase[k] as f64;
        }
        self.xs.push(slack);
        self.ys.push(rt2);
    }

    fn report(&self, tag: &str) {
        let n = self.n as f64;
        let corr = {
            let mx = self.xs.iter().sum::<f64>() / n;
            let my = self.ys.iter().sum::<f64>() / n;
            let mut sxy = 0.0;
            let mut sxx = 0.0;
            let mut syy = 0.0;
            for i in 0..self.xs.len() {
                let dx = self.xs[i] - mx;
                let dy = self.ys[i] - my;
                sxy += dx * dy;
                sxx += dx * dx;
                syy += dy * dy;
            }
            sxy / (sxx.sqrt() * syy.sqrt()).max(1e-12)
        };
        eprintln!("=== slack anatomy aggregate [{tag}] (n={}) ===", self.n);
        eprintln!(
            "  slack d−cWD        {:6.2}   (axis split vs WD: row {:.2} + col {:.2}; cWD surcharge {:.2})",
            self.slack / n,
            self.slack_v / n,
            self.slack_h / n,
            self.surch / n
        );
        eprintln!(
            "  churn 2·RT         {:6.2}   = slack_wd + wd_over ({:.2}); corr(slack, 2RT) = {:.3}",
            self.rt2 / n,
            self.wd_over / n,
            corr
        );
        eprintln!(
            "  RT pairs: home-adjacent {:5.2}   far {:5.2}   (blank RT {:5.2}; cWD demand total {:4.2})",
            self.rt_home / n,
            self.rt_far / n,
            self.blank_rt / n,
            self.dem / n
        );
        let f = |v: &[f64; NB]| {
            v.iter()
                .map(|x| format!("{:.2}", x / n))
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!(
            "  RT by row boundary  [{}]   col boundary [{}]",
            f(&self.rt_b_v),
            f(&self.rt_b_c)
        );
        eprintln!(
            "  excess events by path third: {:.1}% / {:.1}% / {:.1}%",
            100.0 * self.phase[0] / (self.phase.iter().sum::<f64>().max(1.0)),
            100.0 * self.phase[1] / (self.phase.iter().sum::<f64>().max(1.0)),
            100.0 * self.phase[2] / (self.phase.iter().sum::<f64>().max(1.0)),
        );
    }
}

// ---- modes ------------------------------------------------------------------------

fn tsv_header() {
    println!(
        "board\tseed\tlen\td\tv\th\twd_r\twd_c\tcwd\tslack\tfv\tfc\trt_v\trt_c\trt_home\trt_far\trt_blank\tdem_tot\trtb_v\trtb_c\tph\tsec"
    );
}

fn tsv_row(i: u64, seed: u64, len: u32, a: &Anat, sec: f64) {
    let f = |v: &[u32; NB]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{},{},{}\t{:.2}",
        i,
        seed,
        len,
        a.d,
        a.v,
        a.h,
        a.wd_r,
        a.wd_c,
        a.cwd,
        a.d as i64 - a.cwd as i64,
        a.fv,
        a.fc,
        a.rt_v,
        a.rt_c,
        a.rt_home,
        a.rt_far,
        a.blank_rt_v + a.blank_rt_c,
        a.dem_tot,
        f(&a.rt_b_v),
        f(&a.rt_b_c),
        a.phase[0],
        a.phase[1],
        a.phase[2],
        sec
    );
}

fn run_walks(n: u64, len: u32, seed0: u64, threads: usize) {
    eprintln!("loading cWD (solver) + raw WD table… ({threads} worker thread(s))");
    let cwd = Cwd::new();
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");
    let next: std::sync::atomic::AtomicU64 = 0.into();
    let done: std::sync::atomic::AtomicU64 = 0.into();
    let results: std::sync::Mutex<Vec<(u64, u64, Anat, f64)>> =
        std::sync::Mutex::new(Vec::with_capacity(n as usize));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let seed = seed0.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let s = random_walk(seed, len);
                let t = Instant::now();
                let (sol, _st) = Search::new(&s, &cwd).solve_with_stats();
                let path = sol.expect("solvable by construction");
                let sec = t.elapsed().as_secs_f64();
                let a = analyze(&s, &path, &table, &cwd);
                results.lock().unwrap().push((i, seed, a, sec));
                let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if k % 10 == 0 {
                    eprintln!("  {k}/{n} done");
                }
            });
        }
    });
    let mut rows = results.into_inner().unwrap();
    rows.sort_by_key(|r| r.0);
    tsv_header();
    let mut agg = Agg::default();
    for (i, seed, a, sec) in &rows {
        tsv_row(*i, *seed, len, a, *sec);
        agg.push(a);
    }
    agg.report(&format!("walks len={len}"));
}

fn run_rpath(file: &str) {
    eprintln!("loading cWD + raw WD table…");
    let cwd = Cwd::new();
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");
    let text = std::fs::read_to_string(file).expect("read move file");
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let moves = parse_moves(&body);
    eprintln!("R path: {} moves from {file}", moves.len());
    let start = r_board();
    // verify replay legality (analyze asserts GOAL at the end)
    let mut s = start;
    for (i, &m) in moves.iter().enumerate() {
        assert!(
            s.legal_moves().contains(m),
            "move {i} illegal in forward replay"
        );
        s = s.apply(m);
    }
    let a = analyze(&start, &moves, &table, &cwd);
    tsv_header();
    tsv_row(0, 0, 0, &a, 0.0);
    let mut agg = Agg::default();
    agg.push(&a);
    agg.report("R 156-move path");

    // per-tile churn detail
    eprintln!("\nper-tile churn (round-trip pairs, tile ▸ axis:boundary×pairs):");
    // recompute events for the dump (cheap)
    let mut ev: HashMap<(usize, usize, usize), u32> = HashMap::new();
    let mut s = start;
    let mut blank = s.0.iter().position(|&t| t == 0).unwrap();
    for &m in moves.iter() {
        let s2 = s.apply(m);
        let b2 = s2.0.iter().position(|&t| t == 0).unwrap();
        let tile = s2.0[blank] as usize;
        let (r0, c0) = (blank / W, blank % W);
        let (r1, c1) = (b2 / W, b2 % W);
        if r0 != r1 {
            *ev.entry((tile, 0, r0.min(r1))).or_default() += 1;
        } else {
            *ev.entry((tile, 1, c0.min(c1))).or_default() += 1;
        }
        s = s2;
        blank = b2;
    }
    for tile in 1..N_CELLS {
        let pos = start.0.iter().position(|&t| t == tile as u8).unwrap();
        let g = tile - 1;
        let mut parts = Vec::new();
        for ax in 0..2 {
            let (cur, gl) = if ax == 0 {
                (pos / W, g / W)
            } else {
                (pos % W, g % W)
            };
            for b in 0..NB {
                let x = ev.get(&(tile, ax, b)).copied().unwrap_or(0);
                let r = u32::from(b >= cur.min(gl) && b < cur.max(gl));
                if x > r {
                    parts.push(format!(
                        "{}{}×{}",
                        if ax == 0 { "r" } else { "c" },
                        b,
                        (x - r) / 2
                    ));
                }
            }
        }
        if !parts.is_empty() {
            eprintln!("  tile {tile:2}: {}", parts.join(" "));
        }
    }
}

// ---- transit-yield soundness diagnostic (Phase 1) --------------------------
//
// R2/R3 are aggressive transit-yield rules that overshoot d* on some boards
// (unsound). This mode localizes WHERE: for each board it computes the
// board-level unsound flag (cwd_axis value with the rule's demands > d*), and
// buckets every FIRED transit demand by its features (k = goal-g residents in
// line g, thru = tiles whose interval spans g) — measuring how often the
// optimal path made FEWER exits than the rule demanded (the "over-fire", the
// structural cause of unsoundness). A bucket with 0% over-fire is a safe
// (candidate-sound) side-condition; a high-% bucket is the guard we must add.

#[derive(Default, Clone)]
struct DiagBucket {
    fires: u64,
    overfires: u64,  // optimal path made fewer exits than demanded (e < y)
    sum_over: u64,   // total (y - e) across over-fires
    on_unsound: u64, // fires occurring on a board that is unsound overall
}

type DiagAgg = HashMap<(u8, u8, u8), DiagBucket>; // (rule, k, thru_bucket)

fn run_yielddiag(n: u64, len: u32, seed0: u64, threads: usize) {
    eprintln!("loading cWD (solver) + raw WD table… ({threads} worker thread(s))");
    let cwd = Cwd::new();
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");
    let goal = goal_key();
    let next: std::sync::atomic::AtomicU64 = 0.into();
    #[allow(clippy::type_complexity)]
    let shared: std::sync::Mutex<(DiagAgg, [u64; 2], [u64; 2], [u64; 2], [u64; 2], Vec<String>)> =
        // (agg, unsound[rule], fallback[rule], judged[rule], sum_overshoot[rule], samples)
        std::sync::Mutex::new((DiagAgg::new(), [0; 2], [0; 2], [0; 2], [0; 2], Vec::new()));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut agg = DiagAgg::new();
                let mut unsound = [0u64; 2];
                let mut fb = [0u64; 2];
                let mut judged = [0u64; 2];
                let mut overshoot = [0u64; 2];
                let mut samples: Vec<String> = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let seed = seed0.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                    let s = random_walk(seed, len);
                    let (sol, _st) = Search::new(&s, &cwd).solve_with_stats();
                    let path = sol.expect("solvable by construction");
                    let d = path.len() as u32;

                    let (mr, br, dr, mc, bc, dc) = project(&s);
                    let (k_r, t_r, k_c, t_c) = yield_features(&s);
                    let (e_r, e_c) = observed_exits(&s, &path);

                    for (ri, rule) in [2u8, 3u8].into_iter().enumerate() {
                        let y_r = yield_dem(&k_r, &t_r, rule);
                        let y_c = yield_dem(&k_c, &t_c, rule);
                        // board-level unsound flag: cwd_axis with max(LIS, transit) demands
                        let mut fell_back = false;
                        let mut hr = 0u32;
                        for (m, blank, dem) in
                            [(&mr, br, max_dem(&dr, &y_r)), (&mc, bc, max_dem(&dc, &y_c))]
                        {
                            match cwd_axis(&table, m, blank, goal, &dem) {
                                Some(v) => hr += v as u32,
                                None => {
                                    fell_back = true;
                                    hr += *table.get(&pack(m, blank)).unwrap() as u32;
                                }
                            }
                        }
                        if fell_back {
                            fb[ri] += 1;
                            continue;
                        }
                        judged[ri] += 1;
                        let is_unsound = hr > d;
                        if is_unsound {
                            unsound[ri] += 1;
                            overshoot[ri] += (hr - d) as u64;
                        }
                        // line-level over-fire attribution
                        let mut offending = String::new();
                        for (axis, (y, e, k, t)) in
                            [(&y_r, &e_r, &k_r, &t_r), (&y_c, &e_c, &k_c, &t_c)]
                                .into_iter()
                                .enumerate()
                        {
                            for g in 0..W {
                                if y[g] == 0 {
                                    continue;
                                }
                                let key = (rule, k[g], t[g].min(2) as u8);
                                let b = agg.entry(key).or_default();
                                b.fires += 1;
                                if is_unsound {
                                    b.on_unsound += 1;
                                }
                                if e[g] < y[g] {
                                    b.overfires += 1;
                                    b.sum_over += (y[g] - e[g]) as u64;
                                    if is_unsound && offending.is_empty() {
                                        offending = format!(
                                            "{}g{g} k={} thru={} y={} e={}",
                                            if axis == 0 { "r" } else { "c" },
                                            k[g],
                                            t[g],
                                            y[g],
                                            e[g]
                                        );
                                    }
                                }
                            }
                        }
                        if is_unsound && rule == 3 && samples.len() < 12 {
                            samples.push(format!(
                                "d={d} h_r3={hr} (+{}) offend[{offending}] board={:?}",
                                hr - d,
                                s.0
                            ));
                        }
                    }
                }
                let mut g = shared.lock().unwrap();
                for (k, v) in agg {
                    let e = g.0.entry(k).or_default();
                    e.fires += v.fires;
                    e.overfires += v.overfires;
                    e.sum_over += v.sum_over;
                    e.on_unsound += v.on_unsound;
                }
                for i in 0..2 {
                    g.1[i] += unsound[i];
                    g.2[i] += fb[i];
                    g.3[i] += judged[i];
                    g.4[i] += overshoot[i];
                }
                g.5.extend(samples);
            });
        }
    });

    let (agg, unsound, fb, judged, overshoot, samples) = shared.into_inner().unwrap();
    println!("=== transit-yield soundness diagnostic [len={len}] (n={n}, {threads} thr) ===");
    for (ri, rule) in [2u8, 3u8].into_iter().enumerate() {
        let (u, j, f) = (unsound[ri], judged[ri], fb[ri]);
        let rate = 100.0 * u as f64 / j.max(1) as f64;
        let mo = if u > 0 {
            overshoot[ri] as f64 / u as f64
        } else {
            0.0
        };
        println!(
            "  R{rule}: unsound {u}/{j} ({rate:.1}%), fallback {f}; mean overshoot (h−d) when unsound {mo:.2}"
        );
    }
    println!();
    println!("fired transit demands, bucketed by (k=residents, thru=transiters):");
    println!("  rule   k  thru |  fires  overfire  rate   mean_over  %fires_on_unsound_boards");
    let mut keys: Vec<_> = agg.keys().copied().collect();
    keys.sort();
    for key in keys {
        let b = &agg[&key];
        let (rule, k, tb) = key;
        let tb_s = if tb >= 2 { "2+".into() } else { tb.to_string() };
        println!(
            "  R{rule}   {k}  {tb_s:>4} | {:6}  {:8}  {:4.0}%  {:8.2}   {:5.0}%",
            b.fires,
            b.overfires,
            100.0 * b.overfires as f64 / b.fires.max(1) as f64,
            b.sum_over as f64 / b.overfires.max(1) as f64,
            100.0 * b.on_unsound as f64 / b.fires.max(1) as f64,
        );
    }
    println!();
    println!("sample unsound boards (R3):");
    for s in samples.iter().take(12) {
        println!("  {s}");
    }
}

// ---- gap-blocked rule test (Phase 2 first cut) -----------------------------
//
// Measures, over exact-solved bulk boards, the unsound rate (h_rule > d*) and
// mean gain over cWD (LIS-only baseline h0) for two sound-candidate demands:
// R1 (k=5 saturation) and GB (k=5 + interior-gap k<=4). Then counts how often
// each fires along R's 156-move path — "does it activate near R?".

fn run_gbtest(n: u64, len: u32, seed0: u64, threads: usize) {
    eprintln!("loading cWD (solver) + raw WD table… ({threads} worker thread(s))");
    let cwd = Cwd::new();
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");
    let goal = goal_key();
    let next: std::sync::atomic::AtomicU64 = 0.into();
    // per rule [r1, gb]: (judged, unsound, fallback, sum_gain, sum_fires); plus (cnt, sum_d, sum_h0)
    #[allow(clippy::type_complexity)]
    let shared: std::sync::Mutex<(
        [u64; 2],
        [u64; 2],
        [u64; 2],
        [u64; 2],
        [u64; 2],
        u64,
        u64,
        u64,
    )> = std::sync::Mutex::new(([0; 2], [0; 2], [0; 2], [0; 2], [0; 2], 0, 0, 0));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let (mut judged, mut unsound, mut fb, mut gain, mut fires) =
                    ([0u64; 2], [0u64; 2], [0u64; 2], [0u64; 2], [0u64; 2]);
                let (mut cnt, mut sum_d, mut sum_h0) = (0u64, 0u64, 0u64);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let seed = seed0.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                    let s = random_walk(seed, len);
                    let (sol, _st) = Search::new(&s, &cwd).solve_with_stats();
                    let path = sol.expect("solvable by construction");
                    let d = path.len() as u32;
                    let (mr, br, dr, mc, bc, dc) = project(&s);
                    let axis_sum = |dem_r: &[u8; W], dem_c: &[u8; W]| -> Option<u32> {
                        let a = cwd_axis(&table, &mr, br, goal, dem_r)?;
                        let b = cwd_axis(&table, &mc, bc, goal, dem_c)?;
                        Some(a as u32 + b as u32)
                    };
                    let h0 = match axis_sum(&dr, &dc) {
                        Some(v) => v,
                        None => continue, // baseline fell back; skip board
                    };
                    cnt += 1;
                    sum_d += d as u64;
                    sum_h0 += h0 as u64;
                    for (idx, r1_only) in [(0usize, true), (1usize, false)] {
                        let (y_r, y_c) = gap_blocked_dem(&s, r1_only);
                        let f = y_r.iter().chain(y_c.iter()).filter(|&&x| x > 0).count() as u64;
                        match axis_sum(&max_dem(&dr, &y_r), &max_dem(&dc, &y_c)) {
                            Some(h) => {
                                judged[idx] += 1;
                                if h > d {
                                    unsound[idx] += 1;
                                }
                                gain[idx] += (h - h0) as u64;
                                fires[idx] += f;
                            }
                            None => fb[idx] += 1,
                        }
                    }
                }
                let mut g = shared.lock().unwrap();
                for i in 0..2 {
                    g.0[i] += judged[i];
                    g.1[i] += unsound[i];
                    g.2[i] += fb[i];
                    g.3[i] += gain[i];
                    g.4[i] += fires[i];
                }
                g.5 += cnt;
                g.6 += sum_d;
                g.7 += sum_h0;
            });
        }
    });

    let (judged, unsound, fb, gain, fires, cnt, sum_d, sum_h0) = shared.into_inner().unwrap();
    let cf = cnt.max(1) as f64;
    println!("=== gap-blocked rule test [len={len}] (n={n}, {threads} thr) ===");
    println!(
        "  boards judged {cnt}; mean d* {:.2}; mean cWD (LIS baseline h0) {:.2}",
        sum_d as f64 / cf,
        sum_h0 as f64 / cf
    );
    for (idx, name) in [(0usize, "R1 (k=5)"), (1usize, "GB (k=5 + interior-gap)")] {
        let j = judged[idx].max(1);
        println!(
            "  {name:24}: unsound {}/{} ({:.2}%), fallback {}; mean gain over cWD {:.3}; fires/board {:.2}",
            unsound[idx],
            judged[idx],
            100.0 * unsound[idx] as f64 / j as f64,
            fb[idx],
            gain[idx] as f64 / j as f64,
            fires[idx] as f64 / cf,
        );
    }

    // does it fire near R? walk R's 156-move path, count firing states.
    let text = std::fs::read_to_string("data/r156_ours_solution.txt").expect("R path");
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let moves = parse_moves(&body);
    let mut st = r_board();
    let (mut r1_states, mut gb_states, mut r1_f, mut gb_f) = (0u32, 0u32, 0u32, 0u32);
    let (mut r1_gain_sum, mut r1_gain_max, mut r1_bind) = (0i64, 0i64, 0u32);
    let n_states = moves.len() + 1;
    for i in 0..n_states {
        let (yr1r, yr1c) = gap_blocked_dem(&st, true);
        let (ygbr, ygbc) = gap_blocked_dem(&st, false);
        let f1 = yr1r.iter().chain(yr1c.iter()).filter(|&&x| x > 0).count() as u32;
        let fg = ygbr.iter().chain(ygbc.iter()).filter(|&&x| x > 0).count() as u32;
        r1_f += f1;
        gb_f += fg;
        r1_states += u32::from(f1 > 0);
        gb_states += u32::from(fg > 0);
        if f1 > 0 {
            // does R1 actually tighten cWD here? (only compute where it fires)
            let (mr, br, dr, mc, bc, dc) = project(&st);
            let axis_sum = |a: &[u8; W], b: &[u8; W]| -> Option<u32> {
                Some(
                    cwd_axis(&table, &mr, br, goal, a)? as u32
                        + cwd_axis(&table, &mc, bc, goal, b)? as u32,
                )
            };
            if let (Some(h0), Some(h1)) = (
                axis_sum(&dr, &dc),
                axis_sum(&max_dem(&dr, &yr1r), &max_dem(&dc, &yr1c)),
            ) {
                let g = h1 as i64 - h0 as i64;
                r1_gain_sum += g;
                r1_gain_max = r1_gain_max.max(g);
                r1_bind += u32::from(g > 0);
            }
        }
        if i < moves.len() {
            st = st.apply(moves[i]);
        }
    }
    println!();
    println!("R's 156-move path ({n_states} states) — activation near R:");
    println!("  R1: fires on {r1_states}/{n_states} states, {r1_f} total line-fires");
    println!("  GB: fires on {gb_states}/{n_states} states, {gb_f} total line-fires");
    println!(
        "  R1 tightening where it fires: binds on {r1_bind} states, max gain +{r1_gain_max}, total +{r1_gain_sum}"
    );
}

// ---- yield-or-detour ceiling (Phase 2, cross-axis coupling) -----------------
//
// The one coupling §8m/§8n/§8o/§8q did NOT test: a transiter through a row can
// dodge a yield only by slipping through a GAP, but reaching that gap's column
// costs column moves. So each transit crossing is one of:
//   YIELD  — the row was saturated (4 residents + blank, no other gap): a
//            resident must have vacated -> chargeable on the row axis.
//   DETOUR — the crossing happened at a column OUTSIDE the tile's natural
//            [start_col, goal_col] band: extra column travel cWD_col does not
//            charge (it counts only the minimum) -> chargeable on the col axis.
//   FREE   — crossing at an on-band gap: no yield, no detour -> OPTIONAL churn
//            that no sound bound can charge.
// If FREE dominates on R's path, yield-or-detour (and any sound transit charge)
// is dead; if YIELD+DETOUR is a large share of the slack, it has juice.

#[derive(Default)]
struct YdCount {
    yield_c: u32,
    detour_c: u32,
    free_c: u32,
    settle_c: u32,      // crossing into the tile's own goal line (not transit)
    detour_excess: u32, // total off-band distance of detour crossings
}

fn run_ydceiling(file: &str) {
    let text = std::fs::read_to_string(file).expect("read move file");
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let moves = parse_moves(&body);
    let start = r_board();

    let mut row = YdCount::default(); // vertical crossings (transit through a row)
    let mut col = YdCount::default(); // horizontal crossings (transit through a col)
    let mut s = start;
    let mut blank = s.0.iter().position(|&t| t == 0).unwrap();
    for &m in &moves {
        let s2 = s.apply(m);
        let b2 = s2.0.iter().position(|&t| t == 0).unwrap();
        let t = s2.0[blank] as usize; // tile that moved into the old blank cell (r0,c0)
        let (r0, c0) = (blank / W, blank % W);
        let r1 = b2 / W;
        let (gr, gc) = ((t - 1) / W, (t - 1) % W);
        let (sr, sc) = ((25 - t) / W, (25 - t) % W); // start cell on R board

        if r0 != r1 {
            // vertical move: t now sits in row r0, column c0
            if gr == r0 {
                row.settle_c += 1;
            } else {
                // residents of row r0 among its non-blank cells (blank is at c0 in s)
                let res = (0..W)
                    .filter(|&c| {
                        c != c0 && s.0[r0 * W + c] != 0 && (s.0[r0 * W + c] as usize - 1) / W == r0
                    })
                    .count();
                if res == 4 {
                    row.yield_c += 1;
                } else {
                    let (lo, hi) = (sc.min(gc), sc.max(gc));
                    if c0 >= lo && c0 <= hi {
                        row.free_c += 1;
                    } else {
                        row.detour_c += 1;
                        row.detour_excess += if c0 < lo { lo - c0 } else { c0 - hi } as u32;
                    }
                }
            }
        } else {
            // horizontal move: t now sits in column c0, row r0
            if gc == c0 {
                col.settle_c += 1;
            } else {
                let res = (0..W)
                    .filter(|&r| {
                        r != r0 && s.0[r * W + c0] != 0 && (s.0[r * W + c0] as usize - 1) % W == c0
                    })
                    .count();
                if res == 4 {
                    col.yield_c += 1;
                } else {
                    let (lo, hi) = (sr.min(gr), sr.max(gr));
                    if r0 >= lo && r0 <= hi {
                        col.free_c += 1;
                    } else {
                        col.detour_c += 1;
                        col.detour_excess += if r0 < lo { lo - r0 } else { r0 - hi } as u32;
                    }
                }
            }
        }
        s = s2;
        blank = b2;
    }
    assert_eq!(s, GOAL, "path must end at GOAL");

    let show = |name: &str, c: &YdCount| {
        let transit = c.yield_c + c.detour_c + c.free_c;
        let charge = c.yield_c + c.detour_c;
        println!(
            "  {name}: transit crossings {transit} (settle {}) | YIELD {} DETOUR {} FREE {} | chargeable {}/{} = {:.0}%; detour-excess {}",
            c.settle_c,
            c.yield_c,
            c.detour_c,
            c.free_c,
            charge,
            transit.max(1),
            100.0 * charge as f64 / transit.max(1) as f64,
            c.detour_excess,
        );
    };
    println!("== yield-or-detour ceiling on R ({} moves) ==", moves.len());
    println!("classification of transit crossings (through a non-goal line):");
    show("row-transit (vertical)", &row);
    show("col-transit (horizontal)", &col);
    let yields = (row.yield_c + col.yield_c) as i64;
    let detours = (row.detour_c + col.detour_c) as i64;
    let free = (row.free_c + col.free_c) as i64;
    println!();
    println!("  YIELD {yields}  DETOUR {detours}  FREE {free}  (R slack over cWD = 12)");
    println!("  Reading (careful — these are GROSS path crossings, not net-of-cWD):");
    println!(
        "  - YIELDS negligible ({yields}): saturation-forced yields barely occur on R (confirms\n    §8q R1 binds 0). Row-saturation charging is dead here."
    );
    println!(
        "  - DETOURS are the coupling that shows up ({detours}): transiters cross at off-band\n    columns — the ferry/far-churn made concrete. BUT this is gross: cWD_col already\n    prices most column churn (slack is 12, detour-excess is {}), so the NET recoverable\n    is bounded by the 12 slack, and most may be OPTIONAL (a detour on THIS optimal path\n    can be rerouted on another — the R4 trap).",
        row.detour_excess + col.detour_excess
    );
    println!("  - FREE is the plurality ({free}): on-band gap slips no sound bound can reach.");
    println!(
        "  VERDICT: not decisive alone. R's churn is DETOUR-dominated (not yield). Whether any\n  detour is soundly chargeable NET of cWD needs a soundness-checked detour rule (gbtest-\n  style); prior is guarded because path detours are reroutable across optimal solutions."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("walks") => {
            let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
            let len: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(70);
            let seed0: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
            let threads: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
            run_walks(n, len, seed0, threads);
        }
        Some("yieldgap") => {
            let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
            let len: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(70);
            let seed0: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
            let threads: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
            run_yieldgap(n, len, seed0, threads);
        }
        Some("yieldr") => {
            let file = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("data/r156_ours_solution.txt");
            run_yieldr(file);
        }
        Some("rpath") => {
            let file = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("data/r156_ours_solution.txt");
            run_rpath(file);
        }
        Some("yielddiag") => {
            let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
            let len: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(90);
            let seed0: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
            let threads: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
            run_yielddiag(n, len, seed0, threads);
        }
        Some("gbtest") => {
            let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3000);
            let len: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(55);
            let seed0: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
            let threads: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
            run_gbtest(n, len, seed0, threads);
        }
        Some("ydceiling") => {
            let file = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("data/r156_ours_solution.txt");
            run_ydceiling(file);
        }
        other => {
            eprintln!(
                "usage: slack_anatomy walks N LEN SEED0 | yieldgap N LEN SEED0 T | yielddiag N LEN SEED0 T | yieldr [FILE] | rpath [FILE]  (got {other:?})"
            );
        }
    }
}
