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

    let mut s = start.clone();
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
                    let g = (tile - 1) as usize;
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

/// Observed per-line exits along a path: goal-g tile leaving line g (per axis).
fn observed_exits(start: &State, path: &[Move]) -> ([u8; W], [u8; W]) {
    let mut e_r = [0u32; W];
    let mut e_c = [0u32; W];
    let mut s = start.clone();
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
    let table = load_dist_table(Path::new("data/wd24.bin"), WD_KIND_FULL, Some(FULL_WD_ENTRIES))
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
            i, r.d, r.h0, r.h_rule[0], r.h_rule[1], r.h_rule[2], r.h_ceil,
            r.fired[0], r.fired[1], r.fired[2], r.exits_tot, r.fallback as u8
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
    eprintln!("=== transit-yield gap [len={len}] (n={}, fallbacks={nfb}) ===", rows.len());
    for k in 0..3 {
        eprintln!(
            "  R{}: mean gain {:.3}  fired-lines/board {:.2}  UNSOUND on {} boards",
            k + 1, gain[k] / nn, fired[k] / nn, viol[k]
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
    let table = load_dist_table(Path::new("data/wd24.bin"), WD_KIND_FULL, Some(FULL_WD_ENTRIES))
        .expect("wd24.bin load");
    let text = std::fs::read_to_string(file).expect("read move file");
    let body: String =
        text.lines().filter(|l| !l.trim_start().starts_with('#')).collect::<Vec<_>>().join(" ");
    let moves = parse_moves(&body);
    let start = r_board();
    let row = yield_eval(&start, &moves, &table);
    eprintln!("=== transit-yield on R (156-move path; d* unknown, LB 150 / UB 156) ===");
    eprintln!(
        "  h0(ref cWD) {}  R1 {}  R2 {}  R3 {}  CEIL {}  (observed exits {}, fallback {})",
        row.h0, row.h_rule[0], row.h_rule[1], row.h_rule[2], row.h_ceil, row.exits_tot,
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
            .filter(|&m| last.map_or(true, |p: Move| m != p.inverse()))
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
    rt2: f64,        // 2(RT_v + RT_c)
    wd_over: f64,    // (wd_r − fv) + (wd_c − fc)
    surch: f64,      // cwd − wd_r − wd_c
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
            v.iter().map(|x| format!("{:.2}", x / n)).collect::<Vec<_>>().join(" ")
        };
        eprintln!("  RT by row boundary  [{}]   col boundary [{}]", f(&self.rt_b_v), f(&self.rt_b_c));
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
    let f = |v: &[u32; NB]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
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
    let table = load_dist_table(Path::new("data/wd24.bin"), WD_KIND_FULL, Some(FULL_WD_ENTRIES))
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
    let table = load_dist_table(Path::new("data/wd24.bin"), WD_KIND_FULL, Some(FULL_WD_ENTRIES))
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
    let mut s = start.clone();
    for (i, &m) in moves.iter().enumerate() {
        assert!(s.legal_moves().contains(m), "move {i} illegal in forward replay");
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
    let mut s = start.clone();
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
            let (cur, gl) = if ax == 0 { (pos / W, g / W) } else { (pos % W, g % W) };
            for b in 0..NB {
                let x = ev.get(&(tile, ax, b)).copied().unwrap_or(0);
                let r = u32::from(b >= cur.min(gl) && b < cur.max(gl));
                if x > r {
                    parts.push(format!("{}{}×{}", if ax == 0 { "r" } else { "c" }, b, (x - r) / 2));
                }
            }
        }
        if !parts.is_empty() {
            eprintln!("  tile {tile:2}: {}", parts.join(" "));
        }
    }
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
            let file = args.get(2).map(String::as_str).unwrap_or("data/r156_ours_solution.txt");
            run_yieldr(file);
        }
        Some("rpath") => {
            let file = args.get(2).map(String::as_str).unwrap_or("data/r156_ours_solution.txt");
            run_rpath(file);
        }
        other => {
            eprintln!("usage: slack_anatomy walks N LEN SEED0 | rpath [FILE]  (got {other:?})");
        }
    }
}
