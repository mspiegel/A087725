//! cwd_gap — measure the per-node cWD−WD gap (δ) over R's search frontier.
//!
//! `examples/cwd24.rs` proved cWD(R) = 144 = WD(140) + 4 *at the root only*. The
//! decision on whether to build a production escape-constrained WD table and wire
//! it into the solver hinges on ONE unknown: does that +4 propagate to the
//! interior nodes IDA\* actually expands, or is R a favorable special case?
//!
//! **Sharp, not aggregate.** The probe showed run A (sharp, per-goal-line escape
//! demands) gives D_4 = 72 on R while run B (a single pooled Σx_g counter) gives
//! only 70 — the pooled relaxation lets the plan satisfy the count with cheap
//! escapes of the wrong line, so it is worthless on R. The *sound* cWD demands,
//! for EACH goal line g, ≥ x_g escapes OF TYPE g. So the constrained A\* here
//! carries a **vector** of per-line escape counters, each saturating at x_g.
//!
//! This tool: (1) reservoir-samples nodes from a randomized bounded DFS of R's
//! threshold-T tree (tracks the population IDA\* expands, reaching the deep
//! frontier a greedy descent can't); (2) computes WD (lookup) and sharp cWD
//! (per-axis vector-constrained A\* — the measurement form; production would be a
//! table lookup, so this A\*'s cost is irrelevant, only δ matters); (3) reports
//! the δ = cWD−WD distribution by depth and slack. From δ_avg the node-reduction
//! factor ≈ 29^(δ_avg/2). Break-even for a table-based cWD (~2× WD/node) is
//! δ_avg > ~0.4. Run from repo root so `data/wd24.bin` resolves.

use puzzle8::puzzle24::search::walking_distance::{
    load_dist_table, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use puzzle8::puzzle24::state::{Move, State, W};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

type Matrix = [[u8; W]; W];

// ---- WD-table key codec (identical layout to walking_distance::pack) --------

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

/// GOAL contingency: identity with the 4-margin in the blank's axis (index 4).
/// Serves both axes (goal blank is at row 4 and column 4).
fn goal_key() -> u64 {
    let mut m = [[0u8; W]; W];
    for d in 0..W {
        m[d][d] = 5;
    }
    m[W - 1][W - 1] = 4;
    pack(&m, (W - 1) as u8)
}

// ---- per-node projections: contingency matrices + per-line escape demands ---

/// Longest strictly-increasing subsequence length. k ≤ 5 ⇒ trivial O(k²).
fn lis_strict(seq: &[u8]) -> usize {
    let n = seq.len();
    let mut dp = [0usize; W];
    let mut best = 0usize;
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

/// Project `s`. Returns (m_row, br, dem_row, m_col, bc, dem_col) where
/// `dem_row[g] = x_g` = (goal-row-g residents physically in row g) − LIS(their
/// goal columns): the number of type-g tiles that MUST escape row g. Symmetric
/// for columns.
fn project(s: &State) -> (Matrix, u8, [u8; W], Matrix, u8, [u8; W]) {
    let mut m_row = [[0u8; W]; W];
    let mut m_col = [[0u8; W]; W];
    let mut br = 0u8;
    let mut bc = 0u8;
    let mut row_res: [Vec<u8>; W] = Default::default();
    let mut col_res: [Vec<u8>; W] = Default::default();
    for pos in 0..(W * W) {
        let tile = s.0[pos];
        let r = (pos / W) as u8;
        let c = (pos % W) as u8;
        if tile == 0 {
            br = r;
            bc = c;
            continue;
        }
        let goal_pos = (tile - 1) as usize;
        let gr = (goal_pos / W) as u8;
        let gc = (goal_pos % W) as u8;
        m_row[r as usize][gr as usize] += 1;
        m_col[c as usize][gc as usize] += 1;
        if gr == r {
            row_res[r as usize].push(gc); // physical-column order
        }
        if gc == c {
            col_res[c as usize].push(gr); // physical-row order
        }
    }
    let mut dem_row = [0u8; W];
    let mut dem_col = [0u8; W];
    for g in 0..W {
        dem_row[g] = (row_res[g].len() - lis_strict(&row_res[g])) as u8;
        dem_col[g] = (col_res[g].len() - lis_strict(&col_res[g])) as u8;
    }
    (m_row, br, dem_row, m_col, bc, dem_col)
}

// ---- sharp vector-constrained A* on the product graph -----------------------
//
// Product state = (WD-state, per-line escape counters c_g saturating at dem[g]).
// Counters are mixed-radix-encoded over the demanded lines only (radix dem[g]+1),
// so the product stays tiny when few lines are demanded. Goal = reach the goal
// contingency with every counter saturated. h = unconstrained WD (consistent),
// so the first pop of the saturated goal is optimal.

/// Min cost of an abstract plan `start → goal` making ≥ dem[g] escapes of each
/// goal-line g (an escape = a type-g tile leaving physical line g). Returns None
/// only if the pop budget is exhausted first.
fn cwd_axis(table: &WdTable, m: &Matrix, blank: u8, goal: u64, dem: &[u8; W]) -> Option<u8> {
    // Demanded lines and mixed-radix layout.
    let lines: Vec<usize> = (0..W).filter(|&g| dem[g] > 0).collect();
    if lines.is_empty() {
        return Some(*table.get(&pack(m, blank)).expect("start reachable"));
    }
    let mut radix = [1u32; W]; // multiplier for each demanded line's counter
    let mut full: u32 = 1;
    for (i, &g) in lines.iter().enumerate() {
        radix[i] = full;
        full *= dem[g] as u32 + 1;
    }
    let full_index = full - 1; // all counters at max
    let counter_of = |g: usize| -> Option<usize> { lines.iter().position(|&x| x == g) };

    let start_key = pack(m, blank);
    let h0 = *table.get(&start_key).expect("start reachable") as usize;

    // best[(wd_key<<16 | counter_index)] = g (bit 7 closed).
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
                    // escape of line t iff a type-t tile leaves physical row t
                    let mut ci2 = ci;
                    if from == t {
                        if let Some(idx) = counter_of(t) {
                            let cur = (ci / radix[idx]) % (dem[t] as u32 + 1);
                            if cur < dem[t] as u32 {
                                ci2 += radix[idx]; // bump this line's counter
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

/// WD(s) = row-WD + col-WD, both plain table lookups.
fn wd(table: &WdTable, m_row: &Matrix, br: u8, m_col: &Matrix, bc: u8) -> u8 {
    let hr = *table.get(&pack(m_row, br)).expect("row reachable");
    let hc = *table.get(&pack(m_col, bc)).expect("col reachable");
    hr + hc
}

/// Full sharp cWD(s) = cWD_row + cWD_col (each ≥ its WD half). `nanos`/`evals`
/// accumulate the measurement A\* cost.
fn cwd(table: &WdTable, goal: u64, s: &State, nanos: &mut u128, evals: &mut u64) -> (u8, u8) {
    let (mr, br, dr, mc, bc, dc) = project(s);
    let wdv = wd(table, &mr, br, &mc, bc);
    let t = Instant::now();
    let cr = cwd_axis(table, &mr, br, goal, &dr);
    let cc = cwd_axis(table, &mc, bc, goal, &dc);
    *nanos += t.elapsed().as_nanos();
    *evals += 1;
    // If the budget was hit, fall back to the WD half (admissible).
    let cwdv = cr.unwrap_or_else(|| *table.get(&pack(&mr, br)).unwrap())
        + cc.unwrap_or_else(|| *table.get(&pack(&mc, bc)).unwrap());
    (wdv, cwdv.max(wdv))
}

/// Single-line-max cWD for one axis: max over demanded lines g of the cWD with
/// ONLY line g constrained (the buildable-table approximation). Also returns the
/// number of demanded lines (≥2 ⇒ multi-line coupling, where single-line-max can
/// undershoot full). Falls back to WD if any A\* exhausts its budget.
fn cwd_axis_singlemax(
    table: &WdTable,
    m: &Matrix,
    blank: u8,
    goal: u64,
    dem: &[u8; W],
) -> (u8, u8) {
    let wd0 = *table.get(&pack(m, blank)).expect("start reachable");
    let mut best = wd0;
    let mut ndem = 0u8;
    for g in 0..W {
        if dem[g] > 0 {
            ndem += 1;
            let mut single = [0u8; W];
            single[g] = dem[g];
            best = best.max(cwd_axis(table, m, blank, goal, &single).unwrap_or(wd0));
        }
    }
    (best, ndem)
}

/// (wd, full sharp cWD, single-line-max cWD, max demanded-line count over axes).
fn cwd_ext(
    table: &WdTable,
    goal: u64,
    s: &State,
    nanos: &mut u128,
    evals: &mut u64,
) -> (u8, u8, u8, u8) {
    let (mr, br, dr, mc, bc, dc) = project(s);
    let wd_row = *table.get(&pack(&mr, br)).expect("row reachable");
    let wd_col = *table.get(&pack(&mc, bc)).expect("col reachable");
    let wdv = wd_row + wd_col;
    let t = Instant::now();
    let cr_full = cwd_axis(table, &mr, br, goal, &dr).unwrap_or(wd_row);
    let cc_full = cwd_axis(table, &mc, bc, goal, &dc).unwrap_or(wd_col);
    let (cr_s, nr) = cwd_axis_singlemax(table, &mr, br, goal, &dr);
    let (cc_s, nc) = cwd_axis_singlemax(table, &mc, bc, goal, &dc);
    *nanos += t.elapsed().as_nanos();
    *evals += 1;
    let cwd_full = (cr_full + cc_full).max(wdv);
    let cwd_single = (cr_s + cc_s).max(wdv);
    (wdv, cwd_full, cwd_single, nr.max(nc))
}

// ---- randomized bounded DFS with reservoir sampling -------------------------

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
    let mut a = [0u8; 25];
    for i in 1..25u8 {
        a[i as usize] = 25 - i;
    }
    State(a)
}

struct Node {
    state: State,
    depth: u16,
}

struct Sampler {
    table: WdTable,
    threshold: u16,
    rng: Rng,
    cap: u64,
    expansions: u64,
    reservoir: Vec<Node>,
    seen: u64,
    reservoir_size: usize,
    // depth histogram of ALL expanded nodes (not just sampled)
    depth_hist: Vec<u64>,
}

impl Sampler {
    fn consider(&mut self, state: &State, depth: u16) {
        if (depth as usize) >= self.depth_hist.len() {
            self.depth_hist.resize(depth as usize + 1, 0);
        }
        self.depth_hist[depth as usize] += 1;
        self.seen += 1;
        // reservoir sampling
        if self.reservoir.len() < self.reservoir_size {
            self.reservoir.push(Node {
                state: *state,
                depth,
            });
        } else {
            let j = (self.rng.next() % self.seen) as usize;
            if j < self.reservoir_size {
                self.reservoir[j] = Node {
                    state: *state,
                    depth,
                };
            }
        }
    }

    fn dfs(&mut self, s: &State, g: u16, prev_inv: Option<Move>) {
        if self.expansions >= self.cap {
            return;
        }
        self.expansions += 1;
        self.consider(s, g);
        // shuffle children (Fisher–Yates on the up-to-4 legal moves)
        let mut cands: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| Some(m) != prev_inv)
            .collect();
        for k in (1..cands.len()).rev() {
            let j = (self.rng.next() as usize) % (k + 1);
            cands.swap(k, j);
        }
        for m in cands {
            let child = s.apply(m);
            let (mr, br, _, mc, bc, _) = project(&child);
            let cw = wd(&self.table, &mr, br, &mc, bc);
            if g + 1 + cw as u16 <= self.threshold {
                self.dfs(&child, g + 1, Some(m.inverse()));
                if self.expansions >= self.cap {
                    return;
                }
            }
        }
    }
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

/// Replay a move file from R and report δ = cWD−WD at every depth along the
/// trajectory (WD falls 140→0, so this probes the deep near-solved regime the
/// bounded DFS cannot reach). Tries the file as-is and, if it desyncs, its
/// reverse from GOAL.
fn path_mode(table: &WdTable, goal: u64, file: &str) {
    let text = std::fs::read_to_string(file).expect("read move file");
    let non_comment: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let moves = parse_moves(&non_comment);
    eprintln!("path: {} moves from {file}", moves.len());

    // Replay forward from R; if any move is illegal, bail with a message.
    let mut s = r_board();
    let mut traj = vec![s];
    for (i, &m) in moves.iter().enumerate() {
        if !s.legal_moves().contains(m) {
            eprintln!("  move {i} ({m:?}) illegal from R-forward replay — trying GOAL-reverse");
            traj.clear();
            break;
        }
        s = s.apply(m);
        traj.push(s);
    }
    if traj.is_empty() {
        // reverse: replay inverse moves from GOAL
        let mut s = puzzle8::puzzle24::state::GOAL;
        traj.push(s);
        for &m in moves.iter().rev() {
            let inv = m.inverse();
            if !s.legal_moves().contains(inv) {
                eprintln!("  reverse replay also desynced; aborting path mode");
                return;
            }
            s = s.apply(inv);
            traj.push(s);
        }
        traj.reverse();
    }
    let end = traj.last().unwrap();
    let reaches_goal =
        *end == puzzle8::puzzle24::state::GOAL || traj[0] == puzzle8::puzzle24::state::GOAL;
    eprintln!(
        "  replayed {} states; endpoint {} GOAL",
        traj.len(),
        if reaches_goal {
            "reaches"
        } else {
            "does NOT reach"
        }
    );

    println!(
        "=== δ along R's {}-move solution (depth = moves from R) ===",
        moves.len()
    );
    println!("depth   WD  cWD   δ   demand(row/col nonzero)");
    let mut nanos = 0u128;
    let mut evals = 0u64;
    let mut band: Vec<(u8, i16)> = Vec::new(); // (wd, δ)
    for (d, st) in traj.iter().enumerate() {
        let (wdv, cwdv) = cwd(table, goal, st, &mut nanos, &mut evals);
        let (_, _, dr, _, _, dc) = project(st);
        let nz: Vec<u8> = dr
            .iter()
            .chain(dc.iter())
            .copied()
            .filter(|&x| x > 0)
            .collect();
        let delta = cwdv as i16 - wdv as i16;
        band.push((wdv, delta));
        if d % 4 == 0 || d + 1 == traj.len() {
            println!("{d:>4}  {wdv:>4} {cwdv:>4}  {delta:>3}   {nz:?}");
        }
    }
    // δ̄ by WD band along the path
    println!("\n=== δ̄ by WD band along the path (deep = low WD) ===");
    for &(lo, hi) in &[
        (0u8, 20),
        (21, 40),
        (41, 60),
        (61, 80),
        (81, 100),
        (101, 120),
        (121, 140),
    ] {
        let sub: Vec<i16> = band
            .iter()
            .filter(|(w, _)| *w >= lo && *w <= hi)
            .map(|(_, d)| *d)
            .collect();
        if sub.is_empty() {
            println!("WD {lo}-{hi}: n=0");
            continue;
        }
        let mean = sub.iter().map(|&d| d as f64).sum::<f64>() / sub.len() as f64;
        println!("WD {lo:>3}-{hi:<3}: n={:<3} δ̄={mean:.2}", sub.len());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("path") {
        let file = args
            .get(2)
            .map(|s| s.as_str())
            .unwrap_or("data/r156_ours_solution.txt");
        let t0 = Instant::now();
        let table = load_dist_table(
            Path::new("data/wd24.bin"),
            WD_KIND_FULL,
            Some(FULL_WD_ENTRIES),
        )
        .expect("wd24.bin load");
        eprintln!(
            "WD table: {} entries in {:.1}s",
            table.len(),
            t0.elapsed().as_secs_f64()
        );
        path_mode(&table, goal_key(), file);
        return;
    }
    let cap: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    let threshold: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(144);
    let reservoir_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12_000);

    let t0 = Instant::now();
    let path = Path::new("data/wd24.bin");
    let table = load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).expect("wd24.bin load");
    let goal = goal_key();
    eprintln!(
        "WD table: {} entries in {:.1}s. threshold={threshold}, dfs_cap={cap}, reservoir={reservoir_size}",
        table.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- sanity: R must give WD=140, sharp cWD=144, δ=4 ----
    let r = r_board();
    {
        let mut nanos = 0u128;
        let mut evals = 0u64;
        let (wdv, cwdv) = cwd(&table, goal, &r, &mut nanos, &mut evals);
        let (_, _, dr, _, _, dc) = project(&r);
        eprintln!(
            "sanity R: WD={wdv} (exp 140), demand row={dr:?} col={dc:?}, cWD={cwdv} (exp 144), δ={}",
            cwdv as i16 - wdv as i16
        );
        assert_eq!(
            (wdv, cwdv),
            (140, 144),
            "R sanity failed — cWD wiring is wrong"
        );
    }

    // ---- sample the tree ----
    let mut sampler = Sampler {
        table,
        threshold,
        rng: Rng(0x9E37_79B9_7F4A_7C15),
        cap,
        expansions: 0,
        reservoir: Vec::with_capacity(reservoir_size),
        seen: 0,
        reservoir_size,
        depth_hist: Vec::new(),
    };
    let tdfs = Instant::now();
    sampler.dfs(&r, 0, None);
    eprintln!(
        "DFS: {} expansions in {:.1}s, {} sampled, deepest expanded depth {}",
        sampler.expansions,
        tdfs.elapsed().as_secs_f64(),
        sampler.reservoir.len(),
        sampler.depth_hist.len().saturating_sub(1)
    );

    // depth histogram of expanded nodes (coarse bands)
    eprint!("expanded-node depth histogram: ");
    let hb = &sampler.depth_hist;
    for &(lo, hi) in &[(0usize, 39), (40, 79), (80, 109), (110, 129), (130, 200)] {
        let c: u64 = hb
            .iter()
            .enumerate()
            .filter(|(d, _)| *d >= lo && *d <= hi)
            .map(|(_, &v)| v)
            .sum();
        eprint!("[{lo}-{hi}]={c} ");
    }
    eprintln!();

    // ---- compute δ over the reservoir ----
    let table = &sampler.table;
    let mut nanos = 0u128;
    let mut evals = 0u64;
    struct Row {
        depth: u16,
        dfull: i16,   // full sharp surcharge cWD − WD
        dsingle: i16, // single-line-max surcharge
        slack: i16,
        ndem: u8, // max demanded lines over the two axes
    }
    let mut rows: Vec<Row> = Vec::with_capacity(sampler.reservoir.len());
    for node in &sampler.reservoir {
        let (wdv, cfull, csingle, ndem) = cwd_ext(table, goal, &node.state, &mut nanos, &mut evals);
        let slack = threshold as i16 - (node.depth as i16 + wdv as i16);
        rows.push(Row {
            depth: node.depth,
            dfull: cfull as i16 - wdv as i16,
            dsingle: csingle as i16 - wdv as i16,
            slack,
            ndem,
        });
    }
    eprintln!(
        "cWD: {} evals in {:.1}s ({:.0}/s)\n",
        evals,
        nanos as f64 / 1e9,
        evals as f64 / (nanos as f64 / 1e9).max(1e-9)
    );

    // node-weighted δ̄ for any surcharge field (weight each depth's reservoir mean
    // by the true expanded-node count at that depth).
    let hb = &sampler.depth_hist;
    let node_weighted = |fld: &dyn Fn(&Row) -> i16| -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for d in 0..hb.len() {
            let dd = d as u16;
            let vals: Vec<i16> = rows.iter().filter(|r| r.depth == dd).map(fld).collect();
            if vals.is_empty() {
                continue;
            }
            let mean = vals.iter().map(|&x| x as f64).sum::<f64>() / vals.len() as f64;
            num += hb[d] as f64 * mean;
            den += hb[d] as f64;
        }
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    };

    let report = |label: &str, sel: &dyn Fn(&Row) -> bool, fld: &dyn Fn(&Row) -> i16| {
        let sub: Vec<&Row> = rows.iter().filter(|r| sel(r)).collect();
        if sub.is_empty() {
            println!("{label:<18} n=0");
            return;
        }
        let n = sub.len() as f64;
        let mean = sub.iter().map(|r| fld(r) as f64).sum::<f64>() / n;
        let ge2 = sub.iter().filter(|r| fld(r) >= 2).count() as f64 / n * 100.0;
        let ge4 = sub.iter().filter(|r| fld(r) >= 4).count() as f64 / n * 100.0;
        let mut hist = [0u32; 7];
        for r in &sub {
            hist[fld(r).clamp(0, 6) as usize] += 1;
        }
        println!(
            "{label:<18} n={:<6} δ̄={mean:.2}  ≥2:{ge2:>4.0}%  ≥4:{ge4:>4.0}%  node×≈{:>7.1}  hist[0..6]={hist:?}",
            sub.len(),
            29f64.powf(mean / 2.0)
        );
    };
    let full = |r: &Row| r.dfull;
    let single = |r: &Row| r.dsingle;

    println!("=== FULL sharp cWD: δ = cWD − WD by depth ===");
    for &(lo, hi) in &[(0u16, 39), (40, 79), (80, 109), (110, 129), (130, 250)] {
        report(
            &format!("depth {lo}-{hi}"),
            &|r| r.depth >= lo && r.depth <= hi,
            &full,
        );
    }
    println!("\n=== headline (full) ===");
    report("ALL full", &|_| true, &full);
    report("slack≤3 full", &|r| r.slack <= 3, &full);

    println!("\n=== SINGLE-LINE-MAX (the buildable-table approximation) ===");
    for &(lo, hi) in &[(0u16, 39), (40, 79), (80, 109), (110, 129), (130, 250)] {
        report(
            &format!("depth {lo}-{hi}"),
            &|r| r.depth >= lo && r.depth <= hi,
            &single,
        );
    }
    report("ALL single", &|_| true, &single);

    // ---- surcharge distribution + coupling loss ----
    let n = rows.len().max(1) as f64;
    let zero_full = rows.iter().filter(|r| r.dfull == 0).count() as f64 / n * 100.0;
    let multi = rows.iter().filter(|r| r.ndem >= 2).count();
    let multi_pct = multi as f64 / n * 100.0;
    let mut cap_hist = [0u32; 9]; // full surcharge Δ = 0..8
    for r in &rows {
        cap_hist[r.dfull.clamp(0, 8) as usize] += 1;
    }
    // coupling loss: on multi-line nodes, full vs single-line-max
    let ml: Vec<&Row> = rows.iter().filter(|r| r.ndem >= 2).collect();
    let (ml_full, ml_single) = if ml.is_empty() {
        (0.0, 0.0)
    } else {
        (
            ml.iter().map(|r| r.dfull as f64).sum::<f64>() / ml.len() as f64,
            ml.iter().map(|r| r.dsingle as f64).sum::<f64>() / ml.len() as f64,
        )
    };

    println!("\n=== SURCHARGE STRUCTURE (Δ = cWD − WD) — the compression signal ===");
    println!("Δ_full == 0 (compressible-away): {zero_full:.0}% of sampled nodes");
    println!("Δ_full histogram [0..8]:         {cap_hist:?}");
    println!("multi-line demand nodes (ndem≥2, where coupling matters): {multi_pct:.0}%");
    println!(
        "  on those nodes: full δ̄={ml_full:.2}  vs single-line-max δ̄={ml_single:.2}  (coupling loss {:.2})",
        ml_full - ml_single
    );

    let nw_full = node_weighted(&full);
    let nw_single = node_weighted(&single);
    let total_expanded: f64 = hb.iter().map(|&v| v as f64).sum();
    let den_cov: f64 = (0..hb.len())
        .filter(|&d| rows.iter().any(|r| r.depth == d as u16))
        .map(|d| hb[d] as f64)
        .sum();
    println!("\n=== NODE-WEIGHTED HEADLINE (weighted by true expanded-node counts) ===");
    println!(
        "FULL sharp cWD:      δ̄ = {nw_full:.2}   (est. node× ≈ {:.1})",
        29f64.powf(nw_full / 2.0)
    );
    println!(
        "SINGLE-LINE-MAX:     δ̄ = {nw_single:.2}   (est. node× ≈ {:.1})   retains {:.0}% of full's gain",
        29f64.powf(nw_single / 2.0),
        if nw_full > 0.0 { nw_single / nw_full * 100.0 } else { 0.0 }
    );
    println!(
        "(reservoir covers {:.0}% of expanded depths)",
        den_cov / total_expanded * 100.0
    );
    let uncapped = sampler.expansions < cap;
    println!(
        "DFS {} (expansions {} vs cap {}) — weighting is {}",
        if uncapped {
            "FULLY EXHAUSTED"
        } else {
            "CAPPED (depth dist is DFS-biased)"
        },
        sampler.expansions,
        cap,
        if uncapped { "exact" } else { "approximate" }
    );
    println!("\ntotal {:.1}s", t0.elapsed().as_secs_f64());
}
