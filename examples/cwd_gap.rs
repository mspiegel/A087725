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
    wd: u8,
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
    fn consider(&mut self, state: &State, depth: u16, wdv: u8) {
        if (depth as usize) >= self.depth_hist.len() {
            self.depth_hist.resize(depth as usize + 1, 0);
        }
        self.depth_hist[depth as usize] += 1;
        self.seen += 1;
        // reservoir sampling
        if self.reservoir.len() < self.reservoir_size {
            self.reservoir.push(Node { state: state.clone(), depth, wd: wdv });
        } else {
            let j = (self.rng.next() % self.seen) as usize;
            if j < self.reservoir_size {
                self.reservoir[j] = Node { state: state.clone(), depth, wd: wdv };
            }
        }
    }

    fn dfs(&mut self, s: &State, g: u16, prev_inv: Option<Move>, wdv: u8) {
        if self.expansions >= self.cap {
            return;
        }
        self.expansions += 1;
        self.consider(s, g, wdv);
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
                self.dfs(&child, g + 1, Some(m.inverse()), cw);
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
    let non_comment: String = text.lines().filter(|l| !l.trim_start().starts_with('#')).collect::<Vec<_>>().join(" ");
    let moves = parse_moves(&non_comment);
    eprintln!("path: {} moves from {file}", moves.len());

    // Replay forward from R; if any move is illegal, bail with a message.
    let mut s = r_board();
    let mut traj = vec![s.clone()];
    for (i, &m) in moves.iter().enumerate() {
        if !s.legal_moves().contains(m) {
            eprintln!("  move {i} ({m:?}) illegal from R-forward replay — trying GOAL-reverse");
            traj.clear();
            break;
        }
        s = s.apply(m);
        traj.push(s.clone());
    }
    if traj.is_empty() {
        // reverse: replay inverse moves from GOAL
        let mut s = puzzle8::puzzle24::state::GOAL;
        traj.push(s.clone());
        for &m in moves.iter().rev() {
            let inv = m.inverse();
            if !s.legal_moves().contains(inv) {
                eprintln!("  reverse replay also desynced; aborting path mode");
                return;
            }
            s = s.apply(inv);
            traj.push(s.clone());
        }
        traj.reverse();
    }
    let end = traj.last().unwrap();
    let reaches_goal = *end == puzzle8::puzzle24::state::GOAL || traj[0] == puzzle8::puzzle24::state::GOAL;
    eprintln!(
        "  replayed {} states; endpoint {} GOAL",
        traj.len(),
        if reaches_goal { "reaches" } else { "does NOT reach" }
    );

    println!("=== δ along R's {}-move solution (depth = moves from R) ===", moves.len());
    println!("depth   WD  cWD   δ   demand(row/col nonzero)");
    let mut nanos = 0u128;
    let mut evals = 0u64;
    let mut band: Vec<(u8, i16)> = Vec::new(); // (wd, δ)
    for (d, st) in traj.iter().enumerate() {
        let (wdv, cwdv) = cwd(table, goal, st, &mut nanos, &mut evals);
        let (_, _, dr, _, _, dc) = project(st);
        let nz: Vec<u8> = dr.iter().chain(dc.iter()).copied().filter(|&x| x > 0).collect();
        let delta = cwdv as i16 - wdv as i16;
        band.push((wdv, delta));
        if d % 4 == 0 || d + 1 == traj.len() {
            println!("{d:>4}  {wdv:>4} {cwdv:>4}  {delta:>3}   {nz:?}");
        }
    }
    // δ̄ by WD band along the path
    println!("\n=== δ̄ by WD band along the path (deep = low WD) ===");
    for &(lo, hi) in &[(0u8, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 120), (121, 140)] {
        let sub: Vec<i16> = band.iter().filter(|(w, _)| *w >= lo && *w <= hi).map(|(_, d)| *d).collect();
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
        let file = args.get(2).map(|s| s.as_str()).unwrap_or("data/r156_ours_solution.txt");
        let t0 = Instant::now();
        let table = load_dist_table(Path::new("data/wd24.bin"), WD_KIND_FULL, Some(FULL_WD_ENTRIES))
            .expect("wd24.bin load");
        eprintln!("WD table: {} entries in {:.1}s", table.len(), t0.elapsed().as_secs_f64());
        path_mode(&table, goal_key(), file);
        return;
    }
    let cap: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000_000);
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
        assert_eq!((wdv, cwdv), (140, 144), "R sanity failed — cWD wiring is wrong");
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
    let (mr, br, _, mc, bc, _) = project(&r);
    let wr = wd(&sampler.table, &mr, br, &mc, bc);
    let tdfs = Instant::now();
    sampler.dfs(&r, 0, None, wr);
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
        let c: u64 = hb.iter().enumerate().filter(|(d, _)| *d >= lo && *d <= hi).map(|(_, &v)| v).sum();
        eprint!("[{lo}-{hi}]={c} ");
    }
    eprintln!();

    // ---- compute δ over the reservoir ----
    let table = &sampler.table;
    let mut nanos = 0u128;
    let mut evals = 0u64;
    let mut rows: Vec<(u16, u8, i16, i16)> = Vec::with_capacity(sampler.reservoir.len()); // depth, wd, δ, slack
    for node in &sampler.reservoir {
        let (wdv, cwdv) = cwd(table, goal, &node.state, &mut nanos, &mut evals);
        let delta = cwdv as i16 - wdv as i16;
        let slack = threshold as i16 - (node.depth as i16 + wdv as i16);
        rows.push((node.depth, wdv, delta, slack));
    }
    eprintln!(
        "cWD: {} evals in {:.1}s ({:.0}/s)\n",
        evals,
        nanos as f64 / 1e9,
        evals as f64 / (nanos as f64 / 1e9).max(1e-9)
    );

    let report = |label: &str, sel: &dyn Fn(&(u16, u8, i16, i16)) -> bool| {
        let sub: Vec<&(u16, u8, i16, i16)> = rows.iter().filter(|r| sel(r)).collect();
        if sub.is_empty() {
            println!("{label:<18} n=0");
            return;
        }
        let n = sub.len() as f64;
        let mean = sub.iter().map(|r| r.2 as f64).sum::<f64>() / n;
        let ge2 = sub.iter().filter(|r| r.2 >= 2).count() as f64 / n * 100.0;
        let ge4 = sub.iter().filter(|r| r.2 >= 4).count() as f64 / n * 100.0;
        let mut hist = [0u32; 7];
        for r in &sub {
            hist[r.2.clamp(0, 6) as usize] += 1;
        }
        let red = 29f64.powf(mean / 2.0);
        let n_i = sub.len();
        println!(
            "{label:<18} n={n_i:<6} δ̄={mean:.2}  ≥2:{ge2:>4.0}%  ≥4:{ge4:>4.0}%  node×≈{red:>7.1}  δ-hist[0..6]={hist:?}"
        );
    };

    println!("=== δ = cWD − WD by expanded-node depth ===");
    for &(lo, hi) in &[(0u16, 39), (40, 79), (80, 109), (110, 129), (130, 250)] {
        report(&format!("depth {lo}-{hi}"), &|r| r.0 >= lo && r.0 <= hi);
    }
    println!("\n=== δ by slack (T − f); low slack dominates the iteration count ===");
    for &(lo, hi) in &[(0i16, 1), (2, 3), (4, 7), (8, 15), (16, 300)] {
        report(&format!("slack {lo}-{hi}"), &|r| r.3 >= lo && r.3 <= hi);
    }
    println!("\n=== headline ===");
    report("ALL", &|_| true);
    report("depth≥110", &|r| r.0 >= 110);
    report("slack≤3", &|r| r.3 <= 3);

    // Node-weighted δ̄: weight each depth's reservoir-mean δ by the TRUE count of
    // expanded nodes at that depth (depth_hist). Exact when the DFS is uncapped.
    let hb = &sampler.depth_hist;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for d in 0..hb.len() {
        let dd = d as u16;
        let sub: Vec<i16> = rows.iter().filter(|r| r.0 == dd).map(|r| r.2).collect();
        if sub.is_empty() {
            continue;
        }
        let mean = sub.iter().map(|&x| x as f64).sum::<f64>() / sub.len() as f64;
        num += hb[d] as f64 * mean;
        den += hb[d] as f64;
    }
    // depths with reservoir coverage account for `den` of the total expanded
    let total_expanded: f64 = hb.iter().map(|&v| v as f64).sum();
    let weighted = if den > 0.0 { num / den } else { 0.0 };
    println!(
        "\nNODE-WEIGHTED δ̄ = {weighted:.2}  (est. node× ≈ {:.1}; reservoir covers {:.0}% of expanded depths)",
        29f64.powf(weighted / 2.0),
        den / total_expanded * 100.0
    );
    let uncapped = sampler.expansions < cap;
    println!(
        "DFS {} (expansions {} vs cap {}) — weighting is {}",
        if uncapped { "FULLY EXHAUSTED" } else { "CAPPED (depth dist is DFS-biased)" },
        sampler.expansions,
        cap,
        if uncapped { "exact" } else { "approximate" }
    );
    println!("\ntotal {:.1}s", t0.elapsed().as_secs_f64());
}
