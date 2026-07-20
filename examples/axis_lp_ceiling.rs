//! Multi-commodity axis-coupled LP ceiling for the R board — the direction
//! Menger does NOT cap (§8m). Fail-fast: can a fractional cut-covering LP
//! (cuts sharing edge capacity) beat cWD(R)=144?
//!
//! §8m proved the *integral* edge-disjoint cut packing is capped at
//! Manhattan(R)=112 by Menger. The *fractional* relaxation lets cuts share
//! edges, which is exactly the multi-commodity / operator-counting coupling.
//! Concretely:
//!
//!   Let k_e >= 0 be the number of solution moves that cross grid edge e.
//!   Total moves L = sum_e k_e. For every cut S, the moves crossing S equal
//!   sum_{e in boundary(S)} k_e and must be >= lb(S) (§8m: lb(S) = #tiles whose
//!   start/goal straddle S). So any real solution's crossing profile satisfies
//!
//!       COVERING LP:  min  sum_e k_e
//!                     s.t. sum_{e in dS} k_e >= lb(S)   for all cuts S in F
//!                          k_e >= 0
//!
//!   and its optimum is an admissible lower bound on d* (the real profile is
//!   feasible, so the min is <= L). Its LP dual is the fractional cut packing
//!
//!       PACKING LP:   max  sum_S lb(S) y_S
//!                     s.t. sum_{S: e in dS} y_S <= 1     for all edges e
//!                          y_S >= 0
//!
//!   By strong duality the two optima coincide = the coupled ceiling. With only
//!   the 8 axis cuts the optimum is Manhattan=112 (§8m); enriching F with
//!   non-axis cuts that SHARE edges can only raise it. This measures how high.
//!
//! RESULT (it does NOT rise — the fractional LP is Manhattan too). Route each
//! tile on a shortest path; the induced edge-crossing counts k_e are feasible
//! for EVERY cut constraint (a straddling tile crosses each separating cut an
//! odd >=1 number of times, so sum_{e in dS} k_e >= lb(S)) and sum_e k_e =
//! Manhattan. Hence the FULL covering LP <= Manhattan <= (axis lower bound)
//! covering LP, so it equals Manhattan exactly — the multi-commodity coupling
//! gains nothing. The reason: sliding-puzzle edges have UNBOUNDED per-plan
//! capacity (a tile may recross an edge for free in the relaxation), so no
//! cut/flow bound can see what actually makes R hard — intra-line ORDERING,
//! which is a conflict constraint, not a cut. This run exhibits it: the LP
//! optimum is 112 with the 8 axis cuts as its entire support.
//!
//! We solve the PACKING LP (all-slack start is feasible => one-phase simplex)
//! over a rich cut family (rectangles + multi-order prefix sweeps), and report
//! the coupled ceiling against Manhattan(112)/WD(140)/cWD(144)/d*(156). Support
//! cuts (y_S>0) are printed with their observed crossings on R's 156-move
//! optimal path as a soundness witness (obs>=lb).
//!
//! Run from repo root:  cargo run --release --example axis_lp_ceiling [PATHFILE]

use puzzle8::puzzle24::state::{Move, State, N_CELLS, W};

const NCELL: usize = N_CELLS;
const FULL_MASK: u32 = (1u32 << NCELL) - 1;

// ---- board / path (mirror slack_anatomy.rs / mincut_ceiling.rs) ------------

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

// ---- grid geometry ---------------------------------------------------------

fn edges() -> Vec<(usize, usize)> {
    let mut e = Vec::with_capacity(40);
    for r in 0..W {
        for c in 0..W {
            let cell = r * W + c;
            if c + 1 < W {
                e.push((cell, cell + 1));
            }
            if r + 1 < W {
                e.push((cell, cell + W));
            }
        }
    }
    e
}

fn edge_cross_mask(s: u32, edges: &[(usize, usize)]) -> u64 {
    let mut m = 0u64;
    for (i, &(a, b)) in edges.iter().enumerate() {
        if ((s >> a) ^ (s >> b)) & 1 == 1 {
            m |= 1u64 << i;
        }
    }
    m
}

/// lb(S) = straddling tiles (blank excluded). R board: tile t starts 25-t, goal t-1.
fn lb(s: u32) -> u32 {
    let mut n = 0;
    for t in 1u32..=24 {
        let (start, goal) = ((25 - t) as usize, (t - 1) as usize);
        if ((s >> start) ^ (s >> goal)) & 1 == 1 {
            n += 1;
        }
    }
    n
}

fn manhattan_r() -> u32 {
    let mut d = 0;
    for t in 1usize..=24 {
        let (start, goal) = (25 - t, t - 1);
        let (sr, sc) = (start / W, start % W);
        let (gr, gc) = (goal / W, goal % W);
        d += (sr as i32 - gr as i32).unsigned_abs() + (sc as i32 - gc as i32).unsigned_abs();
    }
    d
}

fn observed_crossings(start: &State, path: &[Move], s: u32) -> u32 {
    let mut st = *start;
    let mut blank = st.0.iter().position(|&t| t == 0).unwrap();
    let mut n = 0;
    for &m in path {
        let st2 = st.apply(m);
        let b2 = st2.0.iter().position(|&t| t == 0).unwrap();
        if ((s >> blank) ^ (s >> b2)) & 1 == 1 {
            n += 1;
        }
        st = st2;
        blank = b2;
    }
    n
}

fn describe(mask: u32) -> String {
    let (mut r0, mut r1, mut c0, mut c1) = (W, 0usize, W, 0usize);
    let mut cnt = 0;
    for cell in 0..NCELL {
        if (mask >> cell) & 1 == 1 {
            let (r, c) = (cell / W, cell % W);
            r0 = r0.min(r);
            r1 = r1.max(r);
            c0 = c0.min(c);
            c1 = c1.max(c);
            cnt += 1;
        }
    }
    format!("bbox rows {r0}..={r1} cols {c0}..={c1} ({cnt} cells)")
}

// ---- cut family ------------------------------------------------------------

fn add(out: &mut Vec<u32>, s: u32) {
    if s != 0 && s != FULL_MASK {
        out.push(s);
    }
}

/// Rich family: rectangles + prefix sweeps under several cell orderings.
/// More cuts only raise the covering LP optimum.
fn candidate_cuts() -> Vec<u32> {
    let mut out = Vec::new();

    // all axis-aligned rectangles (convex regions; includes axis half-planes)
    for r0 in 0..W {
        for r1 in r0..W {
            for c0 in 0..W {
                for c1 in c0..W {
                    let mut s = 0u32;
                    for r in r0..=r1 {
                        for c in c0..=c1 {
                            s |= 1u32 << (r * W + c);
                        }
                    }
                    add(&mut out, s);
                }
            }
        }
    }

    // prefix sweeps under several orderings -> non-convex threshold cuts
    let key = |cell: usize, ord: usize| -> i32 {
        let (r, c) = ((cell / W) as i32, (cell % W) as i32);
        match ord {
            0 => r * 5 + c,                        // row-major
            1 => c * 5 + r,                        // col-major
            2 => r + c,                            // diagonal
            3 => r - c,                            // anti-diagonal
            4 => (r - 2).abs() + (c - 2).abs(),    // center-out (Manhattan from center)
            _ => (r - 2).abs().max((c - 2).abs()), // ring (Chebyshev from center)
        }
    };
    for ord in 0..6 {
        let mut cells: Vec<usize> = (0..NCELL).collect();
        cells.sort_by_key(|&cell| key(cell, ord));
        let mut s = 0u32;
        for &cell in &cells[..NCELL - 1] {
            s |= 1u32 << cell;
            add(&mut out, s);
        }
    }

    // NB: scattered random subsets were tried and dropped — they have many
    // cut-edges but few straddlers (low lb/edge), never bind the covering LP,
    // and only bloat the tableau. Congestion bottlenecks are COMPACT regions,
    // captured by the rectangles + prefix sweeps above.

    out
}

// ---- exact LP (one-phase primal simplex; Dantzig + Bland fallback) ---------

/// max c·x  s.t.  A x <= b,  x >= 0,  with b >= 0 (all-slack start feasible).
/// Returns (optimum, x). Dense tableau; sizes here are ~40 rows.
fn simplex_max(a: &[Vec<f64>], b: &[f64], c: &[f64]) -> (f64, Vec<f64>) {
    let m = a.len();
    let n = c.len();
    let total = n + m; // structural + slack
    let mut t = vec![vec![0f64; total + 1]; m + 1];
    for i in 0..m {
        t[i][..n].copy_from_slice(&a[i][..n]);
        t[i][n + i] = 1.0;
        t[i][total] = b[i];
    }
    for j in 0..n {
        t[m][j] = -c[j];
    }
    let mut basis: Vec<usize> = (n..n + m).collect();
    const EPS: f64 = 1e-9;

    for iter in 0..1_000_000 {
        let bland = iter > 20_000; // switch to anti-cycling rule if it drags
                                   // entering column
        let mut pc = None;
        if bland {
            for j in 0..total {
                if t[m][j] < -EPS {
                    pc = Some(j);
                    break;
                }
            }
        } else {
            let mut best = -EPS;
            for j in 0..total {
                if t[m][j] < best {
                    best = t[m][j];
                    pc = Some(j);
                }
            }
        }
        let pc = match pc {
            Some(c) => c,
            None => break, // optimal
        };
        // leaving row: min ratio, Bland tie-break by basis index
        let mut pr = None;
        let mut best_ratio = f64::INFINITY;
        let mut best_basis = usize::MAX;
        for i in 0..m {
            if t[i][pc] > EPS {
                let ratio = t[i][total] / t[i][pc];
                if ratio < best_ratio - EPS || (ratio < best_ratio + EPS && basis[i] < best_basis) {
                    best_ratio = ratio;
                    best_basis = basis[i];
                    pr = Some(i);
                }
            }
        }
        let pr = match pr {
            Some(r) => r,
            None => {
                eprintln!("LP unbounded (should not happen: y_S<=1)");
                break;
            }
        };
        // pivot on (pr, pc)
        let pv = t[pr][pc];
        for j in 0..=total {
            t[pr][j] /= pv;
        }
        for i in 0..=m {
            if i != pr {
                let f = t[i][pc];
                if f.abs() > 1e-15 {
                    for j in 0..=total {
                        t[i][j] -= f * t[pr][j];
                    }
                }
            }
        }
        basis[pr] = pc;
    }

    let val = t[m][total];
    let mut x = vec![0f64; n];
    for i in 0..m {
        if basis[i] < n {
            x[basis[i]] = t[i][total];
        }
    }
    (val, x)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("data/r156_ours_solution.txt");

    let start = r_board();
    let text = std::fs::read_to_string(file).expect("read move file");
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let path = parse_moves(&body);
    {
        let mut s = start;
        for (i, &m) in path.iter().enumerate() {
            assert!(s.legal_moves().contains(m), "move {i} illegal");
            s = s.apply(m);
        }
    }

    let edges = edges();
    let man = manhattan_r();

    // build cuts, dedup by edge-set keeping max lb
    let mut by_edges: std::collections::HashMap<u64, (u32, u32)> = std::collections::HashMap::new();
    for mask in candidate_cuts() {
        let em = edge_cross_mask(mask, &edges);
        if em == 0 {
            continue;
        }
        let l = lb(mask);
        by_edges
            .entry(em)
            .and_modify(|e| {
                if l > e.1 {
                    *e = (mask, l);
                }
            })
            .or_insert((mask, l));
    }
    // keep only lb>0 cuts (lb=0 contributes nothing to the packing objective)
    let cuts: Vec<(u32, u64, u32)> = by_edges
        .into_iter()
        .filter(|(_, (_, l))| *l > 0)
        .map(|(em, (mask, l))| (mask, em, l))
        .collect();

    // PACKING LP:  max sum lb_S y_S  s.t.  for each edge e: sum_{S: e in dS} y_S <= 1
    let m = edges.len(); // 40 edge constraints
    let n = cuts.len();
    let mut a = vec![vec![0f64; n]; m];
    for (j, &(_, em, _)) in cuts.iter().enumerate() {
        for e in 0..m {
            if (em >> e) & 1 == 1 {
                a[e][j] = 1.0;
            }
        }
    }
    let b = vec![1.0f64; m];
    let c: Vec<f64> = cuts.iter().map(|&(_, _, l)| l as f64).collect();

    let (opt, y) = simplex_max(&a, &b, &c);
    let ceiling = opt.round() as i64; // integral data; optimum is rational, report rounded + raw

    println!("== multi-commodity axis-coupled LP ceiling on R ==");
    println!("path: {} moves from {file}", path.len());
    println!("cut family (dedup by edge-set, lb>0): {n}");
    println!();
    println!("  Manhattan(R)              = {man}");
    println!("  WD(R)                     = 140   (established)");
    println!("  cWD(R)                    = 144   (established)");
    println!("  coupled LP CEILING        = {opt:.4}  (~{ceiling})   (fractional cut-covering LP)");
    println!("  d*(R)                     = 156   (best-known; LB 150)");
    println!();

    // support cuts (y_S > 0): weight, lb, observed crossings (soundness witness)
    let mut support: Vec<(f64, u32, u32, u32, String)> = Vec::new();
    for (j, &yj) in y.iter().enumerate() {
        if yj > 1e-6 {
            let (mask, em, l) = cuts[j];
            let obs = observed_crossings(&start, &path, mask);
            support.push((yj, l, obs, em.count_ones(), describe(mask)));
        }
    }
    support.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!(
        "support cuts (y | lb | observed-on-path | #edges | region)  [{} active]:",
        support.len()
    );
    let mut sound_ok = true;
    for (yj, l, obs, ne, desc) in support.iter().take(30) {
        if obs < l {
            sound_ok = false;
        }
        let flag = if obs >= l { "" } else { "  <-- UNSOUND?!" };
        println!("  y {yj:.3} | lb {l:2} | obs {obs:3} | {ne:2}e | {desc}{flag}");
    }
    println!();
    println!(
        "  soundness (all support observed >= lb): {}",
        if sound_ok { "OK" } else { "VIOLATED" }
    );
    println!();

    let verdict = if opt > 144.0 + 1e-6 {
        format!("CEILING {opt:.2} > cWD 144  ->  PURSUE: axis-coupled congestion beats cWD on R")
    } else if (opt - man as f64).abs() < 1e-6 {
        format!(
            "CEILING {opt:.2} == Manhattan {man}  ->  DEAD: the multi-commodity/fractional cut LP\n         equals Manhattan exactly (shortest-path routing is feasible for every cut\n         constraint), so congestion-coupling gains NOTHING over the single-commodity\n         §8m bound. Sliding-puzzle edges have unbounded per-plan capacity, so no\n         cut/flow bound sees intra-line ORDERING (a conflict, not a cut) — the thing\n         cWD already charges. The remaining lever is not a cut/flow LP at all: it is\n         COST-PARTITIONING an ordering bound (cWD) with a cross-axis tile-identity\n         abstraction (PDB) — a different LP, over abstractions, not cuts."
        )
    } else {
        format!(
            "CEILING {opt:.2} in (Manhattan {man}, cWD 144]  ->  coupling helps but does not beat cWD"
        )
    };
    println!("VERDICT: {verdict}");
}
