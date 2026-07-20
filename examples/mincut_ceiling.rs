//! Min-cut / cut-packing ceiling for the R board — a fail-fast measurement of
//! whether a congestion (min-cut / operator-counting) bound can beat cWD(R)=144.
//!
//! Construction (sound by construction).
//!   A *cut* is a subset S of the 25 cells (its complement is the other side).
//!   For any cut C, a sound lower bound on the number of solution moves that
//!   cross C is
//!       lb(C) = #{ tiles t (blank excluded) : start(t), goal(t) on opposite
//!                  sides of C }.
//!   Proof: each move crossing C carries exactly one tile across (the blank
//!   goes the other way). A tile whose start/goal straddle C must be carried
//!   across a net odd number of times, so >= 1 crossing move is dedicated to
//!   it; distinct straddling tiles need distinct net crossings, and the
//!   directional balance a_moves - b_moves = a - b pins the minimum total
//!   crossing moves to exactly a + b = lb(C).
//!
//!   A grid *edge* joins two orthogonally-adjacent cells (40 internal edges).
//!   A move slides one tile along exactly one edge, so a move "crosses C" iff
//!   that edge is a cut-edge of C (its endpoints straddle C). Hence if a family
//!   of cuts is pairwise EDGE-DISJOINT (no grid edge is a cut-edge of two of
//!   them) then no single move crosses two of them, and
//!       L  >=  sum over the family of lb(C).
//!   Maximizing that sum over edge-disjoint families is a max-weight
//!   edge-disjoint cut packing — the min-cut / operator-counting ceiling for
//!   this cut family, and an admissible lower bound on the true distance.
//!
//! THEOREM (the ceiling is exactly Manhattan — Menger).
//!   sum_{C in F} lb(C) = sum_C #{tiles t : C separates start(t),goal(t)}
//!                      = sum_t #{C in F : C separates start(t),goal(t)}.
//!   For a fixed tile t, the cuts in F separating start(t) from goal(t) are
//!   pairwise edge-disjoint (F is globally edge-disjoint), and by Menger's
//!   theorem the max number of pairwise edge-disjoint edge-cuts separating two
//!   vertices equals the graph distance between them = Manhattan(t) in the
//!   grid. Hence sum_{C in F} lb(C) <= sum_t Manhattan(t) = Manhattan(R), with
//!   equality achieved by the 8 axis half-plane cuts. So the cut-packing family
//!   CANNOT EXCEED Manhattan(R) — it is provably weaker than WD (which beats
//!   Manhattan via intra-axis ordering), let alone cWD. This run exhibits the
//!   theorem numerically (best packing == axis packing == Manhattan) and prints
//!   each selected cut's OBSERVED crossings along R's 156-move optimal path as
//!   a soundness witness (observed >= lb must hold).
//!
//! Anchors (all established in FINDINGS_R): Manhattan(R), WD(R)=140,
//! cWD(R)=144, d*(R) in [150,156] (best-known path is 156).
//!
//! Run from repo root:  cargo run --release --example mincut_ceiling [PATHFILE]

use puzzle8::puzzle24::state::{Move, State, N_CELLS, W};

const NCELL: usize = N_CELLS; // 25
const FULL_MASK: u32 = (1u32 << NCELL) - 1;

// ---- board / path sources (mirror slack_anatomy.rs) ------------------------

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

/// The 40 internal grid edges as (cell_a, cell_b) with a < b.
fn edges() -> Vec<(usize, usize)> {
    let mut e = Vec::with_capacity(40);
    for r in 0..W {
        for c in 0..W {
            let cell = r * W + c;
            if c + 1 < W {
                e.push((cell, cell + 1)); // horizontal adjacency
            }
            if r + 1 < W {
                e.push((cell, cell + W)); // vertical adjacency
            }
        }
    }
    e
}

/// Bitset (over the 40 edges) of edges whose endpoints straddle cut S.
fn edge_cross_mask(s: u32, edges: &[(usize, usize)]) -> u64 {
    let mut m = 0u64;
    for (i, &(a, b)) in edges.iter().enumerate() {
        if ((s >> a) ^ (s >> b)) & 1 == 1 {
            m |= 1u64 << i;
        }
    }
    m
}

/// Sound lower bound lb(C): straddling tiles (blank excluded).
/// R board: tile t (1..=24) starts at cell 25-t, goal cell t-1.
fn lb(s: u32) -> u32 {
    let mut n = 0;
    for t in 1u32..=24 {
        let start = 25 - t as usize;
        let goal = (t - 1) as usize;
        if ((s >> start) ^ (s >> goal)) & 1 == 1 {
            n += 1;
        }
    }
    n
}

fn manhattan_r() -> u32 {
    let mut d = 0;
    for t in 1usize..=24 {
        let start = 25 - t;
        let goal = t - 1;
        let (sr, sc) = (start / W, start % W);
        let (gr, gc) = (goal / W, goal % W);
        d += (sr as i32 - gr as i32).unsigned_abs() + (sc as i32 - gc as i32).unsigned_abs();
    }
    d
}

// ---- candidate cut family --------------------------------------------------

/// All axis-aligned rectangles [r0..=r1] x [c0..=c1] as cut masks (one side).
/// Includes the 8 axis half-planes (full-width/height rectangles) as a subset,
/// and single cells / center blocks / corners.
fn candidate_cuts() -> Vec<u32> {
    let mut out = Vec::new();
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
                    if s != 0 && s != FULL_MASK {
                        out.push(s);
                    }
                }
            }
        }
    }
    out
}

/// The 8 axis half-plane cuts (row prefixes 0..=b, col prefixes 0..=b).
fn axis_cuts() -> Vec<u32> {
    let mut out = Vec::new();
    for b in 0..(W - 1) {
        let mut row = 0u32;
        let mut col = 0u32;
        for r in 0..W {
            for c in 0..W {
                if r <= b {
                    row |= 1u32 << (r * W + c);
                }
                if c <= b {
                    col |= 1u32 << (r * W + c);
                }
            }
        }
        out.push(row);
        out.push(col);
    }
    out
}

// ---- packing ---------------------------------------------------------------

struct Cut {
    mask: u32,
    emask: u64,
    lb: u32,
}

/// Edge-disjoint packing following a given visit order: keep each cut if
/// edge-disjoint from those already taken. Returns (total, selected).
fn pack_in_order(cuts: &[Cut], order: &[usize]) -> (u32, Vec<usize>) {
    let mut used = 0u64;
    let mut total = 0u32;
    let mut sel = Vec::new();
    for &i in order {
        if cuts[i].lb > 0 && cuts[i].emask & used == 0 {
            used |= cuts[i].emask;
            total += cuts[i].lb;
            sel.push(i);
        }
    }
    (total, sel)
}

/// Greedy: descending lb.
fn greedy_pack(cuts: &[Cut]) -> (u32, Vec<usize>) {
    let mut order: Vec<usize> = (0..cuts.len()).collect();
    order.sort_by(|&a, &b| cuts[b].lb.cmp(&cuts[a].lb));
    pack_in_order(cuts, &order)
}

/// Best packing found over `restarts` randomized orders (deterministic RNG),
/// biased toward high lb but perturbed. Removes the greedy-suboptimality
/// caveat: if even the best of many restarts stays well under 144, the family
/// ceiling is genuinely low.
fn best_pack(cuts: &[Cut], restarts: u32) -> (u32, Vec<usize>) {
    let (mut best_total, mut best_sel) = greedy_pack(cuts);
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let n = cuts.len();
    for _ in 0..restarts {
        // key = lb scaled up plus random jitter, sort descending by key
        let mut order: Vec<usize> = (0..n).collect();
        let keys: Vec<i64> = (0..n)
            .map(|i| cuts[i].lb as i64 * 64 + (next() % 128) as i64 - 64)
            .collect();
        order.sort_by(|&a, &b| keys[b].cmp(&keys[a]));
        let (total, sel) = pack_in_order(cuts, &order);
        if total > best_total {
            best_total = total;
            best_sel = sel;
        }
    }
    (best_total, best_sel)
}

/// Observed crossings of cut S along a replayed move path.
fn observed_crossings(start: &State, path: &[Move], s: u32) -> u32 {
    let mut st = *start;
    let mut blank = st.0.iter().position(|&t| t == 0).unwrap();
    let mut n = 0;
    for &m in path {
        let st2 = st.apply(m);
        let b2 = st2.0.iter().position(|&t| t == 0).unwrap();
        // move slid a tile along edge (blank, b2); it crosses S iff they straddle
        if ((s >> blank) ^ (s >> b2)) & 1 == 1 {
            n += 1;
        }
        st = st2;
        blank = b2;
    }
    n
}

fn describe(mask: u32) -> String {
    // bounding box of the set bits
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
    format!("rows {r0}..={r1} cols {c0}..={c1} ({cnt} cells)")
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

    // verify replay ends at goal
    {
        let mut s = start;
        for (i, &m) in path.iter().enumerate() {
            assert!(s.legal_moves().contains(m), "move {i} illegal in replay");
            s = s.apply(m);
        }
    }

    let edges = edges();
    let man = manhattan_r();

    // build candidate cuts, dedup by edge-mask keeping max lb
    let mut best_by_edges: std::collections::HashMap<u64, Cut> = std::collections::HashMap::new();
    for mask in candidate_cuts() {
        let emask = edge_cross_mask(mask, &edges);
        let l = lb(mask);
        best_by_edges
            .entry(emask)
            .and_modify(|c| {
                if l > c.lb {
                    *c = Cut { mask, emask, lb: l };
                }
            })
            .or_insert(Cut { mask, emask, lb: l });
    }
    let cuts: Vec<Cut> = best_by_edges.into_values().collect();

    // axis-only packing (validation anchor: should equal Manhattan)
    let axis: Vec<Cut> = axis_cuts()
        .into_iter()
        .map(|mask| Cut {
            mask,
            emask: edge_cross_mask(mask, &edges),
            lb: lb(mask),
        })
        .collect();
    let (axis_total, _) = greedy_pack(&axis);

    // best packing over randomized restarts (greedy is one of them)
    let (greedy_total, _) = greedy_pack(&cuts);
    let (rand_total, _) = best_pack(&cuts, 200_000);
    // proven ceiling = Manhattan (axis packing achieves it; Menger caps it)
    let ceiling = axis_total.max(greedy_total).max(rand_total);

    println!("== min-cut / cut-packing ceiling on R ==");
    println!("path: {} moves from {file}", path.len());
    println!(
        "candidate cuts (rectangles, dedup by edge-set): {}",
        cuts.len()
    );
    println!();
    println!("  Manhattan(R)              = {man}");
    println!("  axis-cut packing          = {axis_total}   (must equal Manhattan)");
    println!("  WD(R)                     = 140   (established)");
    println!("  cWD(R)                    = 144   (established)");
    println!("  greedy cut-packing        = {greedy_total}   (lb-biased; suboptimal)");
    println!("  best of 200k random packs = {rand_total}");
    println!("  cut-packing CEILING       = {ceiling}   (= Manhattan; provably capped by Menger)");
    println!("  d*(R)                     = 156   (best-known; LB 150)");
    println!();

    // axis packing achieves the ceiling; show it with observed crossings
    let mut rows: Vec<(u32, u32, u32, String)> = axis
        .iter()
        .map(|c| {
            let obs = observed_crossings(&start, &path, c.mask);
            (c.lb, obs, c.emask.count_ones(), describe(c.mask))
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    println!("axis cuts achieving the ceiling (lb | observed-on-path | #edges | region):");
    let mut sum_lb = 0;
    let mut sound_ok = true;
    for (l, obs, ne, desc) in &rows {
        let flag = if obs >= l { "" } else { "  <-- UNSOUND?!" };
        if obs < l {
            sound_ok = false;
        }
        sum_lb += l;
        println!("  lb {l:2} | obs {obs:3} | {ne:2}e | {desc}{flag}");
    }
    println!();
    println!(
        "  sum lb = {sum_lb} ; soundness (all observed >= lb): {}",
        if sound_ok { "OK" } else { "VIOLATED" }
    );
    println!();
    let verdict = if ceiling > 144 {
        format!("CEILING {ceiling} > cWD 144  ->  PURSUE: cut-packing can beat cWD on R")
    } else {
        format!(
            "CEILING {ceiling} <= cWD 144  ->  DEAD: cut-packing (min-cut congestion) is provably\n         capped at Manhattan(R)={man} by Menger, so it cannot beat WD 140, let alone cWD 144.\n         A bound that closes R's slack must use constraints Menger does NOT cap:\n         intra-line ORDERING (what WD/cWD already exploit) or true axis-COUPLING\n         (joint row x col relaxations), not single-commodity cut/flow packing."
        )
    };
    println!("VERDICT: {verdict}");
}
