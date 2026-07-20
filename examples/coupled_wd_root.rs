//! Step B of the axis-coupled ("blank-ferry") WD bound: compute the coupled
//! bound at R's root by a single IDA* over the product state, no table.
//!
//! Product abstraction. cWD runs two INDEPENDENT WD automata (row-contingency
//! M_r + blank row; col-contingency M_c + blank col) and sums them. The coupled
//! bound runs ONE automaton over (M_r, M_c, blank_cell): a vertical move slides
//! the blank between adjacent rows (updates M_r, column preserved => M_c
//! untouched); a horizontal move slides it between adjacent columns (updates
//! M_c, row preserved => M_r untouched). Goal = both contingencies solved,
//! blank at cell 24. It is admissible (every real move maps to one abstract
//! move; the abstract search freely CHOOSES which goal-class tile sits in the
//! blank's column — a relaxation), and h = WD_row(M_r)+WD_col(M_c) is a
//! consistent lower bound, so the search is exact.
//!
//! Prediction (and the point of measuring it): this pure column-abstracted
//! coupling is VACUOUS — combined-WD = WD_row + WD_col exactly. Proof: the
//! blank's ROW trajectory is driven solely by the vertical moves and its COLUMN
//! trajectory solely by the horizontal ones; a vertical move's availability
//! depends only on M_r (any class present in the adjacent row, chosen freely),
//! never on the blank's column, and symmetrically for horizontal. So any
//! interleaving of an optimal row-WD sequence with an optimal col-WD sequence
//! is a valid product path => combined <= WD_row+WD_col, and >= h = the same,
//! so = it. The blank position couples nothing because column-abstraction has
//! forgotten which tile is in the blank's column. This run confirms it
//! empirically (expect 140 = WD(R)); a coupling that actually bites must retain
//! tile-column identity, i.e. a PDB — which collapses to <=126 on R.
//!
//! Run from repo root (needs data/wd24.bin):
//!   cargo run --release --example coupled_wd_root

use std::path::Path;

use puzzle8::puzzle24::search::walking_distance::{
    load_dist_table, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use puzzle8::puzzle24::state::{State, N_CELLS, W};

type Matrix = [[u8; W]; W];

// ---- WD-key codec (mirrors slack_anatomy.rs) -------------------------------

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

/// Project R (or any state) onto row + col contingencies.
fn project(s: &State) -> (Matrix, u8, Matrix, u8) {
    let mut m_row = [[0u8; W]; W];
    let mut m_col = [[0u8; W]; W];
    let (mut br, mut bc) = (0u8, 0u8);
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
    }
    (m_row, br, m_col, bc)
}

fn r_board() -> State {
    let mut a = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        a[i] = (25 - i) as u8;
    }
    State(a)
}

// ---- coupled-WD via IDA* over the product (row_key, col_key) ---------------

const NODE_BUDGET: u64 = 200_000_000;

/// Product-state children: vertical moves (update row_key, col_key fixed) and
/// horizontal moves (update col_key, row_key fixed).
fn children(rk: u64, ck: u64) -> Vec<(u64, u64)> {
    let (mr, br) = unpack(rk);
    let (mc, bc) = unpack(ck);
    let mut out = Vec::with_capacity(20);
    for ar in [br.wrapping_sub(1), br + 1] {
        let ar = ar as usize;
        if ar >= W {
            continue;
        }
        for t in 0..W {
            if mr[ar][t] == 0 {
                continue;
            }
            let mut m = mr;
            m[ar][t] -= 1;
            m[br as usize][t] += 1;
            out.push((pack(&m, ar as u8), ck));
        }
    }
    for ac in [bc.wrapping_sub(1), bc + 1] {
        let ac = ac as usize;
        if ac >= W {
            continue;
        }
        for u in 0..W {
            if mc[ac][u] == 0 {
                continue;
            }
            let mut m = mc;
            m[ac][u] -= 1;
            m[bc as usize][u] += 1;
            out.push((rk, pack(&m, ac as u8)));
        }
    }
    out
}

enum Sr {
    Found(u32),
    Exceed(u32), // min f that exceeded threshold in this subtree (u32::MAX if none)
    Budget,
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    rk: u64,
    ck: u64,
    g: u32,
    threshold: u32,
    goal: u64,
    get: &impl Fn(u64) -> u32,
    nodes: &mut u64,
) -> Sr {
    *nodes += 1;
    if *nodes > NODE_BUDGET {
        return Sr::Budget;
    }
    if rk == goal && ck == goal {
        return Sr::Found(g);
    }
    let mut min_ex = u32::MAX;
    for (rk2, ck2) in children(rk, ck) {
        let f = g + 1 + get(rk2) + get(ck2);
        if f > threshold {
            min_ex = min_ex.min(f);
            continue;
        }
        match dfs(rk2, ck2, g + 1, threshold, goal, get, nodes) {
            Sr::Found(v) => return Sr::Found(v),
            Sr::Exceed(e) => min_ex = min_ex.min(e),
            Sr::Budget => return Sr::Budget,
        }
    }
    Sr::Exceed(min_ex)
}

/// IDA* over the product state. Returns Ok(exact combined-WD) or Err(proven LB).
/// With h = WD_row+WD_col (which the vacuousness proof says is EXACT for the
/// product), threshold h0 finds a witnessing path in one downhill plunge.
fn coupled_wd(table: &WdTable, row0: u64, col0: u64) -> Result<u32, u32> {
    let goal = goal_key();
    let get = |k: u64| -> u32 { *table.get(&k).expect("reachable WD state") as u32 };
    let mut threshold = get(row0) + get(col0);
    let mut nodes: u64 = 0;
    loop {
        eprintln!("  IDA* threshold {threshold} (nodes so far {nodes})…");
        match dfs(row0, col0, 0, threshold, goal, &get, &mut nodes) {
            Sr::Found(g) => return Ok(g),
            Sr::Budget => return Err(threshold),
            Sr::Exceed(next) => {
                if next == u32::MAX {
                    return Err(threshold); // no goal reachable (should not happen)
                }
                threshold = next;
            }
        }
    }
}

fn main() {
    eprintln!("loading raw WD table…");
    let table = load_dist_table(
        Path::new("data/wd24.bin"),
        WD_KIND_FULL,
        Some(FULL_WD_ENTRIES),
    )
    .expect("wd24.bin load");

    let start = r_board();
    let (mr, br, mc, bc) = project(&start);
    let row0 = pack(&mr, br);
    let col0 = pack(&mc, bc);
    let wd_r = *table.get(&row0).unwrap() as u32;
    let wd_c = *table.get(&col0).unwrap() as u32;

    let result = coupled_wd(&table, row0, col0);

    println!("== coupled (blank-ferry) WD at R's root — Step B ==");
    println!();
    println!("  WD_row(R)                 = {wd_r}");
    println!("  WD_col(R)                 = {wd_c}");
    println!(
        "  WD(R) = WD_row + WD_col   = {}   (established 140)",
        wd_r + wd_c
    );
    println!("  cWD(R)                    = 144   (established)");
    match result {
        Ok(v) => {
            println!("  coupled-WD(R)             = {v}   (exact, product IDA*)");
            println!("  d*(R)                     = 156   (best-known; LB 150)");
            println!();
            if v == wd_r + wd_c {
                println!(
                    "VERDICT: coupled-WD = WD exactly ({v}) -> DEAD: column-abstracted blank\n         coupling is VACUOUS (as proven — the two axes never constrain each\n         other, so any interleaving of the per-axis optima is a valid product\n         path). A coupling that bites must retain tile-column identity = a PDB,\n         which collapses to <=126 on R. #2 in this form is closed."
                );
            } else if v > 144 {
                println!("VERDICT: coupled-WD {v} > cWD 144 -> PURSUE (surprise: coupling bites!)");
            } else {
                println!(
                    "VERDICT: coupled-WD {v} in (WD {}, cWD 144] -> coupling bites but < cWD",
                    wd_r + wd_c
                );
            }
        }
        Err(lb) => {
            println!("  coupled-WD(R)             >= {lb}   (pop budget hit; proven LB only)");
            println!("  d*(R)                     = 156");
            println!();
            if lb > 144 {
                println!("VERDICT: coupled-WD >= {lb} > cWD 144 -> PURSUE");
            } else {
                println!("VERDICT: budget exhausted at LB {lb} (<=144); rerun with larger budget for exact value");
            }
        }
    }
}
