//! g_histogram — instrument an R lower-bound proof and histogram EXPANDED nodes
//! by path-cost `g`. Answers "how deep a goal-ball would an endgame DB need?":
//! a radius-`r` goal-centered exact table can prune an expanded node only if that
//! node is within `r` of the goal, i.e. `g >= d* - r` (d* = 152 for R). So
//!   coverage_ceiling(r) = #{expanded nodes with g >= 152 - r}
//! is an UPPER BOUND on the nodes such a DB could ever prune.
//!
//! Run: cargo run --release --example g_histogram -- [max_bound]

use puzzle8::puzzle24::search::{Cwd, CwdScratch};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

const D_STAR: usize = 152; // Rokicki's proven optimal depth of R

fn r_board() -> State {
    let mut cells = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        cells[i] = (25 - i) as u8;
    }
    State(cells)
}

struct Instr {
    cwd: Cwd,
    scratch: CwdScratch,
    bound: u8,
    expanded: Vec<u64>, // expanded[g] = # nodes with f<=bound that iterated children
    nodes: u64,         // every entry (matches SearchStats.nodes)
}

impl Instr {
    fn h(&mut self, s: &State) -> u8 {
        self.cwd.eval(s, &mut self.scratch)
    }

    // Faithful replica of search_inc: same inverse-move pruning, same f-prune,
    // threading the blank. Records g for every node that survives f<=bound and
    // therefore expands its children.
    fn dfs(&mut self, s: &State, blank: u8, h_val: u8, g: u8, last: Option<Move>) {
        self.nodes += 1;
        let f = g.saturating_add(h_val);
        if f > self.bound {
            return;
        }
        if s == &GOAL {
            return; // unreachable for R within these bounds, but keep parity with search_inc
        }
        self.expanded[g as usize] += 1;
        for m in State::legal_moves_at(blank).iter() {
            if let Some(prev) = last {
                if m == prev.inverse() {
                    continue;
                }
            }
            let (s_next, next_blank) = s.apply_at(m, blank);
            let child_h = self.h(&s_next);
            self.dfs(&s_next, next_blank, child_h, g + 1, Some(m));
        }
    }
}

fn main() {
    let max_bound: u8 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(144);

    let start = r_board();
    eprintln!("cWD: loading tables…");
    let cwd = Cwd::new();
    let mut scratch = CwdScratch::new();
    let h0 = cwd.eval(&start, &mut scratch);
    eprintln!(
        "cWD(R) = {} (overlay: {})",
        h0,
        if cwd.has_overlay() {
            "fast table"
        } else {
            "reference"
        }
    );

    let mut instr = Instr {
        cwd,
        scratch,
        bound: 0,
        expanded: vec![0u64; 256],
        nodes: 0,
    };

    // Deepening loop, accumulating the histogram across every iteration up to
    // max_bound (matches the real proof's cumulative work). Parity makes the
    // thresholds step by 2.
    let mut bound = h0;
    let t0 = std::time::Instant::now();
    while bound <= max_bound {
        eprintln!("iteration threshold = {bound}");
        instr.bound = bound;
        let hb = instr.h(&start);
        instr.dfs(&start, start.blank_pos(), hb, 0, None);
        bound += 2;
    }
    let elapsed = t0.elapsed();

    let total: u64 = instr.expanded.iter().sum();
    println!(
        "\nProved R >= {} | expanded nodes = {} | total visited = {} | {:.1}s",
        bound,
        total,
        instr.nodes,
        elapsed.as_secs_f64()
    );

    // Per-g histogram (nonzero rows).
    println!("\n g : expanded_nodes : cum_from_top(>= g) : implied_radius r=152-g");
    let mut cum = 0u64;
    let maxg = instr.expanded.iter().rposition(|&c| c > 0).unwrap_or(0);
    // walk high g -> low g so cum = #{expanded : g' >= g}
    let mut rows: Vec<(usize, u64, u64)> = Vec::new();
    for g in (0..=maxg).rev() {
        cum += instr.expanded[g];
        rows.push((g, instr.expanded[g], cum));
    }
    for &(g, c, cumv) in rows.iter().rev() {
        if c == 0 && cumv == 0 {
            continue;
        }
        let r = D_STAR.saturating_sub(g);
        let frac = cumv as f64 / total as f64;
        println!(
            "{:>3} : {:>14} : {:>14} ({:>7.4}%) : r={:>3}",
            g,
            c,
            cumv,
            frac * 100.0,
            r
        );
    }

    // Coverage-ceiling table: for each radius r, the max fraction of expanded
    // nodes a radius-r goal-ball could prune = #{expanded : g >= 152 - r}/total.
    println!("\nEndgame-DB coverage ceiling (upper bound on nodes prunable):");
    println!("  radius r | g>=152-r | ceiling nodes | ceiling %% of proof");
    for &r in &[15usize, 20, 25, 30, 35, 40, 45, 50, 55, 60] {
        let gthr = D_STAR.saturating_sub(r);
        let ceil: u64 = instr.expanded[gthr..].iter().sum();
        println!(
            "  r={:>3}    | g>={:>3}   | {:>14} | {:>8.5}%",
            r,
            gthr,
            ceil,
            ceil as f64 / total as f64 * 100.0
        );
    }
}
