//! r_dedup — measure the duplicate ratio (IDA*-tree nodes / distinct states) of
//! the real R lower-bound proof. This ratio is the CEILING on an A*/transposition
//! hybrid's speedup: A* with a closed list expands each state once; IDA* revisits
//! transpositions. Counts distinct EXPANDED states via an exact HashSet.
//!
//! Run: cargo run --release --example r_dedup -- [max_bound]   (default 144 = prove R>=146)

use std::collections::HashSet;

use puzzle8::puzzle24::search::{Cwd, CwdScratch};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

fn r_board() -> State {
    let mut cells = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        cells[i] = (25 - i) as u8;
    }
    State(cells)
}

fn pack(s: &State) -> u128 {
    let mut x: u128 = 0;
    for i in 0..N_CELLS {
        x |= (s.0[i] as u128) << (5 * i);
    }
    x
}

struct Instr {
    cwd: Cwd,
    scratch: CwdScratch,
    bound: u8,
    seen: HashSet<u128>, // distinct EXPANDED states
    tree_expanded: u64,  // tree nodes expanded (f<=bound, non-goal)
    visited: u64,
}

impl Instr {
    fn h(&mut self, s: &State) -> u8 {
        self.cwd.eval(s, &mut self.scratch)
    }

    fn dfs(&mut self, s: &State, blank: u8, h_val: u8, g: u8, last: Option<Move>) {
        self.visited += 1;
        if g.saturating_add(h_val) > self.bound {
            return;
        }
        if s == &GOAL {
            return;
        }
        self.tree_expanded += 1;
        self.seen.insert(pack(s));
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
    let max_bound: u8 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(144);
    let start = r_board();
    eprintln!("cWD: loading tables…");
    let cwd = Cwd::new();
    let mut scratch = CwdScratch::new();
    let h0 = cwd.eval(&start, &mut scratch);
    eprintln!("cWD(R) = {}", h0);

    let mut instr = Instr {
        cwd,
        scratch,
        bound: 0,
        seen: HashSet::new(),
        tree_expanded: 0,
        visited: 0,
    };

    let mut bound = h0;
    let t0 = std::time::Instant::now();
    while bound <= max_bound {
        eprintln!("iteration threshold = {} (elapsed {:.0}s)", bound, t0.elapsed().as_secs_f64());
        instr.bound = bound;
        let hb = instr.h(&start);
        instr.dfs(&start, start.blank_pos(), hb, 0, None);
        bound += 2;
    }

    let distinct = instr.seen.len() as f64;
    let te = instr.tree_expanded as f64;
    println!("\n=== R proof (>= {}) duplicate ratio ===", bound);
    println!("tree nodes expanded : {}", instr.tree_expanded);
    println!("total visited       : {}", instr.visited);
    println!("distinct expanded   : {}", instr.seen.len());
    println!("DUPLICATE RATIO     : {:.3}x  (A*/TT speedup ceiling)", te / distinct);
    println!("elapsed             : {:.1}s", t0.elapsed().as_secs_f64());
}
