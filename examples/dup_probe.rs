//! Duplicate-node probe: how much headroom is there for duplicate elimination
//! (OPTIMIZATION.md #5 FSM / #6 transposition table)?
//!
//! Runs the *exact* incremental Korf IDA\* (same bounds, same don't-undo
//! pruning, same node set as `korf100_bench`'s default path) over the first N
//! Korf instances, but additionally keeps a per-iteration map from packed board
//! to the shallowest `g` at which it was reached. A node is a *redundant
//! re-expansion* if its board was already reached this iteration at `g' <= g` —
//! exactly the nodes a perfect transposition table (or the duplicate-pruning
//! FSM) could remove. Reports the redundant fraction; that is the ceiling on
//! what #5/#6 can buy on this workload.
//!
//! It does NOT prune — it runs the full real search so `total` must equal the
//! bench's node count (a self-check), and counts what *could* be pruned.
//!
//! ```text
//! cargo run --release --features mmap --example dup_probe -- --pdb-dir data --limit 30
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use puzzle8::puzzle15::pdb::{KorfPdbInc, PatternDb};
use puzzle8::puzzle15::search::IncHeuristic;
use puzzle8::puzzle15::state::{Move, State, GOAL, N_CELLS};

fn to_repo_goal(korf: &[u8; N_CELLS]) -> State {
    let mut cells = [0u8; N_CELLS];
    for i in 0..N_CELLS {
        let v = korf[i];
        cells[N_CELLS - 1 - i] = if v == 0 { 0 } else { 16 - v };
    }
    State(cells)
}

fn pack(s: &State) -> u64 {
    let mut k = 0u64;
    for &v in &s.0 {
        k = (k << 4) | v as u64;
    }
    k
}

fn parse_instances(path: &Path, limit: usize) -> Vec<State> {
    let text = std::fs::read_to_string(path).expect("read instances");
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (_, tiles) = line.split_once(':').expect("':' separator");
        let mut cells = [0u8; N_CELLS];
        for (i, tok) in tiles.split_whitespace().enumerate() {
            cells[i] = tok.parse().expect("tile");
        }
        out.push(to_repo_goal(&cells));
        if out.len() >= limit {
            break;
        }
    }
    out
}

struct Counts {
    total: u64,
    redundant: u64,
}

/// Mirrors `search_inc`: returns `(found, min_exceeding_f)`.
#[allow(clippy::too_many_arguments)]
fn dfs<E: IncHeuristic>(
    s: &State,
    blank: u8,
    ctx: E::Ctx,
    h: u8,
    g: u8,
    bound: u8,
    last: Option<Move>,
    e: &E,
    seen: &mut HashMap<u64, u8>,
    c: &mut Counts,
) -> (bool, u8) {
    c.total += 1;
    let f = g.saturating_add(h);
    if f > bound {
        return (false, f);
    }
    // Redundant if this board was already reached this iteration no deeper.
    let key = pack(s);
    match seen.get(&key) {
        Some(&gp) if gp <= g => c.redundant += 1,
        _ => {
            seen.insert(key, g);
        }
    }
    if s == &GOAL {
        return (true, u8::MAX);
    }
    let mut min_next = u8::MAX;
    for m in State::legal_moves_at(blank).iter() {
        if let Some(p) = last {
            if m == p.inverse() {
                continue;
            }
        }
        let (s_next, nb) = s.apply_at(m, blank);
        let (ch, cc) = e.advance(&ctx, &s_next, m);
        let (found, n) = dfs(&s_next, nb, cc, ch, g + 1, bound, Some(m), e, seen, c);
        if found {
            return (true, u8::MAX);
        }
        if n < min_next {
            min_next = n;
        }
    }
    (false, min_next)
}

fn solve_counting<E: IncHeuristic>(start: &State, e: &E, c: &mut Counts) {
    if start == &GOAL {
        return;
    }
    let (h0, ctx0) = e.root(start);
    let mut bound = h0;
    let blank = start.blank_pos();
    loop {
        let mut seen: HashMap<u64, u8> = HashMap::new();
        let (found, next) = dfs(start, blank, ctx0, h0, 0, bound, None, e, &mut seen, c);
        if found || next == u8::MAX {
            return;
        }
        bound = next;
    }
}

fn main() {
    let mut pdb_dir = PathBuf::from("data");
    let mut limit = 30usize;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => {
                i += 1;
                pdb_dir = PathBuf::from(&argv[i]);
            }
            "--limit" => {
                i += 1;
                limit = argv[i].parse().expect("limit");
            }
            other => panic!("unknown flag {}", other),
        }
        i += 1;
    }

    let p7 = PatternDb::load_mmap(&pdb_dir.join("pdb15_p7_korf.bin")).expect("p7");
    let p8 = PatternDb::load_mmap(&pdb_dir.join("pdb15_p8_korf.bin")).expect("p8");
    let inc = KorfPdbInc::new([&p7, &p8]);

    let instances = parse_instances(&pdb_dir.join("korf100.txt"), limit);
    let mut c = Counts { total: 0, redundant: 0 };
    for s in &instances {
        solve_counting(s, &inc, &mut c);
    }

    println!("instances           : {}", instances.len());
    println!("total nodes         : {}", c.total);
    println!("redundant nodes     : {}", c.redundant);
    println!(
        "redundant fraction  : {:.1}%  (ceiling for #5 FSM / #6 transposition table)",
        100.0 * c.redundant as f64 / c.total as f64
    );
}
