//! Print summary statistics for the 8-puzzle's optimal-move structure.
//!
//! Usage:
//!     cargo run --release --bin stats
//!
//! Builds the table in memory and prints:
//!   - total reachable states
//!   - diameter
//!   - distance histogram
//!   - antipode configurations
//!   - branching-factor distribution (number of optimal moves per state)

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::state::{N_STATES, State};

fn format_state(s: &State) -> String {
    let mut out = String::new();
    for r in 0..3 {
        for c in 0..3 {
            let v = s.0[r * 3 + c];
            if v == 0 {
                out.push('.');
            } else {
                out.push_str(&v.to_string());
            }
            if c < 2 {
                out.push(' ');
            }
        }
        if r < 2 {
            out.push_str(" / ");
        }
    }
    out
}

fn main() {
    eprintln!("building distance table...");
    let t = DistanceTable::build();

    let visited = t.visited_count();
    let diameter = t.diameter();
    println!("== 8-puzzle distance table ==");
    println!("total states     : {}", visited);
    println!("expected         : {}", N_STATES);
    println!("diameter         : {}", diameter);
    println!();

    println!("== distance histogram ==");
    let hist = t.histogram();
    let total: u32 = hist.iter().sum();
    for (d, &count) in hist.iter().enumerate() {
        println!("  d={:2}  {:>8} states", d, count);
    }
    println!("  total {:>8}", total);
    println!();

    println!("== antipodes (distance {}) ==", diameter);
    for (i, s) in t.antipodes().iter().enumerate() {
        println!("  #{}: {}", i + 1, format_state(s));
    }
    println!();

    // Branching factor: count of optimal moves per state.
    let mut bf_hist = [0u32; 5];
    for r in 0..N_STATES {
        let s = unrank(r);
        let n = t.optimal_moves(&s).len() as usize;
        bf_hist[n] += 1;
    }
    println!("== optimal-move branching factor ==");
    for (k, &count) in bf_hist.iter().enumerate() {
        if count > 0 {
            println!("  {} optimal move(s): {:>8} states", k, count);
        }
    }
}
