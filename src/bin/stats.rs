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
use puzzle8::puzzle8::search::{
    Heuristic, LinearConflictHeuristic, ManhattanHeuristic, MaxHeuristic,
    WalkingDistanceHeuristic,
};
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
    println!();

    // Heuristic tightness: for each admissible heuristic, sweep the full state
    // space and report how close it stays to the true distance.
    println!("== heuristic tightness vs. true distance ==");
    WalkingDistanceHeuristic::warm_up();
    let md = ManhattanHeuristic;
    let lc = LinearConflictHeuristic;
    let wd = WalkingDistanceHeuristic;
    let max_lc_wd = MaxHeuristic::new(LinearConflictHeuristic, WalkingDistanceHeuristic);
    let max_all = MaxHeuristic::new(
        ManhattanHeuristic,
        MaxHeuristic::new(LinearConflictHeuristic, WalkingDistanceHeuristic),
    );

    let stats_md = sweep(&md, &t);
    let stats_lc = sweep(&lc, &t);
    let stats_wd = sweep(&wd, &t);
    let stats_lc_wd = sweep(&max_lc_wd, &t);
    let stats_all = sweep(&max_all, &t);

    println!(
        "  {:<22} {:>10} {:>10} {:>10} {:>12}",
        "heuristic", "mean h", "mean gap", "max gap", "exact frac"
    );
    print_row("Manhattan", &stats_md);
    print_row("MD + LinearConflict", &stats_lc);
    print_row("WalkingDistance", &stats_wd);
    print_row("max(MD+LC, WD)", &stats_lc_wd);
    print_row("max(MD, MD+LC, WD)", &stats_all);
}

struct HeuristicStats {
    mean_h: f64,
    mean_gap: f64,
    max_gap: u8,
    exact_count: u32,
}

fn sweep<H: Heuristic>(h: &H, t: &DistanceTable) -> HeuristicStats {
    let mut sum_h: u64 = 0;
    let mut sum_gap: u64 = 0;
    let mut max_gap: u8 = 0;
    let mut exact: u32 = 0;
    for r in 0..N_STATES {
        let s = unrank(r);
        let est = h.h(&s);
        let truth = t.dist(&s);
        let gap = truth - est;
        sum_h += est as u64;
        sum_gap += gap as u64;
        if gap > max_gap {
            max_gap = gap;
        }
        if gap == 0 {
            exact += 1;
        }
    }
    let n = N_STATES as f64;
    HeuristicStats {
        mean_h: sum_h as f64 / n,
        mean_gap: sum_gap as f64 / n,
        max_gap,
        exact_count: exact,
    }
}

fn print_row(label: &str, s: &HeuristicStats) {
    let exact_frac = s.exact_count as f64 / N_STATES as f64;
    println!(
        "  {:<22} {:>10.3} {:>10.3} {:>10} {:>11.2}%",
        label,
        s.mean_h,
        s.mean_gap,
        s.max_gap,
        100.0 * exact_frac
    );
}
