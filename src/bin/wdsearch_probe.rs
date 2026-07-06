//! wdsearch_probe — Phase-1 gate: does a WD-maximizing beam reach high WD?
//!
//! Sweeps beam width × target depth and prints the max/mean Walking Distance of
//! the returned frontier. THE BET: a wide beam escapes WD plateaus and reaches
//! the deep region (WD ~120-140), far past the greedy learned policy's ~90.
//!
//!   cargo run --release --features ml --bin wdsearch_probe

use std::time::Instant;

use puzzle8::puzzle24::ml::scramble::Rng;
use puzzle8::puzzle24::ml::wdsearch::{construct_deep_boards, Diversity, WdSearchConfig};
use puzzle8::puzzle24::search::WalkingDistanceHeuristic;

fn main() {
    print!("warming up Walking Distance table... ");
    let t = Instant::now();
    WalkingDistanceHeuristic::warm_up();
    println!("ready in {:.1}s\n", t.elapsed().as_secs_f64());

    println!("{:>7}  {:>6}  {:>7}  {:>8}  {:>9}", "width", "depth", "max_wd", "mean_wd", "ms");
    for &width in &[2000usize, 8000, 20000] {
        for &depth in &[160usize, 200, 260, 320] {
            let cfg = WdSearchConfig {
                width,
                target_depth: depth,
                node_budget: 0,
                diversity: Diversity::TopK,
            };
            let mut rng = Rng::new(1);
            let t = Instant::now();
            let out = construct_deep_boards(width, &cfg, &mut rng);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let max_wd = out.iter().map(|&(_, w)| w).max().unwrap_or(0);
            let mean_wd = if out.is_empty() {
                0.0
            } else {
                out.iter().map(|&(_, w)| w as f64).sum::<f64>() / out.len() as f64
            };
            println!("{:>7}  {:>6}  {:>7}  {:>8.1}  {:>9.0}", width, depth, max_wd, mean_wd, ms);
        }
    }
}
