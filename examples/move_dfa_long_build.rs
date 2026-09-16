//! Build `LongMoveDfa` for one window and report build time and size.
//!
//! ```text
//! cargo run --release --example move_dfa_long_build -- 14
//! ```

use std::time::Instant;

use puzzle8::puzzle24::eta::Walker;
use puzzle8::puzzle24::search::LongMoveDfa;

fn main() {
    let w: u8 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .expect("usage: move_dfa_long_build WINDOW");
    let t0 = Instant::now();
    let dfa = LongMoveDfa::build(w);
    println!(
        "window {w}: {} states, {:.1} KiB, built and verified in {:.2} s",
        dfa.states(),
        dfa.table_bytes() as f64 / 1024.0,
        t0.elapsed().as_secs_f64()
    );
    let t1 = Instant::now();
    let walker = Walker::new(&dfa, true);
    println!(
        "walker: {} nodes, {} doomed, built in {:.2} s",
        walker.node_count(),
        walker.doomed_nodes(),
        t1.elapsed().as_secs_f64()
    );
}
