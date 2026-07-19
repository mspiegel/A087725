//! Profiling harness: isolate the zero-aware incremental path (`ZpdbInc`) so a
//! sampling profiler's flame graph is dominated by `advance`/`rank`, the per-node
//! hot path we want to optimize. Unlike `zpdb_bench15` this runs ONLY the
//! zero-aware solves (no korf-plus, no LC/WD), so samples aren't split across
//! unrelated heuristics.
//!
//! No board/solve cache is touched: it calls `idastar_inc_with_stats` directly,
//! and IDA* is a memoryless DFS (no transposition table).
//!
//! Run under samply:
//!   CARGO_PROFILE_RELEASE_DEBUG=2 cargo build --release \
//!     --example zpdb_profile15 --features "mmap parallel"
//!   samply record ./target/release/examples/zpdb_profile15 data 3

use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::antipodes;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::idastar_inc_with_stats;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"));
    let n_solves: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let antipode_ranks =
        antipodes::load_ranks(&dir.join("pdb15_antipodes.txt")).expect("antipodes");

    let zdbs = [
        ZPatternDb::load_mmap(&dir.join("zpdb15_p7.zbin")).expect("zp7"),
        ZPatternDb::load_mmap(&dir.join("zpdb15_p8.zbin")).expect("zp8"),
    ];
    let zpdb = ZpdbInc::new([&zdbs[0], &zdbs[1]]);

    let k = n_solves.min(antipode_ranks.len());
    eprintln!("profiling zero-aware idastar on {k} depth-80 antipodes (no cache)");

    let (mut total_nodes, mut total_t) = (0u64, 0.0f64);
    for (i, &r) in antipode_ranks.iter().take(k).enumerate() {
        let s = unrank(r);
        let t0 = Instant::now();
        let (sol, st) = idastar_inc_with_stats(&s, &zpdb);
        let dt = t0.elapsed().as_secs_f64();
        assert_eq!(
            sol.as_ref().map(|v| v.len()),
            Some(80),
            "zero-aware returned non-optimal!"
        );
        total_nodes += st.nodes;
        total_t += dt;
        eprintln!(
            "  {:>2}/{}  {:>13} nodes  {:>7.2}s  {:>5.2} Mnodes/s",
            i + 1,
            k,
            st.nodes,
            dt,
            st.nodes as f64 / dt / 1e6,
        );
    }
    eprintln!(
        "  total  {:>13} nodes  {:>7.2}s  {:>5.2} Mnodes/s",
        total_nodes,
        total_t,
        total_nodes as f64 / total_t / 1e6,
    );
}
