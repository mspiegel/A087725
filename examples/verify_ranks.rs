//! IDA*-verify a list of ranks supplied on the command line. Reports each
//! rank's optimal depth and the cell-grid of the corresponding board.
//!
//! Run: `cargo run --release --features "mmap parallel" --example verify_ranks -- RANK [RANK...]`

use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::unrank;
use puzzle8::puzzle15::search::{
    idastar_inc_with_stats, LinearConflictInc, WalkingDistanceHeuristic,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ranks: Vec<u64> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse::<u64>().ok())
        .collect();
    if ranks.is_empty() {
        eprintln!("usage: verify_ranks RANK [RANK...]");
        std::process::exit(2);
    }

    let pdb_dir = PathBuf::from("data");
    let p7 = ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&pdb_dir.join("zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    for (i, &r) in ranks.iter().enumerate() {
        let t0 = Instant::now();
        let s = unrank(r);
        let (sol, _stats) = idastar_inc_with_stats(&s, &h);
        let d = sol.map(|v| v.len() as u8).unwrap_or(u8::MAX);
        let elapsed = t0.elapsed();

        println!("=== #{} ===", i + 1);
        println!("rank: {}", r);
        println!("depth: {}  (verify took {:.2?})", d, elapsed);
        println!("grid:");
        for row in 0..4 {
            print!("  ");
            for col in 0..4 {
                let t = s.0[row * 4 + col];
                if t == 0 { print!(" __"); } else { print!(" {:>2}", t); }
            }
            println!();
        }
        println!();
    }

    Ok(())
}
