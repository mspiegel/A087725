//! Diff two solve caches: report ranks present in A at depth D but absent
//! in B (any depth). Print board features for the difference set. Used to
//! identify the specific d=77 board that the K=3 probe unlocked but
//! 1-step BFS couldn't reach.
//!
//! Run: `cargo run --release --features "mmap parallel" --example cache_diff -- A B [DEPTH]`

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::pdb::{ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::search::{
    IncHeuristic, LinearConflictInc, SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle15::state::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a_path: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: cache_diff <A> <B> [DEPTH]")?
        .into();
    let b_path: PathBuf = std::env::args()
        .nth(2)
        .ok_or("usage: cache_diff <A> <B> [DEPTH]")?
        .into();
    let depth: u8 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(77);

    let t0 = Instant::now();
    println!("loading A = {} ...", a_path.display());
    let a = cache::load(&a_path)?;
    println!("  {} entries", a.len());
    println!("loading B = {} ...", b_path.display());
    let b = cache::load(&b_path)?;
    println!("  {} entries", b.len());

    let a_at_d: HashSet<u64> = a
        .iter()
        .filter(|(_, &d)| d == depth)
        .map(|(&r, _)| r)
        .collect();
    let b_at_d: HashSet<u64> = b
        .iter()
        .filter(|(_, &d)| d == depth)
        .map(|(&r, _)| r)
        .collect();
    println!(
        "A has {} d={} entries; B has {} d={} entries",
        a_at_d.len(),
        depth,
        b_at_d.len(),
        depth
    );

    // Boards in A at d but NOT in B at d (or not in B at all — we'd want to
    // know if B has them at a different depth too).
    let diff_a_only: Vec<u64> = a_at_d
        .iter()
        .filter(|r| !b_at_d.contains(r))
        .copied()
        .collect();
    println!(
        "\nRanks in A at d={} but not in B at d={}: {}",
        depth,
        depth,
        diff_a_only.len()
    );

    // Set up heuristic for per-board analysis.
    let p7 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p7.zbin"))?;
    let p8 = ZPatternDb::load_mmap(&PathBuf::from("data/zpdb15_p8.zbin"))?;
    WalkingDistanceHeuristic::warm_up();
    LinearConflictInc::warm_up();
    let h = ZpdbPlusInc::new([&p7, &p8]);

    for (i, &r) in diff_a_only.iter().enumerate() {
        let s = unrank(r);
        let blank = s.blank_pos();
        let degree = State::legal_moves_at(blank).len() as u8;
        let mut stats = SearchStats::default();
        let (hv, _) = h.root(&s, &mut stats);

        // Above- and below-degree (using A, the richer cache).
        let mut above = 0u8;
        let mut below = 0u8;
        let mut nb_at = Vec::new();
        for m in State::legal_moves_at(blank).iter() {
            let (ns, _) = s.apply_at(m, blank);
            let nr = rank(&ns);
            let nd = a.get(&nr).copied();
            nb_at.push((nr, nd));
            if let Some(nd) = nd {
                if nd == depth + 1 {
                    above += 1;
                } else if depth > 0 && nd == depth - 1 {
                    below += 1;
                }
            }
        }

        println!("\n=== Residue board #{} ===", i + 1);
        println!("rank: {r}");
        println!(
            "blank cell: {} (row {}, col {}, degree {})",
            blank,
            blank / 4,
            blank % 4,
            degree
        );
        println!(
            "h(s) = {}  (slack = {} - {} = {})",
            hv,
            depth,
            hv,
            depth as i32 - hv as i32
        );
        println!("above-degree (neighbors at d={}): {}", depth + 1, above);
        println!("below-degree (neighbors at d={}): {}", depth - 1, below);
        println!("tile grid:");
        for row in 0..4 {
            print!("  ");
            for col in 0..4 {
                let t = s.0[row * 4 + col];
                if t == 0 {
                    print!(" __");
                } else {
                    print!(" {t:>2}");
                }
            }
            println!();
        }
        println!("neighbor depths (from A):");
        for (nr, nd) in &nb_at {
            match nd {
                Some(d) => println!("  rank {nr} → d={d}"),
                None => println!("  rank {nr} → not in A"),
            }
        }
    }

    println!("\nelapsed {:.1?}", t0.elapsed());
    Ok(())
}
