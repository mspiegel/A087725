//! Insert (rank, depth) pairs into a solve cache. Mirrors each insert under
//! reflection symmetry — if `r != reflect(r)`, both ranks are recorded with
//! the same depth. Backs up the cache file before writing.
//!
//! Modes:
//!   cache_insert CACHE RANK1 DEPTH1 [RANK2 DEPTH2 ...]
//!     Inline (rank, depth) pairs as positional args.
//!
//!   cache_insert CACHE --from-log REVERSE_NEIGHBORHOOD_LOG
//!     Parse a reverse_neighborhood log and extract every (rank, depth) pair
//!     listed in `=== NEW d=NN BOARDS ===` sections.

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle15::enumerate::cache;
use puzzle8::puzzle15::rank::{rank, unrank};
use puzzle8::puzzle15::symmetry::reflect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: cache_insert CACHE (RANK DEPTH ...|--from-log LOG)");
        std::process::exit(2);
    }
    let cache_path: PathBuf = args.remove(0).into();

    let pairs: Vec<(u64, u8)> = if args.first().map(|s| s == "--from-log").unwrap_or(false) {
        if args.len() < 2 {
            eprintln!("--from-log requires a log path");
            std::process::exit(2);
        }
        let log_path = &args[1];
        let p = parse_reverse_neighborhood_log(log_path)?;
        println!("parsed {} (rank, depth) pairs from {}", p.len(), log_path);
        p
    } else {
        // Positional (RANK DEPTH ...) args.
        let mut it = args.into_iter();
        let mut p: Vec<(u64, u8)> = Vec::new();
        loop {
            let r_str = match it.next() {
                Some(s) => s,
                None => break,
            };
            let d_str = it.next().ok_or("trailing rank with no depth")?;
            let r: u64 = r_str.parse()?;
            let d: u8 = d_str.parse()?;
            p.push((r, d));
        }
        p
    };

    if pairs.is_empty() {
        eprintln!("no pairs to insert");
        std::process::exit(2);
    }

    println!("loading {} ...", cache_path.display());
    let mut c = cache::load(&cache_path)?;
    let n_before = c.len();
    println!("loaded {} entries", n_before);

    // Backup before mutation.
    let backup_path = cache_path.with_extension("bin.bak");
    println!("backing up to {} ...", backup_path.display());
    cache::save(&backup_path, &c)?;

    let t = Instant::now();
    let mut new_inserts = 0u32;
    let mut already_present = 0u32;
    let mut mirror_inserts = 0u32;
    let mut conflicts = 0u32;
    for &(r, d) in &pairs {
        match c.get(&r).copied() {
            Some(prev) if prev != d => {
                println!("CONFLICT at rank {}: existing d={}, new d={} — skipping", r, prev, d);
                conflicts += 1;
                continue;
            }
            Some(_) => { already_present += 1; }
            None => { c.insert(r, d); new_inserts += 1; }
        }
        let r_refl = rank(&reflect(&unrank(r)));
        if r_refl != r {
            match c.get(&r_refl).copied() {
                Some(prev) if prev != d => {
                    println!("CONFLICT at reflected rank {}: existing d={}, new d={} — skipping mirror", r_refl, prev, d);
                    conflicts += 1;
                }
                Some(_) => {}
                None => { c.insert(r_refl, d); mirror_inserts += 1; }
            }
        }
    }

    println!("\nsummary:");
    println!("  pairs requested:    {}", pairs.len());
    println!("  new direct inserts: {}", new_inserts);
    println!("  mirror inserts:     {}", mirror_inserts);
    println!("  already present:    {}", already_present);
    println!("  conflicts (skipped):{}", conflicts);

    if new_inserts + mirror_inserts > 0 {
        println!("\nsaving {} (was {}, now {}) ...",
                 cache_path.display(), n_before, c.len());
        cache::save(&cache_path, &c)?;
        println!("done in {:.1?}", t.elapsed());
    } else {
        println!("\nno new entries; cache unchanged");
    }

    Ok(())
}

/// Parse a `reverse_neighborhood` example's log file and extract (rank, depth)
/// pairs from every `=== NEW d=NN BOARDS ===` section.
fn parse_reverse_neighborhood_log(path: &str) -> Result<Vec<(u64, u8)>, Box<dyn std::error::Error>> {
    let f = std::fs::File::open(path)?;
    let r = std::io::BufReader::new(f);
    let mut depth: Option<u8> = None;
    let mut pairs: Vec<(u64, u8)> = Vec::new();
    for line in r.lines() {
        let line = line?;
        // Section header: "=== NEW d=NN BOARDS: M ===" (reverse_neighborhood)
        // or              "=== d=NN finds: M ==="          (residue_pocket)
        let rest_opt = line.strip_prefix("=== NEW d=")
            .or_else(|| line.strip_prefix("=== d="));
        if let Some(rest) = rest_opt {
            let n: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            depth = n.parse().ok();
            continue;
        }
        // Board listing: "  # NNN rank XXXXX blank@N: ..."
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('#') {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            // parts: [N, "rank", X, "blank@N:", ...]
            if parts.len() >= 3 && parts[1] == "rank" {
                if let (Ok(r), Some(d)) = (parts[2].parse::<u64>(), depth) {
                    pairs.push((r, d));
                }
            }
        }
    }
    Ok(pairs)
}
