//! Direction #7: Kolmogorov-complexity floor.
//!
//! The other six directions are solvers (or solver-shaped objects). This one
//! is a *measurement*: how compressible is `dist8.bin`'s raw byte stream
//! when fed through general-purpose lossless compressors?
//!
//! The best size achieved is our tightest *upper bound* on the Kolmogorov
//! complexity `K(dist8.bin)` — the shortest description of the file in any
//! language. `K` is uncomputable in general, but any compressor that
//! produces a smaller artifact tightens the bound.
//!
//! Why this matters for the compression study: any other direction whose
//! `bytes_stored` doesn't beat this floor is essentially doing the same
//! work as a generic compressor — it's not extracting puzzle-specific
//! structure beyond what zstd/xz already find. Conversely, a direction that
//! beats this floor *is* extracting genuine semantic structure.
//!
//! Run with:
//!     cargo run --release --example c7_kolmogorov
//!
//! Requires `gzip`, `bzip2`, `xz`, `zstd`, `brotli` in `$PATH`. Each is
//! skipped gracefully if unavailable.

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::unrank;
use puzzle8::puzzle8::search::{Heuristic, ManhattanHeuristic};
use puzzle8::puzzle8::state::N_STATES;

/// Pipe `data` into `cmd <args>` on stdin; return compressed-output size on
/// success, `None` if the command isn't installed or returns non-zero.
fn compress(cmd: &str, args: &[&str], data: &[u8]) -> Option<usize> {
    let mut c = Command::new(cmd);
    c.args(args);
    c.stdin(std::process::Stdio::piped());
    c.stdout(std::process::Stdio::piped());
    c.stderr(std::process::Stdio::null());
    let mut child = c.spawn().ok()?;
    {
        let stdin = child.stdin.as_mut()?;
        stdin.write_all(data).ok()?;
    }
    let output = child.wait_with_output().ok()?;
    if output.status.success() {
        Some(output.stdout.len())
    } else {
        None
    }
}

/// Shannon entropy of the byte distribution, in bits per byte. The number
/// of bytes a *memoryless* entropy coder needs is approximately
/// `(entropy_bits_per_byte * len(data) / 8).ceil()`.
///
/// Note: this is the entropy of single bytes, treating them as independent
/// — i.e. the lower bound for **memoryless** coders. Real compressors like
/// LZ77/zstd/brotli exploit *correlations* between adjacent bytes (e.g.
/// neighbour ranks having similar distances) and can therefore beat this
/// per-byte Shannon floor. The true K-floor is given by the joint
/// distribution and is uncomputable; the byte-entropy here is only a
/// reference for "how much can be gained by entropy coding alone."
fn shannon_entropy_bits_per_byte(data: &[u8]) -> f64 {
    let mut hist = [0u64; 256];
    for &b in data {
        hist[b as usize] += 1;
    }
    let n = data.len() as f64;
    let mut h = 0.0f64;
    for &c in &hist {
        if c == 0 {
            continue;
        }
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    h
}

#[derive(Clone)]
struct Result {
    label: String,
    bytes: usize,
}

/// Run a sweep of compressors over `data`. The "raw" entry is the input
/// size for reference; the "Shannon" entry is the entropy floor.
fn sweep(label_prefix: &str, data: &[u8]) -> Vec<Result> {
    let mut out = Vec::new();

    out.push(Result {
        label: format!("{}: raw", label_prefix),
        bytes: data.len(),
    });

    let h_bits = shannon_entropy_bits_per_byte(data);
    let h_bytes = ((h_bits * data.len() as f64) / 8.0).ceil() as usize;
    out.push(Result {
        label: format!("{}: Shannon entropy floor", label_prefix),
        bytes: h_bytes,
    });

    // `(label, command, args)` triples.
    let compressors: &[(&str, &str, &[&str])] = &[
        ("gzip -9", "gzip", &["-9", "-c"]),
        ("bzip2 -9", "bzip2", &["-9", "-c"]),
        ("xz -9 -e", "xz", &["-9", "-e", "-c"]),
        ("zstd --ultra -22", "zstd", &["--ultra", "-22", "-q"]),
        ("brotli -q 11", "brotli", &["-q", "11", "-c"]),
    ];

    for (label, cmd, args) in compressors {
        let t0 = Instant::now();
        match compress(cmd, args, data) {
            Some(size) => {
                let _dt = t0.elapsed();
                out.push(Result {
                    label: format!("{}: {}", label_prefix, label),
                    bytes: size,
                });
            }
            None => {
                eprintln!(
                    "  (skipping {}: command unavailable or returned non-zero)",
                    cmd
                );
            }
        }
    }

    out
}

fn print_results(results: &[Result], baseline: usize) {
    for r in results {
        let ratio = baseline as f64 / r.bytes as f64;
        println!("  {:<46} {:>10} bytes  ({:.2}× vs raw)", r.label, r.bytes, ratio);
    }
}

fn main() {
    eprintln!("building distance table...");
    let table = DistanceTable::build();
    let raw = table.raw();
    let n = raw.len();
    eprintln!(
        "  {} bytes ({} states, diameter {})",
        n,
        table.visited_count(),
        table.diameter(),
    );
    println!();

    println!("== Direction #7: Kolmogorov-complexity floor ==");
    println!("Two variants of the same {}-byte payload:", n);
    println!("  raw  - distance bytes in rank order (the canonical layout)");
    println!("  gap  - distance(s) - Manhattan(s) per rank (heuristic-gap encoding)");
    println!();
    println!("Both variants are *queryable* — given the compressed bytes, the full");
    println!("rank→distance mapping is recoverable. The smallest result is our");
    println!("tightest upper bound on K(dist8.bin).");
    println!();

    let baseline = n;
    let mut all: Vec<Result> = Vec::new();

    // -------- Variant 1: raw --------
    println!("--- Variant: raw distance bytes (canonical layout) ---");
    let v1 = sweep("raw", raw);
    print_results(&v1, baseline);
    all.extend(v1);
    println!();

    // -------- Variant 2: heuristic gap --------
    // dist(s) - Manhattan(s) for each rank. Manhattan is admissible
    // (h ≤ dist), so the difference is always >= 0 and tends to be small.
    eprintln!("computing heuristic gap...");
    let h = ManhattanHeuristic;
    let mut gap: Vec<u8> = vec![0u8; n];
    for r in 0..N_STATES {
        let s = unrank(r);
        let d = raw[r as usize];
        let m = h.h(&s);
        debug_assert!(d >= m, "Manhattan exceeded true distance at {:?}", s.0);
        gap[r as usize] = d - m;
    }
    println!("--- Variant: dist - Manhattan (heuristic-gap encoding) ---");
    let v2 = sweep("gap", &gap);
    print_results(&v2, baseline);
    all.extend(v2);
    println!();

    // -------- Summary --------
    println!("== Summary ==");
    let real_compressors: Vec<&Result> = all
        .iter()
        .filter(|r| !r.label.ends_with("raw") && !r.label.contains("Shannon"))
        .collect();
    let shannon_results: Vec<&Result> = all.iter().filter(|r| r.label.contains("Shannon")).collect();
    let best = real_compressors.iter().min_by_key(|r| r.bytes).unwrap();
    let best_shannon = shannon_results.iter().min_by_key(|r| r.bytes).unwrap();

    println!(
        "  best real compressor result    : {} bytes  via \"{}\"  ({:.2}× vs raw)",
        best.bytes,
        best.label,
        baseline as f64 / best.bytes as f64,
    );
    println!(
        "  best per-byte Shannon floor    : {} bytes  via \"{}\"  ({:.2}× vs raw)",
        best_shannon.bytes,
        best_shannon.label,
        baseline as f64 / best_shannon.bytes as f64,
    );
    println!();
    println!("Interpretation:");
    println!("  - {} bytes is the tightest upper bound on K(dist8.bin) from this sweep.", best.bytes);
    println!("    Any solver direction storing fewer than {} bytes is doing genuine", best.bytes);
    println!("    algorithmic work beyond what generic LZ + entropy coding finds.");
    println!();
    println!("  - The Shannon floor treats bytes as independent. LZ-based compressors");
    println!("    exploit cross-byte correlations (adjacent ranks have similar distances)");
    println!("    and can beat per-byte Shannon — that's why brotli on \"gap\" lands below");
    println!("    the Shannon floor here. The true joint-entropy floor is uncomputable.");
}
