//! Build `data/dist8.bin` — the persistent full distance table for the 8-puzzle.
//!
//! Usage:
//!     cargo run --release --bin build_table [--out PATH]
//!
//! Default output path is `data/dist8.bin`. Prints a brief summary on stderr.

use std::path::PathBuf;
use std::time::Instant;

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::io::save;
use puzzle8::puzzle8::state::{DIAMETER, N_STATES};

fn parse_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut out = PathBuf::from("data/dist8.bin");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" | "-o" => {
                let v = args.next().expect("--out requires a path argument");
                out = PathBuf::from(v);
            }
            "-h" | "--help" => {
                eprintln!("Usage: build_table [--out PATH]");
                eprintln!("Default output: data/dist8.bin");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {}", other);
                std::process::exit(2);
            }
        }
    }
    out
}

fn main() {
    let out = parse_args();

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }

    eprintln!("building 8-puzzle distance table ({} states)...", N_STATES);
    let t_start = Instant::now();
    let table = DistanceTable::build();
    let build_ms = t_start.elapsed().as_millis();

    let visited = table.visited_count();
    let diameter = table.diameter();
    assert_eq!(visited, N_STATES, "BFS missed states: {}/{}", visited, N_STATES);
    assert_eq!(diameter, DIAMETER, "unexpected diameter: {}", diameter);

    eprintln!("build OK: {} states, diameter {}, {} ms", visited, diameter, build_ms);

    save(&table, &out).expect("save distance table");
    let meta = std::fs::metadata(&out).expect("stat output");
    eprintln!("wrote {} ({} bytes)", out.display(), meta.len());
}
