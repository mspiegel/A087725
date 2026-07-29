//! Does a *demanded* line always cost something? — a scan of the cWD surcharge
//! curves, sizing the tighter neighbour pre-prune.
//!
//! # Why
//!
//! [`Cwd::child_h_lb`] bounds a child from its parent's cached neighbour-WD and
//! is loose by *exactly* the changed axis's surcharge:
//!
//! ```text
//! lb = h(child) − surcharge(child, changed axis)
//! ```
//!
//! Instrumenting the flat engine on `R` at exhaust-144 shows the price of that
//! looseness: of 422,379,805 children built, **103,397,347 (24.48%) are over the
//! bound the moment they are built** and thrown away — every one of them with a
//! nonzero surcharge on the axis its move touched, since that is precisely the
//! term the pre-prune dropped.
//!
//! Closing the gap needs a lower bound on the child's surcharge *without* probing
//! the table for its curves. The child's **demand** is free (board-derived, via
//! the demand LUT); its **curves** are not. So the cheapest possible fix rests on
//! a premise:
//!
//! > a line carrying demand `d ≥ 1` always has curve nibble `≥ 1`
//!
//! If that held universally, "the child has any demand" would prove
//! `surcharge ≥ 2`, and the pre-prune could add 2 for nothing.
//!
//! There is reason to doubt it. `WD_row = displacement + 2·retreats` already
//! charges for tiles that step out of their line and back, so an escape can ride
//! along on a retreat WD has *already* paid for — leaving the curve at 0 despite
//! nonzero demand.
//!
//! # What this prints
//!
//! - The global answer (does the premise hold?).
//! - `floor[g][d]` — the minimum nibble over **every** state, per goal line and
//!   demand. Any entry `≥ 1` is a *sound, zero-memory* rule: "demand `d` on line
//!   `g` ⇒ surcharge `≥ 2·floor`". That is the partial win to look for.
//! - The nibble distribution, which bounds what a per-state (index-variant)
//!   pre-prune could recover.
//!
//! Run: `cargo run --release --example cwd_curve_floor`

use puzzle8::puzzle24::search::load_cwd_overlay;
use puzzle8::puzzle24::state::W;
use std::path::Path;
use std::time::Instant;

/// Curve nibbles are indexed by demand `1..=4`.
const D_MAX: usize = 4;

fn main() {
    let path = Path::new("data/cwd_single.bin");
    if !path.exists() {
        eprintln!("data/cwd_single.bin absent — build it with `cargo run --release --example build_cwd_table`");
        return;
    }
    let t0 = Instant::now();
    let overlay = match load_cwd_overlay(path) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("failed to load {}: {e}", path.display());
            return;
        }
    };
    println!(
        "overlay: {} states loaded in {:.1?}\n",
        overlay.len(),
        t0.elapsed()
    );

    // floor[g][d-1]: the smallest nibble any state shows for that (line, demand).
    let mut floor = [[15u8; D_MAX]; W];
    let mut ceil = [[0u8; D_MAX]; W];
    let mut zeros = [[0u64; D_MAX]; W];
    let mut total = [[0u64; D_MAX]; W];
    let mut hist = [0u64; 16];
    // How many states could ever levy a surcharge at all?
    let mut states_all_zero = 0u64;

    for curves in overlay.values() {
        let mut any = false;
        for (g, &cv) in curves.iter().enumerate() {
            for d in 1..=D_MAX {
                let nib = ((cv >> (4 * (d - 1))) & 0xF) as u8;
                hist[nib as usize] += 1;
                total[g][d - 1] += 1;
                if nib == 0 {
                    zeros[g][d - 1] += 1;
                } else {
                    any = true;
                }
                if nib < floor[g][d - 1] {
                    floor[g][d - 1] = nib;
                }
                if nib > ceil[g][d - 1] {
                    ceil[g][d - 1] = nib;
                }
            }
        }
        if !any {
            states_all_zero += 1;
        }
    }

    // ---- the premise ------------------------------------------------------
    let any_zero = floor.iter().flatten().any(|&f| f == 0);
    println!(
        "PREMISE  \"a demanded line always costs >= 1 nibble\": {}",
        if any_zero { "FALSE" } else { "TRUE" }
    );

    // ---- the partial win --------------------------------------------------
    println!("\nfloor[line][demand] — min nibble over ALL states.");
    println!(
        "Any entry >= 1 is a sound zero-memory rule: demand d on line g => surcharge >= 2*floor."
    );
    print!("{:>6}", "line");
    for d in 1..=D_MAX {
        print!("{:>10}", format!("d={d}"));
    }
    println!();
    for g in 0..W {
        print!("{:>6}", g + 1);
        for d in 1..=D_MAX {
            print!("{:>10}", floor[g][d - 1]);
        }
        println!();
    }

    println!("\nzero-rate[line][demand] — share of states whose nibble is 0 there.");
    print!("{:>6}", "line");
    for d in 1..=D_MAX {
        print!("{:>10}", format!("d={d}"));
    }
    println!();
    for g in 0..W {
        print!("{:>6}", g + 1);
        for d in 1..=D_MAX {
            let z = zeros[g][d - 1] as f64 / total[g][d - 1].max(1) as f64;
            print!("{:>9.2}%", 100.0 * z);
        }
        println!();
    }

    println!("\nmax nibble seen[line][demand] — the ceiling an exact per-state bound could reach.");
    print!("{:>6}", "line");
    for d in 1..=D_MAX {
        print!("{:>10}", format!("d={d}"));
    }
    println!();
    for g in 0..W {
        print!("{:>6}", g + 1);
        for d in 1..=D_MAX {
            print!("{:>10}", ceil[g][d - 1]);
        }
        println!();
    }

    // ---- what an exact (index-variant) pre-prune could recover ------------
    let n: u64 = hist.iter().sum();
    println!("\nnibble distribution over all (state, line, demand) triples:");
    for (v, &c) in hist.iter().enumerate() {
        if c > 0 {
            println!(
                "  nibble {v:>2}: {c:>14}  ({:>6.2}%)  => surcharge {:>2}",
                100.0 * c as f64 / n as f64,
                2 * v
            );
        }
    }
    println!(
        "\nstates whose every curve is zero (no surcharge possible at any demand): {} ({:.2}%)",
        states_all_zero,
        100.0 * states_all_zero as f64 / overlay.len() as f64
    );
}
