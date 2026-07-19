//! Count (state, tile) pairs where the tile sits at its goal position in the
//! state, yet *every* optimal solution from that state must move the tile off
//! its goal cell at some point.
//!
//! Algorithm: load `data/dist8.bin`, then for each state s and each non-blank
//! tile t at goal in s, decide whether some optimal s → GOAL path keeps t at
//! goal throughout. Process states in increasing-distance order so each
//! successor's verdict is known when its predecessors are visited.
//!
//! Recurrence: `keep(s, t) = true` iff t is at goal in s AND there exists an
//! optimal successor s' = s.apply(m) such that m does not slide t AND
//! `keep(s', t) = true`. Base case: `keep(GOAL, t) = true` for all t.
//!
//! A (state, tile) pair counts as *forcibly displaced* when t is at goal in s
//! but `keep(s, t) = false`.

use std::path::Path;

use puzzle8::puzzle8::{io, rank, unrank, DistanceTable, GOAL, N_STATES};

fn main() {
    let path = Path::new("data/dist8.bin");
    let table: DistanceTable = io::load(path).expect("load data/dist8.bin");

    let mut by_dist: Vec<Vec<u32>> = vec![Vec::new(); 32];
    for r in 0..N_STATES {
        let d = table.dist_of_rank(r) as usize;
        by_dist[d].push(r);
    }

    // keep[r]: bit (t-1) set means tile t is at goal in state r AND can stay
    // at goal along some optimal path.
    let mut keep = vec![0u8; N_STATES as usize];
    keep[rank(&GOAL) as usize] = 0xFF;

    let max_d = (by_dist.len() - 1) as u8;
    for d in 1..=max_d {
        if by_dist[d as usize].is_empty() {
            continue;
        }
        for &r in &by_dist[d as usize] {
            let s = unrank(r);
            let mut mask: u8 = 0;
            // Per-tile flag: was this tile at goal in s AND did the loop find
            // at least one optimal-and-non-displacing successor that keeps it?
            for m in s.legal_moves().iter() {
                let s_next = s.apply(m);
                if table.dist(&s_next) != d - 1 {
                    continue;
                }
                // The tile that just slid into the old blank cell is the one
                // sitting at the *new* blank cell in s.
                let b_new = s_next.blank_pos() as usize;
                let moved_tile = s.0[b_new];
                let keep_next = keep[rank(&s_next) as usize];
                for t in 1..=8u8 {
                    if s.0[t as usize - 1] != t {
                        continue;
                    }
                    if moved_tile == t {
                        continue;
                    }
                    if keep_next & (1 << (t - 1)) != 0 {
                        mask |= 1 << (t - 1);
                    }
                }
            }
            keep[r as usize] = mask;
        }
    }

    let goal_rank = rank(&GOAL);
    let mut total_pairs: u64 = 0;
    let mut hist_by_correct_count = [0u64; 9];
    let mut states_by_correct_count = [0u64; 9];
    // For c = 1: which tile is the lone correct-and-forced one?
    let mut c1_per_tile = [0u64; 9];
    for r in 0..N_STATES {
        if r == goal_rank {
            continue;
        }
        let s = unrank(r);
        let k = keep[r as usize];
        let mut correct_count: u8 = 0;
        let mut forced_in_state: u8 = 0;
        let mut lone_forced_tile: u8 = 0;
        for t in 1..=8u8 {
            if s.0[t as usize - 1] != t {
                continue;
            }
            correct_count += 1;
            if k & (1 << (t - 1)) == 0 {
                forced_in_state += 1;
                lone_forced_tile = t;
            }
        }
        if forced_in_state > 0 {
            total_pairs += forced_in_state as u64;
            hist_by_correct_count[correct_count as usize] += forced_in_state as u64;
            states_by_correct_count[correct_count as usize] += 1;
            if correct_count == 1 {
                c1_per_tile[lone_forced_tile as usize] += 1;
            }
        }
    }

    println!("8-puzzle forced-displacement analysis");
    println!("  forcibly-displaced pairs : {total_pairs}");
    println!();
    println!("For each forced (state, tile) pair, how many tiles total are at");
    println!("their goal cells in that state?");
    println!();
    println!("  correct-tile count c | forced-pairs | distinct states with ≥1 forced");
    println!("  ---------------------+--------------+--------------------------------");
    for c in 0..=8 {
        println!(
            "                    {} | {:>12} | {:>30}",
            c, hist_by_correct_count[c], states_by_correct_count[c]
        );
    }
    println!();
    println!("c = 1: identity of the lone correct-and-forced tile");
    println!("  tile | goal cell | states");
    for t in 1..=8 {
        println!("  {:>4} | {:>9} | {:>6}", t, t - 1, c1_per_tile[t]);
    }

    // Find the shallowest state where tile 1 is the lone correct-and-forced tile.
    let mut best: Option<(u8, u32)> = None;
    for r in 0..N_STATES {
        if r == goal_rank {
            continue;
        }
        let s = unrank(r);
        if s.0[0] != 1 {
            continue;
        }
        let mut correct = 0u8;
        for t in 1..=8u8 {
            if s.0[t as usize - 1] == t {
                correct += 1;
            }
        }
        if correct != 1 {
            continue;
        }
        if keep[r as usize] & 1 != 0 {
            continue;
        } // tile 1 not forced
        let d = table.dist_of_rank(r);
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, r));
        }
    }
    if let Some((d, r)) = best {
        let s = unrank(r);
        println!();
        println!("Shallowest state where tile 1 is the lone correct-and-forced tile:");
        println!("  dist = {d}, board (row-major):");
        for row in 0..3 {
            print!("   ");
            for col in 0..3 {
                let v = s.0[row * 3 + col];
                if v == 0 {
                    print!("  _");
                } else {
                    print!(" {v:>2}");
                }
            }
            println!();
        }
    }
}
