//! End-to-end optimality verification for the 15-puzzle PDB + IDA\* pipeline.
//!
//! Strategy:
//! - Build a small PDB (k ≤ 5 tiles — both BFS and PDB tables fit comfortably).
//! - Use it (either alone or in an additive composition) as the IDA\* heuristic.
//! - Compute true distances for every state reachable within depth `D` via
//!   a forward BFS from `GOAL`.
//! - For every such state, run IDA\* and assert the solution length matches
//!   the BFS-known true distance, and that applying the solution restores `GOAL`.

use puzzle8::puzzle15::pdb::{AdditivePdbHeuristic, PatternDb, PdbHeuristic};
use puzzle8::puzzle15::pdb::pattern::Pattern;
use puzzle8::puzzle15::search::{idastar, Heuristic, ManhattanHeuristic};
use puzzle8::puzzle15::state::{State, GOAL, N_CELLS};

use std::collections::{HashMap, VecDeque};

fn bfs_distances(depth_limit: u8) -> HashMap<[u8; N_CELLS], u8> {
    let mut dist: HashMap<[u8; N_CELLS], u8> = HashMap::new();
    dist.insert(GOAL.0, 0);
    let mut frontier: VecDeque<State> = VecDeque::new();
    frontier.push_back(GOAL);
    while let Some(s) = frontier.pop_front() {
        let d = dist[&s.0];
        if d >= depth_limit {
            continue;
        }
        for m in s.legal_moves().iter() {
            let n = s.apply(m);
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(n.0) {
                e.insert(d + 1);
                frontier.push_back(n);
            }
        }
    }
    dist
}

fn assert_idastar_optimal_for_each<H: Heuristic>(h: &H, truth: &HashMap<[u8; N_CELLS], u8>) {
    let mut count = 0u32;
    for (raw, &true_dist) in truth {
        let s = State(*raw);
        let sol = idastar(&s, h).expect("must solve a reachable state");
        assert_eq!(
            sol.len() as u8,
            true_dist,
            "IDA* returned len {} but true distance is {} for state {:?}",
            sol.len(),
            true_dist,
            raw
        );
        let mut cur = s;
        for m in &sol {
            cur = cur.apply(*m);
        }
        assert_eq!(cur, GOAL, "applying solution doesn't reach GOAL from {raw:?}");
        count += 1;
    }
    assert!(count > 0, "no states were tested");
}

#[test]
fn idastar_with_single_3tile_pdb_is_optimal_on_shallow_states() {
    let pdb = PatternDb::build(Pattern::new(&[1, 2, 3]));
    let h = PdbHeuristic::new(&pdb);
    let truth = bfs_distances(10);
    assert_idastar_optimal_for_each(&h, &truth);
}

#[test]
fn idastar_with_single_5tile_pdb_is_optimal_on_shallow_states() {
    let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 6, 7]));
    let h = PdbHeuristic::new(&pdb);
    let truth = bfs_distances(10);
    assert_idastar_optimal_for_each(&h, &truth);
}

#[test]
fn idastar_with_additive_3plus3_pdb_is_optimal_on_shallow_states() {
    let a = PatternDb::build(Pattern::new(&[1, 2, 3]));
    let b = PatternDb::build(Pattern::new(&[4, 5, 6]));
    let dbs = vec![a, b];
    let h = AdditivePdbHeuristic::new(&dbs);
    let truth = bfs_distances(10);
    assert_idastar_optimal_for_each(&h, &truth);
}

#[test]
fn idastar_with_additive_partition_covering_full_15_tile_set() {
    // Disjoint partition that covers all 15 tiles via small patterns: a 4+4+4+3
    // split. This is a *correctness* check rather than the production
    // configuration (which uses 7+8 patterns).
    let a = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
    let b = PatternDb::build(Pattern::new(&[5, 6, 7, 8]));
    let c = PatternDb::build(Pattern::new(&[9, 10, 11, 12]));
    let d = PatternDb::build(Pattern::new(&[13, 14, 15]));
    let dbs = vec![a, b, c, d];
    let h = AdditivePdbHeuristic::new(&dbs);
    let truth = bfs_distances(10);
    assert_idastar_optimal_for_each(&h, &truth);
}

#[test]
fn manhattan_and_pdb_agree_on_shallow_states() {
    // Both heuristics are admissible; IDA* with either should return the same
    // optimal length for every state.
    let pdb = PatternDb::build(Pattern::new(&[1, 2, 3, 4]));
    let h_pdb = PdbHeuristic::new(&pdb);
    let h_man = ManhattanHeuristic;
    let truth = bfs_distances(8);
    for (raw, &true_dist) in &truth {
        let s = State(*raw);
        let len_pdb = idastar(&s, &h_pdb).unwrap().len() as u8;
        let len_man = idastar(&s, &h_man).unwrap().len() as u8;
        assert_eq!(len_pdb, true_dist);
        assert_eq!(len_man, true_dist);
    }
}
