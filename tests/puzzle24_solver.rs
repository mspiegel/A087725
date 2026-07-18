//! Phase 3a verification: the 24-puzzle IDA* solver returns optimal solutions.
//!
//! Builds small disjoint additive PDBs in-process (the full 6-6-6-6 build is a
//! `build_pdb24` job) and checks that both the Manhattan solver and the
//! incremental Korf-max solver recover the exact BFS distance for every shallow
//! state, and that the returned path reaches the goal.
//!
//! Run with `cargo test --test puzzle24_solver --features parallel,mmap`.

use std::collections::HashMap;

use puzzle8::puzzle24::pdb::pattern::Pattern;
use puzzle8::puzzle24::pdb::{KorfPdbInc, PatternDb};
use puzzle8::puzzle24::search::{idastar, idastar_inc, ManhattanHeuristic};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

/// BFS from GOAL to `depth_limit`, mapping each reached state to its distance.
fn bfs(depth_limit: u8) -> HashMap<[u8; N_CELLS], u8> {
    let mut dist: HashMap<[u8; N_CELLS], u8> = HashMap::new();
    dist.insert(GOAL.0, 0);
    let mut frontier = vec![GOAL];
    let mut depth = 0u8;
    while !frontier.is_empty() && depth < depth_limit {
        let nd = depth + 1;
        let mut next = Vec::new();
        for s in &frontier {
            for m in s.legal_moves().iter() {
                let s2 = s.apply(m);
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(s2.0) {
                    e.insert(nd);
                    next.push(s2);
                }
            }
        }
        depth = nd;
        frontier = next;
    }
    dist
}

fn reaches_goal(start: &State, sol: &[Move]) -> bool {
    let mut cur = *start;
    for m in sol {
        cur = cur.apply(*m);
    }
    cur == GOAL
}

#[test]
fn manhattan_solver_optimal_on_shallow_states() {
    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let sol = idastar(&s, &ManhattanHeuristic).expect("solvable");
        assert_eq!(sol.len() as u8, d, "manhattan non-optimal for {raw:?}");
        assert!(reaches_goal(&s, &sol));
    }
}

#[test]
fn korf_inc_solver_optimal_on_shallow_states() {
    // Four small pairwise-disjoint additive PDBs (a sub-scale stand-in for the
    // 6-6-6-6 partition — enough to exercise the full Korf-max incremental path).
    let dbs = [
        PatternDb::build(Pattern::new(&[1, 2, 3])),
        PatternDb::build(Pattern::new(&[4, 5, 9])),
        PatternDb::build(Pattern::new(&[6, 7, 8])),
        PatternDb::build(Pattern::new(&[21, 22, 23, 24])),
    ];
    let inc = KorfPdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);

    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let sol = idastar_inc(&s, &inc).expect("solvable");
        assert_eq!(sol.len() as u8, d, "korf-inc non-optimal for {raw:?}");
        assert!(reaches_goal(&s, &sol));
    }
}

#[test]
fn korf_and_manhattan_agree_on_optimal_length() {
    // Independent heuristics must yield the same optimal length on deeper,
    // hand-scrambled positions (where exhaustive BFS is infeasible).
    let dbs = [
        PatternDb::build(Pattern::new(&[1, 2, 6, 7])),
        PatternDb::build(Pattern::new(&[3, 4, 5])),
        PatternDb::build(Pattern::new(&[11, 16, 21])),
        PatternDb::build(Pattern::new(&[20, 24])),
    ];
    let inc = KorfPdbInc::new([&dbs[0], &dbs[1], &dbs[2], &dbs[3]]);

    // Build positions by scrambling GOAL a fixed number of non-undo steps.
    let mut rng: u64 = 0xABCD_1234_5678_9F01;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..40 {
        let mut s = GOAL;
        let mut last: Option<Move> = None;
        for _ in 0..14 {
            let opts: Vec<Move> = s
                .legal_moves()
                .iter()
                .filter(|&m| last.is_none_or(|p: Move| m != p.inverse()))
                .collect();
            let m = opts[(next() as usize) % opts.len()];
            s = s.apply(m);
            last = Some(m);
        }
        let sol_m = idastar(&s, &ManhattanHeuristic).expect("solvable");
        let sol_k = idastar_inc(&s, &inc).expect("solvable");
        assert_eq!(sol_m.len(), sol_k.len(), "heuristics disagree on optimal length for {:?}", s.0);
        assert!(reaches_goal(&s, &sol_m) && reaches_goal(&s, &sol_k));
    }
}
