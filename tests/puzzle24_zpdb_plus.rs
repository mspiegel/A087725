//! Integration test: 24-puzzle `ZpdbPlusInc` = max(zpdb, refl-zpdb, MD+LC, WD).
//!
//! Builds small in-process zero-aware PDBs (a sub-scale stand-in for the
//! production 6-6-6-6 / 7-7-7-3 partitions) and checks, against exhaustive
//! shallow BFS, that the combined heuristic is admissible, dominates the additive
//! korf-plus, and drives IDA* to the exact optimal depth (replaying to GOAL).
//!
//! Run with `cargo test --release --test puzzle24_zpdb_plus --features parallel,mmap`.

use std::collections::HashMap;

use puzzle8::puzzle24::pdb::heuristic::{AdditivePdbHeuristic, ReflectedHeuristic};
use puzzle8::puzzle24::pdb::pattern::Pattern;
use puzzle8::puzzle24::pdb::{PatternDb, ZPatternDb, ZpdbPlusInc};
use puzzle8::puzzle24::search::{
    idastar_inc, idastar_inc_with_stats, Heuristic, IncHeuristic, LinearConflictHeuristic,
    SearchStats, WalkingDistanceHeuristic,
};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS};

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
fn zpdb_plus_admissible_and_dominates_korf_plus() {
    WalkingDistanceHeuristic::warm_up();
    let pats = [Pattern::new(&[1, 2, 3]), Pattern::new(&[6, 7, 8])];
    let zdbs = [ZPatternDb::build(pats[0]), ZPatternDb::build(pats[1])];
    let adbs = [PatternDb::build(pats[0]), PatternDb::build(pats[1])];
    let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1]]);
    let add = AdditivePdbHeuristic::new(&adbs);
    let refl = ReflectedHeuristic::new(AdditivePdbHeuristic::new(&adbs));

    let truth = bfs(9);
    let mut stats = SearchStats::default();
    for (raw, &d) in &truth {
        let s = State(*raw);
        let hz = IncHeuristic::root(&zplus, &s, &mut stats).0;
        let korf_plus = add
            .h(&s)
            .max(refl.h(&s))
            .max(LinearConflictHeuristic.h(&s))
            .max(WalkingDistanceHeuristic.h(&s));
        assert!(hz >= korf_plus, "zpdb-plus {hz} < korf-plus {korf_plus} for {raw:?}");
        assert!(hz <= d, "zpdb-plus {hz} inadmissible (truth {d}) for {raw:?}");
    }
}

#[test]
fn zpdb_plus_solver_optimal_on_shallow_states() {
    WalkingDistanceHeuristic::warm_up();
    let zdbs = [
        ZPatternDb::build(Pattern::new(&[1, 2, 3, 6])),
        ZPatternDb::build(Pattern::new(&[4, 5, 9, 10])),
        ZPatternDb::build(Pattern::new(&[21, 22, 23, 24])),
    ];
    let zplus = ZpdbPlusInc::new([&zdbs[0], &zdbs[1], &zdbs[2]]);

    let truth = bfs(8);
    for (raw, &d) in &truth {
        let s = State(*raw);
        let sol = idastar_inc(&s, &zplus).expect("solvable");
        assert_eq!(sol.len() as u8, d, "zpdb-plus non-optimal for {raw:?}");
        assert!(reaches_goal(&s, &sol));
    }

    // A few deeper hand-scrambles: optimal length must replay to GOAL.
    let mut rng: u64 = 0x5151_2424_3636_4848;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..10 {
        let mut s = GOAL;
        let mut last: Option<Move> = None;
        for _ in 0..16 {
            let opts: Vec<Move> = s
                .legal_moves()
                .iter()
                .filter(|&m| last.is_none_or(|p: Move| m != p.inverse()))
                .collect();
            let m = opts[(next() as usize) % opts.len()];
            s = s.apply(m);
            last = Some(m);
        }
        let (sol, _) = idastar_inc_with_stats(&s, &zplus);
        let sol = sol.expect("solvable");
        assert!(reaches_goal(&s, &sol), "deep scramble solution missed GOAL: {:?}", s.0);
    }
}
