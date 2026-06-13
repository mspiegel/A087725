//! Iterative-deepening A* over [`State`].
//!
//! Standard IDA*: at iteration `k`, depth-first search with `f = g + h` cut
//! off at threshold `bound_k`. If the goal is found, return the path; else
//! `bound_{k+1}` is the minimum `f` value that exceeded `bound_k`. Repeat.
//!
//! With an admissible heuristic, the first solution found is optimal. We
//! additionally prune immediate-undo moves (never apply `m.inverse()`
//! immediately after `m`), which roughly halves the branching factor without
//! sacrificing optimality.

use crate::puzzle8::state::{Move, State, GOAL};

use super::heuristic::Heuristic;

/// Return an optimal move sequence from `start` to [`GOAL`].
///
/// Returns `Some(vec![])` if `start == GOAL`. Returns `None` only if `start`
/// is unreachable from `GOAL`, which is impossible for solvable states.
pub fn idastar<H: Heuristic>(start: &State, h: &H) -> Option<Vec<Move>> {
    if start == &GOAL {
        return Some(Vec::new());
    }

    let mut bound = h.h(start);
    let mut path: Vec<Move> = Vec::with_capacity(64);

    loop {
        match search(start, 0, bound, &mut path, None, h) {
            Step::Found => return Some(path),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return None;
                }
                bound = next;
            }
        }
    }
}

enum Step {
    Found,
    /// Smallest `f` value seen at this iteration that exceeded `bound`.
    Bound(u8),
}

fn search<H: Heuristic>(
    s: &State,
    g: u8,
    bound: u8,
    path: &mut Vec<Move>,
    last: Option<Move>,
    h: &H,
) -> Step {
    let f = g.saturating_add(h.h(s));
    if f > bound {
        return Step::Bound(f);
    }
    if s == &GOAL {
        return Step::Found;
    }

    let mut min_next = u8::MAX;
    for m in s.legal_moves().iter() {
        if let Some(prev) = last {
            if m == prev.inverse() {
                continue;
            }
        }
        let s_next = s.apply(m);
        path.push(m);
        match search(&s_next, g + 1, bound, path, Some(m), h) {
            Step::Found => return Step::Found,
            Step::Bound(n) => {
                if n < min_next {
                    min_next = n;
                }
            }
        }
        path.pop();
    }

    Step::Bound(min_next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::bfs::DistanceTable;
    use crate::puzzle8::search::heuristic::{ManhattanHeuristic, TableHeuristic};

    #[test]
    fn solves_goal_with_empty_path() {
        let sol = idastar(&GOAL, &ManhattanHeuristic).unwrap();
        assert!(sol.is_empty());
    }

    #[test]
    fn applying_solution_reaches_goal() {
        let t = DistanceTable::build();
        let antipodes = t.antipodes();
        for s in &antipodes {
            let sol = idastar(s, &ManhattanHeuristic).expect("should solve");
            assert_eq!(sol.len() as u8, t.dist(s));
            let mut cur = *s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL);
        }
    }

    #[test]
    fn manhattan_and_table_agree_on_antipodes() {
        let t = DistanceTable::build();
        let th = TableHeuristic::new(&t);
        for s in t.antipodes() {
            let a = idastar(&s, &ManhattanHeuristic).unwrap().len() as u8;
            let b = idastar(&s, &th).unwrap().len() as u8;
            assert_eq!(a, b);
            assert_eq!(a, t.dist(&s));
        }
    }

    #[test]
    fn small_sample_matches_table_distance() {
        // 200 states by random walk from GOAL.
        let t = DistanceTable::build();
        let mut s = GOAL;
        let mut prng: u32 = 0x9E3779B9;
        let mut tested = 0u32;
        for _ in 0..200 {
            let moves: Vec<Move> = s.legal_moves().iter().collect();
            prng = prng.wrapping_mul(1664525).wrapping_add(1013904223);
            let m = moves[(prng as usize) % moves.len()];
            s = s.apply(m);

            let sol = idastar(&s, &ManhattanHeuristic).unwrap();
            assert_eq!(sol.len() as u8, t.dist(&s), "mismatch at state {:?}", s.0);
            tested += 1;
        }
        assert!(tested > 0);
    }
}
