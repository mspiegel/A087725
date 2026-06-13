//! Iterative-deepening A* over the 15-puzzle [`State`].
//!
//! Standard IDA*: at iteration `k`, depth-first search with `f = g + h` cut
//! off at threshold `bound_k`. If the goal is found, return the path; else
//! `bound_{k+1}` is the minimum `f` value that exceeded `bound_k`. Repeat.
//!
//! With an admissible heuristic, the first solution found is optimal. We
//! additionally prune immediate-undo moves (never apply `m.inverse()`
//! immediately after `m`), which roughly halves the branching factor without
//! sacrificing optimality.
//!
//! Cost types stay `u8` — the 15-puzzle diameter is 80, well below 255.

use crate::puzzle15::state::{Move, State, GOAL};

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
    let mut path: Vec<Move> = Vec::with_capacity(96);

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
    use crate::puzzle15::search::heuristic::ManhattanHeuristic;
    use crate::puzzle15::search::tests_util::bfs_distances;

    #[test]
    fn solves_goal_with_empty_path() {
        let sol = idastar(&GOAL, &ManhattanHeuristic).unwrap();
        assert!(sol.is_empty());
    }

    #[test]
    fn one_step_neighbors_solved_in_one_move() {
        for m in Move::ALL {
            if GOAL.legal_moves().contains(m) {
                let s = GOAL.apply(m);
                let sol = idastar(&s, &ManhattanHeuristic).unwrap();
                assert_eq!(sol.len(), 1);
                // Applying solution must reach GOAL.
                let mut cur = s;
                for mv in &sol {
                    cur = cur.apply(*mv);
                }
                assert_eq!(cur, GOAL);
            }
        }
    }

    #[test]
    fn applying_solution_always_reaches_goal_on_shallow_walks() {
        // For each state at known true distance ≤ 10, IDA* must:
        //   1. return a solution of exactly that length (optimality), and
        //   2. that solution must apply to the state and reach GOAL.
        let table = bfs_distances(10);
        for (raw, &truth) in &table {
            let s = State(*raw);
            let sol = idastar(&s, &ManhattanHeuristic).expect("solver should solve");
            assert_eq!(
                sol.len() as u8,
                truth,
                "IDA* returned len {} but true dist is {} for {:?}",
                sol.len(),
                truth,
                raw
            );
            let mut cur = s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL, "solution doesn't reach goal from {:?}", raw);
        }
    }

    #[test]
    fn random_walk_of_depth_25_solvable_optimally() {
        // A random walk gives an upper bound; IDA* must return ≤ walk length
        // and applying the solution must reach GOAL.
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        // Construct a position by walking 25 random steps from GOAL.
        let mut s = GOAL;
        let mut walk_len = 0u8;
        for i in 0u32..25 {
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    walk_len += 1;
                    break;
                }
            }
        }
        let sol = idastar(&s, &ManhattanHeuristic).expect("should solve");
        assert!(
            (sol.len() as u8) <= walk_len,
            "solution len {} exceeds walk len {}",
            sol.len(),
            walk_len
        );
        let mut cur = s;
        for m in &sol {
            cur = cur.apply(*m);
        }
        assert_eq!(cur, GOAL);
    }
}
