//! Max-of-two-admissibles combinator.
//!
//! `max(a, b)` of two admissible heuristics is itself admissible: each term is
//! a lower bound on true distance, so the larger of the two is also a (tighter)
//! lower bound. Used to compose Manhattan with Linear Conflict and Walking
//! Distance — and, generally, any chain of admissible heuristics by nesting.

use super::Heuristic;
use crate::puzzle8::state::State;

/// Max of two admissible heuristics. Always admissible.
pub struct MaxHeuristic<A: Heuristic, B: Heuristic> {
    a: A,
    b: B,
}

impl<A: Heuristic, B: Heuristic> MaxHeuristic<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Heuristic, B: Heuristic> Heuristic for MaxHeuristic<A, B> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        let ha = self.a.h(s);
        let hb = self.b.h(s);
        ha.max(hb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::search::ManhattanHeuristic;
    use crate::puzzle8::state::{Move, GOAL};

    #[test]
    fn max_of_goal_is_zero() {
        let h = MaxHeuristic::new(ManhattanHeuristic, ManhattanHeuristic);
        assert_eq!(h.h(&GOAL), 0);
    }

    #[test]
    fn max_picks_larger_of_two() {
        struct Zero;
        impl Heuristic for Zero {
            fn h(&self, _: &State) -> u8 {
                0
            }
        }
        // After one move, Manhattan > 0; max(Manhattan, Zero) == Manhattan.
        let s = GOAL.apply(Move::Up);
        let h = MaxHeuristic::new(ManhattanHeuristic, Zero);
        assert_eq!(h.h(&s), ManhattanHeuristic.h(&s));
        assert!(h.h(&s) > 0);
    }

    #[test]
    fn max_accepts_borrowed_heuristics() {
        // The blanket `&H: Heuristic` impl lets us pass references through.
        let a = ManhattanHeuristic;
        let b = ManhattanHeuristic;
        let h = MaxHeuristic::new(&a, &b);
        assert_eq!(h.h(&GOAL), 0);
    }
}
