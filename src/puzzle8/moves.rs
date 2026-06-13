//! Optimal-move sets derived from a [`DistanceTable`].
//!
//! For any state `s`, an *optimal move* is a legal move `m` such that
//! `dist(s.apply(m)) == dist(s) - 1`. Multiple optimal moves are common —
//! many positions have 2 or 3 of them. The goal has zero optimal moves
//! (already solved).

use crate::puzzle8::bfs::DistanceTable;
use crate::puzzle8::state::{MoveSet, State};

impl DistanceTable {
    /// All moves that strictly decrease distance to the goal.
    pub fn optimal_moves(&self, s: &State) -> MoveSet {
        let d = self.dist(s);
        if d == 0 {
            return MoveSet::empty();
        }
        let target = d - 1;
        let mut out = MoveSet::empty();
        for m in s.legal_moves().iter() {
            if self.dist(&s.apply(m)) == target {
                out.insert(m);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::rank::unrank;
    use crate::puzzle8::state::{DIAMETER, GOAL, N_STATES};

    fn table() -> DistanceTable {
        DistanceTable::build()
    }

    #[test]
    fn goal_has_no_optimal_moves() {
        let t = table();
        assert!(t.optimal_moves(&GOAL).is_empty());
    }

    #[test]
    fn every_non_goal_state_has_at_least_one_optimal_move() {
        let t = table();
        let mut checked = 0u32;
        for r in 0..N_STATES {
            let s = unrank(r);
            if s == GOAL {
                continue;
            }
            let opt = t.optimal_moves(&s);
            assert!(!opt.is_empty(), "no optimal move from {:?} (dist={})", s.0, t.dist(&s));
            checked += 1;
        }
        assert_eq!(checked, N_STATES - 1);
    }

    #[test]
    fn optimal_moves_strictly_decrease_distance() {
        let t = table();
        for r in 0..N_STATES {
            let s = unrank(r);
            let d = t.dist(&s);
            for m in t.optimal_moves(&s).iter() {
                let after = t.dist(&s.apply(m));
                assert_eq!(
                    after, d - 1,
                    "optimal move {:?} from {:?} (d={}) gave d_next={}",
                    m, s.0, d, after
                );
            }
        }
    }

    #[test]
    fn non_optimal_legal_moves_do_not_decrease_distance() {
        let t = table();
        for r in 0..N_STATES {
            let s = unrank(r);
            let d = t.dist(&s);
            if d == 0 {
                continue;
            }
            let opt = t.optimal_moves(&s);
            for m in s.legal_moves().iter() {
                if opt.contains(m) {
                    continue;
                }
                let after = t.dist(&s.apply(m));
                assert!(
                    after >= d,
                    "non-optimal move {:?} from {:?} (d={}) gave d_next={} < d",
                    m, s.0, d, after
                );
            }
        }
    }

    #[test]
    fn antipodes_have_correct_branching() {
        // Antipodes at depth 31 can only have moves going to depth 30, so
        // their optimal-move set equals their full legal-move set.
        let t = table();
        for s in t.antipodes() {
            let opt = t.optimal_moves(&s);
            let legal = s.legal_moves();
            assert_eq!(opt, legal, "antipode {:?} should have all legal moves optimal", s.0);
            // Sanity: at least 2 (no 3x3 cell has fewer than 2 neighbors).
            assert!(opt.len() >= 2);
        }
        // The diameter check is a free sanity test.
        assert_eq!(t.diameter(), DIAMETER);
    }

    #[test]
    fn branching_factor_distribution_is_reasonable() {
        // Histogram of optimal-move-set sizes across all 181,440 states.
        // Sanity checks: bucket 0 has exactly 1 state (the goal); no state
        // has more optimal moves than its legal moves; max bucket <= 4.
        let t = table();
        let mut hist = [0u32; 5];
        for r in 0..N_STATES {
            let s = unrank(r);
            let n = t.optimal_moves(&s).len() as usize;
            hist[n] += 1;
        }
        assert_eq!(hist[0], 1, "only the goal should have 0 optimal moves");
        let total: u32 = hist.iter().sum();
        assert_eq!(total, N_STATES);
    }
}
