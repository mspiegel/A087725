//! The 8-puzzle's only nontrivial goal-preserving symmetry: reflection through
//! the main diagonal (the one passing through the goal's blank corner).
//!
//! Acting on the board, the position permutation is `σ: (r, c) → (c, r)`. To
//! preserve the goal `1 2 3 / 4 5 6 / 7 8 _`, this must be paired with a tile
//! relabeling `τ` that sends tile `k` to whichever tile sits at `σ(position of
//! k)` in the goal. Concretely:
//!
//! ```text
//! τ:  0 ↔ 0    (blank is fixed)
//!     1 ↔ 1, 5 ↔ 5    (fixed by σ)
//!     2 ↔ 4
//!     3 ↔ 7
//!     6 ↔ 8
//! ```
//!
//! Both σ and τ are involutions. The reflection of a state `s` is
//! `reflect(s)[p] = τ(s[σ(p)])`, since `σ⁻¹ = σ`.
//!
//! For any state `s`, `dist(s) == dist(reflect(s))` — the symmetry preserves
//! distance to the goal, so we can quotient the state space by it (factor-of-2
//! storage reduction, compression direction #1).

use crate::puzzle8::state::State;

/// Position permutation: `SIGMA[p]` is the position that ends up at index `p`
/// in the reflected state (equivalently, σ⁻¹(p), since σ is self-inverse).
const SIGMA: [u8; 9] = [0, 3, 6, 1, 4, 7, 2, 5, 8];

/// Tile relabeling: `TAU[t]` is the new label for tile `t` under the
/// reflection.
const TAU: [u8; 9] = [0, 1, 4, 7, 2, 5, 8, 3, 6];

/// Reflect `s` through the main diagonal (the one passing through the goal's
/// blank corner).
pub fn reflect(s: &State) -> State {
    let mut out = [0u8; 9];
    for p in 0..9 {
        let src = SIGMA[p] as usize;
        out[p] = TAU[s.0[src] as usize];
    }
    State(out)
}

/// Canonical representative of `s` under the diagonal symmetry. Returns the
/// lex-smaller of `(s, reflect(s))`, plus a flag indicating which was chosen.
pub fn canonical(s: &State) -> (State, bool) {
    let r = reflect(s);
    if r.0 < s.0 {
        (r, true)
    } else {
        (*s, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::bfs::DistanceTable;
    use crate::puzzle8::rank::unrank;
    use crate::puzzle8::state::{GOAL, N_STATES};

    #[test]
    fn sigma_is_self_inverse() {
        for p in 0..9 {
            assert_eq!(SIGMA[SIGMA[p as usize] as usize], p);
        }
    }

    #[test]
    fn tau_is_self_inverse() {
        for t in 0..9 {
            assert_eq!(TAU[TAU[t as usize] as usize], t);
        }
    }

    #[test]
    fn tau_is_a_permutation_of_0_to_8() {
        let mut seen = [false; 9];
        for &t in &TAU {
            assert!(!seen[t as usize], "TAU repeats {}", t);
            seen[t as usize] = true;
        }
    }

    #[test]
    fn reflect_fixes_goal() {
        assert_eq!(reflect(&GOAL), GOAL);
    }

    #[test]
    fn reflect_is_involution_on_full_state_space() {
        for r in 0..N_STATES {
            let s = unrank(r);
            let r2 = reflect(&reflect(&s));
            assert_eq!(r2, s, "reflect²({:?}) = {:?}", s.0, r2.0);
        }
    }

    #[test]
    fn reflect_preserves_solvability() {
        for r in 0..N_STATES {
            let s = unrank(r);
            assert!(s.is_solvable());
            assert!(reflect(&s).is_solvable(), "reflect made unsolvable: {:?}", s.0);
        }
    }

    #[test]
    fn reflect_preserves_distance() {
        let t = DistanceTable::build();
        for r in 0..N_STATES {
            let s = unrank(r);
            let rs = reflect(&s);
            assert_eq!(
                t.dist(&s),
                t.dist(&rs),
                "dist({:?}) = {} vs dist({:?}) = {}",
                s.0,
                t.dist(&s),
                rs.0,
                t.dist(&rs),
            );
        }
    }

    #[test]
    fn antipodes_are_each_others_reflection() {
        // The two known antipodes are diagonal reflections of each other.
        let a1 = State([8, 6, 7, 2, 5, 4, 3, 0, 1]);
        let a2 = State([6, 4, 7, 8, 5, 0, 3, 2, 1]);
        assert_eq!(reflect(&a1), a2);
        assert_eq!(reflect(&a2), a1);
    }

    #[test]
    fn canonical_is_idempotent_on_full_state_space() {
        for r in 0..N_STATES {
            let s = unrank(r);
            let (c1, _) = canonical(&s);
            let (c2, _) = canonical(&c1);
            assert_eq!(c1, c2, "canonical not idempotent on {:?}", s.0);
        }
    }

    #[test]
    fn canonical_picks_lex_smaller() {
        for r in 0..N_STATES {
            let s = unrank(r);
            let (c, _) = canonical(&s);
            let r = reflect(&s);
            assert!(c.0 <= s.0);
            assert!(c.0 <= r.0);
            assert!(c == s || c == r);
        }
    }

    #[test]
    fn canonical_orbit_count_is_close_to_half() {
        // The orbit count is (N + |Fix(σ)|) / 2 by Burnside. A small number
        // of states are self-symmetric (lie on the diagonal of fixed
        // configurations). Verify the count is in the expected range.
        use std::collections::HashSet;
        let mut canonicals: HashSet<[u8; 9]> = HashSet::with_capacity(N_STATES as usize / 2);
        for r in 0..N_STATES {
            let (c, _) = canonical(&unrank(r));
            canonicals.insert(c.0);
        }
        let n = canonicals.len() as u32;
        // Orbits are either size 1 (self-symmetric) or size 2. So
        // 2*n - (self-symmetric count) == N_STATES. n must be between
        // N/2 and N/2 + small.
        assert!(n >= N_STATES / 2);
        assert!(n <= N_STATES / 2 + 1000, "too many orbits: {}", n);
    }
}
