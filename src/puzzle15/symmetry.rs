//! The 15-puzzle's goal-preserving diagonal symmetry: reflection through the
//! main diagonal (the one passing through the goal's blank corner).
//!
//! Acting on the board, the position permutation is `σ: (r, c) → (c, r)`. To
//! preserve the goal layout (tile `i` at position `i-1`, blank at position 15),
//! this must be paired with a tile relabeling `τ` that sends tile `k` to the
//! tile sitting at `σ(position of k)` in the goal. Both σ and τ are
//! involutions; the reflection of a state `s` is
//! `reflect(s)[p] = τ(s[σ(p)])`, since `σ⁻¹ = σ`.
//!
//! For any state `s`, `dist(s) == dist(reflect(s))` — the symmetry preserves
//! distance to the goal, so quotienting by it gives factor-of-2 storage
//! reduction (compression direction #1). We use it indirectly via *reflected
//! PDBs* in the M3 Korf heuristic: `h(s) = max(h_normal(s), h_reflected(s))`.

use crate::puzzle15::state::{Move, State, N_CELLS};

/// Position permutation: `SIGMA[p]` is the position whose value lands at index
/// `p` after reflection. Equivalently, σ⁻¹(p), but σ is self-inverse so this
/// also equals σ(p).
///
/// Derivation: for position `p = r*W + c` (row-major), the transpose sends
/// `(r,c) → (c,r)`, giving `σ(p) = c*W + r`.
const SIGMA: [u8; N_CELLS] = [
    0, 4, 8, 12,
    1, 5, 9, 13,
    2, 6, 10, 14,
    3, 7, 11, 15,
];

/// Tile relabeling: `TAU[t]` is the new label for tile `t` under the
/// reflection. Derived so that goal tiles are permuted onto their σ-images:
/// tile `k` (at goal position `k-1`) is renamed to whichever tile sits at
/// position `σ(k-1)` in the goal.
const TAU: [u8; N_CELLS] = [
    0,           // blank fixed
    1,           // diagonal tile (goal pos 0 → 0)
    5,  9,  13,  // tile 2 → 5, 3 → 9, 4 → 13
    2,           // 5 → 2
    6,           // diagonal (goal pos 5 → 5)
    10, 14,      // 7 → 10, 8 → 14
    3,  7,       // 9 → 3, 10 → 7
    11,          // diagonal (goal pos 10 → 10)
    15,          // 12 → 15
    4,  8,  12,  // 13 → 4, 14 → 8, 15 → 12
];

/// Reflect `s` through the main diagonal (the one passing through the goal's
/// blank corner).
pub fn reflect(s: &State) -> State {
    let mut out = [0u8; N_CELLS];
    for p in 0..N_CELLS {
        let src = SIGMA[p] as usize;
        out[p] = TAU[s.0[src] as usize];
    }
    State(out)
}

/// The blank move on `reflect(s)` corresponding to move `m` on `s`.
///
/// The reflection is the transpose `σ: (r, c) → (c, r)`, which swaps the row and
/// column axes, so a vertical blank move becomes horizontal and vice versa:
/// `Up ↔ Left`, `Down ↔ Right`. Formally, for every `s` and every legal `m`,
/// `reflect(s.apply(m)) == reflect(s).apply(transpose_move(m))`. This lets a
/// search maintain the reflected view incrementally without recomputing
/// [`reflect`] at each node. The map is its own inverse.
pub fn transpose_move(m: Move) -> Move {
    match m {
        Move::Up => Move::Left,
        Move::Left => Move::Up,
        Move::Down => Move::Right,
        Move::Right => Move::Down,
    }
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
    use crate::puzzle15::state::{GOAL, Move, W};

    #[test]
    fn sigma_is_self_inverse() {
        for p in 0..N_CELLS {
            assert_eq!(SIGMA[SIGMA[p] as usize], p as u8);
        }
    }

    #[test]
    fn sigma_matches_transpose() {
        // SIGMA[p] should equal (c, r) viewed in row-major, where p = (r, c).
        for p in 0..N_CELLS {
            let r = p / W;
            let c = p % W;
            assert_eq!(SIGMA[p] as usize, c * W + r);
        }
    }

    #[test]
    fn tau_is_self_inverse() {
        for t in 0..N_CELLS {
            assert_eq!(TAU[TAU[t] as usize], t as u8);
        }
    }

    #[test]
    fn tau_is_a_permutation_of_0_to_15() {
        let mut seen = [false; N_CELLS];
        for &t in &TAU {
            assert!(!seen[t as usize], "TAU repeats {t}");
            seen[t as usize] = true;
        }
    }

    #[test]
    fn tau_derived_from_sigma_and_goal() {
        // For each tile k (1..=15), τ(k) must equal the tile sitting at
        // σ(goal_pos_of(k)) in the goal — i.e., goal[σ(k-1)].
        // GOAL[i] is i+1 for i in 0..15; GOAL[15] is 0. So for k in 1..=15,
        // its goal position is k-1, and goal[σ(k-1)] should equal τ(k).
        for k in 1u8..=15 {
            let goal_pos = (k - 1) as usize;
            let sigma_image = SIGMA[goal_pos] as usize;
            let goal_tile_at_sigma = GOAL.0[sigma_image];
            assert_eq!(
                TAU[k as usize], goal_tile_at_sigma,
                "τ({k}) inconsistent with σ image of its goal position"
            );
        }
    }

    #[test]
    fn reflect_fixes_goal() {
        assert_eq!(reflect(&GOAL), GOAL);
    }

    #[test]
    fn reflect_is_involution_on_random_walks() {
        // Walk pseudo-randomly from GOAL and verify reflect² = id at each step.
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        let mut s = GOAL;
        for i in 0u32..2000 {
            let r2 = reflect(&reflect(&s));
            assert_eq!(r2, s, "reflect² broken at step {}: {:?}", i, s.0);
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    #[test]
    fn reflect_preserves_solvability() {
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        let mut s = GOAL;
        for i in 0u32..2000 {
            assert!(s.is_solvable());
            assert!(reflect(&s).is_solvable(), "reflect made unsolvable: {:?}", s.0);
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    #[test]
    fn canonical_is_idempotent() {
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        let mut s = GOAL;
        for i in 0u32..1000 {
            let (c1, _) = canonical(&s);
            let (c2, _) = canonical(&c1);
            assert_eq!(c1, c2);
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    #[test]
    fn canonical_picks_lex_smaller() {
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        let mut s = GOAL;
        for i in 0u32..1000 {
            let (c, _) = canonical(&s);
            let r = reflect(&s);
            assert!(c.0 <= s.0);
            assert!(c.0 <= r.0);
            assert!(c == s || c == r);
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    #[test]
    fn transpose_move_is_self_inverse() {
        for m in Move::ALL {
            assert_eq!(transpose_move(transpose_move(m)), m);
        }
    }

    #[test]
    fn transpose_move_commutes_reflection_on_random_walks() {
        // The defining identity: reflect(s.apply(m)) == reflect(s).apply(tm)
        // for every legal move m along a pseudo-random walk.
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        let mut s = GOAL;
        for i in 0u32..2000 {
            for m in s.legal_moves().iter() {
                let lhs = reflect(&s.apply(m));
                let rhs = reflect(&s).apply(transpose_move(m));
                assert_eq!(lhs, rhs, "transpose_move broke commutation for {:?} at {:?}", m, s.0);
            }
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    #[test]
    fn reflect_commutes_with_a_known_move_pair() {
        // Reflection swaps row/col, hence Move::Up of the blank in the original
        // corresponds to Move::Left in the reflected board (the blank moves
        // along the transposed axis). Verify: reflect(apply(Up, s)) ==
        // apply(Left, reflect(s)), when Up is legal in s and Left is legal in
        // reflect(s).
        // Construct a non-trivial position by moving the blank up once from
        // GOAL, then verify the commuter on the reflected.
        let s = GOAL.apply(Move::Up); // blank now at pos 11
        let lhs = reflect(&s.apply(Move::Up));
        let rs = reflect(&s);
        // After reflecting, the blank that was at 11 is now at σ(11) = 14, an
        // edge cell. Move::Left of the blank moves it from col 2 to col 1, so
        // the rhs side applies Left to the reflected.
        let rhs = rs.apply(Move::Left);
        assert_eq!(lhs, rhs);
    }
}
