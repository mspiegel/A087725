//! The 24-puzzle's goal-preserving diagonal symmetry: reflection through the
//! main diagonal (the one passing through the goal's blank corner).
//!
//! Acting on the board, the position permutation is `σ: (r, c) → (c, r)`. To
//! preserve the goal layout (tile `i` at position `i-1`, blank at position 24),
//! this must be paired with a tile relabeling `τ` that sends tile `k` to the
//! tile sitting at `σ(position of k)` in the goal. Both σ and τ are
//! involutions; the reflection of a state `s` is
//! `reflect(s)[p] = τ(s[σ(p)])`, since `σ⁻¹ = σ`.
//!
//! For any state `s`, `dist(s) == dist(reflect(s))` — the symmetry preserves
//! distance to the goal, so it gives a factor-of-2 storage reduction. We use it
//! via *reflected PDBs* in the Korf-max heuristic: `h(s) = max(h(s),
//! h(reflect(s)))`. Both tables (`SIGMA`, `TAU`) are derived at compile time
//! from the board geometry to avoid hand-transcription errors at 5×5 scale.

use crate::puzzle24::state::{Move, State, N_CELLS, W};

/// Position permutation: `SIGMA[p]` is the position whose value lands at index
/// `p` after reflection. For `p = r*W + c`, the transpose gives
/// `σ(p) = c*W + r`. σ is self-inverse.
const SIGMA: [u8; N_CELLS] = {
    let mut t = [0u8; N_CELLS];
    let mut p = 0;
    while p < N_CELLS {
        let r = p / W;
        let c = p % W;
        t[p] = (c * W + r) as u8;
        p += 1;
    }
    t
};

/// Tile relabeling: `TAU[t]` is the new label for tile `t` under the reflection.
/// Derived so goal tiles map onto their σ-images: tile `k` (at goal position
/// `k-1`) is renamed to whichever tile sits at position `σ(k-1)` in the goal.
/// `GOAL[i] = i+1` for `i < 24` and `GOAL[24] = 0`, so the goal value at
/// position `q` is `0` when `q = 24` and `q+1` otherwise. The blank (`t = 0`)
/// is fixed because `σ(24) = 24`.
const TAU: [u8; N_CELLS] = {
    let mut t = [0u8; N_CELLS];
    let mut k = 1;
    while k < N_CELLS {
        let sigma_img = SIGMA[k - 1] as usize; // σ(goal position of tile k)
        t[k] = if sigma_img == N_CELLS - 1 { 0 } else { (sigma_img + 1) as u8 };
        k += 1;
    }
    t
};

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
/// The reflection is the transpose `σ: (r, c) → (c, r)`, swapping the row and
/// column axes, so a vertical blank move becomes horizontal and vice versa:
/// `Up ↔ Left`, `Down ↔ Right`. For every `s` and legal `m`,
/// `reflect(s.apply(m)) == reflect(s).apply(transpose_move(m))`, letting a
/// search maintain the reflected view incrementally. Self-inverse.
pub fn transpose_move(m: Move) -> Move {
    match m {
        Move::Up => Move::Left,
        Move::Left => Move::Up,
        Move::Down => Move::Right,
        Move::Right => Move::Down,
    }
}

/// True iff `s` is a fixpoint of the diagonal reflection — i.e. `reflect(s) == s`.
///
/// A σ-fixed state has σ as an automorphism of its search tree (`σ` preserves
/// distance-to-goal and commutes with moves), so its root children fall into
/// σ-orbits and a lower-bound proof may search one representative per orbit. The
/// blank of any σ-fixed state necessarily sits on the main diagonal (the only
/// positions fixed by `SIGMA`), where the legal-move set is closed under
/// [`transpose_move`]. Board `R` is such a fixpoint.
pub fn is_symmetric(s: &State) -> bool {
    reflect(s) == *s
}

/// Whether `m` is the canonical member of its σ-orbit `{m, transpose_move(m)}`:
/// the one with the smaller move index (`Up < Down < Left < Right`, matching
/// `move_dfa::mcode`). `transpose_move` pairs `Up↔Left` and `Down↔Right` and has
/// no fixpoint, so exactly one move of each pair is a representative — the reps
/// are `{Up, Down}`. Used to keep one root child per orbit on a σ-fixed board.
pub fn is_orbit_representative(m: Move) -> bool {
    (m as u8) <= (transpose_move(m) as u8)
}

/// Canonical representative of `s` under the diagonal symmetry: the lex-smaller
/// of `(s, reflect(s))`, plus a flag indicating whether the reflection was
/// chosen.
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
    use crate::puzzle24::state::{Move, GOAL};

    #[test]
    fn sigma_is_self_inverse() {
        for p in 0..N_CELLS {
            assert_eq!(SIGMA[SIGMA[p] as usize], p as u8);
        }
    }

    #[test]
    fn sigma_matches_transpose() {
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
    fn tau_is_a_permutation() {
        let mut seen = [false; N_CELLS];
        for &t in &TAU {
            assert!(!seen[t as usize], "TAU repeats {t}");
            seen[t as usize] = true;
        }
    }

    #[test]
    fn tau_derived_from_sigma_and_goal() {
        // τ(k) must equal the goal tile at σ(goal_pos_of(k)) = goal[σ(k-1)].
        for k in 1u8..=24 {
            let sigma_image = SIGMA[(k - 1) as usize] as usize;
            let goal_tile_at_sigma = GOAL.0[sigma_image];
            assert_eq!(TAU[k as usize], goal_tile_at_sigma, "τ({k}) inconsistent");
        }
    }

    #[test]
    fn reflect_fixes_goal() {
        assert_eq!(reflect(&GOAL), GOAL);
    }

    #[test]
    fn reflect_is_involution_on_random_walks() {
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
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
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
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
    fn transpose_move_is_self_inverse() {
        for m in Move::ALL {
            assert_eq!(transpose_move(transpose_move(m)), m);
        }
    }

    #[test]
    fn transpose_move_commutes_reflection_on_random_walks() {
        // reflect(s.apply(m)) == reflect(s).apply(transpose_move(m)).
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        for i in 0u32..2000 {
            for m in s.legal_moves().iter() {
                let lhs = reflect(&s.apply(m));
                let rhs = reflect(&s).apply(transpose_move(m));
                assert_eq!(lhs, rhs, "transpose_move broke commutation for {m:?}");
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
    fn canonical_is_idempotent_and_lex_smaller() {
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        for i in 0u32..1000 {
            let (c1, _) = canonical(&s);
            let (c2, _) = canonical(&c1);
            assert_eq!(c1, c2);
            let r = reflect(&s);
            assert!(c1.0 <= s.0 && c1.0 <= r.0);
            assert!(c1 == s || c1 == r);
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }

    /// Board R: blank at cell 0, tile `25-i` at cell `i` (the 180° rotation).
    fn r_board() -> State {
        let mut c = [0u8; N_CELLS];
        for i in 1..N_CELLS {
            c[i] = (25 - i) as u8;
        }
        State(c)
    }

    #[test]
    fn orbit_reps_partition_moves() {
        // transpose_move pairs Up↔Left and Down↔Right (no fixpoint); the reps are
        // exactly {Up, Down}.
        assert!(is_orbit_representative(Move::Up));
        assert!(is_orbit_representative(Move::Down));
        assert!(!is_orbit_representative(Move::Left));
        assert!(!is_orbit_representative(Move::Right));
        // For every move, exactly one of {m, transpose_move(m)} is a representative.
        for m in Move::ALL {
            let t = transpose_move(m);
            assert_ne!(m, t, "transpose_move must have no fixpoint");
            assert!(
                is_orbit_representative(m) ^ is_orbit_representative(t),
                "exactly one of {{{m:?}, {t:?}}} is a representative"
            );
        }
        // R's corner orbit is {Down, Right}; Down is the kept representative.
        assert_eq!(transpose_move(Move::Down), Move::Right);
        assert!(is_orbit_representative(Move::Down));
        assert!(!is_orbit_representative(Move::Right));
    }

    #[test]
    fn is_symmetric_detects_sigma_fixpoints() {
        assert!(is_symmetric(&GOAL), "GOAL is σ-fixed");
        assert!(is_symmetric(&r_board()), "R is σ-fixed");
        // One move off GOAL breaks the symmetry (blank leaves the diagonal).
        let off = GOAL.apply(Move::Up);
        assert!(!is_symmetric(&off), "a single move off GOAL is not σ-fixed");
    }

    #[test]
    fn orbit_split_keeps_one_rep_per_legal_orbit() {
        // R (blank in corner 0): legal moves {Down, Right} form one orbit -> keep 1.
        let r = r_board();
        let r_kept: Vec<Move> = r
            .legal_moves()
            .iter()
            .filter(|&m| is_orbit_representative(m))
            .collect();
        assert_eq!(r_kept, vec![Move::Down], "R keeps 1 of 2 root children");

        // GOAL (blank in corner 24): legal moves {Up, Left} form one orbit -> keep 1.
        let goal_kept = GOAL
            .legal_moves()
            .iter()
            .filter(|&m| is_orbit_representative(m))
            .count();
        assert_eq!(goal_kept, 1usize, "GOAL keeps 1 of 2 root children");

        // Interior σ-fixed board (blank on the main diagonal at cell 18): 4 legal
        // moves {U,D,L,R} form 2 orbits -> keep 2. Fixture found by BFS from GOAL.
        let interior = parse_test_board(
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 _ 24 21 22 23 20 19",
        );
        assert!(is_symmetric(&interior), "fixture must be σ-symmetric");
        assert_eq!(interior.blank_pos(), 18, "fixture blank on the diagonal");
        let interior_kept = interior
            .legal_moves()
            .iter()
            .filter(|&m| is_orbit_representative(m))
            .count();
        assert_eq!(interior_kept, 2, "interior diagonal board keeps 2 of 4");
    }

    /// Parse a 25-token row-major board (`_` = blank) for test fixtures.
    fn parse_test_board(s: &str) -> State {
        let mut cells = [0u8; N_CELLS];
        for (i, tok) in s.split_whitespace().enumerate() {
            cells[i] = if tok == "_" { 0 } else { tok.parse().unwrap() };
        }
        State(cells)
    }
}
