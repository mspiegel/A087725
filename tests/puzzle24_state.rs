//! Phase 1 verification for the 24-puzzle scaffold (`src/puzzle24/`).
//!
//! Covers the four checks called for in the plan: goal state, odd-width
//! solvability parity, a hand-traced 5-move sequence, and reflection
//! involution. Run with `cargo test --test puzzle24_state`.

use puzzle8::puzzle24::rank::{rank, unrank};
use puzzle8::puzzle24::state::{Move, State, GOAL, N_CELLS, N_STATES, W};
use puzzle8::puzzle24::symmetry::{reflect, transpose_move};

/// Goal layout: tile `i` at position `i-1`, blank (0) at the bottom-right.
#[test]
fn goal_state_is_correct() {
    for i in 0..(N_CELLS - 1) {
        assert_eq!(GOAL.0[i], (i + 1) as u8, "goal tile at {} should be {}", i, i + 1);
    }
    assert_eq!(GOAL.0[N_CELLS - 1], 0, "blank should be last");
    assert_eq!(GOAL.blank_pos() as usize, N_CELLS - 1);
    assert!(GOAL.is_solvable());
    assert_eq!(GOAL.inversions(), 0);
}

/// Odd-width (5×5) solvability rule: a state is solvable iff its inversion count
/// is even, **independent of the blank's position**.
#[test]
fn solvability_parity_is_odd_width_rule() {
    // A single adjacent transposition of two tiles → 1 inversion → unsolvable.
    let mut one_swap = GOAL.0;
    one_swap.swap(0, 1);
    assert!(!State(one_swap).is_solvable());
    assert_eq!(State(one_swap).inversions() % 2, 1);

    // A 3-cycle of tiles is an even permutation → solvable.
    let mut three_cycle = GOAL.0;
    let tmp = three_cycle[0];
    three_cycle[0] = three_cycle[1];
    three_cycle[1] = three_cycle[2];
    three_cycle[2] = tmp;
    assert!(State(three_cycle).is_solvable());

    // Blank-independence: sliding only the blank never changes solvability,
    // wherever the blank ends up. Walk a long well-mixed path (xorshift64, so
    // the blank actually wanders rather than oscillating in place).
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = GOAL;
    let mut blanks_seen = [false; N_CELLS];
    for i in 0u32..6000 {
        assert!(s.is_solvable(), "blank slide broke solvability at step {}", i);
        blanks_seen[s.blank_pos() as usize] = true;
        let moves = s.legal_moves();
        let opts: Vec<Move> = moves.iter().collect();
        let m = opts[(next() as usize) % opts.len()];
        s = s.apply(m);
    }
    // The walk should have driven the blank through every cell — confirms the
    // invariant was actually exercised across all blank positions.
    assert!(blanks_seen.iter().all(|&b| b), "walk did not reach every blank cell");
}

/// Hand-traced 5-move sequence from GOAL: `Up, Left, Down, Right, Up`.
///
/// Board cells (row-major):
/// ```text
///  0  1  2  3  4
///  5  6  7  8  9
/// 10 11 12 13 14
/// 15 16 17 18 19
/// 20 21 22 23 24
/// ```
/// Blank starts at 24. Trace (blank cell → cell):
/// 1. Up:    24→19, swap tile 20 down   (pos19=0, pos24=20)
/// 2. Left:  19→18, swap tile 19 right  (pos18=0, pos19=19)
/// 3. Down:  18→23, swap tile 24 up     (pos18=24, pos23=0)
/// 4. Right: 23→24, swap tile 20 left   (pos23=20, pos24=0)
/// 5. Up:    24→19, swap tile 19 down   (pos19=0, pos24=19)
#[test]
fn hand_traced_five_move_sequence() {
    let seq = [Move::Up, Move::Left, Move::Down, Move::Right, Move::Up];

    let mut s = GOAL;
    let mut blank = s.blank_pos();
    for m in seq {
        assert!(State::legal_moves_at(blank).contains(m), "illegal {:?} at blank {}", m, blank);
        let (next, nb) = s.apply_at(m, blank);
        s = next;
        blank = nb;
    }

    // Hand-derived final board.
    let expected = State([
        1, 2, 3, 4, 5, //
        6, 7, 8, 9, 10, //
        11, 12, 13, 14, 15, //
        16, 17, 18, 24, 0, //
        21, 22, 23, 20, 19,
    ]);
    assert_eq!(s, expected, "5-move trace produced unexpected board");
    assert_eq!(blank, 19, "blank should end at cell 19");
    assert!(s.is_solvable());

    // Independent cross-check: undoing the sequence in reverse restores GOAL.
    let mut back = s;
    for m in seq.iter().rev() {
        back = back.apply(m.inverse());
    }
    assert_eq!(back, GOAL, "reverse-inverse of the trace must restore GOAL");

    // And the displaced state round-trips through the rank bijection.
    assert_eq!(unrank(rank(&s)), s);
    assert!(rank(&s) < N_STATES);
}

/// Reflection through the main (blank-corner) diagonal is an involution that
/// fixes the goal, preserves solvability, and commutes with moves via the
/// transposed move.
#[test]
fn reflection_involution() {
    assert_eq!(reflect(&GOAL), GOAL, "reflection must fix the goal");

    let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
    let mut s = GOAL;
    for i in 0u32..3000 {
        // reflect ∘ reflect == identity.
        assert_eq!(reflect(&reflect(&s)), s, "reflect² broke at step {}", i);
        // Distance-preserving symmetry keeps states solvable.
        assert_eq!(reflect(&s).is_solvable(), s.is_solvable());
        // reflect(s.apply(m)) == reflect(s).apply(transpose_move(m)).
        for m in s.legal_moves().iter() {
            assert_eq!(
                reflect(&s.apply(m)),
                reflect(&s).apply(transpose_move(m)),
                "reflection/move commutation broke for {:?}",
                m
            );
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

/// Sanity that the diagonal axis really is the 5×5 transpose (catches a wrong-W
/// port of the symmetry tables).
#[test]
fn reflection_is_board_transpose() {
    // Place a distinguishable state and check reflect maps position p's value
    // through the transpose σ(r,c)=(c,r) plus the tile relabel τ. We verify the
    // weaker, table-free property: reflecting GOAL after one Up equals applying
    // Left to reflected GOAL (= GOAL), per transpose_move(Up)=Left.
    let s = GOAL.apply(Move::Up);
    assert_eq!(reflect(&s), GOAL.apply(transpose_move(Move::Up)));
    assert_eq!(W, 5);
}
