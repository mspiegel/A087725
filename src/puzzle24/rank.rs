//! Bijection between solvable [`State`]s and integers in `[0, N_STATES)`.
//!
//! This is the generic permutation-ranking scaffold for the 24-puzzle. The full
//! state space (`25!/2 ≈ 7.76 × 10²⁴`) is far too large to tabulate — we never
//! build a distance table over it — but the Lehmer rank/unrank below is the
//! shared primitive the PDB `(m, p, r)` index (Phase 2) is built from, and it
//! gives exact round-trip tests for the state machinery now.
//!
//! # Construction
//!
//! ```text
//! rank = blank_pos * EVEN_BLOCK + even_rank
//! ```
//! where `blank_pos ∈ [0, 25)` and `even_rank ∈ [0, 24!/2)` ranks the 24-tile
//! permutation among the `24!/2` permutations of the **even** parity class.
//! Total: `25 · 24!/2 = 25!/2` distinct ranks, one per solvable state. Values
//! exceed `u64`, so the rank is a `u128`.
//!
//! ## Odd-width solvability
//!
//! Unlike the even-width 15-puzzle (whose parity couples to the blank's row),
//! the 24-puzzle has **odd width 5**: a state is solvable iff its permutation is
//! even, *independent of the blank's position* (Johnson 1879). So for every
//! `blank_pos` the reachable half of the `24!` permutations is the same — the
//! even class — and `required_parity` is always `0`.
//!
//! ## Lehmer encoding
//!
//! For a permutation `π` of `{1..=24}`, compute Lehmer digits `(d_0, …, d_23)`
//! with `d_i ∈ [0, 24-i)`. Always `d_23 = 0`, and `parity(π) = (d_0+…+d_22) mod
//! 2`. For a permutation of known parity, `d_0..d_21` determine `d_22`. We pack
//! `d_0..d_21` mixed-radix with bases `(24, 23, …, 3)`, giving
//! `24·23·…·3 = 24!/2 = EVEN_BLOCK` distinct values.

use crate::puzzle24::state::{State, N_CELLS, N_STATES, N_TILES};

/// `24! / 2`. The number of permutations of `{1..=24}` of the even parity class
/// (the solvable class, for every blank position on the odd-width board).
const EVEN_BLOCK: u128 = {
    let mut f: u128 = 1;
    let mut k: u128 = 2;
    while k <= 24 {
        f *= k;
        k += 1;
    }
    f / 2
};

/// Map a solvable [`State`] to a unique integer in `[0, N_STATES)`.
///
/// Debug-asserts that `s` is solvable.
pub fn rank(s: &State) -> u128 {
    debug_assert!(s.is_solvable(), "rank() called on unsolvable state: {:?}", s.0);

    let blank_pos = s.blank_pos() as u128;

    // Collect the 24 non-blank tiles in order of position.
    let mut tiles = [0u8; N_TILES];
    let mut idx = 0;
    for &v in &s.0 {
        if v != 0 {
            tiles[idx] = v;
            idx += 1;
        }
    }

    // Compute Lehmer digits d_0..d_21. `available` is a bitmask over values
    // 1..=24; bit `v` set means tile value `v` has not been placed yet.
    let mut available: u32 = ((1u32 << N_CELLS) - 1) & !1; // bits 1..=24
    let mut even_rank: u128 = 0;
    for i in 0..(N_TILES - 2) {
        let tile = tiles[i];
        let lo_mask = (1u32 << tile) - 1;
        let d = (available & lo_mask).count_ones() as u128;
        let radix = (N_TILES - i) as u128; // 24, 23, …, 3
        even_rank = even_rank * radix + d;
        available &= !(1u32 << tile);
    }

    blank_pos * EVEN_BLOCK + even_rank
}

/// Inverse of [`rank`]: reconstruct the state from its rank.
///
/// Panics if `r >= N_STATES`.
pub fn unrank(r: u128) -> State {
    assert!(r < N_STATES, "rank {r} out of range [0, {N_STATES})");

    let blank_pos = (r / EVEN_BLOCK) as usize;
    let even_rank = r % EVEN_BLOCK;

    // Recover d_0..d_21 by repeated division (least-significant first).
    let mut digits = [0u8; N_TILES]; // d_0..d_23
    let mut r = even_rank;
    for i in (0..(N_TILES - 2)).rev() {
        let radix = (N_TILES - i) as u128; // 24, 23, …, 3
        digits[i] = (r % radix) as u8;
        r /= radix;
    }

    // d_22 is forced by the solvability parity rule. On the odd-width board the
    // required permutation parity is 0 (even) for every blank position:
    //   sum(d_0..d_22) ≡ 0 (mod 2)  ⇒  d_22 = sum(d_0..d_21) mod 2.
    // d_23 = 0 (only one value remains at that step).
    let parity_so_far: u32 = digits[..(N_TILES - 2)].iter().map(|&x| x as u32).sum();
    digits[N_TILES - 2] = (parity_so_far & 1) as u8;
    // digits[N_TILES - 1] stays 0.

    // Reconstruct the 24-permutation from the Lehmer digits.
    let mut available = [0u8; N_TILES];
    for (i, slot) in available.iter_mut().enumerate() {
        *slot = (i + 1) as u8;
    }
    let mut n_available = N_TILES;
    let mut tiles = [0u8; N_TILES];
    for i in 0..N_TILES {
        let d = digits[i] as usize;
        debug_assert!(d < n_available);
        tiles[i] = available[d];
        for k in d..(n_available - 1) {
            available[k] = available[k + 1];
        }
        n_available -= 1;
    }

    // Splice the 24-permutation into a 25-state with the blank at `blank_pos`.
    let mut cells = [0u8; N_CELLS];
    let mut idx = 0;
    for pos in 0..N_CELLS {
        if pos == blank_pos {
            cells[pos] = 0;
        } else {
            cells[pos] = tiles[idx];
            idx += 1;
        }
    }

    State(cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::{Move, GOAL};

    #[test]
    fn even_block_value() {
        let mut f: u128 = 1;
        for k in 2..=24u128 {
            f *= k;
        }
        assert_eq!(EVEN_BLOCK, f / 2);
        // 25 blocks of EVEN_BLOCK must equal the full solvable count.
        assert_eq!(25 * EVEN_BLOCK, N_STATES);
    }

    #[test]
    fn goal_roundtrips() {
        let r = rank(&GOAL);
        assert_eq!(unrank(r), GOAL, "unrank(rank(GOAL)) must equal GOAL");
    }

    #[test]
    fn goal_rank_is_known() {
        // GOAL: blank at position 24, identity permutation (even). even_rank = 0.
        assert_eq!(rank(&GOAL), 24 * EVEN_BLOCK);
    }

    #[test]
    fn rank_zero_roundtrips() {
        let s = unrank(0);
        assert!(s.is_solvable());
        assert_eq!(rank(&s), 0);
    }

    #[test]
    fn rank_max_roundtrips() {
        let r = N_STATES - 1;
        let s = unrank(r);
        assert!(s.is_solvable());
        assert_eq!(rank(&s), r);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn unrank_panics_on_oversize() {
        let _ = unrank(N_STATES);
    }

    #[test]
    fn random_walk_states_roundtrip() {
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        for i in 0u32..1000 {
            assert!(s.is_solvable());
            let r = rank(&s);
            assert!(r < N_STATES);
            assert_eq!(unrank(r), s, "round-trip failed at step {}: {:?}", i, s.0);
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
    fn blank_position_recovered_from_rank() {
        for bp in 0..N_CELLS {
            let r = (bp as u128) * EVEN_BLOCK;
            let s = unrank(r);
            assert_eq!(s.blank_pos() as usize, bp);
            assert_eq!(rank(&s), r);
        }
    }

    #[test]
    fn stratified_sample_distinct_states() {
        use std::collections::HashSet;
        let mut seen: HashSet<[u8; N_CELLS]> = HashSet::new();
        const PER_BLOCK: u128 = 400;
        for bp in 0u128..25 {
            let base = bp * EVEN_BLOCK;
            for k in 0u128..PER_BLOCK {
                let r = base + (k * EVEN_BLOCK / PER_BLOCK);
                let s = unrank(r);
                assert!(s.is_solvable(), "unrank({}) unsolvable: {:?}", r, s.0);
                assert_eq!(rank(&s), r, "rank/unrank mismatch at r={r}");
                assert!(seen.insert(s.0), "duplicate state at rank {r}");
            }
        }
        assert_eq!(seen.len() as u128, 25 * PER_BLOCK);
    }
}
