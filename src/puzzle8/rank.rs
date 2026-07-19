//! Bijection between solvable [`State`]s and integers in `[0, N_STATES)`.
//!
//! # Construction
//!
//! Each state is encoded as
//! ```text
//! rank = blank_pos * 20160 + even_rank
//! ```
//! where `blank_pos ∈ [0, 9)` and `even_rank ∈ [0, 20160)` is the rank of the
//! 8-tile permutation among the `8!/2 = 20_160` even permutations of `{1..=8}`.
//! Total: `9 * 20160 = 181_440` distinct ranks, one per solvable state.
//!
//! A state is solvable iff its non-blank tiles form an even permutation of
//! `{1..=8}` (since the 8-puzzle has odd width 3, inversions of non-blank
//! tiles are a move-invariant). So each solvable state is reachable, and each
//! `(blank_pos, even_perm)` pair corresponds to exactly one solvable state.
//!
//! ## Even-permutation rank
//!
//! For a permutation `π` of `{1..=8}`, compute its Lehmer digits
//! `(d_0, …, d_7)` with `d_i ∈ [0, 8-i)`. We have `d_7 = 0` and the parity of
//! `π` equals `(d_0 + … + d_6) mod 2`. So for even `π`, knowing `d_0..d_5`
//! determines `d_6 = (d_0 + … + d_5) mod 2`. We pack `d_0..d_5` into a
//! mixed-radix integer with bases `(8, 7, 6, 5, 4, 3)`:
//! ```text
//! even_rank = (((((d_0 * 7 + d_1) * 6 + d_2) * 5 + d_3) * 4 + d_4) * 3 + d_5)
//! ```
//! giving `8·7·6·5·4·3 = 20_160` distinct values.

use crate::puzzle8::state::{State, N_STATES};

const EVEN_BLOCK: u32 = 20_160; // 8! / 2

/// Map a solvable [`State`] to a unique integer in `[0, N_STATES)`.
///
/// Debug-asserts that `s` is solvable.
pub fn rank(s: &State) -> u32 {
    debug_assert!(
        s.is_solvable(),
        "rank() called on unsolvable state: {:?}",
        s.0
    );

    let blank_pos = s.blank_pos() as u32;

    // Collect the 8 non-blank tiles in order of position; they form a
    // permutation of {1..=8} (with even parity, since `s` is solvable).
    let mut tiles8 = [0u8; 8];
    let mut idx = 0;
    for &v in &s.0 {
        if v != 0 {
            tiles8[idx] = v;
            idx += 1;
        }
    }

    // Compute Lehmer digits d_0..d_5. `available` is a bitmask over values
    // 1..=8; bit `v` set means tile value `v` has not been placed yet.
    let mut available: u16 = 0b1_1111_1110; // bits 1..=8
    let mut even_rank: u32 = 0;
    for i in 0..6 {
        let tile = tiles8[i];
        let lo_mask = (1u16 << tile) - 1;
        let d = (available & lo_mask).count_ones();
        let radix = 8 - i as u32;
        even_rank = even_rank * radix + d;
        available &= !(1u16 << tile);
    }

    blank_pos * EVEN_BLOCK + even_rank
}

/// Inverse of [`rank`]: reconstruct the state from its rank.
///
/// Panics if `r >= N_STATES`.
pub fn unrank(r: u32) -> State {
    assert!(r < N_STATES, "rank {r} out of range [0, {N_STATES})");

    let blank_pos = (r / EVEN_BLOCK) as usize;
    let even_rank = r % EVEN_BLOCK;

    // Recover d_0..d_5 by repeated division. Bases are 8, 7, 6, 5, 4, 3 for
    // d_0, d_1, …, d_5. Decoding extracts least-significant first.
    let mut digits = [0u8; 8]; // d_0..d_7
    let mut r = even_rank;
    for i in (0..6).rev() {
        let radix = 8 - i as u32;
        digits[i] = (r % radix) as u8;
        r /= radix;
    }
    // d_6 is forced by parity (the 8-permutation must be even).
    let parity_so_far: u32 = digits[..6].iter().map(|&x| x as u32).sum();
    digits[6] = (parity_so_far % 2) as u8;
    // d_7 = 0 (already).

    // Reconstruct the 8-permutation from the Lehmer digits.
    let mut available: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut n_available = 8usize;
    let mut tiles8 = [0u8; 8];
    for i in 0..8 {
        let d = digits[i] as usize;
        debug_assert!(d < n_available);
        tiles8[i] = available[d];
        for k in d..(n_available - 1) {
            available[k] = available[k + 1];
        }
        n_available -= 1;
    }

    // Splice the 8-permutation into a 9-state with the blank at `blank_pos`.
    let mut s = [0u8; 9];
    let mut idx = 0;
    for pos in 0..9 {
        if pos == blank_pos {
            s[pos] = 0;
        } else {
            s[pos] = tiles8[idx];
            idx += 1;
        }
    }

    State(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle8::state::GOAL;

    #[test]
    fn goal_roundtrips() {
        let r = rank(&GOAL);
        assert_eq!(unrank(r), GOAL, "unrank(rank(GOAL)) must equal GOAL");
    }

    #[test]
    fn goal_rank_is_in_range() {
        let r = rank(&GOAL);
        assert!(r < N_STATES);
        // GOAL has blank at position 8 and identity 8-perm, so:
        assert_eq!(r, 8 * EVEN_BLOCK);
    }

    #[test]
    fn rank_zero_roundtrips() {
        let s = unrank(0);
        assert!(s.is_solvable());
        assert_eq!(rank(&s), 0);
    }

    #[test]
    fn rank_unrank_roundtrip_exhaustive() {
        // Every rank in [0, N_STATES) round-trips, and every produced state is
        // solvable.
        for r in 0..N_STATES {
            let s = unrank(r);
            assert!(
                s.is_solvable(),
                "unrank({}) produced unsolvable: {:?}",
                r,
                s.0
            );
            let r2 = rank(&s);
            assert_eq!(
                r2, r,
                "roundtrip failed at r={}: unrank gave {:?}, rerank gave {}",
                r, s.0, r2
            );
        }
    }

    #[test]
    fn ranks_cover_distinct_states_exhaustively() {
        // 181_440 ranks must yield 181_440 distinct states.
        use std::collections::HashSet;
        let mut seen: HashSet<[u8; 9]> = HashSet::with_capacity(N_STATES as usize);
        for r in 0..N_STATES {
            let s = unrank(r);
            assert!(seen.insert(s.0), "duplicate state at rank {}: {:?}", r, s.0);
        }
        assert_eq!(seen.len() as u32, N_STATES);
    }
}
