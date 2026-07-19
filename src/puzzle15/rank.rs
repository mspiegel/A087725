//! Bijection between solvable [`State`]s and integers in `[0, N_STATES)`.
//!
//! # Construction
//!
//! Each state is encoded as
//! ```text
//! rank = blank_pos * 653_837_184_000 + even_rank
//! ```
//! where `blank_pos ∈ [0, 16)` and `even_rank ∈ [0, 15!/2)` is the rank of the
//! 15-tile permutation among the `15!/2 = 653_837_184_000` permutations whose
//! parity matches the solvability constraint for `blank_pos` (see below).
//! Total: `16 · 15!/2 = 16!/2 = 10_461_394_944_000` distinct ranks, one per
//! solvable state.
//!
//! ## Even-width solvability
//!
//! Unlike the odd-width 8-puzzle, the 15-puzzle's parity invariant couples
//! the permutation parity to the blank's row (Johnson 1879):
//! ```text
//! inversions(π) + blank_row_from_bottom  ≡  0  (mod 2)
//! ```
//! Equivalently, `parity(π) ≡ blank_row_from_bottom (mod 2)`. So for each
//! `blank_pos`, exactly half of the `15!` permutations are reachable, and that
//! half is determined by which row the blank sits in. The encoding is uniform
//! across blank positions — same `EVEN_BLOCK`, different parity bit — so the
//! product `16 · 15!/2` exactly equals `16!/2`.
//!
//! ## Lehmer encoding
//!
//! For a permutation `π` of `{1..=15}`, compute its Lehmer digits
//! `(d_0, …, d_14)` with `d_i ∈ [0, 15-i)`. We have `d_14 = 0` always, and
//! `parity(π) = (d_0 + … + d_13) mod 2`. So for a permutation of known parity,
//! knowing `d_0..d_12` determines `d_13`. We pack `d_0..d_12` into a
//! mixed-radix integer with bases `(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3)`:
//! ```text
//! even_rank = ((((…((d_0 * 14 + d_1) * 13 + d_2) … ) * 4 + d_11) * 3 + d_12)
//! ```
//! giving `15·14·13·…·3 = 15!/2 = 653_837_184_000` distinct values.

use crate::puzzle15::state::{State, N_CELLS, N_STATES, N_TILES, W};

/// `15! / 2 = 653_837_184_000`. The number of permutations of `{1..=15}` of
/// either parity (one fixed parity per blank position).
const EVEN_BLOCK: u64 = 653_837_184_000;

/// Map a solvable [`State`] to a unique integer in `[0, N_STATES)`.
///
/// Debug-asserts that `s` is solvable.
pub fn rank(s: &State) -> u64 {
    debug_assert!(
        s.is_solvable(),
        "rank() called on unsolvable state: {:?}",
        s.0
    );

    let blank_pos = s.blank_pos() as u64;

    // Collect the 15 non-blank tiles in order of position.
    let mut tiles = [0u8; N_TILES];
    let mut idx = 0;
    for &v in &s.0 {
        if v != 0 {
            tiles[idx] = v;
            idx += 1;
        }
    }

    // Compute Lehmer digits d_0..d_12. `available` is a bitmask over values
    // 1..=15; bit `v` set means tile value `v` has not been placed yet.
    let mut available: u32 = 0b1111_1111_1111_1110; // bits 1..=15
    let mut even_rank: u64 = 0;
    for i in 0..(N_TILES - 2) {
        let tile = tiles[i];
        let lo_mask = (1u32 << tile) - 1;
        let d = (available & lo_mask).count_ones() as u64;
        let radix = (N_TILES - i) as u64; // 15, 14, …, 3
        even_rank = even_rank * radix + d;
        available &= !(1u32 << tile);
    }

    blank_pos * EVEN_BLOCK + even_rank
}

/// Inverse of [`rank`]: reconstruct the state from its rank.
///
/// Panics if `r >= N_STATES`.
pub fn unrank(r: u64) -> State {
    assert!(r < N_STATES, "rank {r} out of range [0, {N_STATES})");

    let blank_pos = (r / EVEN_BLOCK) as usize;
    let even_rank = r % EVEN_BLOCK;

    // Recover d_0..d_12 by repeated division (least-significant first).
    let mut digits = [0u8; N_TILES]; // d_0..d_14
    let mut r = even_rank;
    for i in (0..(N_TILES - 2)).rev() {
        let radix = (N_TILES - i) as u64; // 15, 14, …, 3
        digits[i] = (r % radix) as u8;
        r /= radix;
    }

    // d_13 is forced by the solvability parity rule:
    //   parity(π) ≡ blank_row_from_bottom (mod 2)
    //   sum(d_0..d_13) ≡ blank_row_from_bottom (mod 2)
    // So d_13 = required_parity XOR (sum d_0..d_12 mod 2). d_14 = 0 (only one
    // value will remain at that step).
    let blank_row_from_bottom = (W - 1) - blank_pos / W;
    let required_parity = (blank_row_from_bottom as u32) & 1;
    let parity_so_far: u32 = digits[..(N_TILES - 2)].iter().map(|&x| x as u32).sum();
    digits[N_TILES - 2] = ((required_parity ^ (parity_so_far & 1)) & 1) as u8;
    // digits[N_TILES - 1] stays 0.

    // Reconstruct the 15-permutation from the Lehmer digits.
    let mut available: [u8; N_TILES] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
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

    // Splice the 15-permutation into a 16-state with the blank at `blank_pos`.
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
    use crate::puzzle15::state::{Move, GOAL};

    #[test]
    fn even_block_value() {
        // 15! / 2 by direct factorial.
        let mut f: u64 = 1;
        for k in 2..=15u64 {
            f *= k;
        }
        assert_eq!(EVEN_BLOCK, f / 2);
    }

    #[test]
    fn goal_roundtrips() {
        let r = rank(&GOAL);
        assert_eq!(unrank(r), GOAL, "unrank(rank(GOAL)) must equal GOAL");
    }

    #[test]
    fn goal_rank_is_in_range_and_known() {
        let r = rank(&GOAL);
        assert!(r < N_STATES);
        // GOAL has blank at position 15 (row 3, bottom-right) and identity
        // permutation. blank_row_from_bottom = 0 → required_parity = 0 →
        // identity permutation (all Lehmer digits = 0) satisfies parity →
        // even_rank = 0.
        assert_eq!(r, 15 * EVEN_BLOCK);
    }

    #[test]
    fn rank_zero_roundtrips() {
        let s = unrank(0);
        assert!(s.is_solvable());
        assert_eq!(rank(&s), 0);
    }

    #[test]
    fn rank_max_roundtrips() {
        // Last valid rank.
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
        // Walk a thousand pseudo-random legal moves from GOAL; every reached
        // state must rank into [0, N_STATES), unrank back exactly, and remain
        // solvable.
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        for i in 0u32..1000 {
            assert!(s.is_solvable());
            let r = rank(&s);
            assert!(r < N_STATES);
            assert_eq!(unrank(r), s, "round-trip failed at step {}: {:?}", i, s.0);

            // Take one legal pseudo-random step.
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
    fn stratified_rank_sample_distinct_states() {
        // Sample ranks across the full range (every blank position) and
        // confirm distinct ranks produce distinct states.
        use std::collections::HashSet;
        let mut seen: HashSet<[u8; N_CELLS]> = HashSet::with_capacity(16_000);

        // Sample 1000 ranks from each blank-position block, evenly spaced.
        const PER_BLOCK: u64 = 1000;
        for bp in 0u64..16 {
            let base = bp * EVEN_BLOCK;
            for k in 0u64..PER_BLOCK {
                let r = base + (k * EVEN_BLOCK / PER_BLOCK);
                let s = unrank(r);
                assert!(s.is_solvable(), "unrank({}) unsolvable: {:?}", r, s.0);
                assert_eq!(rank(&s), r, "rank/unrank mismatch at r={r}");
                assert!(seen.insert(s.0), "duplicate state at rank {}: {:?}", r, s.0);
            }
        }
        assert_eq!(seen.len() as u64, 16 * PER_BLOCK);
    }

    #[test]
    fn blank_position_recovered_from_rank() {
        // For each blank position, an unrank lands the blank there.
        for bp in 0..N_CELLS {
            let r = (bp as u64) * EVEN_BLOCK;
            let s = unrank(r);
            assert_eq!(s.blank_pos() as usize, bp);
            assert_eq!(rank(&s), r);
        }
    }

    #[test]
    fn rank_is_injective_on_random_walks() {
        // Visit 5000 states via random walks restarting periodically; the
        // ranks must be pairwise distinct iff the states are.
        use std::collections::HashMap;
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut state_to_rank: HashMap<[u8; N_CELLS], u64> = HashMap::with_capacity(5000);
        let mut s = GOAL;
        for i in 0u32..5000 {
            let r = rank(&s);
            if let Some(&prior) = state_to_rank.get(&s.0) {
                assert_eq!(prior, r, "same state mapped to two different ranks");
            } else {
                for (k, &existing_r) in state_to_rank.values().zip(state_to_rank.values()).take(0) {
                    let _ = (k, existing_r);
                }
                state_to_rank.insert(s.0, r);
            }
            // Restart from GOAL every 100 steps to diversify the sample.
            if i % 100 == 99 {
                s = GOAL;
            }
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
        // Sanity: every value in the map is distinct.
        use std::collections::HashSet;
        let unique_ranks: HashSet<u64> = state_to_rank.values().copied().collect();
        assert_eq!(unique_ranks.len(), state_to_rank.len());
    }
}
