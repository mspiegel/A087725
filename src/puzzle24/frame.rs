//! Frame-conformant board construction (the m=5 **frame rule**, PUZZLE24.md 2B).
//!
//! Generalized from the 15-puzzle (where all 17 depth-80 antipodes satisfy it):
//! deep boards are *frame-conformant* — (a) the corner pieces `{_, 1, 5, 21}`
//! sit at their antipodal corners `{0, 24, 20, 4}`, and (b) the 8 corner-neighbor
//! tiles sit within Chebyshev ≤ 1 of their assigned anti-corners. Tier-1/Tier-2
//! (2B) confirmed this shifts the proven-LB distribution far above random and
//! yields certified-deep non-R boards — so it seeds the Phase-2 hunt generator.
//!
//! This module is **ungated** (no `ml` feature) so the classical hunt tooling
//! (`examples/candidates24.rs`) can construct seeds without pulling in candle.
//! `ml/corridor.rs::construct_frame` and `examples/frame24.rs` both delegate here
//! (each supplying its own RNG via the `below` closure), preserving their exact
//! draw sequences.

use crate::puzzle24::state::{State, N_CELLS};

/// Corner pieces and their antipodal corner cells: blank→0, 1→24, 5→20, 21→4.
pub const CORNER_ANTI: [(u8, u8); 4] = [(0, 0), (1, 24), (5, 20), (21, 4)];
/// The 8 corner-neighbor tiles and each one's assigned anti-corner
/// (goal-neighbors of tile 1 = {2,6}→24; of 5 = {4,10}→20; of 21 = {16,22}→4;
/// of the blank = {20,24}→0).
pub const NBR_ANTI: [(u8, u8); 8] =
    [(2, 24), (6, 24), (4, 20), (10, 20), (16, 4), (22, 4), (20, 0), (24, 0)];

/// Cells within Chebyshev ≤ 1 of `c` on the 5×5 grid.
pub fn cheb1(c: u8) -> Vec<u8> {
    let (r0, c0) = ((c / 5) as i32, (c % 5) as i32);
    let mut out = Vec::new();
    for dr in -1..=1i32 {
        for dc in -1..=1i32 {
            let (r, cc) = (r0 + dr, c0 + dc);
            if (0..5).contains(&r) && (0..5).contains(&cc) {
                out.push((r * 5 + cc) as u8);
            }
        }
    }
    out
}

/// Construct one frame-conformant board: corner pieces at anti-corners, the 8
/// corner-neighbor tiles within Chebyshev ≤ 1 of theirs, interior tiles uniform,
/// parity fixed by an interior swap. Returns `None` on rare placement contention
/// (caller retries).
///
/// RNG-agnostic: `below(n)` must return a uniform index in `[0, n)`. Callers pass
/// their own generator; the internal draw sequence is `below(i+1)` per shuffle
/// step and `below(free.len())` per neighbor placement, so a faithful wrapper
/// reproduces any existing caller's stream exactly.
pub fn construct_frame_with(below: &mut dyn FnMut(usize) -> usize) -> Option<State> {
    let mut cells = [u8::MAX; N_CELLS]; // MAX = empty
    let mut placed = [false; 25];
    // (a) corner pieces at anti-corners.
    for &(t, cell) in &CORNER_ANTI {
        cells[cell as usize] = t;
        placed[t as usize] = true;
    }
    // (b) each neighbor tile at a random free cell within Chebyshev 1 of its
    // assigned anti-corner (shuffled order so contention resolves randomly).
    let mut order: Vec<usize> = (0..NBR_ANTI.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, below(i + 1));
    }
    for &i in &order {
        let (t, anti) = NBR_ANTI[i];
        let free: Vec<u8> =
            cheb1(anti).into_iter().filter(|&c| cells[c as usize] == u8::MAX).collect();
        if free.is_empty() {
            return None; // contention (rare) — caller retries
        }
        cells[free[below(free.len())] as usize] = t;
        placed[t as usize] = true;
    }
    // Interior: remaining 13 tiles shuffled into the remaining 13 cells.
    let mut rest: Vec<u8> = (1..25u8).filter(|&t| !placed[t as usize]).collect();
    for i in (1..rest.len()).rev() {
        rest.swap(i, below(i + 1));
    }
    let free_cells: Vec<usize> = (0..N_CELLS).filter(|&c| cells[c] == u8::MAX).collect();
    debug_assert_eq!(free_cells.len(), rest.len());
    for (c, t) in free_cells.iter().zip(rest.iter()) {
        cells[*c] = *t;
    }
    let mut s = State(cells);
    if !s.is_solvable() {
        // One interior transposition flips permutation parity.
        s.0.swap(free_cells[0], free_cells[1]);
    }
    debug_assert!(s.is_solvable());
    Some(s)
}

/// `true` iff `s` is frame-conformant: the 4 corner pieces sit exactly at their
/// anti-corners and each of the 8 corner-neighbor tiles is within Chebyshev ≤ 1
/// of its assigned anti-corner. (The interior 13 tiles are unconstrained.)
pub fn is_frame_conformant(s: &State) -> bool {
    // Tile → cell lookup.
    let mut pos = [0u8; N_CELLS];
    for (cell, &tile) in s.0.iter().enumerate() {
        pos[tile as usize] = cell as u8;
    }
    for &(t, cell) in &CORNER_ANTI {
        if pos[t as usize] != cell {
            return false;
        }
    }
    for &(t, anti) in &NBR_ANTI {
        if !cheb1(anti).contains(&pos[t as usize]) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::GOAL;

    /// Deterministic xorshift for tests (independent of any caller's RNG).
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
        }
    }

    #[test]
    fn constructions_are_solvable_and_conformant() {
        let mut rng = Rng(0x1234_5678);
        let mut made = 0;
        for _ in 0..5000 {
            if let Some(s) = construct_frame_with(&mut |n| rng.below(n)) {
                assert!(s.is_solvable(), "unsolvable frame board: {:?}", s.0);
                assert!(is_frame_conformant(&s), "non-conformant: {:?}", s.0);
                made += 1;
            }
        }
        assert!(made > 4000, "contention too high: only {} of 5000 succeeded", made);
    }

    #[test]
    fn goal_is_not_frame_conformant() {
        // GOAL has tile 1 at cell 0, not at anti-corner cell 24.
        assert!(!is_frame_conformant(&GOAL));
    }

    #[test]
    fn deterministic_for_fixed_below_sequence() {
        let mut a = Rng(42);
        let mut b = Rng(42);
        let sa = construct_frame_with(&mut |n| a.below(n));
        let sb = construct_frame_with(&mut |n| b.below(n));
        assert_eq!(sa, sb);
    }

    #[test]
    fn corners_exact_neighbors_within_cheb1() {
        let mut rng = Rng(99);
        let s = loop {
            if let Some(s) = construct_frame_with(&mut |n| rng.below(n)) {
                break s;
            }
        };
        // Corners exact.
        assert_eq!(s.0[0], 0);
        assert_eq!(s.0[24], 1);
        assert_eq!(s.0[20], 5);
        assert_eq!(s.0[4], 21);
    }
}
