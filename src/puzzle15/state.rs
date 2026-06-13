//! Core 15-puzzle state, moves, and solvability invariants.
//!
//! Positions are indexed in row-major order:
//! ```text
//!  0  1  2  3
//!  4  5  6  7
//!  8  9 10 11
//! 12 13 14 15
//! ```
//! Tile values are `1..=15`; the blank is `0`. The goal state has tile `i` at
//! position `i-1` and the blank at position 15.
//!
//! A `Move` describes which direction the **blank** moves. Equivalently, the
//! tile adjacent to the blank in the opposite direction slides into the blank's
//! cell.

/// Board width (and height). Used throughout for blank-coordinate arithmetic.
pub const W: usize = 4;

/// Number of cells on the board: 16.
pub const N_CELLS: usize = W * W;

/// Number of tiles (excluding the blank): 15.
pub const N_TILES: usize = N_CELLS - 1;

/// A puzzle configuration: 16 tile values in row-major order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State(pub [u8; N_CELLS]);

/// Solved configuration: `1 2 3 4 / 5 6 7 8 / 9 10 11 12 / 13 14 15 _`.
pub const GOAL: State =
    State([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]);

/// Number of solvable states: 16!/2.
pub const N_STATES: u64 = 10_461_394_944_000;

/// Maximum optimal solution length (Korf & Schultze 2005).
pub const DIAMETER: u8 = 80;

/// One move of the blank.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Move {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Move {
    pub const ALL: [Move; 4] = [Move::Up, Move::Down, Move::Left, Move::Right];

    /// The move that undoes this one.
    pub fn inverse(self) -> Move {
        match self {
            Move::Up => Move::Down,
            Move::Down => Move::Up,
            Move::Left => Move::Right,
            Move::Right => Move::Left,
        }
    }
}

/// A set of moves stored as a bitmask. Bit `i` set ⇔ `Move::ALL[i]` present.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MoveSet(pub u8);

impl MoveSet {
    pub const fn empty() -> Self { MoveSet(0) }

    pub fn contains(self, m: Move) -> bool {
        self.0 & (1 << m as u8) != 0
    }

    pub fn insert(&mut self, m: Move) {
        self.0 |= 1 << m as u8;
    }

    pub fn len(self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> MoveSetIter {
        MoveSetIter { bits: self.0, idx: 0 }
    }
}

pub struct MoveSetIter {
    bits: u8,
    idx: u8,
}

impl Iterator for MoveSetIter {
    type Item = Move;
    fn next(&mut self) -> Option<Move> {
        while self.idx < 4 {
            let i = self.idx;
            self.idx += 1;
            if self.bits & (1 << i) != 0 {
                return Some(Move::ALL[i as usize]);
            }
        }
        None
    }
}

impl State {
    /// Position of the blank (value `0`), in `0..16`.
    pub fn blank_pos(&self) -> u8 {
        for (i, &v) in self.0.iter().enumerate() {
            if v == 0 {
                return i as u8;
            }
        }
        panic!("State has no blank: {:?}", self.0);
    }

    /// The (up to 4) legal moves of the blank from this state.
    pub fn legal_moves(&self) -> MoveSet {
        let b = self.blank_pos() as usize;
        let row = b / W;
        let col = b % W;
        let mut moves = MoveSet::empty();
        if row > 0 { moves.insert(Move::Up); }
        if row < W - 1 { moves.insert(Move::Down); }
        if col > 0 { moves.insert(Move::Left); }
        if col < W - 1 { moves.insert(Move::Right); }
        moves
    }

    /// Apply `m`. Panics if `m` is not legal (blank would move off the board).
    pub fn apply(&self, m: Move) -> Self {
        let b = self.blank_pos() as usize;
        let br = b / W;
        let bc = b % W;
        let (nr, nc) = match m {
            Move::Up => {
                assert!(br > 0, "illegal Up from blank at {}", b);
                (br - 1, bc)
            }
            Move::Down => {
                assert!(br < W - 1, "illegal Down from blank at {}", b);
                (br + 1, bc)
            }
            Move::Left => {
                assert!(bc > 0, "illegal Left from blank at {}", b);
                (br, bc - 1)
            }
            Move::Right => {
                assert!(bc < W - 1, "illegal Right from blank at {}", b);
                (br, bc + 1)
            }
        };
        let n = nr * W + nc;
        let mut next = self.0;
        next.swap(b, n);
        State(next)
    }

    /// Number of inversions in the non-blank tile sequence.
    pub fn inversions(&self) -> u32 {
        let mut count = 0u32;
        for i in 0..N_CELLS {
            for j in (i + 1)..N_CELLS {
                if self.0[i] > 0 && self.0[j] > 0 && self.0[i] > self.0[j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// `true` iff this state is reachable from `GOAL`.
    ///
    /// For an **even-width** sliding puzzle (4 here), the move invariant is
    /// `inversions + blank_row_from_bottom` mod 2. A horizontal blank move
    /// involves no non-blank reordering and doesn't change the blank's row; a
    /// vertical move flips inversions by an odd amount (carrying one tile past
    /// W-1 = 3 others) **and** changes the blank's row by 1, so the parity of
    /// the sum is preserved. `GOAL` has zero inversions and the blank in the
    /// bottom row (row_from_bottom = 0), so a state is solvable iff that sum
    /// is even (Johnson 1879).
    pub fn is_solvable(&self) -> bool {
        let blank_row = self.blank_pos() as usize / W;
        let blank_row_from_bottom = (W - 1) - blank_row;
        (self.inversions() as usize + blank_row_from_bottom) % 2 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_is_solvable() {
        assert!(GOAL.is_solvable());
        assert_eq!(GOAL.inversions(), 0);
        assert_eq!(GOAL.blank_pos(), 15);
    }

    #[test]
    fn n_states_value() {
        // 16! / 2 — sanity-check the constant against a direct factorial.
        let mut f: u64 = 1;
        for k in 2..=16u64 {
            f *= k;
        }
        assert_eq!(N_STATES, f / 2);
    }

    #[test]
    fn goal_legal_moves() {
        // Blank in bottom-right corner: only Up and Left are legal.
        let m = GOAL.legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Up));
        assert!(m.contains(Move::Left));
        assert!(!m.contains(Move::Down));
        assert!(!m.contains(Move::Right));
    }

    #[test]
    fn apply_then_inverse_is_identity() {
        for m in Move::ALL {
            if GOAL.legal_moves().contains(m) {
                let after = GOAL.apply(m);
                let back = after.apply(m.inverse());
                assert_eq!(back, GOAL, "apply({:?}) then apply({:?}) should restore", m, m.inverse());
            }
        }
    }

    #[test]
    fn apply_preserves_solvability() {
        let mut s = GOAL;
        let path = [
            Move::Up, Move::Up, Move::Left, Move::Left, Move::Left,
            Move::Down, Move::Right, Move::Up, Move::Right, Move::Down,
        ];
        for m in path {
            assert!(s.legal_moves().contains(m));
            s = s.apply(m);
            assert!(s.is_solvable(), "unsolvable after apply: {:?}", s);
        }
    }

    #[test]
    fn legal_moves_always_2_to_4() {
        // Walk a few hundred pseudo-random positions; every reached state has
        // 2..=4 legal moves.
        let mut s = GOAL;
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        for i in 0u32..500 {
            let n = s.legal_moves().len();
            assert!((2..=4).contains(&n), "blank at {} has {} moves", s.blank_pos(), n);
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
    fn legal_moves_at_each_corner() {
        // Construct states with the blank at each of the four corners and
        // verify the expected 2-move set.
        let mut corner = GOAL.0;
        // Blank at top-left: only Down + Right.
        corner.swap(0, 15);
        let m = State(corner).legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Down) && m.contains(Move::Right));

        // Blank at top-right: Down + Left.
        let mut corner = GOAL.0;
        corner.swap(3, 15);
        let m = State(corner).legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Down) && m.contains(Move::Left));

        // Blank at bottom-left: Up + Right.
        let mut corner = GOAL.0;
        corner.swap(12, 15);
        let m = State(corner).legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Up) && m.contains(Move::Right));

        // GOAL itself is bottom-right: Up + Left (already covered).
    }

    #[test]
    fn legal_moves_at_edge_non_corner() {
        // Blank in the middle of the top edge (position 1 or 2): 3 moves.
        let mut s = GOAL.0;
        s.swap(1, 15);
        let m = State(s).legal_moves();
        assert_eq!(m.len(), 3);
        assert!(m.contains(Move::Down) && m.contains(Move::Left) && m.contains(Move::Right));
        assert!(!m.contains(Move::Up));
    }

    #[test]
    fn legal_moves_in_interior() {
        // Blank in an interior cell (e.g., position 5): all 4 moves.
        let mut s = GOAL.0;
        s.swap(5, 15);
        let m = State(s).legal_moves();
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn move_inverse_is_involution() {
        for m in Move::ALL {
            assert_eq!(m.inverse().inverse(), m);
        }
    }

    #[test]
    fn unsolvable_state_detected() {
        // Swap two adjacent non-blank tiles in the goal — should be unsolvable
        // (inversions=1, blank_row_from_bottom=0, sum odd).
        let mut bad = GOAL.0;
        bad.swap(0, 1); // swap tiles 1 and 2
        assert!(!State(bad).is_solvable());
    }

    #[test]
    fn solvability_invariant_under_moves() {
        // Walk a long path and verify solvability holds at every step (the
        // whole point of the invariant).
        let mut s = GOAL;
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        for i in 0u32..1000 {
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
            assert!(s.is_solvable(), "lost solvability after step {}", i);
        }
    }
}
