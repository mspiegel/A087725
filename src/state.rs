//! Core 8-puzzle state, moves, and solvability invariants.
//!
//! Positions are indexed in row-major order:
//! ```text
//! 0 1 2
//! 3 4 5
//! 6 7 8
//! ```
//! Tile values are `1..=8`; the blank is `0`. The goal state has tile `i` at
//! position `i-1` and the blank at position 8.
//!
//! A `Move` describes which direction the **blank** moves. Equivalently, the
//! tile adjacent to the blank in the opposite direction slides into the blank's
//! cell.

/// A puzzle configuration: 9 tile values in row-major order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State(pub [u8; 9]);

/// Solved configuration: `1 2 3 / 4 5 6 / 7 8 _`.
pub const GOAL: State = State([1, 2, 3, 4, 5, 6, 7, 8, 0]);

/// Number of solvable states: 9!/2.
pub const N_STATES: u32 = 181_440;

/// Maximum optimal solution length (Reinefeld 1993).
pub const DIAMETER: u8 = 31;

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
    /// Position of the blank (value `0`), in `0..9`.
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
        let b = self.blank_pos();
        let row = b / 3;
        let col = b % 3;
        let mut moves = MoveSet::empty();
        if row > 0 { moves.insert(Move::Up); }
        if row < 2 { moves.insert(Move::Down); }
        if col > 0 { moves.insert(Move::Left); }
        if col < 2 { moves.insert(Move::Right); }
        moves
    }

    /// Apply `m`. Panics if `m` is not legal (blank would move off the board).
    pub fn apply(&self, m: Move) -> Self {
        let b = self.blank_pos() as usize;
        let br = b / 3;
        let bc = b % 3;
        let (nr, nc) = match m {
            Move::Up => {
                assert!(br > 0, "illegal Up from blank at {}", b);
                (br - 1, bc)
            }
            Move::Down => {
                assert!(br < 2, "illegal Down from blank at {}", b);
                (br + 1, bc)
            }
            Move::Left => {
                assert!(bc > 0, "illegal Left from blank at {}", b);
                (br, bc - 1)
            }
            Move::Right => {
                assert!(bc < 2, "illegal Right from blank at {}", b);
                (br, bc + 1)
            }
        };
        let n = nr * 3 + nc;
        let mut next = self.0;
        next.swap(b, n);
        State(next)
    }

    /// Number of inversions in the tile sequence (counting the blank as 0).
    /// `permutation_parity` is this count mod 2.
    pub fn inversions(&self) -> u32 {
        let mut count = 0u32;
        for i in 0..9 {
            for j in (i + 1)..9 {
                if self.0[i] > 0 && self.0[j] > 0 && self.0[i] > self.0[j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// `true` iff this state is reachable from `GOAL`.
    ///
    /// For a sliding puzzle with **odd width** (3 here), the inversion count
    /// of the non-blank tiles is a move-invariant — horizontal blank moves
    /// involve no non-blank reordering, and vertical moves carry one tile past
    /// exactly two others (flipping two inversions, net parity zero). The goal
    /// has zero inversions, so a state is solvable iff its inversion count is
    /// even.
    pub fn is_solvable(&self) -> bool {
        self.inversions() % 2 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_is_solvable() {
        assert!(GOAL.is_solvable());
        assert_eq!(GOAL.inversions(), 0);
        assert_eq!(GOAL.blank_pos(), 8);
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
        // Walk a deterministic path; every reached state must be solvable.
        let path = [Move::Up, Move::Up, Move::Left, Move::Left, Move::Down, Move::Right];
        for m in path {
            assert!(s.legal_moves().contains(m));
            s = s.apply(m);
            assert!(s.is_solvable(), "unsolvable after apply: {:?}", s);
        }
    }

    #[test]
    fn legal_moves_always_2_to_4() {
        // Walk a few hundred random-ish positions; every state we reach has
        // 2..=4 legal moves.
        let mut s = GOAL;
        let pseudo = |i: u32| -> Move {
            Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize]
        };
        for i in 0u32..500 {
            let n = s.legal_moves().len();
            assert!((2..=4).contains(&n), "blank at {} has {} moves", s.blank_pos(), n);
            // Try to apply a pseudo-random legal move
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
    fn move_inverse_is_involution() {
        for m in Move::ALL {
            assert_eq!(m.inverse().inverse(), m);
        }
    }

    #[test]
    fn unsolvable_state_detected() {
        // Swap two adjacent non-blank tiles in the goal — should be unsolvable.
        let mut bad = GOAL.0;
        bad.swap(0, 1); // swap tiles 1 and 2
        assert!(!State(bad).is_solvable());
    }
}
