//! Core 24-puzzle state, moves, and solvability invariants.
//!
//! Positions are indexed in row-major order:
//! ```text
//!  0  1  2  3  4
//!  5  6  7  8  9
//! 10 11 12 13 14
//! 15 16 17 18 19
//! 20 21 22 23 24
//! ```
//! Tile values are `1..=24`; the blank is `0`. The goal state has tile `i` at
//! position `i-1` and the blank at position 24.
//!
//! A `Move` describes which direction the **blank** moves. Equivalently, the
//! tile adjacent to the blank in the opposite direction slides into the blank's
//! cell.
//!
//! The 24-puzzle is the project's Milestone 5 research target: its STM diameter
//! is known only to lie in `[152, 205]`. This module ports the verified
//! 15-puzzle state machinery to the 5×5 board; unlike the 15-puzzle (even
//! width) it shares the 8-puzzle's **odd-width** solvability rule.

/// Board width (and height). Used throughout for blank-coordinate arithmetic.
pub const W: usize = 5;

/// Number of cells on the board: 25.
pub const N_CELLS: usize = W * W;

/// Number of tiles (excluding the blank): 24.
pub const N_TILES: usize = N_CELLS - 1;

/// A puzzle configuration: 25 tile values in row-major order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State(pub [u8; N_CELLS]);

/// Solved configuration: `1..5 / 6..10 / 11..15 / 16..20 / 21..24 _`.
pub const GOAL: State = State([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,
]);

/// Number of solvable states: `25! / 2 ≈ 7.76 × 10²⁴`. Exceeds `u64`, so this
/// is a `u128`. We never materialize a table of this size — it bounds the
/// abstract rank space only.
pub const N_STATES: u128 = {
    let mut f: u128 = 1;
    let mut k: u128 = 2;
    while k <= 25 {
        f *= k;
        k += 1;
    }
    f / 2
};

/// Best known bounds on the 24-puzzle's STM diameter (`[152, 205]`; upper bound
/// Rokicki 2016). The exact value is an open problem — this project's goal is to
/// tighten these, so we record the interval rather than a single constant.
pub const DIAMETER_LOWER: u16 = 152;
pub const DIAMETER_UPPER: u16 = 205;

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
    pub const fn empty() -> Self {
        MoveSet(0)
    }

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
        MoveSetIter {
            bits: self.0,
            idx: 0,
        }
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

/// Sentinel in [`NEIGHBOR`] meaning "no neighbor in that direction" (the move
/// would push the blank off the board).
const NO_NEIGHBOR: u8 = u8::MAX;

/// `LEGAL_MOVES[b]` is the set of legal blank moves when the blank is at cell
/// `b`. Precomputed at compile time so move generation is a single array read
/// rather than re-derived row/column arithmetic at every node.
const LEGAL_MOVES: [MoveSet; N_CELLS] = {
    let mut table = [MoveSet::empty(); N_CELLS];
    let mut b = 0;
    while b < N_CELLS {
        let row = b / W;
        let col = b % W;
        let mut bits = 0u8;
        if row > 0 {
            bits |= 1 << (Move::Up as u8);
        }
        if row < W - 1 {
            bits |= 1 << (Move::Down as u8);
        }
        if col > 0 {
            bits |= 1 << (Move::Left as u8);
        }
        if col < W - 1 {
            bits |= 1 << (Move::Right as u8);
        }
        table[b] = MoveSet(bits);
        b += 1;
    }
    table
};

/// `NEIGHBOR[b][m as usize]` is the cell the blank lands on after applying move
/// `m` from cell `b`, or [`NO_NEIGHBOR`] if that move is illegal. Lets `apply`
/// avoid the blank scan and the coordinate match entirely.
const NEIGHBOR: [[u8; 4]; N_CELLS] = {
    let mut table = [[NO_NEIGHBOR; 4]; N_CELLS];
    let mut b = 0;
    while b < N_CELLS {
        let row = b / W;
        let col = b % W;
        if row > 0 {
            table[b][Move::Up as usize] = (b - W) as u8;
        }
        if row < W - 1 {
            table[b][Move::Down as usize] = (b + W) as u8;
        }
        if col > 0 {
            table[b][Move::Left as usize] = (b - 1) as u8;
        }
        if col < W - 1 {
            table[b][Move::Right as usize] = (b + 1) as u8;
        }
        b += 1;
    }
    table
};

impl State {
    /// Position of the blank (value `0`), in `0..25`.
    pub fn blank_pos(&self) -> u8 {
        for (i, &v) in self.0.iter().enumerate() {
            if v == 0 {
                return i as u8;
            }
        }
        panic!("State has no blank: {:?}", self.0);
    }

    /// The legal blank moves when the blank is known to be at cell `blank`.
    ///
    /// Hot-path variant of [`legal_moves`](Self::legal_moves) for callers (the
    /// IDA\* search) that already track the blank position, skipping the scan.
    #[inline]
    pub fn legal_moves_at(blank: u8) -> MoveSet {
        LEGAL_MOVES[blank as usize]
    }

    /// The (up to 4) legal moves of the blank from this state.
    #[inline]
    pub fn legal_moves(&self) -> MoveSet {
        LEGAL_MOVES[self.blank_pos() as usize]
    }

    /// Apply `m` given the blank's current cell `blank`. Returns the new state
    /// **and the blank's new cell**, so a search threading the blank through
    /// the recursion never rescans for it.
    ///
    /// In debug builds, panics if `m` is illegal from `blank`; in release the
    /// out-of-bounds swap would itself panic, so an illegal move never silently
    /// corrupts state.
    #[inline]
    pub fn apply_at(&self, m: Move, blank: u8) -> (Self, u8) {
        let nb = NEIGHBOR[blank as usize][m as usize];
        debug_assert!(nb != NO_NEIGHBOR, "illegal {m:?} from blank at {blank}");
        let mut next = self.0;
        next.swap(blank as usize, nb as usize);
        (State(next), nb)
    }

    /// Apply `m`. In debug builds, panics if `m` is not legal.
    #[inline]
    pub fn apply(&self, m: Move) -> Self {
        self.apply_at(m, self.blank_pos()).0
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
    /// For a sliding puzzle with **odd width** (5 here), the inversion count of
    /// the non-blank tiles is a move-invariant — horizontal blank moves involve
    /// no non-blank reordering, and a vertical move carries one tile past
    /// exactly `W - 1 = 4` others (an even number, so the inversion parity is
    /// unchanged). `GOAL` has zero inversions, so a state is solvable iff its
    /// inversion count is even — **independent of the blank's position**, unlike
    /// the even-width 15-puzzle.
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
        assert_eq!(GOAL.blank_pos(), 24);
    }

    #[test]
    fn n_states_value() {
        // 25! / 2 — sanity-check the constant against a direct factorial.
        let mut f: u128 = 1;
        for k in 2..=25u128 {
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
                assert_eq!(
                    back,
                    GOAL,
                    "apply({:?}) then apply({:?}) should restore",
                    m,
                    m.inverse()
                );
            }
        }
    }

    #[test]
    fn apply_preserves_solvability() {
        let mut s = GOAL;
        let path = [
            Move::Up,
            Move::Up,
            Move::Left,
            Move::Left,
            Move::Left,
            Move::Left,
            Move::Down,
            Move::Right,
            Move::Up,
            Move::Right,
            Move::Down,
        ];
        for m in path {
            assert!(s.legal_moves().contains(m));
            s = s.apply(m);
            assert!(s.is_solvable(), "unsolvable after apply: {s:?}");
        }
    }

    #[test]
    fn legal_moves_always_2_to_4() {
        let mut s = GOAL;
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        for i in 0u32..500 {
            let n = s.legal_moves().len();
            assert!(
                (2..=4).contains(&n),
                "blank at {} has {} moves",
                s.blank_pos(),
                n
            );
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
        // Blank at top-left (0): Down + Right.
        let mut c = GOAL.0;
        c.swap(0, 24);
        let m = State(c).legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Down) && m.contains(Move::Right));

        // Blank at top-right (4): Down + Left.
        let mut c = GOAL.0;
        c.swap(4, 24);
        let m = State(c).legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Down) && m.contains(Move::Left));

        // Blank at bottom-left (20): Up + Right.
        let mut c = GOAL.0;
        c.swap(20, 24);
        let m = State(c).legal_moves();
        assert_eq!(m.len(), 2);
        assert!(m.contains(Move::Up) && m.contains(Move::Right));
    }

    #[test]
    fn legal_moves_in_interior() {
        // Blank at center (12): all 4 moves.
        let mut s = GOAL.0;
        s.swap(12, 24);
        let m = State(s).legal_moves();
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn legal_moves_at_matches_method_for_every_blank() {
        for b in 0..N_CELLS {
            let mut cells = GOAL.0;
            cells.swap(24, b);
            let s = State(cells);
            assert_eq!(s.blank_pos() as usize, b);
            assert_eq!(State::legal_moves_at(b as u8), s.legal_moves());
        }
    }

    #[test]
    fn apply_at_matches_apply_and_tracks_blank() {
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        let mut blank = s.blank_pos();
        for i in 0u32..1000 {
            for m in State::legal_moves_at(blank).iter() {
                let (via_at, nb) = s.apply_at(m, blank);
                assert_eq!(via_at, s.apply(m), "apply_at disagrees with apply");
                assert_eq!(nb, via_at.blank_pos(), "apply_at returned wrong blank");
            }
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if State::legal_moves_at(blank).contains(m) {
                    let (next, nb) = s.apply_at(m, blank);
                    s = next;
                    blank = nb;
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
        // Swap two adjacent non-blank tiles in the goal — inversions = 1 (odd).
        let mut bad = GOAL.0;
        bad.swap(0, 1);
        assert!(!State(bad).is_solvable());
    }

    #[test]
    fn solvability_is_blank_independent() {
        // Odd-width hallmark: moving only the blank never changes solvability,
        // for the blank at any cell. Start from GOAL (solvable) and walk.
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        let mut s = GOAL;
        for i in 0u32..1000 {
            assert!(s.is_solvable(), "lost solvability after step {i}");
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    break;
                }
            }
        }
    }
}
