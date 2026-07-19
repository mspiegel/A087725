//! Random-walk scrambling from GOAL, plus a tiny dependency-free RNG.
//!
//! DAVI training states are produced by scrambling the goal `k` times with `k`
//! uniform in `[1, k_max]` (a fixed range, not an advancing curriculum — see
//! TRAINING.md). We reuse the repo's inline-xorshift convention rather than
//! pulling in the `rand` crate, matching how the existing tests generate
//! randomness (e.g. `state.rs` tests, `walking_distance.rs` tests).

use crate::puzzle15::state::{Move, State, GOAL};

/// Small, fast, deterministic xorshift64 PRNG. Not cryptographic — just enough
/// for reproducible scrambles and categorical sampling.
pub struct Rng(u64);

impl Rng {
    /// Seed the RNG. A zero seed is remapped to a fixed nonzero constant so the
    /// generator never degenerates to always returning 0.
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `[lo, hi_inclusive]`. `lo <= hi_inclusive` required.
    #[inline]
    pub fn gen_range(&mut self, lo: u32, hi_inclusive: u32) -> u32 {
        debug_assert!(lo <= hi_inclusive);
        let span = (hi_inclusive - lo + 1) as u64;
        lo + (self.next_u64() % span) as u32
    }

    /// Uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn gen_f32(&mut self) -> f32 {
        // Top 24 bits -> [0,1); 2^24 = 16_777_216.
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }
}

/// Random-walk scramble from GOAL: draw `k ~ Uniform(1, k_max)`, then take `k`
/// legal blank moves, never immediately undoing the previous move (mirrors the
/// parent-move pruning in `idastar`). Returns the resulting state and the actual
/// `k` taken. The result is always solvable (every move preserves solvability).
pub fn scramble(rng: &mut Rng, k_max: u32) -> (State, u32) {
    let k = rng.gen_range(1, k_max.max(1));
    let choose = |_s: &State, blank: u8, rng: &mut Rng| -> Move {
        // Uniform over legal moves (caller of scramble_with handles undo pruning).
        let legal: Vec<Move> = State::legal_moves_at(blank).iter().collect();
        legal[rng.gen_range(0, (legal.len() - 1) as u32) as usize]
    };
    let s = walk(rng, k, choose);
    (s, k)
}

/// Like [`scramble`] but each step's move is chosen by a caller-supplied policy
/// `choose(state, blank, rng) -> Move` — the hook the generator's inference-only
/// rollout plugs into. The undo-pruning is applied *around* `choose`: `choose`
/// is re-invoked (up to a few tries) if it returns the inverse of the last move,
/// so the policy still expresses a distribution over the non-undo moves.
pub fn scramble_with<F>(rng: &mut Rng, k_max: u32, mut choose: F) -> (State, u32)
where
    F: FnMut(&State, u8, &mut Rng) -> Move,
{
    let k = rng.gen_range(1, k_max.max(1));
    let s = walk(rng, k, &mut choose);
    (s, k)
}

/// Shared walk driver: `k` moves from GOAL, skipping the inverse of the last
/// move. `choose` proposes a move; if it proposes the undo, we resample it up to
/// a small bound, then fall back to any non-undo legal move.
fn walk<F>(rng: &mut Rng, k: u32, mut choose: F) -> State
where
    F: FnMut(&State, u8, &mut Rng) -> Move,
{
    let mut s = GOAL;
    let mut blank = s.blank_pos();
    let mut last: Option<Move> = None;
    for _ in 0..k {
        let banned = last.map(|m| m.inverse());
        let mut m = choose(&s, blank, rng);
        // Resample a few times if the proposed move is the undo or illegal.
        for _ in 0..8 {
            let legal = State::legal_moves_at(blank).contains(m);
            if legal && Some(m) != banned {
                break;
            }
            m = choose(&s, blank, rng);
        }
        // Guaranteed-valid fallback: first legal non-undo move.
        if !State::legal_moves_at(blank).contains(m) || Some(m) == banned {
            m = State::legal_moves_at(blank)
                .iter()
                .find(|&mv| Some(mv) != banned)
                .expect("blank always has >= 1 non-undo legal move");
        }
        let (ns, nb) = s.apply_at(m, blank);
        s = ns;
        blank = nb;
        last = Some(m);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrambles_are_solvable_and_length_in_range() {
        let mut rng = Rng::new(12345);
        for _ in 0..500 {
            let (s, k) = scramble(&mut rng, 30);
            assert!((1..=30).contains(&k));
            assert!(
                s.is_solvable(),
                "scramble produced unsolvable state: {:?}",
                s.0
            );
        }
    }

    #[test]
    fn no_immediate_undo_in_walk() {
        // A length-2 walk from GOAL must never return to GOAL (that would require
        // the second move to undo the first).
        let mut rng = Rng::new(999);
        let mut returned_to_goal = 0;
        for _ in 0..2000 {
            let s = walk(&mut rng, 2, |_s, blank, rng| {
                let legal: Vec<Move> = State::legal_moves_at(blank).iter().collect();
                legal[rng.gen_range(0, (legal.len() - 1) as u32) as usize]
            });
            if s == GOAL {
                returned_to_goal += 1;
            }
        }
        assert_eq!(
            returned_to_goal, 0,
            "undo pruning failed: walk returned to GOAL"
        );
    }

    #[test]
    fn rng_gen_range_bounds() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let v = rng.gen_range(3, 9);
            assert!((3..=9).contains(&v));
        }
        for _ in 0..10_000 {
            let f = rng.gen_f32();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn scramble_with_policy_hook_runs() {
        // A deterministic "always try Up" policy still produces a valid solvable
        // board (fallback handles when Up is illegal / an undo).
        let mut rng = Rng::new(42);
        let (s, _k) = scramble_with(&mut rng, 20, |_s, _b, _r| Move::Up);
        assert!(s.is_solvable());
    }
}
