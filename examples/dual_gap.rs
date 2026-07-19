//! dual_gap — witness that DUAL heuristic lookups are INVALID for the 24-puzzle.
//!
//! Tempting idea (Felner et al., "Dual lookups in pattern databases", IJCAI'05):
//! normalize states to the identity-goal frame via π_G (tile t ↦ t−1, blank ↦
//! cell 24); if `dist(s, GOAL) = dist(dual(s), GOAL)` for `dual(s) = π_G⁻¹ ∘
//! (π_G ∘ s)⁻¹`, then `max(h(s), h(dual(s)))` would be a free second admissible
//! probe for every heuristic we own (cWD, ZPDB). R is even self-dual (its
//! normalized permutation is the involution x_R(j) = 24−j), which made this look
//! tailor-made for R's tree.
//!
//! **It is unsound for sliding-tile puzzles.** The dual-lookup theorem needs the
//! reversed transposition sequence to be legal; the blank's adjacency constraint
//! breaks that. Concretely, the dual of a solvable state can be UNSOLVABLE:
//! from GOAL apply L then U — the normalized permutation x has x(24) = 23 (odd
//! cell color) while x⁻¹(24) = 18 (even cell color), so `dual(s)` violates the
//! permutation-parity/blank-color invariant. This tool prints that witness and
//! the violation rate over random scrambles (~50%), so nobody wires a dual probe
//! into an LB proof and certifies a false bound. Even among solvable duals the
//! distances need not match (the path-reversal argument already failed).
//!
//! Verdict: dual lookups are CLOSED for this codebase. The only valid "second
//! frame" tricks remain the goal-preserving automorphism σ (identical by
//! construction for WD/cWD) and the R↔GOAL swaps ν/σν (FINDINGS_R §8c).
//!
//! Usage: cargo run --release --example dual_gap [nscrambles]

use puzzle8::puzzle24::state::{Move, State, GOAL};

fn r_board() -> State {
    let mut a = [0u8; 25];
    for i in 1..25u8 {
        a[i as usize] = 25 - i;
    }
    State(a)
}

/// dual(s): x = π_G ∘ s (content → its goal cell, blank → 24), invert, map back.
fn dual(s: &State) -> State {
    let mut x = [0u8; 25];
    for cell in 0..25 {
        let c = s.0[cell];
        x[cell] = if c == 0 { 24 } else { c - 1 };
    }
    let mut xi = [0u8; 25];
    for (cell, &v) in x.iter().enumerate() {
        xi[v as usize] = cell as u8;
    }
    let mut d = [0u8; 25];
    for cell in 0..25 {
        let v = xi[cell];
        d[cell] = if v == 24 { 0 } else { v + 1 };
    }
    State(d)
}

/// Solvability: permutation parity of the normalized state must equal the
/// parity of the blank's taxicab distance from its goal cell (24).
fn solvable(s: &State) -> bool {
    let mut x = [0u8; 25];
    let mut blank = 0usize;
    for cell in 0..25 {
        let c = s.0[cell];
        x[cell] = if c == 0 { 24 } else { c - 1 };
        if c == 0 {
            blank = cell;
        }
    }
    // permutation parity via cycle decomposition
    let mut seen = [false; 25];
    let mut transpositions = 0u32;
    for i in 0..25 {
        if seen[i] {
            continue;
        }
        let mut j = i;
        let mut len = 0u32;
        while !seen[j] {
            seen[j] = true;
            j = x[j] as usize;
            len += 1;
        }
        transpositions += len - 1;
    }
    let taxicab = (blank / 5).abs_diff(4) + (blank % 5).abs_diff(4);
    transpositions % 2 == taxicab as u32 % 2
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn scramble(rng: &mut Rng, steps: u32) -> State {
    let mut s = GOAL;
    let mut prev: Option<Move> = None;
    for _ in 0..steps {
        let ms: Vec<Move> = s
            .legal_moves()
            .iter()
            .filter(|&m| Some(m.inverse()) != prev)
            .collect();
        let m = ms[(rng.next() as usize) % ms.len()];
        s = s.apply(m);
        prev = Some(m);
    }
    s
}

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    // structural sanity: dual is an involution, GOAL and R are self-dual
    let r = r_board();
    assert_eq!(dual(&GOAL), GOAL);
    assert_eq!(dual(&r), r, "R must be self-dual");
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    for _ in 0..1000 {
        let s = scramble(&mut rng, 60);
        assert_eq!(dual(&dual(&s)), s, "dual must be an involution");
    }

    // the 2-move witness: GOAL, then Left, then Up — dual is unsolvable
    let w = GOAL.apply(Move::Left).apply(Move::Up);
    let dw = dual(&w);
    println!(
        "witness (GOAL + L + U): solvable(s) = {}, solvable(dual(s)) = {}",
        solvable(&w),
        solvable(&dw)
    );
    assert!(solvable(&w));
    assert!(
        !solvable(&dw),
        "expected the witness dual to be unsolvable — if this fires, re-derive; do NOT enable dual probes on faith"
    );

    // violation rate over random scrambles
    let mut bad = 0u64;
    for i in 0..n {
        let steps = 10 + (rng.next() % 90) as u32;
        let s = scramble(&mut rng, steps);
        debug_assert!(solvable(&s), "scramble {i} unsolvable?!");
        if !solvable(&dual(&s)) {
            bad += 1;
        }
    }
    println!(
        "dual unsolvable on {bad}/{n} random scrambles ({:.1}%) — dual lookups are UNSOUND for sliding-tile; do not use",
        bad as f64 / n as f64 * 100.0
    );
}
