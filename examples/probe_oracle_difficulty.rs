//! probe_oracle_difficulty — find the walk length where cWD stops being exact.
//!
//! Throwaway calibration for `gen_flat_oracle`. Short no-immediate-undo walks
//! give boards where cWD is exact (h0 = d), so `bound = h0` *solves* in ~d nodes
//! and nothing exhausts — useless as a tree-shape oracle. Uniformly random states
//! are the opposite problem: h0 ≈ 85-95 and exhausting `bound = h0` could be 10^11
//! nodes.
//!
//! This walks the length up and prints, per board, the root h and the node count
//! the flat engine needs to exhaust `bound = h0`. Escalation stops as soon as a
//! board crosses the ceiling, so no single probe can run away.

use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::flat::flat_bounded;
use puzzle8::puzzle24::search::idastar::BoundedOutcome;
use puzzle8::puzzle24::search::move_dfa::MoveDfa;
use puzzle8::puzzle24::state::{Move, State, GOAL};

fn scramble(seed: u64, steps: u32) -> State {
    let mut s = GOAL;
    let mut blank = s.blank_pos();
    let mut last: Option<Move> = None;
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for _ in 0..steps {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let legal: Vec<Move> = State::legal_moves_at(blank)
            .iter()
            .filter(|&m| last.is_none_or(|p| m != p.inverse()))
            .collect();
        let m = legal[(x % legal.len() as u64) as usize];
        let (next, nb) = s.apply_at(m, blank);
        s = next;
        blank = nb;
        last = Some(m);
    }
    s
}

fn main() {
    let cwd = Cwd::new().with_neighbor_prune(true);
    assert!(cwd.has_overlay(), "needs the cWD tables");
    let dfa = MoveDfa::build_default();

    // Stop escalating a seed once exhausting bound = h0 costs more than this.
    const CEILING: u64 = 8_000_000;

    // cWD shares the true distance's parity, so f only takes h0, h0+2, h0+4, ...
    // Odd increments are duplicates of the even one below them; escalate by 2.
    for seed in 0..4u64 {
        for steps in [200u32, 400, 700] {
            let s = scramble(seed, steps);
            let h0 = match flat_bounded(&s, &cwd, &dfa, false, 0).0 {
                BoundedOutcome::ProvedAtLeast(k) => k,
                other => panic!("unexpected {other:?}"),
            };
            for k in 0..6u8 {
                let bound = h0 + 2 * k;
                let t = std::time::Instant::now();
                let (oc, st) = flat_bounded(&s, &cwd, &dfa, false, bound);
                let tag = match oc {
                    BoundedOutcome::Solved(ref m) => format!("Solved({})", m.len()),
                    BoundedOutcome::ProvedAtLeast(k) => format!("ProvedAtLeast({k})"),
                    BoundedOutcome::Unsolvable => "Unsolvable".into(),
                    BoundedOutcome::BudgetExhausted(t) => format!("BudgetExhausted({t})"),
                };
                println!(
                    "seed={seed} steps={steps:4} h0={h0:3} bound=h0+{:<2} -> {tag:20} \
                     nodes={:>12} iters={:2} {:?}",
                    2 * k,
                    st.nodes,
                    st.iterations,
                    t.elapsed()
                );
                if st.nodes > CEILING || matches!(oc, BoundedOutcome::Solved(_)) {
                    println!("  (stop: ceiling crossed or solved)");
                    break;
                }
            }
        }
    }
}
