//! gen_flat_oracle — freeze the recursive engine's answers as a committed fixture.
//!
//! The flat engine's differential gate compared it against the recursive engine
//! node-for-node. The recursive engine is being deleted, so its answers are
//! captured here **once**, while it still exists, and the gate is rewritten to
//! compare against these frozen values instead.
//!
//! This is a one-shot tool. Once `idastar.rs`'s search drivers are gone it can no
//! longer run, which is precisely why its output is committed source rather than
//! something regenerated on demand.
//!
//!   cargo run --release --example gen_flat_oracle > src/puzzle24/search/flat_oracle.rs
//!
//! Boards and bounds are emitted as **literals** so the fixture stands alone: it
//! must not depend on `scramble`, nor on the flat engine's own root `h`, staying
//! the same — a regression in either could otherwise shift the grid and hide
//! itself.
//!
//! # Sizing the grid
//!
//! The differential test this replaces searched **1,570 nodes in total**, which
//! is not a tree-shape check at all. Two properties of cWD make a naive grid
//! vacuous, and both had to be measured before the grid could be sized (see
//! `examples/probe_oracle_difficulty.rs`):
//!
//! 1. **Parity.** cWD shares the true distance's parity, so `f` only takes
//!    `h0, h0+2, h0+4, …`. A bound of `h0+1` is the *same search* as `h0`.
//! 2. **`bound = h0` is a narrow corridor.** At the root `f = h0`; every child
//!    has `h = h0 ± 1`, so `f` is `h0` or `h0+2` and the latter is pruned
//!    immediately. Only strictly-`h`-decreasing paths survive — tens of nodes.
//!
//! Also, cWD is *exact* on short no-immediate-undo random walks (`h0 = d`), so
//! such boards solve on the first threshold instead of exhausting. Real trees
//! need long walks *and* `h0 + 2k` for `k ≥ 2`; growth is ~8-12× per step of 2.
//!
//! The emitted grid is ~3.7 × 10^7 nodes over 180 cases.

use puzzle8::puzzle24::search::cwd::Cwd;
use puzzle8::puzzle24::search::flat::flat_bounded;
use puzzle8::puzzle24::search::idastar::{BoundedOutcome, LadderOutcome, Search};
use puzzle8::puzzle24::search::move_dfa::MoveDfa;
use puzzle8::puzzle24::state::{Move, State, GOAL};

/// Deterministic scramble by a random walk with no immediate undo. Copied
/// verbatim from `flat.rs`'s test module — the grid must not move.
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

fn board_lit(s: &State) -> String {
    let cells: Vec<String> = s.0.iter().map(|c| c.to_string()).collect();
    format!("[{}]", cells.join(", "))
}

/// `(tag, val)` — 0 = `Solved(len)`, 1 = `ProvedAtLeast(k)`, 2 = `Unsolvable`.
fn encode(o: &LadderOutcome) -> (u8, u8) {
    match o {
        LadderOutcome::Solved(mv) => (0, mv.len() as u8),
        LadderOutcome::ProvedAtLeast(k) => (1, *k),
        LadderOutcome::Unsolvable => (2, 0),
        LadderOutcome::TimedOut(_) => panic!("no deadline was set; TimedOut is impossible"),
    }
}

fn main() {
    let cwd = Cwd::new().with_neighbor_prune(true);
    assert!(
        cwd.has_overlay(),
        "needs data/wd24.bin + data/cwd_single.bin"
    );
    let dfa = MoveDfa::build_default();

    let mut rows: Vec<String> = Vec::new();
    let mut total_nodes: u64 = 0;

    let case =
        |rows: &mut Vec<String>, total: &mut u64, s: &State, bound: u8, orbit: bool, what: &str| {
            let (ro, rs) = Search::new(s, &cwd)
                .bound(bound)
                .pruner(&dfa)
                .orbit_split(orbit)
                .run();
            let (tag, val) = encode(&ro);
            *total += rs.nodes;
            eprintln!(
                "{what} bound={bound} orbit={orbit} -> tag={tag} val={val} nodes={} iters={}",
                rs.nodes, rs.iterations
            );
            rows.push(format!(
                "    OracleCase {{ board: {}, bound: {}, orbit_split: {}, \
             tag: {}, val: {}, nodes: {}, iterations: {} }},",
                board_lit(s),
                bound,
                orbit,
                tag,
                val,
                rs.nodes,
                rs.iterations
            ));
        };

    // ---- tier 1: shallow grid, cheap, mostly `Solved` on the first threshold --
    // Kept because it pins the easy path, but it exhausts almost nothing: with a
    // no-immediate-undo walk the optimum equals the walk length, so `bound = h0`
    // solves at once. Tier 2 is what actually exercises the tree.
    for seed in 0..8u64 {
        for steps in [4u32, 9, 14, 19] {
            let s = scramble(seed, steps);
            // Root h only selects which bounds to probe; the value emitted below
            // is a literal, so the fixture does not depend on this call.
            let h0 = match flat_bounded(&s, &cwd, &dfa, false, 0).0 {
                BoundedOutcome::ProvedAtLeast(k) => k,
                other => panic!("unexpected root outcome {other:?}"),
            };
            // h0-2 exercises the zero-node path (bound below the root h returns
            // ProvedAtLeast immediately); h0 and h0+2 solve, since cWD is exact
            // on these. Steps of 2 because of the parity below.
            for b in [h0.saturating_sub(2), h0, h0 + 2] {
                case(
                    &mut rows,
                    &mut total_nodes,
                    &s,
                    b,
                    false,
                    &format!("t1 seed={seed} steps={steps}"),
                );
            }
        }
    }

    // ---- tier 2: long walks, exhausting bounds well above the root h ----------
    // This is the tier that actually constrains tree shape. Two facts set it up,
    // both measured (examples/probe_oracle_difficulty.rs):
    //
    // 1. **Parity.** cWD shares the true distance's parity, so f only takes
    //    h0, h0+2, h0+4, ... A bound of h0+1 is indistinguishable from h0.
    // 2. **`bound = h0` is nearly free.** At the root f = h0; each child has
    //    h = h0 ± 1, so f is h0 or h0+2 and the latter is pruned at once. Only
    //    strictly-h-decreasing paths survive — a narrow corridor, tens of nodes.
    //
    // Real trees therefore need h0 + 2k for k ≥ 2. Measured growth is ~8-12x per
    // step of 2, so h0+{4,6,8} spans roughly 10^3 to 10^6 nodes per board. Short
    // walks are useless here (cWD is exact on them, so the search solves instead
    // of exhausting); 200-700 steps opens real slack between h0 and d.
    const BUDGET: u64 = 5_000_000;
    for seed in 0..8u64 {
        for steps in [200u32, 400, 700] {
            let s = scramble(seed, steps);
            let h0 = match flat_bounded(&s, &cwd, &dfa, false, 0).0 {
                BoundedOutcome::ProvedAtLeast(k) => k,
                other => panic!("unexpected root outcome {other:?}"),
            };
            for k in [4u8, 6, 8] {
                let before = total_nodes;
                case(
                    &mut rows,
                    &mut total_nodes,
                    &s,
                    h0 + k,
                    false,
                    &format!("t2 seed={seed} steps={steps} h0={h0}"),
                );
                // Stop escalating this board once a case gets expensive; the next
                // one would be ~10x larger and the test has to run it too.
                if total_nodes - before > BUDGET {
                    break;
                }
            }
        }
    }

    // ---- the σ-symmetric orbit-split fixture ---------------------------------
    let fix = State([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 14, 15, 16, 17, 18, 13, 20, 21, 22, 23, 24, 19,
    ]);
    // Both settings at each bound: the split's node count is only meaningful
    // against the unsplit count at the same bound, and the flat test asserts the
    // split strictly reduces nodes.
    for bound in [10u8, 12, 14, 16, 18, 20] {
        for orbit in [true, false] {
            case(
                &mut rows,
                &mut total_nodes,
                &fix,
                bound,
                orbit,
                "t3 orbit-fixture",
            );
        }
    }

    println!(
        "//! Frozen answers from the deleted recursive IDA\\* engine.\n\
         //!\n\
         //! Generated once by `examples/gen_flat_oracle.rs` against the recursive\n\
         //! engine before it was removed, and committed as source: it **cannot be\n\
         //! regenerated**, because the engine that produced it no longer exists.\n\
         //! Treat every value here as ground truth, not as something to refresh if a\n\
         //! test starts failing — a mismatch means the flat engine changed its tree.\n\
         //!\n\
         //! Node counts are the load-bearing column. Outcomes alone would let a\n\
         //! tree-shape divergence through that still happens to reach the same\n\
         //! bound; node counts will not.\n\
         \n\
         /// One frozen `(board, bound, orbit_split)` -> `(outcome, nodes, iterations)`.\n\
         #[derive(Clone, Copy)]\n\
         pub(crate) struct OracleCase {{\n\
         \x20   pub board: [u8; 25],\n\
         \x20   pub bound: u8,\n\
         \x20   pub orbit_split: bool,\n\
         \x20   /// 0 = `Solved` with length `val`, 1 = `ProvedAtLeast(val)`, 2 = `Unsolvable`.\n\
         \x20   pub tag: u8,\n\
         \x20   pub val: u8,\n\
         \x20   pub nodes: u64,\n\
         \x20   pub iterations: u32,\n\
         }}\n\
         \n\
         pub(crate) const CASES: &[OracleCase] = &[\n{}\n];",
        rows.join("\n")
    );
    eprintln!("emitted {} cases, {} total nodes", rows.len(), total_nodes);
}
