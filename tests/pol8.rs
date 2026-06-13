//! Invariants for the optimal-policy payload produced by `build_pol8`.
//!
//! Tests rebuild [`DistanceTable`] in memory (matching the convention in
//! `tests/roundtrip.rs`) so they don't depend on `data/dist8.bin` being
//! present.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::{rank, unrank};
use puzzle8::puzzle8::state::{GOAL, N_STATES};

/// Reproduce the exact byte buffer that `build_pol8` writes (just the payload,
/// not the magic + version header).
fn build_payload(table: &DistanceTable) -> Vec<u8> {
    let mut payload = vec![0u8; N_STATES as usize];
    for r in 0..N_STATES {
        let s = unrank(r);
        payload[r as usize] = table.optimal_moves(&s).0;
    }
    payload
}

#[test]
fn goal_has_zero_mask() {
    let t = DistanceTable::build();
    let payload = build_payload(&t);
    assert_eq!(payload[rank(&GOAL) as usize], 0, "GOAL must have empty optimal-move mask");
}

#[test]
fn every_non_goal_has_at_least_one_optimal_move() {
    let t = DistanceTable::build();
    let payload = build_payload(&t);
    let goal_rank = rank(&GOAL);
    for r in 0..N_STATES {
        if r == goal_rank {
            continue;
        }
        assert!(payload[r as usize] != 0,
            "rank {} (state {:?}) has empty optimal-move mask", r, unrank(r).0);
    }
}

#[test]
fn no_illegal_bits_set() {
    let t = DistanceTable::build();
    let payload = build_payload(&t);
    for r in 0..N_STATES {
        let s = unrank(r);
        let legal_mask = s.legal_moves().0;
        let mask = payload[r as usize];
        assert_eq!(mask & !legal_mask, 0,
            "rank {} (state {:?}): mask 0b{:04b} has illegal bits beyond legal 0b{:04b}",
            r, s.0, mask, legal_mask);
        // Also: high 4 bits always zero (MoveSet layout uses only low 4 bits).
        assert_eq!(mask & 0xF0, 0, "rank {}: high 4 bits should be zero (got 0b{:08b})", r, mask);
    }
}

#[test]
fn every_set_bit_strictly_decreases_distance() {
    let t = DistanceTable::build();
    let payload = build_payload(&t);
    for r in 0..N_STATES {
        let s = unrank(r);
        let d = t.dist(&s);
        if d == 0 {
            continue;
        }
        let mask = payload[r as usize];
        for m in s.legal_moves().iter() {
            let bit = 1u8 << (m as u8);
            if mask & bit == 0 {
                continue;
            }
            let d_next = t.dist(&s.apply(m));
            assert_eq!(d_next, d - 1,
                "rank {} (state {:?}, d={}): optimal move {:?} -> d_next={}",
                r, s.0, d, m, d_next);
        }
    }
}

#[test]
fn determinism_two_builds_identical() {
    let t = DistanceTable::build();
    let a = build_payload(&t);
    let b = build_payload(&t);
    assert_eq!(a, b, "two payload builds must be byte-identical");
}
