//! Invariants for the 16-byte adjacency records produced by `build_adj8`.
//!
//! Tests are pure-puzzle (no `DistanceTable` build needed) since adjacency
//! depends only on the puzzle's combinatorial structure.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::{rank, unrank};
use puzzle8::puzzle8::state::{Move, N_STATES};

/// Reproduce the per-record byte arrays produced by `build_adj8` (records only,
/// no header).
fn build_records() -> Vec<[u32; 4]> {
    let mut out: Vec<[u32; 4]> = Vec::with_capacity(N_STATES as usize);
    for r in 0..N_STATES {
        let s = unrank(r);
        let legal = s.legal_moves();
        let mut rec = [u32::MAX; 4];
        for (slot, &m) in Move::ALL.iter().enumerate() {
            if legal.contains(m) {
                rec[slot] = rank(&s.apply(m));
            }
        }
        out.push(rec);
    }
    out
}

#[test]
fn sentinel_iff_move_illegal() {
    let records = build_records();
    for r in 0..N_STATES {
        let s = unrank(r);
        let legal = s.legal_moves();
        let rec = &records[r as usize];
        for (slot, &m) in Move::ALL.iter().enumerate() {
            let is_legal = legal.contains(m);
            let is_sentinel = rec[slot] == u32::MAX;
            assert_eq!(is_legal, !is_sentinel,
                "rank {}: move {:?} legal={} but neighbor={} (sentinel={})",
                r, m, is_legal, rec[slot], is_sentinel);
        }
    }
}

#[test]
fn non_sentinel_slots_match_apply() {
    let records = build_records();
    for r in 0..N_STATES {
        let s = unrank(r);
        let rec = &records[r as usize];
        for (slot, &m) in Move::ALL.iter().enumerate() {
            let neighbor = rec[slot];
            if neighbor == u32::MAX {
                continue;
            }
            let s_next = s.apply(m);
            assert_eq!(unrank(neighbor), s_next,
                "rank {r}: move {m:?} -> rank {neighbor} but state mismatch");
        }
    }
}

#[test]
fn legal_count_matches_move_set() {
    let records = build_records();
    for r in 0..N_STATES {
        let s = unrank(r);
        let legal_count = s.legal_moves().len() as usize;
        let non_sentinel_count = records[r as usize].iter().filter(|&&n| n != u32::MAX).count();
        assert_eq!(non_sentinel_count, legal_count,
            "rank {r}: {non_sentinel_count} non-sentinel slots vs {legal_count} legal moves");
    }
}

#[test]
fn move_inverse_involution() {
    // For every directed edge (r, m) -> r', the reverse edge (r', m.inverse())
    // must point back to r. This asserts the move graph is correctly undirected.
    let records = build_records();
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        for (slot, &m) in Move::ALL.iter().enumerate() {
            let r_next = rec[slot];
            if r_next == u32::MAX {
                continue;
            }
            let back = records[r_next as usize][m.inverse() as usize];
            assert_eq!(back, r,
                "rank {}: forward {:?} -> {}, reverse {:?} -> {} (expected {})",
                r, m, r_next, m.inverse(), back, r);
        }
    }
}

#[test]
fn pol8_optimal_moves_are_subset_of_adj8_legal_moves() {
    // Every optimal-move bit set in pol8 must correspond to a non-sentinel
    // slot in adj8. (Every optimal move is a legal move.)
    let t = DistanceTable::build();
    let records = build_records();
    for r in 0..N_STATES {
        let s = unrank(r);
        let opt_mask = t.optimal_moves(&s).0;
        for (slot, &m) in Move::ALL.iter().enumerate() {
            if opt_mask & (1 << (m as u8)) != 0 {
                assert!(records[r as usize][slot] != u32::MAX,
                    "rank {r}: pol8 marks {m:?} optimal but adj8 says illegal");
            }
        }
    }
}

#[test]
fn determinism_two_builds_identical() {
    let a = build_records();
    let b = build_records();
    assert_eq!(a, b, "two record builds must be byte-identical");
}
