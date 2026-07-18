//! Invariants for the 32-byte feature records produced by `build_feat8`.
//!
//! Tests rebuild [`DistanceTable`] in memory so they don't depend on
//! `data/dist8.bin`.

use puzzle8::puzzle8::bfs::DistanceTable;
use puzzle8::puzzle8::rank::{rank, unrank};
use puzzle8::puzzle8::state::N_STATES;
use puzzle8::puzzle8::symmetry::reflect;

const RECORD_SIZE: usize = 32;

#[inline]
fn tile_manhattan(t: u8, p: usize) -> u8 {
    let goal_pos = (t - 1) as usize;
    let cur_row = (p / 3) as i32;
    let cur_col = (p % 3) as i32;
    let goal_row = (goal_pos / 3) as i32;
    let goal_col = (goal_pos % 3) as i32;
    ((cur_row - goal_row).unsigned_abs() + (cur_col - goal_col).unsigned_abs()) as u8
}

/// Reproduce the byte buffer that `build_feat8` writes per record (records
/// only, no header).
fn build_records(table: &DistanceTable) -> Vec<[u8; RECORD_SIZE]> {
    let mut out: Vec<[u8; RECORD_SIZE]> = Vec::with_capacity(N_STATES as usize);
    for r in 0..N_STATES {
        let s = unrank(r);
        let mut rec = [0u8; RECORD_SIZE];
        rec[0..9].copy_from_slice(&s.0);
        rec[9] = s.blank_pos();
        rec[10] = table.dist_of_rank(r);

        let mut manhattan_sum: u32 = 0;
        let mut correct_count: u8 = 0;
        let mut correct_mask: u8 = 0;
        for t in 1u8..=8u8 {
            let mut pos = 9;
            for (i, &v) in s.0.iter().enumerate() {
                if v == t {
                    pos = i;
                    break;
                }
            }
            assert!(pos < 9);
            let m = tile_manhattan(t, pos);
            rec[24 + (t as usize - 1)] = m;
            manhattan_sum += m as u32;
            if pos == (t as usize - 1) {
                correct_count += 1;
                correct_mask |= 1 << (t - 1);
            }
        }
        rec[11] = manhattan_sum as u8;
        rec[13] = correct_count;
        rec[14] = correct_mask;
        rec[12] = s.inversions() as u8;
        rec[15] = table.optimal_moves(&s).len() as u8;
        let s_refl = reflect(&s);
        rec[16] = (s_refl == s) as u8;
        let r_refl = rank(&s_refl);
        let canonical = r.min(r_refl);
        rec[20..24].copy_from_slice(&canonical.to_le_bytes());

        out.push(rec);
    }
    out
}

#[test]
fn distance_field_matches_table() {
    let t = DistanceTable::build();
    let records = build_records(&t);
    for r in 0..N_STATES {
        assert_eq!(records[r as usize][10], t.dist_of_rank(r),
            "rank {r}: feat8 distance doesn't match table");
    }
}

#[test]
fn board_field_matches_unrank() {
    let t = DistanceTable::build();
    let records = build_records(&t);
    for r in 0..N_STATES {
        let s = unrank(r);
        assert_eq!(&records[r as usize][0..9], &s.0,
            "rank {r}: feat8 board doesn't match unrank({r})");
    }
}

#[test]
fn blank_pos_matches_board() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        let blank_pos = rec[9] as usize;
        assert!(blank_pos < 9);
        assert_eq!(rec[blank_pos], 0,
            "rank {}: blank_pos {} doesn't index a 0 in the board {:?}",
            r, blank_pos, &rec[0..9]);
    }
}

#[test]
fn manhattan_sum_equals_per_tile_sum() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        let sum: u32 = (0..8).map(|i| rec[24 + i] as u32).sum();
        assert_eq!(rec[11] as u32, sum,
            "rank {}: manhattan_sum {} != sum of per-tile {}", r, rec[11], sum);
    }
}

#[test]
fn correct_tile_count_matches_mask_popcount() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        let popcount = rec[14].count_ones() as u8;
        assert_eq!(rec[13], popcount,
            "rank {}: correct_tile_count {} != popcount(correct_tile_mask 0b{:08b}) = {}",
            r, rec[13], rec[14], popcount);
    }
}

#[test]
fn correct_tile_mask_matches_board_placements() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        let board = &rec[0..9];
        for t in 1u8..=8u8 {
            let bit = (rec[14] >> (t - 1)) & 1;
            let at_goal = board[(t - 1) as usize] == t;
            assert_eq!(bit == 1, at_goal,
                "rank {r}: tile {t} bit={bit} but at_goal={at_goal}");
        }
    }
}

#[test]
fn num_optimal_moves_matches_table() {
    let t = DistanceTable::build();
    let records = build_records(&t);
    for r in 0..N_STATES {
        let s = unrank(r);
        assert_eq!(records[r as usize][15], t.optimal_moves(&s).len() as u8,
            "rank {r}: num_optimal_moves disagrees with table");
    }
}

#[test]
fn reflected_canonical_rank_le_r() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        let canonical = u32::from_le_bytes(rec[20..24].try_into().unwrap());
        assert!(canonical <= r,
            "rank {r}: reflected_canonical_rank {canonical} > r");
    }
}

#[test]
fn self_symmetric_flag_iff_state_equals_its_reflection() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        let s = unrank(r);
        let is_self = reflect(&s) == s;
        let flag = rec[16];
        assert!(flag == 0 || flag == 1, "rank {r}: self_symmetric_flag must be 0 or 1");
        assert_eq!(flag == 1, is_self,
            "rank {r}: flag={flag} but reflect==self? {is_self}");
    }
}

#[test]
fn padding_bytes_are_zero() {
    let records = build_records(&DistanceTable::build());
    for r in 0..N_STATES {
        let rec = &records[r as usize];
        for offset in 17..20 {
            assert_eq!(rec[offset], 0,
                "rank {}: padding byte at offset {} not zero ({})", r, offset, rec[offset]);
        }
    }
}

#[test]
fn cross_consistency_with_pol8_optimal_move_count() {
    // num_optimal_moves field must equal popcount of pol8's MoveSet bitmask
    // for the same rank (both derived from optimal_moves).
    let t = DistanceTable::build();
    let records = build_records(&t);
    for r in 0..N_STATES {
        let s = unrank(r);
        let mask = t.optimal_moves(&s).0;
        assert_eq!(records[r as usize][15], mask.count_ones() as u8,
            "rank {r}: num_optimal_moves != popcount(pol8 mask)");
    }
}

#[test]
fn determinism_two_builds_identical() {
    let t = DistanceTable::build();
    let a = build_records(&t);
    let b = build_records(&t);
    assert_eq!(a, b, "two record builds must be byte-identical");
}
