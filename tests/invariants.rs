//! Integration tests verifying the 8-puzzle's published invariants.
//!
//! Sources:
//!   - Reinefeld 1993, "Complete Solution of the Eight-Puzzle and the Benefit
//!     of Node Ordering in IDA*", IJCAI-93.
//!   - OEIS A087725 commentary by Dan Hoey.

use puzzle8::bfs::DistanceTable;
use puzzle8::state::{State, DIAMETER, GOAL, N_STATES};

#[test]
fn diameter_count_and_antipodes_match_reinefeld() {
    let t = DistanceTable::build();

    assert_eq!(t.visited_count(), N_STATES);
    assert_eq!(t.diameter(), DIAMETER);
    assert_eq!(t.dist(&GOAL), 0);

    let antipodes = t.antipodes();
    let arr: Vec<[u8; 9]> = antipodes.iter().map(|s| s.0).collect();
    assert_eq!(arr.len(), 2);
    assert!(arr.contains(&[8, 6, 7, 2, 5, 4, 3, 0, 1]));
    assert!(arr.contains(&[6, 4, 7, 8, 5, 0, 3, 2, 1]));
}

#[test]
fn distance_histogram_matches_published() {
    // Reinefeld 1993, Table 1 (goal blank in a corner). Verified
    // independently by every modern 8-puzzle solver.
    const EXPECTED: [u32; 32] = [
        1,     // d= 0
        2,     // d= 1
        4,     // d= 2
        8,     // d= 3
        16,    // d= 4
        20,    // d= 5
        39,    // d= 6
        62,    // d= 7
        116,   // d= 8
        152,   // d= 9
        286,   // d=10
        396,   // d=11
        748,   // d=12
        1024,  // d=13
        1893,  // d=14
        2512,  // d=15
        4485,  // d=16
        5638,  // d=17
        9529,  // d=18
        10878, // d=19
        16993, // d=20
        17110, // d=21
        23952, // d=22
        20224, // d=23
        24047, // d=24
        15578, // d=25
        14560, // d=26
        6274,  // d=27
        3910,  // d=28
        760,   // d=29
        221,   // d=30
        2,     // d=31
    ];
    let t = DistanceTable::build();
    let hist = t.histogram();
    assert_eq!(hist.len(), EXPECTED.len());
    for (d, (&got, &want)) in hist.iter().zip(EXPECTED.iter()).enumerate() {
        assert_eq!(got, want, "histogram mismatch at d={}: got {}, want {}", d, got, want);
    }
    let total: u32 = hist.iter().sum();
    assert_eq!(total, N_STATES);
}

#[test]
fn antipodes_share_odd_tile_skeleton() {
    // Folklore observation: the two 8-puzzle antipodes hold odd-numbered
    // tiles (1, 3, 5, 7) at the same positions. This is the structural
    // generalization we discussed across antipode sets.
    let a1 = State([8, 6, 7, 2, 5, 4, 3, 0, 1]);
    let a2 = State([6, 4, 7, 8, 5, 0, 3, 2, 1]);
    for tile in [1u8, 3, 5, 7] {
        let p1 = a1.0.iter().position(|&t| t == tile).unwrap();
        let p2 = a2.0.iter().position(|&t| t == tile).unwrap();
        assert_eq!(p1, p2, "tile {} differs: a1 at {}, a2 at {}", tile, p1, p2);
    }
}
