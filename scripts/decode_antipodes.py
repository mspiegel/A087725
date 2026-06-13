#!/usr/bin/env python3
"""Decode the 17 4x4 antipodes from stannic's 5x5-extended encoding.

Source: http://forum.cubeman.org/?q=node/view/555 (comment "Nodecounts" by
stannic, 04/24/2017). stannic posted the 17 known antipodes of the 4x4
sliding puzzle, embedded in 5x5 form using his 5x5 goal convention (blank
at top-left, tiles 1..24 row-major).

We need them in the standard 4x4 convention: blank at bottom-right, tiles
1..15 row-major. The 4x4 sub-puzzle sits in the upper-left 4x4 of the 5x5
(rows 0-3, cols 0-3); the right column (col 4) and bottom row (row 4) hold
fringe tiles in their goal positions, undisturbed by the 4x4 antipode.

Translation: 180-degree rotation + relabel.

  stannic 4x4 sub-puzzle GOAL (blank at (0,0)):
      0  1  2  3
      5  6  7  8
     10 11 12 13
     15 16 17 18

  Standard 4x4 GOAL (blank at (3,3)):
      1  2  3  4
      5  6  7  8
      9 10 11 12
     13 14 15  0

  Composing 180-rotation with relabel gives:
      0->0, 1->15, 2->14, 3->13, 5->12, 6->11, 7->10, 8->9,
     10->8, 11->7, 12->6, 13->5, 15->4, 16->3, 17->2, 18->1
"""

import sys

# stannic's 17 antipodes as 5x5 instances (row-major; 0 = blank).
RAW = [
    "18 12 10 15 4 13 17 11 16 9 2 7 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 1
    "18 12 10 15 4 13 17 11 16 9 8 2 6 1 14 3 7 5 0 19 20 21 22 23 24",  # 2
    "18 13 10 15 4 17 12 11 16 9 2 7 1 5 14 3 8 6 0 19 20 21 22 23 24",  # 3
    "18 13 10 15 4 17 12 11 16 9 2 7 5 6 14 3 8 1 0 19 20 21 22 23 24",  # 4
    "18 13 10 15 4 17 12 11 16 9 2 7 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 5
    "18 13 10 15 4 17 12 11 16 9 7 8 6 1 14 3 2 5 0 19 20 21 22 23 24",  # 6
    "18 13 10 15 4 17 12 11 16 9 8 2 6 1 14 3 7 5 0 19 20 21 22 23 24",  # 7
    "18 13 10 15 4 17 12 11 16 9 8 7 2 1 14 3 6 5 0 19 20 21 22 23 24",  # 8
    "18 13 10 15 4 17 12 16 11 9 2 8 6 1 14 3 7 5 0 19 20 21 22 23 24",  # 9
    "18 13 11 15 4 17 12 10 16 9 7 2 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 10
    "18 13 11 15 4 17 12 16 10 9 2 7 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 11
    "18 13 11 15 4 17 12 16 10 9 7 8 6 1 14 3 2 5 0 19 20 21 22 23 24",  # 12
    "18 13 16 15 4 17 12 10 11 9 2 7 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 13
    "18 13 16 15 4 17 12 10 11 9 8 2 6 1 14 3 7 5 0 19 20 21 22 23 24",  # 14
    "18 13 16 15 4 17 12 11 6 9 2 7 10 1 14 3 8 5 0 19 20 21 22 23 24",  # 15
    "18 17 10 15 4 12 13 11 16 9 2 7 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 16
    "18 17 16 15 4 12 13 10 11 9 2 7 6 1 14 3 8 5 0 19 20 21 22 23 24",  # 17
]

# stannic 5x5 tile -> standard 4x4 tile (only defined for tiles appearing in
# the upper-left 4x4 sub-region).
RELABEL = {
    0: 0, 1: 15, 2: 14, 3: 13,
    5: 12, 6: 11, 7: 10, 8: 9,
    10: 8, 11: 7, 12: 6, 13: 5,
    15: 4, 16: 3, 17: 2, 18: 1,
}


def decode(raw: str, idx: int) -> list[int]:
    cells5 = [int(x) for x in raw.split()]
    assert len(cells5) == 25, f"#{idx}: expected 25 cells, got {len(cells5)}"

    # Extract upper-left 4x4 of the 5x5 (rows 0-3, cols 0-3).
    sub4x4 = []
    for r in range(4):
        for c in range(4):
            sub4x4.append(cells5[r * 5 + c])
    assert len(sub4x4) == 16

    # Sanity: the right column (col 4) and bottom row (row 4) of the 5x5
    # must be at their goal positions.
    goal_5x5 = list(range(25))
    for r in range(5):
        assert cells5[r * 5 + 4] == goal_5x5[r * 5 + 4], \
            f"#{idx}: col 4 disturbed at row {r}"
    for c in range(5):
        assert cells5[4 * 5 + c] == goal_5x5[4 * 5 + c], \
            f"#{idx}: row 4 disturbed at col {c}"

    # Apply 180-degree rotation: new[(r,c)] = old[(3-r, 3-c)].
    rotated = [0] * 16
    for r in range(4):
        for c in range(4):
            rotated[r * 4 + c] = sub4x4[(3 - r) * 4 + (3 - c)]

    # Relabel from stannic-5x5 tile values to standard-4x4 tile values.
    relabeled = []
    for v in rotated:
        if v not in RELABEL:
            raise ValueError(f"#{idx}: tile {v} has no relabeling")
        relabeled.append(RELABEL[v])

    # Final sanity: tiles {0..15} all present.
    assert sorted(relabeled) == list(range(16)), \
        f"#{idx}: relabeled board missing tiles: {sorted(relabeled)}"
    return relabeled


def main() -> int:
    print("# 15-puzzle antipodes (depth 80) decoded from stannic's 5x5-extended encoding.")
    print("# Source: http://forum.cubeman.org/?q=node/view/555 (comment by stannic, 2017-04-24)")
    print("# Each line is a 16-token row-major 4x4 board: tiles 1..15 plus '_' for the blank.")
    print("# Standard 4x4 goal convention (blank at position 15, bottom-right).")
    print()
    for idx, raw in enumerate(RAW, start=1):
        cells = decode(raw, idx)
        tokens = [("_" if v == 0 else str(v)) for v in cells]
        print(" ".join(tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
