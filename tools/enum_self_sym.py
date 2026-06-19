#!/usr/bin/env python3
"""Enumerate every self-symmetric (reflect(B) == B) solvable 15-puzzle board.

Output: 6-byte little-endian u48 ranks to stdout (the standard .ranks format
consumed by enumerate15 --seed-ranks).

Decomposition:
- Diagonal positions {0, 5, 10, 15} must hold TAU-fixed tiles {0, 1, 6, 11}.
- The 6 off-diagonal SIGMA pairs each hold a TAU 2-orbit of tiles, with a
  choice of orientation.
"""

import itertools
import struct
import sys

W = 4
N_CELLS = 16
N_TILES = 15
EVEN_BLOCK = 653_837_184_000

SIGMA = [(p % W) * W + (p // W) for p in range(N_CELLS)]
def goal_at(p):
    return 0 if p == N_CELLS - 1 else p + 1
TAU = [0] * N_CELLS
for k in range(1, N_CELLS):
    TAU[k] = goal_at(SIGMA[k - 1])

DIAGONAL = [p for p in range(N_CELLS) if SIGMA[p] == p]      # {0,5,10,15}
FIXED_TILES = [t for t in range(N_CELLS) if TAU[t] == t]     # {0,1,6,11}
OFF_PAIRS = sorted({tuple(sorted((p, SIGMA[p]))) for p in range(N_CELLS) if SIGMA[p] != p})
TWO_ORBITS = []
seen = set()
for t in range(N_CELLS):
    if t in FIXED_TILES or t in seen:
        continue
    TWO_ORBITS.append((t, TAU[t]))
    seen.add(t); seen.add(TAU[t])

assert len(DIAGONAL) == 4 and len(FIXED_TILES) == 4
assert len(OFF_PAIRS) == 6 and len(TWO_ORBITS) == 6, (OFF_PAIRS, TWO_ORBITS)

def rank_board(b):
    blank_pos = b.index(0)
    tiles = [v for v in b if v != 0]
    available = (1 << N_CELLS) - 2  # bits 1..15
    even_rank = 0
    for i in range(N_TILES - 2):
        tile = tiles[i]
        lo_mask = (1 << tile) - 1
        d = bin(available & lo_mask).count('1')
        radix = N_TILES - i
        even_rank = even_rank * radix + d
        available &= ~(1 << tile)
    return blank_pos * EVEN_BLOCK + even_rank

def is_solvable(b):
    blank_pos = b.index(0)
    blank_row_from_bottom = (W - 1) - blank_pos // W
    # inversions over non-blank tile sequence
    inv = 0
    nb = [v for v in b if v != 0]
    for i in range(len(nb)):
        for j in range(i + 1, len(nb)):
            if nb[i] > nb[j]:
                inv += 1
    return (inv + blank_row_from_bottom) % 2 == 0

def main():
    out = set()
    total = 0
    solvable = 0
    for diag_perm in itertools.permutations(FIXED_TILES):
        # Assign each diagonal cell to a fixed-tile.
        b_template = [None] * N_CELLS
        for cell, tile in zip(DIAGONAL, diag_perm):
            b_template[cell] = tile
        for pair_perm in itertools.permutations(TWO_ORBITS):
            # pair_perm[i] = 2-orbit assigned to OFF_PAIRS[i]
            for orient_mask in range(1 << len(OFF_PAIRS)):
                b = b_template[:]
                for i, (p, q) in enumerate(OFF_PAIRS):
                    t, tau_t = pair_perm[i]
                    if (orient_mask >> i) & 1:
                        b[p], b[q] = tau_t, t
                    else:
                        b[p], b[q] = t, tau_t
                total += 1
                if not is_solvable(b):
                    continue
                solvable += 1
                out.add(rank_board(b))
    sys.stderr.write(
        f"enumerated {total} self-symmetric boards "
        f"({solvable} solvable, {len(out)} distinct ranks)\n"
    )
    # Emit sorted, 6-byte LE.
    buf = bytearray()
    for r in sorted(out):
        buf.extend(r.to_bytes(8, 'little')[:6])
    sys.stdout.buffer.write(buf)

if __name__ == "__main__":
    main()
