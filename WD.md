# Walking Distance, explained

An intuition-level account of the Walking Distance heuristic (Ken'ichiro
Takahashi) for the 24-puzzle, using Greek and Latin letters instead of tile
numbers. Implementation lives in `src/puzzle24/search/walking_distance.rs`.

## Two alphabets

Give every tile two names. Its **Greek** name is the row it belongs to; its
**Latin** name is the column it belongs to.

```
goal board          Greek names          Latin names
 1  2  3  4  5      α α α α α            A B C D E
 6  7  8  9 10      β β β β β            A B C D E
11 12 13 14 15  →   γ γ γ γ γ      and   A B C D E
16 17 18 19 20      δ δ δ δ δ            A B C D E
21 22 23 24  _      ε ε ε ε _            A B C D _
```

So tile 13 is `γ` and `C`; tile 20 is `δ` and `E`. There are five of each Greek
letter except `ε` (only four — the blank eats a slot in row 4), and likewise
five of each Latin letter except `E`.

## The Greek puzzle

Play the puzzle with only the Greek names visible. The rule change that matters:
**horizontal moves are free.** They cost nothing.

Once sliding sideways is free, position within a row is meaningless — a row can
be shuffled into any order for zero. So a row isn't a sequence of letters, it's
a **bag** of letters. The whole board is five bags plus "which bag holds the
blank":

```
row 0: {α α α α α}
row 1: {β β β β β}
row 2: {γ γ γ γ γ}     ← the Greek goal
row 3: {δ δ δ δ δ}
row 4: {ε ε ε ε} + blank
```

A (paid) move is: the blank is in bag `r`; pick an adjacent bag; move one letter
from it into bag `r`; the blank goes the other way. That's the entire abstract
puzzle — five bags, one blank, letters hopping between neighbours.

That collapse is the whole trick. Merely *relabeling* the board `α β α α α / …`
leaves ~3×10¹⁵ boards. Turning each row into a bag leaves **65,650,495**, which
is BFS-able — that's `FULL_WD_ENTRIES` at `walking_distance.rs:232`. The bag is
exactly `M[r][g]` = how many of letter `g` sit in bag `r`.

The Latin puzzle is the same thing rotated 90°: five column-bags of Latin
letters, vertical moves free, blank's column tracked.

## Why the two halves add

Manhattan distance sums each tile's `|row gap| + |col gap|`. The Greek puzzle
bounds only the `|row gap|` half — but it bounds it *properly*, because it knows
about the blank.

`WD = Greek + Latin` (`walking_distance.rs:678`) is admissible because the two
puzzles bill **disjoint moves**: Greek charges only for vertical moves, Latin
only for horizontal. Any real solution with `V` vertical and `H` horizontal
moves projects to a legal Greek solution costing `V` (its horizontal moves
become no-ops) and a legal Latin one costing `H`, so `Greek + Latin ≤ V + H`.
Never an overestimate.

Note this is a **move partition**, not the tile partition that additive pattern
databases use.

Because the 5×5 goal is symmetric under transposition, the Greek and Latin
bag-puzzles are literally the same puzzle: both lookups hit one shared table,
with the projection transposed before the query (`walking_distance.rs:6`).

## Worked example 1 — near-solved, where WD crushes Manhattan

Two tiles from looking solved (tiles 1, 2, 6 rotated in a 3-cycle — an even
permutation with the blank home, so genuinely reachable):

```
 2  6  3  4  5        α β α α α        B A C D E
 1  7  8  9 10        α β β β β        A B C D E
11 12 13 14 15   →    γ γ γ γ γ  and   A B C D E
16 17 18 19 20        δ δ δ δ δ        A B C D E
21 22 23 24  _        ε ε ε ε _        A B C D _
```

**Manhattan says 4.** Tile 1 is one off, tile 2 is one off, tile 6 is two off.

Now play the Greek puzzle. Bag 0 is `{α α α α β}`, bag 1 is `{α β β β β}`, bags
2–4 are clean, blank is in bag 4. To get that stray `β` out of bag 0, the blank
must *be* in bag 0 at some point (the letter slides into the blank's cell, and
that cell is in bag 0). Same for lifting the stray `α` out of bag 1. Every paid
move shifts the blank by exactly one bag, and the blank must end back in bag 4.
So:

> bag 4 → bag 0 → bag 4 = **at least 8 moves**

And 8 is achievable: walk the blank up, dragging one `δ`, one `γ`, one `β` down
behind it; swap the stray `β` down and the stray `α` up; walk back, and the
return trip drags exactly those three letters home again. **Greek distance = 8.**

The Latin board has the identical shape — `{B A A A A}` in column 0, `{A B B B B}`
in column 1, blank in column 4 — so **Latin distance = 8** too.

```
WD = Greek + Latin = 8 + 8 = 16      vs Manhattan's 4
```

Manhattan sees three barely-displaced tiles. WD sees that the blank is in the
far corner and must make a round trip on both axes, wrecking and repairing a
diagonal of tiles as it goes.

## Worked example 2 — a genuinely scrambled board

A uniform-random solvable state:

```
   numbers            Greek (row)         Latin (col)
 22  2 16  9  3       ε α δ β α           B B A D C
 12 10  7  8 15       γ β β β γ           B E B C E
 14  5 13  6 23       γ α γ β ε           D E C A C
 19  4 11 20 17       δ α γ δ δ           D D A E B
 24 18  1 21  _       ε δ α ε _           D C A A _
```

Top-left cell: tile 22 belongs at row 4, column 1 — `ε` in the Greek view, `B`
in the Latin view — and it currently sits at row 0, column 0. Far from home on
both axes.

Squash each Greek row into a bag and count:

```
M_row              α  β  γ  δ  ε        (blank is in row 4)
  row 0  ε α δ β α  2  1  0  1  1
  row 1  γ β β β γ  0  3  2  0  0
  row 2  γ α γ β ε  1  1  2  0  1
  row 3  δ α γ δ δ  1  0  1  3  0
  row 4  ε δ α ε _  1  0  0  1  2
```

Row sums are `5 5 5 5 4` — four in the blank's row. Column sums are `5 5 5 5 4`
too, since there are five of each Greek letter but only four `ε`. Those margins
are what `pack()` exploits at `walking_distance.rs:99`: the last column of each
row is derivable, so only 20 of the 25 counts need storing (20 cells × 3 bits +
3-bit blank index = 63 bits, fits a `u64`).

The Latin view, bagged per column:

```
M_col              A  B  C  D  E        (blank is in col 4)
  col 0  B B D D D  0  2  0  3  0
  col 1  B E E D C  0  1  1  1  2
  col 2  A B C A A  3  1  1  0  0
  col 3  D C A E A  2  0  1  1  1
  col 4  C E C B _  0  1  2  0  1
```

The goal for both is the identity — `5` down the diagonal, `4` in the blank's
slot.

Scores:

```
                     Manhattan     WD
  vertical / Greek        24        30     (+6)
  horizontal / Latin      42        46     (+4)
  ─────────────────────────────────────
  total                   66        76     (+10)
```

The Latin puzzle is the expensive one, and `M_col` shows why: column 0 holds
three `D`s and two `B`s and not a single `A`, while column 2 holds three `A`s.
That is a lot of lateral traffic, and the blank — parked in column 4 — must make
several round trips to move any of it, dragging tiles the wrong way on each leg.

## How much WD actually buys on random boards

That +10 board was *hunted* for. Over 300 uniform random solvable 24-puzzle
states:

```
  mean Manhattan   75.9
  mean WD          78.1
  WD − MD          mean 2.2, min 0, max 8
```

On a thoroughly scrambled board the count matrices come out nearly flat — almost
every entry 0–3, roughly one of each letter per bag — and a flat matrix has
little structure to charge for beyond what Manhattan already sees. Some random
boards score `WD = MD` exactly.

WD's leverage is largest on **structured** boards: near-solved ones like example
1, and the deep/adversarial region this project targets, where rows and columns
are coherent but interlocked. That gap between "cheap on random boards, valuable
on structured ones" is much of why the surcharge machinery in
`src/puzzle24/search/cwd.rs` exists.

## Where to look in the code

| What | Where |
|---|---|
| State shape (`M[r][g]` + blank axis index) | `walking_distance.rs:31` |
| `u64` key packing, last-column-derivable trick | `walking_distance.rs:99` |
| Reachable state count (65,650,495) | `walking_distance.rs:232` |
| Shared row/col table by transposition symmetry | `walking_distance.rs:6` |
| `h = h_row + h_col` | `walking_distance.rs:678` |
| Table build / persist tool | `src/bin/build_wd24.rs` |
