# Walking Distance, explained

An intuition-level account of the Walking Distance heuristic (Ken'ichiro
Takahashi) for the 24-puzzle — and of **cWD**, this project's escape-constrained
sharpening of it — using Greek and Latin letters instead of tile numbers.
Implementations live in `src/puzzle24/search/walking_distance.rs` and
`src/puzzle24/search/cwd.rs`.

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
are coherent but interlocked. But there is structure the bags *cannot* see, and
that blind spot is where the rest of this document goes: cWD charges the bag
puzzle for it.

## Where to look in the code (WD)

| What | Where |
|---|---|
| State shape (`M[r][g]` + blank axis index) | `walking_distance.rs:31` |
| `u64` key packing, last-column-derivable trick | `walking_distance.rs:99` |
| Reachable state count (65,650,495) | `walking_distance.rs:232` |
| Shared row/col table by transposition symmetry | `walking_distance.rs:6` |
| `h = h_row + h_col` | `walking_distance.rs:678` |
| Table build / persist tool | `src/bin/build_wd24.rs` |

## What the bags forget

Turning each row into a bag is what made the Greek puzzle small — and it is
also its blind spot. Take a board that is solved except the top row holds
`2 3 1` (a 3-cycle: an even permutation with the blank home, so reachable):

```
 2  3  1  4  5        α α α α α        B C A D E
 6  7  8  9 10        β β β β β        A B C D E
11 12 13 14 15   →    γ γ γ γ γ  and   A B C D E
16 17 18 19 20        δ δ δ δ δ        A B C D E
21 22 23 24  _        ε ε ε ε _        A B C D _
```

Every tile is in its home row, so every Greek bag is clean and the Greek
distance is **0**. The scramble is purely horizontal and horizontal moves are
free — case closed, says the bag.

But the row as a *sequence*, which the bag threw away, is scrambled. In a
sliding puzzle two tiles in the same row cannot pass each other while both stay
in the row: one of them must step out, let the other by, and re-enter. Those
forced exits are **vertical** moves — exactly the kind the Greek puzzle is
supposed to bill — and the bag charges nothing for them, because a bag has no
notion of "pass each other."

## The escape demand

cWD recovers that lost fact as one number per goal line. For a line `g`, look
at its **residents** — tiles that belong to line `g` and currently sit in line
`g` — and read off their goal positions in physical order. Residents that never
leave the line can never reorder among themselves, so the set that stays put
must already read in increasing goal order. The most that can stay is the
longest increasing subsequence; everyone else must leave at least once:

```
x_g  =  residents of g  −  LIS(their goal-cross order)
```

For the row above: the residents' goal columns read `1 2 0 3 4`, the LIS is
`1 2 3 4` (length 4), so `x_α = 5 − 4 = 1` — **at least one α must escape
row 0** in any real solution. (This bound is machine-checked in Lean 4:
`proofs/puzzle15-wd`.)

## Escapes constrain the bag puzzle — they don't add

Classic linear conflict would stop here and *add* 2 per escape on top of
Manhattan. cWD does something sharper. An escape is visible in the bag puzzle —
it is a paid move in which a type-`g` letter leaves bag `g` — so the demand can
be handed to the bag puzzle as a side condition on its own moves. For each
demanded line `g`, ask:

> What is the cheapest bag-puzzle solution that **also** makes at least `x_g`
> escapes of line `g`?

Each answer is still a relaxation of the real puzzle: any real solution
projects to a bag solution, and the projection performs the real solution's
escapes — in particular at least `x_g` of line `g`. So no single-line answer
ever overestimates, and the solver charges the **strongest one**. And because
the escapes constrain WD's own move budget instead of being added on top,
nothing is double-billed: if the WD-optimal plan happened to make the required
escapes anyway, the surcharge is zero. `cWD ≥ WD` always, and the row and
column halves still sum for the same reason as before — they bill disjoint
move classes.

## Worked example 3 — paying for a forced escape

Back to the 3-cycle board. The Greek bags are clean, but the constraint demands
one α escape from bag 0. In the bag puzzle, the only move that takes an α out
of bag 0 is the blank stepping *into* bag 0 (the α drops into the bag the blank
just left). So the blank must visit bag 0, and it starts and must end in bag 4:

> bag 4 → bag 0 → bag 4 = **at least 8 moves**

And 8 is achievable, by the same dance as example 1: walk the blank up dragging
a δ, a γ, a β down behind it; the last step up kicks the α out into bag 1; the
first step down hands it straight back; the rest of the return trip re-homes
the dragged letters. **Constrained Greek distance = 8**, where WD said 0.

The columns make no demands (each column's residents already read in
increasing order), so the Latin half is untouched — and the Latin bags see this
horizontal scramble directly, pricing it at 10. Scores:

```
Manhattan 4      Manhattan + linear conflict 6      WD = 0 + 10 = 10
cWD = 8 + 10 = 18
```

## The escape counter, state by state

How does a shortest-path search even express "make at least one α escape"?
Not as a statement about which state to reach — the start and goal bags are
both clean — but as a statement about what must happen *along the way*, and
shortest-path algorithms cannot see path history. So the history is made part
of the state: the constrained search runs over pairs `(σ, c)`, where `σ` is
the bag state and `c` counts the escapes of the constrained line made so far,
saturating at the demand. Every move updates `σ` as usual; a move in which an
α leaves bag 0 also ticks `c`. The start is `(σ₀, c=0)` and the goal is
`(σ₀, c=1)` — **the same bags, different counter**. That one bit is the entire
difference: the start is no longer the goal, so the search must actually move.

Example 3's optimal path through this product graph:

| # | blank move | letter kicked | bags after | `c` |
|---|-----------|---------------|-----------|---|
| 1 | bag 4 → 3 | a δ drops into bag 4 | bag 3 = 4δ, bag 4 = 4ε+δ | 0 |
| 2 | bag 3 → 2 | a γ drops into bag 3 | bag 2 = 4γ, bag 3 = 4δ+γ | 0 |
| 3 | bag 2 → 1 | a β drops into bag 2 | bag 1 = 4β, bag 2 = 4γ+β | 0 |
| 4 | bag 1 → 0 | **an α drops into bag 1** | bag 0 = 4α, bag 1 = 4β+α | **1** ✓ |
| 5 | bag 0 → 1 | the α drops back into bag 0 | bag 0 clean | 1 |
| 6 | bag 1 → 2 | the β back into bag 1 | bag 1 clean | 1 |
| 7 | bag 2 → 3 | the γ back into bag 2 | bag 2 clean | 1 |
| 8 | bag 3 → 4 | the δ back into bag 3 | all clean | 1 |

After move 8 the state is `(σ₀, c=1)`: clean bags *and* a satisfied counter,
at cost 8 — the `Δ(σ₀, g=0, d=1) = 8` that the surcharge table stores.

The walk makes three things visible:

- **The counter never remembers *which* α left or where it sat.** At move 4
  the state does not record that tile 1 escaped from column 2 — just "bags as
  shown, one escape banked." All the order reasoning already happened in the
  LIS on the concrete board; the counter enforces its conclusion, nothing more.
- **Saturation.** Had the path kicked a second α out, `c` would stay pinned at
  the demand (`c ← min(c + 1, d)`): extra escapes buy nothing, so the product
  space is only `(d+1)×` the bag space — and `d ≤ 4`, so the counter fits in
  3 bits.
- **Constraint, not addend.** Move 4 costs 1 like every other bag move — no
  per-escape "+2" is ever added on. The +8 emerges entirely from the counter
  making the start and goal distinct, which is why a pre-billed excursion
  returns marginal 0 and double-billing is impossible by construction.

## Worked example 4 — two scrambled rows, one blank tour

Now scramble row 2 the same way (`12 13 11`, another 3-cycle). The Greek bags
are still all clean, and there are two demands: `x_α = 1` and `x_γ = 1`.

The solver prices each demand **separately**, one line constrained at a time:

- **α alone**: cheapest bag solution with ≥ 1 α escape — the blank must tour
  bag 4 → bag 0 → bag 4, as in example 3: surcharge **8**;
- **γ alone**: cheapest bag solution with ≥ 1 γ escape — a shorter tour,
  bag 4 → bag 2 → bag 4, kicking a γ out on the way in and handing it back on
  the way out: surcharge **4**.

Each single-line answer is a true lower bound on its own — any real solution
performs *all* the demanded escapes, so in particular it satisfies each
constraint alone. The charge is the **strongest one**: max(8, 4) = 8. Not the
sum: a single 8-move tour to bag 0 does row 2's escape *in passing* (the step
from bag 3 into bag 2 kicks a γ out, the step from bag 1 into bag 0 kicks the
α out, and the way down hands both back), so 8 + 4 = 12 would be charging for
a detour no optimal solution makes.

```
Manhattan 8      WD = 0 + 14 = 14      cWD = 8 + 14 = 22
```

The surcharge is the **same 8** as example 3 — and here the strongest
single-line bound is also the exact jointly-constrained cost, because bag 2
lies on the blank's walk to bag 0. When demands genuinely cannot share the
walk, the max undercharges the joint truth; that gap is measured below.

## Looking it up instead of searching

Running the bags-×-counter search at every node of a real search is far too
slow, so cWD precomputes the answer. For each of the 65,650,495 bag states,
each line `g`, and each demand `d = 1..4` (five residents and LIS ≥ 1, so a
demand never exceeds 4), a table stores the **surcharge** `Δ(σ, g, d)` =
constrained cost − WD, one line constrained at a time. At a node: look up the state's WD and its
surcharge curves, compute the demands, and take the **largest single-line
surcharge** among the demanded lines — exactly the max(8, 4) of example 4.

Largest — not sum, as example 4 showed. Each single-line `Δ` is a true lower
bound on its own, so the strongest one is a safe charge; the sum is not.

The price of "one line at a time" is that when demands genuinely cannot share
the blank's walk, the jointly-constrained cost exceeds every single-line bound
and the lookup undercharges. (The measured case: the blank's own home row
demanding an escape — the kicked-out letter must also be walked back, which
one sweep cannot absorb.) The tighter score is recoverable: run the
constrained A\* with **all** of an axis's demands at once — still admissible,
by the same projection argument. But that is a search per node instead of an
O(1) lookup, and a table of joint values would be indexed by whole demand
vectors — memory-infeasible. Measured on R, the tightness it would buy is
marginal anyway: the gap is zero at the root, the single-line max retains
**98%** of the node-weighted joint gain tree-wide, and wiring in the pairwise
joint value (memoized, essentially free per node) still bought only ~2.6%
fewer nodes in an exhaustive A/B — the extra tightness barely prunes. The
compact table is the right trade. See `FINDINGS_R.md` §8 for those
measurements.

## How much cWD actually buys

On the 180°-rotated board R — this project's target — WD says 140 and cWD says
**144**, and the +4 holds tree-wide (node-weighted mean surcharge ≈ 4.26 over
R's search tree). That translated into 20.7× fewer nodes exhausting threshold
144 and 11.7× at 146, re-proving the R ≥ 148 record in 11 minutes instead of
2 hours. On random boards the gain is small, as with WD itself: example 2's
scrambled board picks up +2 (cWD 78 vs WD 76). The leverage lives where the
bags are clean but the lines are internally interlocked — precisely the
structured region WD alone under-charges.

## Where to look in the code (cWD)

| What | Where |
|---|---|
| Projection + per-line escape demands `x_g` | `cwd.rs:370` |
| Surcharge curves, largest-single-line lookup | `cwd.rs:200` |
| Surcharge table builder | `src/bin/build_cwd_single.rs` |
| Escape-bound soundness proof (Lean 4) | `proofs/puzzle15-wd` |
| Joint vs single-line-max gap measurements | `FINDINGS_R.md` §8 |
