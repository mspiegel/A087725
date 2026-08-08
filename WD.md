# Walking Distance, explained

An intuition-level account of the Walking Distance heuristic (Ken'ichiro
Takahashi) for the 24-puzzle — and of **cWD**, this project's escape-constrained
sharpening of it — using Greek and Latin letters instead of tile numbers.
The document ends with the last-move refinements (`--lm`, `--lm2`, `--clm2`)
built on top of cWD. Implementations live in
`src/puzzle24/search/walking_distance.rs`, `src/puzzle24/search/cwd.rs`,
`src/puzzle24/search/cwd_lm.rs`, and `src/puzzle24/search/cwd_lm1l.rs`.

## Contents

**Walking Distance**

- [Two alphabets](#two-alphabets)
- [The Greek puzzle](#the-greek-puzzle)
- [Why the two halves add](#why-the-two-halves-add)
- [Worked example 1 — near-solved, where WD crushes Manhattan](#worked-example-1--near-solved-where-wd-crushes-manhattan)
- [Worked example 2 — a genuinely scrambled board](#worked-example-2--a-genuinely-scrambled-board)
- [How much WD actually buys on random boards](#how-much-wd-actually-buys-on-random-boards)
- [Where to look in the code (WD)](#where-to-look-in-the-code-wd)

**cWD — the escape-constrained sharpening**

- [What the bags forget](#what-the-bags-forget)
- [The escape demand](#the-escape-demand)
- [Escapes constrain the bag puzzle — they don't add](#escapes-constrain-the-bag-puzzle--they-dont-add)
- [Worked example 3 — paying for a forced escape](#worked-example-3--paying-for-a-forced-escape)
- [The escape counter, state by state](#the-escape-counter-state-by-state)
- [Worked example 4 — two scrambled rows, one blank tour](#worked-example-4--two-scrambled-rows-one-blank-tour)
- [Looking it up instead of searching](#looking-it-up-instead-of-searching)
- [How much cWD actually buys](#how-much-cwd-actually-buys)
- [Where to look in the code (cWD)](#where-to-look-in-the-code-cwd)

**The last-move refinements**

- [The last move — naming one letter (`--lm`)](#the-last-move--naming-one-letter---lm)
- [Worked example 5 — the two branches on the 3-cycle board](#worked-example-5--the-two-branches-on-the-3-cycle-board)
- [The last two moves (`--lm2`)](#the-last-two-moves---lm2)
- [Worked example 6 — four corridors on the same board](#worked-example-6--four-corridors-on-the-same-board)
- [cLM2 — obligations and demands, priced together (`--clm2`)](#clm2--obligations-and-demands-priced-together---clm2)
- [Where to look in the code (LM / LM2 / cLM2)](#where-to-look-in-the-code-lm--lm2--clm2)

## Two alphabets

Give every tile two names. Its **Greek** name is the row it belongs to; its
**Latin** name is the column it belongs to.

```
goal board
 1  2  3  4  5
 6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24  _

Greek names      Latin names
α α α α α        A B C D E
β β β β β        A B C D E
γ γ γ γ γ        A B C D E
δ δ δ δ δ        A B C D E
ε ε ε ε _        A B C D _
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
 2  6  3  4  5
 1  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24  _

Greek            Latin
α β α α α        B A C D E
α β β β β        A B C D E
γ γ γ γ γ        A B C D E
δ δ δ δ δ        A B C D E
ε ε ε ε _        A B C D _
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
WD = Greek + Latin = 8 + 8 = 16
Manhattan = 4
```

Manhattan sees three barely-displaced tiles. WD sees that the blank is in the
far corner and must make a round trip on both axes, wrecking and repairing a
diagonal of tiles as it goes.

## Worked example 2 — a genuinely scrambled board

A uniform-random solvable state:

```
numbers
 22  2 16  9  3
 12 10  7  8 15
 14  5 13  6 23
 19  4 11 20 17
 24 18  1 21  _

Greek (row)      Latin (col)
ε α δ β α        B B A D C
γ β β β γ        B E B C E
γ α γ β ε        D E C A C
δ α γ δ δ        D D A E B
ε δ α ε _        D C A A _
```

Top-left cell: tile 22 belongs at row 4, column 1 — `ε` in the Greek view, `B`
in the Latin view — and it currently sits at row 0, column 0. Far from home on
both axes.

Squash each Greek row into a bag and count:

```
M_row (blank in row 4)
                    α  β  γ  δ  ε
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
M_col (blank in col 4)
                    A  B  C  D  E
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
                 Manhattan   WD
vertical/Greek        24     30   (+6)
horizontal/Latin      42     46   (+4)
──────────────────────────────────────
total                 66     76  (+10)
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
 2  3  1  4  5
 6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24  _

Greek            Latin
α α α α α        B C A D E
β β β β β        A B C D E
γ γ γ γ γ        A B C D E
δ δ δ δ δ        A B C D E
ε ε ε ε _        A B C D _
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
x_g = residents of g − LIS(goal-cross order)
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
Manhattan                     4
Manhattan + linear conflict   6
WD  = 0 + 10 = 10
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
| 1 | bag 4 → 3 | a δ drops into bag 4 | 3: 4δ, 4: 4ε+δ | 0 |
| 2 | bag 3 → 2 | a γ drops into bag 3 | 2: 4γ, 3: 4δ+γ | 0 |
| 3 | bag 2 → 1 | a β drops into bag 2 | 1: 4β, 2: 4γ+β | 0 |
| 4 | bag 1 → 0 | **an α drops into bag 1** | 0: 4α, 1: 4β+α | **1** ✓ |
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
Manhattan = 8
WD  = 0 + 14 = 14
cWD = 8 + 14 = 22
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

## The last move — naming one letter (`--lm`)

Everything so far treats the letters of a type as interchangeable: the bag
knows *how many* δs it holds, never *which*. The next refinement breaks that
anonymity for exactly one tile per axis, and the reason it can afford to is an
endgame fact.

The blank's home is the bottom-right corner, and every solution ends with the
blank stepping into it. Rewind the film one frame. One move before the end the
board is solved except for that corner, and the last move slides a single tile
out of it — either tile 20 dropping back up into its home above the corner, or
tile 24 sliding back left into its home beside it:

```
ending R:   19  _        19 20
            24 20   →    24  _

ending C:   19 20        19 20
             _ 24   →    24  _
```

There is no third ending. And each ending carries an obligation that starts
long before the final frame: tile 20's home is row 3, so ending R requires
tile 20 to **enter row 4 and come back** — a 3→4 crossing on the row axis.
Symmetrically, tile 24's home is column 3, so ending C requires tile 24 to
cross column 3→4 and return.

The row bags cannot express this. "Some δ dips into bag 4" is visible — but
tile 20 is just one of five δs, and the bag has no way to insist the dipper be
*that one*. So the refinement tracks it by name: the product state becomes

> (bags, **tile 20's bag**, **crossed yet?**)

and a backward BFS from the goal `(clean bags, bag 3, crossed = true)` prices
every such state — the same make-history-part-of-the-state trick as the escape
counter, with the counter replaced by one monotone bit. 65,650,495 bag states
× 5 lines × 2 flag values ≈ 656 M product states; the stored layer is
`crossed = false` (queries always start there), giving a refined distance
`D(key, line) ≥ WD(key)`. One table serves both axes, by the same
transposition symmetry as WD itself.

At a search node the two endings become two branch values:

```
branch20 = D_row(row key, row of 20)
           + full Latin cWD term
branch24 = D_col(col key, col of 24)
           + full Greek cWD term

h = max( cWD, min(branch20, branch24) )
```

Each branch swaps *one* axis's WD term for the refined distance and keeps the
other axis's full cWD term — still additive, because the two axes still bill
disjoint move classes. The `min` is forced: every solution takes one of the
two endings, so only the weaker branch is a bound on all of them. The outer
`max` restores cWD when the swap loses (the refined `D` sharpens WD but not
the escape surcharge, so a surcharge-rich axis can make a branch dip below
cWD). A branch also degenerates to cWD outright when its tile already sits in
line 4 — the crossing is a free ride from there — or when the table has no
entry for the placement.

One honest caveat: the obligation presumes the solution still *has* a last
move, so the bound is wrong at the goal board itself (where the true distance
is 0). The engine never prices the goal — a search node is consulted only
after it fails the solved test — so this never bites.

## Worked example 5 — the two branches on the 3-cycle board

Back to example 3's board (top row `2 3 1`, everything else home):
cWD = 8 + 10 = 18, and both tracked tiles sit at home — tile 20 in bag 3,
tile 24 in column-bag 3.

**branch20** asks for the cheapest row-bag plan in which tile 20 *itself*
dips into bag 4 and returns. The row bags are clean, so the plan is nothing
but the dip — the same two-column walk as the escape counter's, with the
counter replaced by the crossed bit:

| # | blank move | letter kicked | tile 20 is in | crossed |
|---|-----------|---------------|---------------|---------|
| 1 | bag 4 → 3 | **tile 20's δ drops into bag 4** | bag 4 | **✓** |
| 2 | bag 3 → 4 | the same δ climbs home | bag 3 | ✓ |

`D_row = 2` where WD_row said 0. But the branch swaps out the row's *full*
cWD term, surcharge included: branch20 = 2 + 10 = **12**, below cWD's 18 —
the refined table sharpens WD, not the escape surcharge, and this is exactly
the dip the outer `max` exists to absorb.

**branch24** asks for a Latin plan in which tile 24 dips through column 4.
Example 3's Latin tour already crosses the column-3/4 boundary on its way
out — its first move kicks *some* D-letter from column 3 into column 4, and
the bags never cared which. Nominate tile 24: the obligation is absorbed
into a walk the plan was making anyway, `D_col = 10 = WD_col`, and
branch24 = 10 + 8 = **18**.

```
h = max(18, min(12, 18)) = 18
```

No gain — and that is the honest, typical outcome. Most plans can nominate
the tracked tile for free: any row plan whose blank crosses the 3/4 boundary
kicks some δ into bag 4 on the way, and if tile 20 can be that δ, `D = WD`
and the branch adds nothing. The refinement binds when tile 20 sits far from
the boundary — its personal journey to row 4 and back then costs
line-changes the anonymous accounting never paid — which means deep,
structured endgame states, exactly this project's territory. Measured on R:
exhaust-144 falls from 422,379,806 nodes (cWD) to **267,641,675** (1.58×),
and on 507 deep prune-event boards the precomputable form certifies 43% of
the +2s that the 30.5 GB k8 pattern databases prove.

## The last two moves (`--lm2`)

If one frame of rewind buys an obligation, rewind two. The blank's final two
steps approach its home — the corner — through the bottom-right block:

```
        col 2   col 3   col 4
row 2 |  13      14      15
row 3 |  18      19      20
row 4 |  23      24       _
```

Chasing both frames backward pins the blank's suffix to one of exactly **four
corridors**, and each corridor commits two tiles to a crossing-and-return:

- **A** — blank runs 15's home → 20's home → corner. Last move: 20 comes
  home from row 4; the move before: 15 comes home from row 3. Both
  obligations on the **row** axis: 20 crosses 3→4, 15 crosses 2→3.
- **B** — 19's home → 20's home → corner. 20 comes home from row 4; 19
  comes home from column 4. One obligation per axis: 20 crosses row 3→4,
  19 crosses col 3→4.
- **C** — 19's home → 24's home → corner. 24 comes home from column 4; 19
  comes home from row 4. One per axis, mirrored: 19 crosses row 3→4, 24
  crosses col 3→4.
- **D** — 23's home → 24's home → corner. 24 comes home from column 4; 23
  comes home from column 3. Both on the **column** axis: 24 crosses 3→4,
  23 crosses 2→3.

Two shapes appear. In corridors B and C the two obligations live on
*different* axes — and tile 19, whose home is row 3 **and** column 3, is a
type-3 letter in both alphabets, so both of its appearances read the same
single-tracked 656 M-state table that `--lm` built. Nothing new to
precompute: those branches are `D_row + D_col`, refined on both axes at once.

Corridors A and D put both obligations on **one** axis, and there a second
single-tracked lookup would be unsound — the two excursions might want to
share the same blank walk, and pricing them separately then adding would
charge a detour no optimal solution makes (the same reason cWD takes the max
of single-line surcharges, not the sum). So A and D get their own table:
track *two* named tiles on one axis — the type-3 token (20 or 24, crossing
3→4) and a type-2 token (15 or 23, crossing 2→3) — through the product

> (bags, line of A, crossed_A, line of B, crossed_B)

≈ 6.57 B states, backward-BFS'd the same way, stored as 25 pair values per
bag state. The two tracked tokens have different types, so their identities
never collide on a move. `D2(key, la, lb)` prices both excursions *jointly* —
shared blank-walk and all.

The node evaluation is the four-way version of the same shape:

```
A = D2_row(rkey, row20, row15)
    + full Latin term
B = D_row(rkey, row20) + D_col(ckey, col19)
C = D_row(rkey, row19) + D_col(ckey, col24)
D = D2_col(ckey, col24, col23)
    + full Greek term

h = max( cWD, min(A, B, C, D) )
```

with the same degeneracy ladder as before: a tile already past its crossing
line (or a missing table entry) collapses its lookup to that axis's baseline
term, so a fully degenerate branch is exactly cWD and the bound never falls
below it. The goal-board caveat now extends one frame: obligations presume
≥ 2 remaining moves, and the two boards one move from the goal stay safe
because their displaced tile sits in line 4, collapsing that branch to
cWD = 1, the true distance.

## Worked example 6 — four corridors on the same board

Example 3's board again; every tracked tile is home, so all four branches
are live. Corridor **A** prices two *nested* dips through one blank walk —
the pair table's whole reason to exist:

| # | blank move | letter kicked | 20 crossed | 15 crossed |
|---|-----------|---------------|------------|------------|
| 1 | bag 4 → 3 | **tile 20's δ drops into bag 4** | **✓** | |
| 2 | bag 3 → 2 | **tile 15's γ drops into bag 3** | ✓ | **✓** |
| 3 | bag 2 → 3 | the γ climbs home | ✓ | ✓ |
| 4 | bag 3 → 4 | the δ climbs home | ✓ | ✓ |

`D2_row = 4`: the inner excursion rides inside the outer one, and the joint
value prices that sharing — two separate single-tracked lookups could not
promise the walks combine. A = 4 + 10 = **14**.

The other three corridors are nomination stories. **B**: tile 20's two-move
row dip plus tile 19's column crossing — and 19, like 24, is a D-letter
sitting in column 3, so the Latin tour's outbound kick absorbs it:
B = 2 + 10 = **12**. **C** is the mirror: tile 19's row dip (2) plus tile
24's absorbed column crossing: C = 2 + 10 = **12**. **D** wants both column
obligations at once, and the Latin tour happens to offer both kicks — its
outbound walk crosses the 3/4 boundary (nominate tile 24) *and* the 2/3
boundary (the kick sends one of column 2's C-letters into column 3:
nominate tile 23), and the return walk hands both back: `D2_col = 10`,
D = 10 + 8 = **18**.

```
h = max(18, min(14, 12, 12, 18)) = 18
```

Again no gain: on this board every obligation is either absorbed into a walk
the bags already paid for, or undercut by an axis whose surcharge the branch
dropped — and again the `max` holds the floor. The four-way `min` starts
biting where no corridor gets its excursions for free.

Measured on R: exhaust-144 at **216,158,191** nodes — 1.95× under plain cWD
and 1.24× under `--lm` — with the whole tier costing two table probes per
surviving child through a small front cache.

## cLM2 — obligations and demands, priced together (`--clm2`)

Examples 5 and 6 exposed a seam. A branch value refines WD by the corridor's
obligations but *drops the escape surcharge* on the refined axis; cWD keeps
the surcharge but knows nothing of corridors. The `max` picks the better of
two half-truths — it never prices the conjunction. The honest question is:

> What is the cheapest bag plan that makes the demanded escapes **and** the
> corridor's excursions?

On example 3's board the conjunction happens to collapse (the 8-move escape
tour can nominate tile 20 as its dragged δ, so demands and obligation share
one walk) — but on boards where they cannot share, the joint answer exceeds
both halves, and neither table built so far can see it. cLM2 —
*constrained* LM2 — closes that seam. Unlike the previous sections, the
implementation route matters here, because the project built it twice.

**The search we could be running.** Every state-augmentation trick in this
document composes into one product space:

> (bags, one saturating escape counter per demanded line,
>  tracked line(s), crossed bit(s))

and the joint value is a shortest path in it — computable at a search node
by running the constrained A\* *right there*, on the node's actual demand
vector, one axis at a time. The internal heuristic `wd + strongest
single-line surcharge over the remaining demands` is admissible and
consistent, so a bucketed A\* without reopening is exact. This is not
hypothetical: the tier first shipped in exactly this form (2026-08-06), a
per-node memoized A\*, and it measured the joint heuristic's worth — 1.38×
fewer nodes than `--lm2` at threshold 144, and on the survivor census 545 of
its 780 slack-0 prunes were beyond both the LM2 tables *and* the 30.5 GB k8
pattern databases. The cost is the same one that killed the joint-demands
idea back in "Looking it up instead of searching": a search per node instead
of a probe, with a memo that must absorb an open-ended space of
(key, demand-vector, placement) queries. And a full joint *table* is
unbuildable for the same reason as before — the key would include the whole
demand vector.

**The measurement that opened the door.** Cross-tabbing those 780 joint
prunes: restricting the demands to **one line at a time** — the same
retreat cWD itself made — retains **778 of 780**
(`data/lm2j1l_survivors148.txt`). One line `g`, one demand `d ≤ 4`, one
tracked placement *is* enumerable: 5 lines × 4 demands × 19 variants
(4 single-A lines, 3 single-B lines, 12 pairs) + 3 zero-demand single-B
slots = **383 values per bag state**.

**The table we run instead.** `cwd_lm1l` closes those 383 answers per key
over all 65,650,495 bag states. Three implementation choices carry it:

- *Layered build.* For fixed `g`, let `D_r` price "≥ r escapes of `g` plus
  the tracked obligations". Any optimal path's first move either escapes
  `g` (the rest needs `r−1`) or doesn't (still needs `r`) — so `D_r` is
  computed from the complete `D_{r−1}` by seeding every escape edge at
  `D_{r−1} + 1` and relaxing only non-escape edges with a multi-source
  dial. Four layers per line, reusing one reversed-edge infrastructure.
  Two hazards the port had to get right, both admissibility-critical: the
  plain BFS's write-once guard must become a **min-write** (a state seeded
  at 10 but propagation-reachable at 6 must take 6, or the table ships an
  overestimate), and termination must wait for an empty frontier **and**
  exhausted seed depths.
- *2-bit storage.* `D − base` is even (blank-line parity is fixed by the
  key) and never negative (constraints only add), where
  `base(key, g, d) = wd + 2·curve_nibble(g, d)` is recomputed from the
  merged cWD cell at consult time. So each value compresses to
  `min((D − base)/2, 3)` — an advantage of 0/+2/+4/+6, saturating — and
  saturation clamps *down*, which keeps it admissible. 383 fields → 96
  payload bytes per key; the mmap'd artifact (`data/cwd_lm1l_mm.bin`,
  magic `CWJ1`) is 10.5 GB at 0.70 load.
- *The consult.* Per corridor variant:
  `max(zero-demand LM2 floor, max over demanded lines g of
  base(g, x_g) + 2·field)` — one probe, no search. A runtime gate skips
  the whole consult when it provably cannot move the bound (no demands on
  either axis, no tb-only variant live, `h ≤ 1`); the three zero-demand
  single-B fields cover the one placement — a type-2 token tracked while
  its type-3 partner is home — that no other table stores.

The A\* did not die: it survives as the **reference oracle**
(`cwd_lm_joint.rs`), demoted from the engine to the build gates — the table
builder's validation cross-checks probe values against fresh exact searches.
The engine itself only ever probes.

Measured on R: exhaust-144 at **134,801,951** nodes — 1.60× under `--lm2`
at wall-clock parity — and composed with the k8 tier (consult order
cWD → cLM2 → k8) the cascade reaches **115,436,814**, the strongest stack
this project runs.

## Where to look in the code (LM / LM2 / cLM2)

| What | Where |
|---|---|
| Single-tracked refined table `D(key, line)` | `cwd_lm.rs:37` |
| Backward BFS over (key × line × crossed) | `cwd_lm.rs:104` |
| Pair table `D2(key, la, lb)` | `cwd_lm.rs:224` |
| Pair builder over (key × la × ca × lb × cb) | `cwd_lm.rs:308` |
| Combined mmap artifact (`data/cwd_lm_mm.bin`) | `cwd_lm.rs:513` |
| Two-branch evaluation (`--lm`) | `flat.rs:1212` |
| Four-branch evaluation (`--lm2`) | `flat.rs:1496` |
| Tracked-line maintenance per move | `flat.rs:1192`, `flat.rs:1466` |
| Joint constrained A\* (reference oracle) | `cwd_lm_joint.rs:66` |
| Layered `D_r` build (seed + dial relax) | `cwd_lm1l.rs:593` |
| 2-bit field layout, payload probe | `cwd_lm1l.rs:67`, `cwd_lm1l.rs:818` |
| cLM2 consult (floor + delta lift) | `flat.rs:1631` |
| Table builder + oracle validation gate | `cwd_lm1l.rs:878`, `cwd_lm1l.rs:1029` |
| Builder CLI (all five cWD artifacts) | `src/bin/build_cwd_artifacts.rs` |
| Single-line-retention measurement | `data/lm2j1l_survivors148.txt` |
