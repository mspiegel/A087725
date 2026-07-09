# puzzle15-wd — machine-checked admissibility for the 15-puzzle

Lean 4 / mathlib formalization of heuristic **admissibility** (`h(s) ≤ dist(s)`)
for the 15-puzzle, as correctness insurance for the WD/cWD heuristics used to
bound the 24-puzzle diameter. The admissibility argument is dimension-agnostic,
so a 15-puzzle proof transfers to the 24-puzzle unchanged.

## Status

| Result | File | Verified |
|---|---|---|
| Manhattan distance is admissible (`MD s ≤ n` for any `n`-move solution) | `Puzzle15Wd/Basic.lean` | ✅ no `sorry`; axioms = `{propext, Classical.choice, Quot.sound}` |
| Walking Distance is admissible (`WD s ≤ n`) | `Puzzle15Wd/WD.lean` | ✅ no `sorry`; axioms = `{propext, Classical.choice, Quot.sound}` |
| cWD escape machinery (`Sol`, `rowEscapes`, `escapeDemand`, `rowEscapes_le`) | `Puzzle15Wd/CWD.lean` | ✅ sorry-free |
| cWD LIS kernel (`maxKeptCard_insert_le`/`_le_insert`/`_congr`/`_full`) | `Puzzle15Wd/CWD.lean` | ✅ sorry-free (fills a mathlib gap) |
| cWD forced-escape bound (`move_demand_le` → `row_escape_bound`) | `Puzzle15Wd/CWD.lean` | ✅ no `sorry`; axioms = `{propext, Classical.choice, Quot.sound}` |

The model: `State := Equiv.Perm (Fin 16)` (tile ↦ cell), `goal := Equiv.refl`,
`blank := 0`; a `Move` swaps the blank's cell with an adjacent tile's cell;
`StepPath s t n` = reachable in exactly `n` moves.

**Manhattan** (`Basic.lean`) is proved as a monovariant: `MD goal = 0` and
`move_md_le : Move s s' → MD s ≤ MD s' + 1`, so `MD s ≤ n` by induction on
`StepPath`.

**Walking Distance** (`WD.lean`) uses the direct projection argument.
`projRow s` is the multiset of `(physical row, goal row)` pairs over non-blank
tiles; `AG` is the abstract graph whose edges shift one tile one step between
adjacent physical rows; `WDrow s := AG.dist (projRow s) (projRow goal)`,
`WD := WDrow + WDcol`. The crux (`proj_change`) is that one move changes exactly
one tile's projection entry; hence each move is an `AG`-edge on one axis and a
no-op on the other (`move_proj`). An `n`-move solution therefore yields `AG`-walks
of total length `≤ n`, and since `dist ≤ walk length`, `WD s ≤ n`.

**cWD** (`CWD.lean`) proves the soundness of the escape side-constraint: any
solution makes at least `escapeDemand g s` type-`g` escapes (`row_escape_bound`),
where `escapeDemand` = residents of goal line `g` minus the largest
order-preserving subset (the LIS / linear-conflict demand). Proved as a
monovariant: the demand is `0` at the goal and rises by at most one per move —
and only on a genuine escape (`move_demand_le`, a 4-case dispatch on whether the
moved tile is a line-`g` resident before/after, built on the small verified LIS
kernel `maxKeptCard_*` that mathlib lacks). The in-row case is closed by
adjacent-swap order preservation: every other resident's column avoids both the
mover's and the blank's adjacent columns, so the induced order is unchanged.

Together, WD's walk projection and the forced-escape bound are the two verified
ingredients of cWD admissibility. The remaining (unformalized) step is assembly
only: define the escape-constrained abstract distance on the product graph
(`AG` × escape counters) and thread these two facts through it.

## Build

Depends on mathlib `v4.29.1` (toolchain `leanprover/lean4:v4.29.1`). To avoid a
cold mathlib build, `.lake/packages` is a **symlink** to the already-built
packages in `../../../A002845/proofs/skip-split/.lake/packages`. If that project
moves or is cleaned, run `lake exe cache get` here instead.

```sh
cd proofs/puzzle15-wd
lake build
# axiom audit:
echo 'import Puzzle15Wd.Basic
#print axioms Puzzle15.md_admissible' | lake env lean --stdin
```
