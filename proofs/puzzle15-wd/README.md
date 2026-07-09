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
| cWD (escape-constrained WD) admissible | — | ⬜ planned |

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
