import Mathlib

/-!
# Admissibility of Manhattan distance for the 15-puzzle

A warm-up formalization establishing the model (state / move / solution length)
and the *monovariant* proof pattern that also drives Walking Distance:

* `md_goal`      : `MD goal = 0`
* `move_md_le`   : one move changes `MD` by at most 1 (`MD s ≤ MD s' + 1`)
* `md_admissible`: `MD s ≤ n` for every `n`-move solution `s ↝ goal`

The heuristic is therefore an admissible lower bound on the optimal solution
length. No `sorry`.
-/

namespace Puzzle15

open Finset

/-- Cells of the 4×4 board, row-major. -/
abbrev Cell := Fin 16
/-- Tiles; tile `0` is the blank. -/
abbrev Tile := Fin 16

/-- The blank tile. -/
def blank : Tile := 0

/-- Physical row of a cell. -/
def row (c : Cell) : ℕ := c.val / 4
/-- Physical column of a cell. -/
def col (c : Cell) : ℕ := c.val % 4

/-- A board: which cell each tile occupies (a bijection). The solved board is the
    identity — tile `t` sits at cell `t`. -/
abbrev State := Equiv.Perm (Fin 16)

/-- The solved board. -/
def goal : State := Equiv.refl _

/-- Two cells are grid-adjacent iff they differ by one step along one axis. -/
def Adjacent (a b : Cell) : Prop :=
  (row a = row b ∧ Nat.dist (col a) (col b) = 1) ∨
  (col a = col b ∧ Nat.dist (row a) (row b) = 1)

theorem Adjacent.symm {a b : Cell} (h : Adjacent a b) : Adjacent b a := by
  rcases h with ⟨hr, hc⟩ | ⟨hc, hr⟩
  · exact Or.inl ⟨hr.symm, by rw [Nat.dist_comm]; exact hc⟩
  · exact Or.inr ⟨hc.symm, by rw [Nat.dist_comm]; exact hr⟩

/-- Manhattan distance from a cell `c` to a tile `t`'s goal cell (which is `t`). -/
def mdCell (c : Cell) (t : Tile) : ℕ :=
  Nat.dist (row c) (row t) + Nat.dist (col c) (col t)

/-- Total Manhattan distance of a board (the blank is not counted). -/
def MD (s : State) : ℕ :=
  ∑ t ∈ univ.filter (· ≠ blank), mdCell (s t) t

/-- One legal move: slide the tile `u` (currently adjacent to the blank) into the
    blank's cell — i.e. swap the cells occupied by `u` and the blank. -/
def Move (s s' : State) : Prop :=
  ∃ u : Tile, u ≠ blank ∧ Adjacent (s blank) (s u) ∧
    s' = s.trans (Equiv.swap (s u) (s blank))

/-- Reachability of `t` from `s` in exactly `n` moves. -/
inductive StepPath : State → State → ℕ → Prop
  | refl (s) : StepPath s s 0
  | step {s s' t : State} {n : ℕ} : Move s s' → StepPath s' t n → StepPath s t (n + 1)

/-- Moving to an adjacent cell changes one tile's Manhattan term by at most 1. -/
theorem mdCell_le_of_adj {a b : Cell} (h : Adjacent a b) (u : Tile) :
    mdCell a u ≤ mdCell b u + 1 := by
  unfold mdCell
  rcases h with ⟨hr, hc⟩ | ⟨hc, hr⟩ <;>
    · unfold Nat.dist at *
      omega

@[simp] theorem md_goal : MD goal = 0 := by
  unfold MD
  apply Finset.sum_eq_zero
  intro t _
  simp [mdCell, goal, Nat.dist_self]

/-- A single move changes the total Manhattan distance by at most 1. -/
theorem move_md_le {s s' : State} (h : Move s s') : MD s ≤ MD s' + 1 := by
  obtain ⟨u, hu, hadj, rfl⟩ := h
  set a := s u with ha
  set b := s blank with hb
  set s'' : State := s.trans (Equiv.swap a b) with hs''
  have huF : u ∈ univ.filter (· ≠ blank) := by simp [hu]
  -- the value of s'' at each tile
  have hval : ∀ t, s'' t = (Equiv.swap a b) (s t) := fun t => Equiv.trans_apply _ _ t
  -- termwise: each Manhattan term is preserved except tile u, which shifts by ≤1
  have key : ∀ t ∈ univ.filter (· ≠ blank),
      mdCell (s t) t ≤ mdCell (s'' t) t + (if t = u then 1 else 0) := by
    intro t ht
    simp only [mem_filter, mem_univ, true_and] at ht
    by_cases htu : t = u
    · subst htu
      have h1 : s'' t = b := by rw [hval, ← ha, Equiv.swap_apply_left]
      rw [h1, if_pos rfl, ← ha]
      exact mdCell_le_of_adj hadj.symm t
    · have hne_a : s t ≠ a := fun hc => htu (s.injective (hc.trans ha.symm ▸ rfl))
      have hne_b : s t ≠ b := fun hc => ht (s.injective (hc.trans hb.symm ▸ rfl))
      have h1 : s'' t = s t := by
        rw [hval, Equiv.swap_apply_of_ne_of_ne hne_a hne_b]
      rw [h1, if_neg htu]
      omega
  calc
    MD s = ∑ t ∈ univ.filter (· ≠ blank), mdCell (s t) t := rfl
    _ ≤ ∑ t ∈ univ.filter (· ≠ blank),
          (mdCell (s'' t) t + (if t = u then 1 else 0)) := Finset.sum_le_sum key
    _ = (∑ t ∈ univ.filter (· ≠ blank), mdCell (s'' t) t)
          + ∑ t ∈ univ.filter (· ≠ blank), (if t = u then (1:ℕ) else 0) := by
        rw [Finset.sum_add_distrib]
    _ = MD s'' + 1 := by
        rw [Finset.sum_ite_eq' (univ.filter (· ≠ blank)) u (fun _ => (1:ℕ))]
        simp [huF, MD]

/-- Manhattan distance lower-bounds the length of any solution from `s` to `t`,
    offset by `MD t`. -/
theorem md_le_of_path {s t : State} {n : ℕ} (h : StepPath s t n) : MD s ≤ MD t + n := by
  induction h with
  | refl s => simp
  | step hmove _ ih =>
      have := move_md_le hmove
      omega

/-- **Admissibility.** Manhattan distance never exceeds the number of moves in any
    solution taking `s` to the solved board. -/
theorem md_admissible {s : State} {n : ℕ} (h : StepPath s goal n) : MD s ≤ n := by
  have := md_le_of_path h
  simpa using this

end Puzzle15
