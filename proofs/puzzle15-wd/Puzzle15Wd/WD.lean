import Mathlib
import Puzzle15Wd.Basic

/-!
# Admissibility of Walking Distance for the 15-puzzle

Walking Distance projects the board onto one axis at a time. `projRow s` is the
multiset of `(physical row, goal row)` pairs over the non-blank tiles; `WDrow s`
is the shortest-path distance, in the abstract graph `AG` whose edges move one
tile one step between adjacent physical rows, from `projRow s` to `projRow goal`.
`WD := WDrow + WDcol`.

The proof is the **direct projection argument**: any `n`-move solution splits
into vertical and horizontal moves; the vertical moves project to an `AG`-walk
`projRow s ↝ projRow goal`, the horizontal ones to an `AG`-walk on columns, and
the two walk lengths sum to `n`. Since graph distance never exceeds a walk's
length, `WD s ≤ n`. No `sorry`.
-/

namespace Puzzle15

open Finset

/-- The non-blank tiles. -/
def NB : Finset Tile := univ.filter (· ≠ blank)

@[simp] theorem mem_NB {t : Tile} : t ∈ NB ↔ t ≠ blank := by simp [NB]

/-- Abstract contingency: the multiset of `(physical line, goal line)` pairs. -/
abbrev Abs := Multiset (ℕ × ℕ)

/-- Row projection. -/
def projRow (s : State) : Abs := NB.val.map (fun t => (row (s t), row t))
/-- Column projection. -/
def projCol (s : State) : Abs := NB.val.map (fun t => (col (s t), col t))

/-- One abstract move: shift one tile of goal-line `j` between the adjacent
    physical lines `i` and `i'`. -/
def AbsRel (m m' : Abs) : Prop :=
  ∃ (i i' j : ℕ) (M : Abs), Nat.dist i i' = 1 ∧ m = (i, j) ::ₘ M ∧ m' = (i', j) ::ₘ M

/-- The abstract Walking-Distance graph. -/
def AG : SimpleGraph Abs := SimpleGraph.fromRel AbsRel

/-- Row / column Walking Distance and their sum. -/
noncomputable def WDrow (s : State) : ℕ := AG.dist (projRow s) (projRow goal)
noncomputable def WDcol (s : State) : ℕ := AG.dist (projCol s) (projCol goal)
noncomputable def WD (s : State) : ℕ := WDrow s + WDcol s

/-- Distinct-headed conses of the same tail are distinct multisets. -/
theorem cons_ne {ra rb j : ℕ} (h : ra ≠ rb) (M : Abs) :
    ((ra, j) ::ₘ M) ≠ ((rb, j) ::ₘ M) := by
  intro heq
  have hc := congrArg (Multiset.count (ra, j)) heq
  rw [Multiset.count_cons_self,
      Multiset.count_cons_of_ne (fun hp => h (congrArg Prod.fst hp))] at hc
  omega

theorem dist_ne {a b : ℕ} (h : Nat.dist a b = 1) : a ≠ b := by
  intro he; rw [he, Nat.dist_self] at h; exact absurd h (by norm_num)

/-- Build an `AG` edge from the underlying relation and distinctness. -/
theorem AG_adj {x y : Abs} (hr : AbsRel x y) (hne : x ≠ y) : AG.Adj x y :=
  (SimpleGraph.fromRel_adj AbsRel x y).mpr ⟨hne, Or.inl hr⟩

/-- One move changes a single tile's cell, so each axis projection changes by
    exactly one `(line, goalLine)` entry: tile `u`'s pair moves from `s u`'s line
    to `s blank`'s line. -/
theorem proj_change (coord : Cell → ℕ) {s s' : State} {u : Tile}
    (hu : u ≠ blank) (hs' : s' = s.trans (Equiv.swap (s u) (s blank))) :
    ∃ M : Abs,
      NB.val.map (fun t => (coord (s t), coord t)) = (coord (s u), coord u) ::ₘ M ∧
      NB.val.map (fun t => (coord (s' t), coord t)) = (coord (s blank), coord u) ::ₘ M := by
  have huNB : u ∈ NB := mem_NB.mpr hu
  have hNBval : NB.val = u ::ₘ (NB.erase u).val := by
    conv_lhs => rw [← Finset.insert_erase huNB]
    rw [Finset.insert_val_of_notMem (Finset.notMem_erase u NB)]
  refine ⟨(NB.erase u).val.map (fun t => (coord (s t), coord t)), ?_, ?_⟩
  · rw [hNBval, Multiset.map_cons]
  · rw [hNBval, Multiset.map_cons]
    have hu' : s' u = s blank := by rw [hs', Equiv.trans_apply, Equiv.swap_apply_left]
    rw [hu']
    congr 1
    apply Multiset.map_congr rfl
    intro t ht
    have htE : t ∈ NB.erase u := Finset.mem_val.mp ht
    have htu : t ≠ u := Finset.ne_of_mem_erase htE
    have htb : t ≠ blank := mem_NB.mp (Finset.mem_of_mem_erase htE)
    have hst : s' t = s t := by
      rw [hs', Equiv.trans_apply,
          Equiv.swap_apply_of_ne_of_ne (s.injective.ne htu) (s.injective.ne htb)]
    rw [hst]

/-- Each move either shifts the row projection by an edge and fixes the column
    projection (a vertical move) or vice versa (a horizontal move). -/
theorem move_proj {s s' : State} (h : Move s s') :
    (AG.Adj (projRow s) (projRow s') ∧ projCol s = projCol s') ∨
    (projRow s = projRow s' ∧ AG.Adj (projCol s) (projCol s')) := by
  obtain ⟨u, hu, hadj, hs'⟩ := h
  obtain ⟨Mr, hr1, hr2⟩ := proj_change row hu hs'
  obtain ⟨Mc, hc1, hc2⟩ := proj_change col hu hs'
  -- restate the projections in folded form (defeq) so rewrites fire
  have hr1' : projRow s = (row (s u), row u) ::ₘ Mr := hr1
  have hr2' : projRow s' = (row (s blank), row u) ::ₘ Mr := hr2
  have hc1' : projCol s = (col (s u), col u) ::ₘ Mc := hc1
  have hc2' : projCol s' = (col (s blank), col u) ::ₘ Mc := hc2
  rcases hadj with ⟨hrow, hcol⟩ | ⟨hcol, hrow⟩
  · -- horizontal move: rows equal, columns adjacent
    right
    refine ⟨by rw [hr1', hr2', hrow], ?_⟩
    apply AG_adj
    · exact ⟨col (s u), col (s blank), col u, Mc, by rw [Nat.dist_comm]; exact hcol, hc1', hc2'⟩
    · rw [hc1', hc2']; exact cons_ne (dist_ne hcol).symm Mc
  · -- vertical move: columns equal, rows adjacent
    left
    refine ⟨?_, by rw [hc1', hc2', hcol]⟩
    apply AG_adj
    · exact ⟨row (s u), row (s blank), row u, Mr, by rw [Nat.dist_comm]; exact hrow, hr1', hr2'⟩
    · rw [hr1', hr2']; exact cons_ne (dist_ne hrow).symm Mr

/-- From an `n`-move solution, walks in `AG` on each axis whose lengths sum to
    `≤ n`. -/
theorem wd_le_of_path {s t : State} {n : ℕ} (h : StepPath s t n) :
    ∃ (pr : AG.Walk (projRow s) (projRow t)) (pc : AG.Walk (projCol s) (projCol t)),
      pr.length + pc.length ≤ n := by
  induction h with
  | refl s => exact ⟨SimpleGraph.Walk.nil, SimpleGraph.Walk.nil, by simp⟩
  | @step s s' t n hmove _ ih =>
      obtain ⟨pr, pc, hlen⟩ := ih
      rcases move_proj hmove with ⟨hadjR, hcolEq⟩ | ⟨hrowEq, hadjC⟩
      · refine ⟨SimpleGraph.Walk.cons hadjR pr, pc.copy hcolEq.symm rfl, ?_⟩
        rw [SimpleGraph.Walk.length_cons, SimpleGraph.Walk.length_copy]
        omega
      · refine ⟨pr.copy hrowEq.symm rfl, SimpleGraph.Walk.cons hadjC pc, ?_⟩
        rw [SimpleGraph.Walk.length_copy, SimpleGraph.Walk.length_cons]
        omega

@[simp] theorem wd_goal : WD goal = 0 := by
  simp [WD, WDrow, WDcol, SimpleGraph.dist_self]

/-- **Admissibility.** Walking Distance never exceeds the number of moves in any
    solution taking `s` to the solved board. -/
theorem wd_admissible {s : State} {n : ℕ} (h : StepPath s goal n) : WD s ≤ n := by
  obtain ⟨pr, pc, hlen⟩ := wd_le_of_path h
  have h1 : WDrow s ≤ pr.length := SimpleGraph.dist_le pr
  have h2 : WDcol s ≤ pc.length := SimpleGraph.dist_le pc
  unfold WD
  omega

end Puzzle15
