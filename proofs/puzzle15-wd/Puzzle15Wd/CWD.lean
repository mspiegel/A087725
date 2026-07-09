import Mathlib
import Puzzle15Wd.WD

/-!
# Toward cWD (escape-constrained WD) admissibility — 15-puzzle

cWD strengthens WD with a *side constraint*: for each goal line `g`, any solution
must make at least `escapeDemand g s` **type-`g` escapes** (moves taking a
goal-line-`g` tile out of physical line `g`), where
`escapeDemand g s = (#residents of line g) − (largest order-preserving subset)`
is the `n`-tile linear-conflict / longest-increasing-subsequence bound.

This file proves the **forced-escape bound** `row_escape_bound`: any solution
makes at least `escapeDemand g s` type-`g` escapes. That is exactly the
soundness of cWD's side constraint. The proof is a monovariant — `escapeDemand`
is `0` at the goal and rises by at most one per move, and only on a genuine
escape (`move_demand_le`) — built on a small LIS kernel (`maxKeptCard_*`) that
mathlib lacks (no longest-increasing-subsequence / Dilworth API). No `sorry`.

Note: solutions are carried here by a **`Type`-valued** `Sol` (not the `Prop`
`StepPath`), because counting escapes eliminates a path into `ℕ`, which a `Prop`
cannot do. `Sol.toPath` recovers a `StepPath`, so WD admissibility still applies.
-/

namespace Puzzle15

open Finset

attribute [local instance] Classical.propDecidable

/-- Type-level solution carrier (so escapes can be counted). -/
inductive Sol : State → State → ℕ → Type
  | refl (s : State) : Sol s s 0
  | step {s s' t : State} {n : ℕ} : Move s s' → Sol s' t n → Sol s t (n + 1)

/-- Forget the move data to recover a `StepPath`; lets WD admissibility apply. -/
def Sol.toPath : {s t : State} → {n : ℕ} → Sol s t n → StepPath s t n
  | _, _, _, .refl s => .refl s
  | _, _, _, .step hm p => .step hm p.toPath

/-- The move `s → s'` (standard witness) is a **type-`g` escape**: a tile whose
    goal row is `g`, currently in physical row `g`, that leaves row `g`. -/
def RowEscape (g : ℕ) (s s' : State) : Prop :=
  ∃ u : Tile, u ≠ blank ∧ row u = g ∧ row (s u) = g ∧ row (s blank) ≠ g ∧
    s' = s.trans (Equiv.swap (s u) (s blank))

/-- Number of type-`g` escapes along a solution. -/
noncomputable def rowEscapes (g : ℕ) : {s t : State} → {n : ℕ} → Sol s t n → ℕ
  | _, _, _, .refl _ => 0
  | _, _, _, .step (s := s) (s' := s') _ p =>
      (if RowEscape g s s' then 1 else 0) + rowEscapes g p

/-- Escapes never exceed the solution length (each move is at most one escape). -/
theorem rowEscapes_le (g : ℕ) {s t : State} {n : ℕ} (p : Sol s t n) :
    rowEscapes g p ≤ n := by
  induction p with
  | refl s => simp [rowEscapes]
  | step hm q ih => simp only [rowEscapes]; split <;> omega

/-- Residents of goal line `g`: non-blank tiles whose goal row is `g` and that
    currently sit in physical row `g`. -/
def residents (g : ℕ) (s : State) : Finset Tile :=
  NB.filter (fun t => row t = g ∧ row (s t) = g)

/-- The "kept" predicate for an abstract physical-order `phys`: on `T`, `phys`
    order implies goal-column order. -/
def Kept (phys : Tile → ℕ) (T : Finset Tile) : Prop :=
  ∀ t₁ ∈ T, ∀ t₂ ∈ T, phys t₁ < phys t₂ → col t₁ < col t₂

/-- Largest kept subset of `R` under `phys` — the abstract LIS bound (mathlib has
    no such API, so we develop the pieces we need). -/
noncomputable def maxKeptCard (R : Finset Tile) (phys : Tile → ℕ) : ℕ :=
  (R.powerset.filter (Kept phys)).sup Finset.card

theorem maxKeptCard_le (R : Finset Tile) (phys : Tile → ℕ) :
    maxKeptCard R phys ≤ R.card := by
  apply Finset.sup_le
  intro T hT
  rw [Finset.mem_filter, Finset.mem_powerset] at hT
  exact Finset.card_le_card hT.1

/-- If everything is kept, the whole set is the max. -/
theorem maxKeptCard_full {R : Finset Tile} {phys : Tile → ℕ} (h : Kept phys R) :
    maxKeptCard R phys = R.card := by
  refine le_antisymm (maxKeptCard_le R phys) ?_
  apply Finset.le_sup (f := Finset.card)
  rw [Finset.mem_filter, Finset.mem_powerset]
  exact ⟨subset_rfl, h⟩

/-- Adding one element raises the max by at most one. -/
theorem maxKeptCard_insert_le (u : Tile) (R : Finset Tile) (phys : Tile → ℕ) :
    maxKeptCard (insert u R) phys ≤ maxKeptCard R phys + 1 := by
  apply Finset.sup_le
  intro T hT
  rw [Finset.mem_filter, Finset.mem_powerset] at hT
  obtain ⟨hsub, hkept⟩ := hT
  have hkept' : Kept phys (T.erase u) :=
    fun t₁ h₁ t₂ h₂ => hkept t₁ (Finset.mem_of_mem_erase h₁) t₂ (Finset.mem_of_mem_erase h₂)
  have hmem : T.erase u ∈ R.powerset.filter (Kept phys) := by
    rw [Finset.mem_filter, Finset.mem_powerset]
    refine ⟨fun x hx => ?_, hkept'⟩
    have := hsub (Finset.mem_of_mem_erase hx)
    rcases Finset.mem_insert.1 this with h | h
    · exact absurd h (Finset.ne_of_mem_erase hx)
    · exact h
  have hle : (T.erase u).card ≤ maxKeptCard R phys := Finset.le_sup hmem
  have : T.card ≤ (T.erase u).card + 1 := by
    by_cases hu : u ∈ T
    · rw [Finset.card_erase_of_mem hu]; omega
    · rw [Finset.erase_eq_of_notMem hu]; omega
  omega

/-- Adding one element never lowers the max. -/
theorem maxKeptCard_le_insert (u : Tile) (R : Finset Tile) (phys : Tile → ℕ) :
    maxKeptCard R phys ≤ maxKeptCard (insert u R) phys := by
  apply Finset.sup_le
  intro T hT
  rw [Finset.mem_filter, Finset.mem_powerset] at hT
  apply Finset.le_sup (f := Finset.card)
  rw [Finset.mem_filter, Finset.mem_powerset]
  exact ⟨hT.1.trans (Finset.subset_insert u R), hT.2⟩

/-- The max depends on `phys` only through the order it induces on `R`. -/
theorem maxKeptCard_congr {R : Finset Tile} {phys phys' : Tile → ℕ}
    (h : ∀ t₁ ∈ R, ∀ t₂ ∈ R, (phys t₁ < phys t₂ ↔ phys' t₁ < phys' t₂)) :
    maxKeptCard R phys = maxKeptCard R phys' := by
  unfold maxKeptCard
  congr 1
  apply Finset.filter_congr
  intro T hT
  rw [Finset.mem_powerset] at hT
  constructor
  · exact fun hk t₁ h₁ t₂ h₂ hlt => hk t₁ h₁ t₂ h₂ ((h t₁ (hT h₁) t₂ (hT h₂)).2 hlt)
  · exact fun hk t₁ h₁ t₂ h₂ hlt => hk t₁ h₁ t₂ h₂ ((h t₁ (hT h₁) t₂ (hT h₂)).1 hlt)

/-- The largest order-preserving subset of residents — the forced-escape LIS. -/
noncomputable def keepable (g : ℕ) (s : State) : ℕ :=
  maxKeptCard (residents g s) (fun t => col (s t))

/-- The forced-escape demand for goal line `g`. -/
noncomputable def escapeDemand (g : ℕ) (s : State) : ℕ :=
  (residents g s).card - keepable g s

/-- At the goal every resident is already sorted, so `keepable` is the full
    resident set. -/
theorem keepable_goal (g : ℕ) : keepable g goal = (residents g goal).card := by
  apply maxKeptCard_full
  intro t₁ _ t₂ _ h
  simpa [goal] using h

@[simp] theorem demand_goal (g : ℕ) : escapeDemand g goal = 0 := by
  unfold escapeDemand; rw [keepable_goal]; omega

/-! ## The per-move monovariant

The whole-trajectory order-preservation + LIS argument is captured by
`move_demand_le`: one move raises `escapeDemand` by at most 1, and only a genuine
type-`g` escape can raise it at all. The case split is on whether the moved tile
`u` is a resident of line `g` before/after the move; the LIS kernel
(`maxKeptCard_*`) supplies every combinatorial step, and the horizontal in-row
case is closed by adjacent-swap order preservation.
-/

theorem mem_residents {g : ℕ} {t : Tile} {σ : State} :
    t ∈ residents g σ ↔ t ≠ blank ∧ row t = g ∧ row (σ t) = g := by
  simp only [residents, NB, Finset.mem_filter, Finset.mem_univ, true_and]

/-- Two cells in the same row and the same column coincide. -/
theorem cell_eq_of_row_col {a b : Cell} (hr : row a = row b) (hc : col a = col b) :
    a = b := by
  simp only [row, col] at hr hc
  exact Fin.val_injective (by omega)

/-- **Per-move escape monovariant.** One move raises `escapeDemand` by at most
one, and only a genuine type-`g` escape can raise it at all. Case split on
whether the moved tile `u` is a resident of goal line `g` before/after the move
(`Rs = residents g s`, `Rs' = residents g s'` agree away from `u`, and every
non-`u` resident keeps its column):

* `u ∈ Rs, u ∈ Rs'` — a horizontal slide within row `g`; the mover crosses no
  other resident (all residents' cells lie in row `g`, so their columns avoid
  both `col (s u)` and the adjacent `col (s blank)`), hence the column order is
  preserved and `keepable` is unchanged.
* `u ∈ Rs, u ∉ Rs'` — a genuine type-`g` escape; `Rs = insert u Rs'`, so the
  demand rises by at most the allowed `+1`.
* `u ∉ Rs, u ∈ Rs'` — an entry; `Rs' = insert u Rs`, and the demand cannot rise.
* `u ∉ Rs, u ∉ Rs'` — residents and their columns unchanged. -/
theorem move_demand_le (g : ℕ) {s s' : State} (h : Move s s') :
    escapeDemand g s ≤ escapeDemand g s' + (if RowEscape g s s' then 1 else 0) := by
  obtain ⟨u, hu, hadj, hs'⟩ := h
  have su' : s' u = s blank := by rw [hs', Equiv.trans_apply, Equiv.swap_apply_left]
  have hval : ∀ t : Tile, t ≠ u → t ≠ blank → s' t = s t := by
    intro t htu htb
    rw [hs', Equiv.trans_apply,
        Equiv.swap_apply_of_ne_of_ne (s.injective.ne htu) (s.injective.ne htb)]
  have hmem : ∀ t : Tile, t ≠ u → (t ∈ residents g s ↔ t ∈ residents g s') := by
    intro t htu
    by_cases htb : t = blank
    · subst htb; simp [mem_residents]
    · rw [mem_residents, mem_residents, hval t htu htb]
  by_cases huS : u ∈ residents g s <;> by_cases huS' : u ∈ residents g s'
  · -- u resident before AND after ⇒ horizontal slide within row g: order preserved
    have hRR : residents g s = residents g s' := by
      ext x
      by_cases hx : x = u
      · subst hx; exact ⟨fun _ => huS', fun _ => huS⟩
      · exact hmem x hx
    have hrgu : row (s u) = g := (mem_residents.1 huS).2.2
    have hrgb : row (s blank) = g := by
      have h2 := (mem_residents.1 huS').2.2
      rwa [su'] at h2
    have hdist : Nat.dist (col (s blank)) (col (s u)) = 1 := by
      rcases hadj with ⟨_, hc⟩ | ⟨_, hr⟩
      · exact hc
      · rw [hrgb, hrgu] at hr
        simp [Nat.dist_self] at hr
    unfold Nat.dist at hdist
    have hkeq : keepable g s = keepable g s' := by
      unfold keepable
      rw [← hRR]
      apply maxKeptCard_congr
      intro t₁ h₁ t₂ h₂
      show col (s t₁) < col (s t₂) ↔ col (s' t₁) < col (s' t₂)
      by_cases e₁ : t₁ = u <;> by_cases e₂ : t₂ = u
      · rw [e₁, e₂]; simp
      · -- t₁ = u, t₂ ≠ u: only u's column moves, and only to the adjacent free slot
        rw [e₁]
        have hb₂ : t₂ ≠ blank := (mem_residents.1 h₂).1
        have hr₂ : row (s t₂) = g := (mem_residents.1 h₂).2.2
        have hna : col (s t₂) ≠ col (s u) := fun hc =>
          e₂ (s.injective (cell_eq_of_row_col (by rw [hr₂, hrgu]) hc))
        have hnb : col (s t₂) ≠ col (s blank) := fun hc =>
          hb₂ (s.injective (cell_eq_of_row_col (by rw [hr₂, hrgb]) hc))
        rw [su', hval t₂ e₂ hb₂]
        omega
      · -- t₁ ≠ u, t₂ = u: symmetric
        rw [e₂]
        have hb₁ : t₁ ≠ blank := (mem_residents.1 h₁).1
        have hr₁ : row (s t₁) = g := (mem_residents.1 h₁).2.2
        have hna : col (s t₁) ≠ col (s u) := fun hc =>
          e₁ (s.injective (cell_eq_of_row_col (by rw [hr₁, hrgu]) hc))
        have hnb : col (s t₁) ≠ col (s blank) := fun hc =>
          hb₁ (s.injective (cell_eq_of_row_col (by rw [hr₁, hrgb]) hc))
        rw [su', hval t₁ e₁ hb₁]
        omega
      · rw [hval t₁ e₁ (mem_residents.1 h₁).1, hval t₂ e₂ (mem_residents.1 h₂).1]
    unfold escapeDemand
    rw [hRR, hkeq]
    exact Nat.le_add_right _ _
  · -- u resident before, not after ⇒ a genuine type-g escape: demand may rise by 1
    obtain ⟨hub, hrug, hrsug⟩ := mem_residents.1 huS
    have hesc : RowEscape g s s' := by
      refine ⟨u, hub, hrug, hrsug, fun hbg => huS' ?_, hs'⟩
      rw [mem_residents]
      refine ⟨hub, hrug, ?_⟩
      rw [su']; exact hbg
    have hRR : residents g s = insert u (residents g s') := by
      ext x
      by_cases hx : x = u
      · rw [hx]; exact ⟨fun _ => Finset.mem_insert_self u _, fun _ => huS⟩
      · rw [Finset.mem_insert]
        constructor
        · intro hx'; exact Or.inr ((hmem x hx).1 hx')
        · rintro (hh | hh)
          · exact absurd hh hx
          · exact (hmem x hx).2 hh
    have hcard : (residents g s).card = (residents g s').card + 1 := by
      rw [hRR, Finset.card_insert_of_notMem huS']
    have hcongr : maxKeptCard (residents g s') (fun t => col (s t))
                = maxKeptCard (residents g s') (fun t => col (s' t)) := by
      apply maxKeptCard_congr
      intro t₁ h₁ t₂ h₂
      show col (s t₁) < col (s t₂) ↔ col (s' t₁) < col (s' t₂)
      rw [hval t₁ (fun he => huS' (he ▸ h₁)) (mem_residents.1 h₁).1,
          hval t₂ (fun he => huS' (he ▸ h₂)) (mem_residents.1 h₂).1]
    have hkle : keepable g s' ≤ keepable g s := by
      unfold keepable
      rw [hRR, ← hcongr]
      exact maxKeptCard_le_insert u _ _
    have h2 : keepable g s' ≤ (residents g s').card := maxKeptCard_le _ _
    rw [if_pos hesc]
    unfold escapeDemand
    omega
  · -- u resident after, not before (an entry): demand cannot rise
    have hRR : residents g s' = insert u (residents g s) := by
      ext x
      by_cases hx : x = u
      · rw [hx]; exact ⟨fun _ => Finset.mem_insert_self u _, fun _ => huS'⟩
      · rw [Finset.mem_insert]
        constructor
        · intro hx'; exact Or.inr ((hmem x hx).2 hx')
        · rintro (hh | hh)
          · exact absurd hh hx
          · exact (hmem x hx).1 hh
    have hcard : (residents g s').card = (residents g s).card + 1 := by
      rw [hRR, Finset.card_insert_of_notMem huS]
    have hcongr : maxKeptCard (residents g s) (fun t => col (s t))
                = maxKeptCard (residents g s) (fun t => col (s' t)) := by
      apply maxKeptCard_congr
      intro t₁ h₁ t₂ h₂
      show col (s t₁) < col (s t₂) ↔ col (s' t₁) < col (s' t₂)
      rw [hval t₁ (fun he => huS (he ▸ h₁)) (mem_residents.1 h₁).1,
          hval t₂ (fun he => huS (he ▸ h₂)) (mem_residents.1 h₂).1]
    have hkle : keepable g s' ≤ keepable g s + 1 := by
      unfold keepable
      rw [hRR, hcongr]
      exact maxKeptCard_insert_le u _ _
    have h1 : keepable g s ≤ (residents g s).card := maxKeptCard_le _ _
    have hle : escapeDemand g s ≤ escapeDemand g s' := by
      unfold escapeDemand; omega
    exact hle.trans (Nat.le_add_right _ _)
  · -- u resident in neither: residents and demand unchanged
    have hRR : residents g s = residents g s' := by
      ext x
      by_cases hx : x = u
      · subst hx; exact ⟨fun hh => absurd hh huS, fun hh => absurd hh huS'⟩
      · exact hmem x hx
    have hcongr : maxKeptCard (residents g s) (fun t => col (s t))
                = maxKeptCard (residents g s) (fun t => col (s' t)) := by
      apply maxKeptCard_congr
      intro t₁ h₁ t₂ h₂
      show col (s t₁) < col (s t₂) ↔ col (s' t₁) < col (s' t₂)
      rw [hval t₁ (fun he => huS (he ▸ h₁)) (mem_residents.1 h₁).1,
          hval t₂ (fun he => huS (he ▸ h₂)) (mem_residents.1 h₂).1]
    have hkeq : keepable g s = keepable g s' := by
      unfold keepable
      rw [← hRR, hcongr]
    unfold escapeDemand
    rw [hRR, hkeq]
    exact Nat.le_add_right _ _

/-- Telescoped monovariant: `escapeDemand` at the start is bounded by its value at
    the end plus the escapes made in between. -/
theorem demand_le_of_sol (g : ℕ) {s t : State} {n : ℕ} (p : Sol s t n) :
    escapeDemand g s ≤ escapeDemand g t + rowEscapes g p := by
  induction p with
  | refl s => simp [rowEscapes]
  | @step s s' t n hm p' ih =>
      simp only [rowEscapes]
      have hmove := move_demand_le g hm
      by_cases hesc : RowEscape g s s'
      · rw [if_pos hesc] at hmove ⊢; omega
      · rw [if_neg hesc] at hmove ⊢; omega

/-- **Forced-escape bound.** Any solution makes at least `escapeDemand g s`
    type-`g` escapes — modulo the single local lemma `move_demand_le`. -/
theorem row_escape_bound (g : ℕ) {s : State} {n : ℕ} (p : Sol s goal n) :
    escapeDemand g s ≤ rowEscapes g p := by
  have h := demand_le_of_sol g p
  rwa [demand_goal, Nat.zero_add] at h

end Puzzle15
