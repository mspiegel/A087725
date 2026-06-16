//! Zero-aware PDB heuristics for the 15-puzzle IDA\*. Ported from
//! `puzzle24::pdb::heuristic`'s `ZpdbInc`/`ZpdbCtx`, plus a non-incremental
//! [`ZpdbHeuristic`] and an additive/max combiner.
//!
//! - [`ZpdbHeuristic`]: single zero-aware PDB lookup (implements [`Heuristic`]).
//! - [`AdditiveZpdbHeuristic`]: sum over a disjoint pattern partition.
//! - [`ZpdbInc`]: incremental Korf-max evaluator —
//!   `max(zpdb(s), zpdb(reflect(s)))` advanced one move per node — implements
//!   [`IncHeuristic`].
//!
//! Admissibility: each ZPDB value lower-bounds the true distance (it is a BFS
//! over a contracted quotient of the real move graph) and pointwise dominates
//! the blank-agnostic additive PDB; the sum over disjoint patterns is
//! admissible; max of the normal and diagonal-reflected views is admissible
//! because the reflection preserves distance to GOAL.

use super::pattern::{Pattern, ProjectedState};
use super::zdb::ZPatternDb;
use crate::puzzle15::search::{Heuristic, IncHeuristic};
use crate::puzzle15::state::{Move, State};
use crate::puzzle15::symmetry::{reflect, transpose_move};

/// Single zero-aware-PDB heuristic.
pub struct ZpdbHeuristic<'a> {
    db: &'a ZPatternDb,
}

impl<'a> ZpdbHeuristic<'a> {
    pub fn new(db: &'a ZPatternDb) -> Self {
        Self { db }
    }
}

impl<'a> Heuristic for ZpdbHeuristic<'a> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        self.db.cold_lookup(s)
    }
}

/// Sum-of-ZPDBs heuristic over a disjoint pattern partition.
pub struct AdditiveZpdbHeuristic<'a> {
    dbs: &'a [ZPatternDb],
}

impl<'a> AdditiveZpdbHeuristic<'a> {
    /// Verifies pairwise disjointness at construction.
    pub fn new(dbs: &'a [ZPatternDb]) -> Self {
        let mut union: u32 = 0;
        for db in dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "additive ZPDB patterns must be disjoint");
            union |= p;
        }
        Self { dbs }
    }

    pub fn coverage(&self) -> Pattern {
        let mut union: u32 = 0;
        for db in self.dbs {
            union |= db.pattern().0;
        }
        Pattern(union)
    }
}

impl<'a> Heuristic for AdditiveZpdbHeuristic<'a> {
    #[inline]
    fn h(&self, s: &State) -> u8 {
        let mut sum: u32 = 0;
        for db in self.dbs {
            sum += db.cold_lookup(s) as u32;
        }
        sum.min(u8::MAX as u32) as u8
    }
}

/// Incremental zero-aware-PDB Korf-max heuristic:
/// `max(zpdb(s), zpdb(reflect(s)))` over `N` pairwise-disjoint ZPDBs. Per-node
/// context tracks each PDB's projected board, abstract `(m, p, r)` index, and
/// running absolute `h`, in both the normal and reflected views.
///
/// On every puzzle move `m`:
/// - the normal projected state slides by `m`;
/// - the reflected projected state slides by [`transpose_move`]`(m)`;
/// - if the index is unchanged (the swap was with an *anon* tile — the blank
///   wandered inside its region in the abstract ZPDB graph), `h` carries
///   through unchanged;
/// - otherwise the swap is a unit ZPDB edge, and the new `h` comes from
///   [`ZPatternDb::diff_lookup`] in O(1).
pub struct ZpdbInc<'a, const N: usize> {
    dbs: [&'a ZPatternDb; N],
}

#[derive(Clone, Copy)]
pub struct ZpdbCtx<const N: usize> {
    normal: [ProjectedState; N],
    reflected: [ProjectedState; N],
    n_idx: [u64; N],
    r_idx: [u64; N],
    n_h: [u8; N],
    r_h: [u8; N],
}

impl<'a, const N: usize> ZpdbInc<'a, N> {
    /// Build from `N` ZPDBs; panics unless the patterns are pairwise disjoint.
    pub fn new(dbs: [&'a ZPatternDb; N]) -> Self {
        let mut union: u32 = 0;
        for db in &dbs {
            let p = db.pattern().0;
            assert_eq!(union & p, 0, "ZPDB patterns must be disjoint");
            union |= p;
        }
        Self { dbs }
    }

    #[inline]
    fn value(&self, ctx: &ZpdbCtx<N>) -> u8 {
        let mut hn: u32 = 0;
        let mut hr: u32 = 0;
        for i in 0..N {
            hn += ctx.n_h[i] as u32;
            hr += ctx.r_h[i] as u32;
        }
        hn.max(hr).min(u8::MAX as u32) as u8
    }
}

impl<'a, const N: usize> IncHeuristic for ZpdbInc<'a, N> {
    type Ctx = ZpdbCtx<N>;

    fn root(&self, s: &State) -> (u8, Self::Ctx) {
        let rs = reflect(s);
        let mut ctx = ZpdbCtx {
            normal: std::array::from_fn(|i| ProjectedState::from_state(s, self.dbs[i].pattern())),
            reflected: std::array::from_fn(|i| {
                ProjectedState::from_state(&rs, self.dbs[i].pattern())
            }),
            n_idx: [0u64; N],
            r_idx: [0u64; N],
            n_h: [0u8; N],
            r_h: [0u8; N],
        };
        for i in 0..N {
            let db = self.dbs[i];
            ctx.n_idx[i] = db.layout().rank(&ctx.normal[i], db.pattern());
            ctx.r_idx[i] = db.layout().rank(&ctx.reflected[i], db.pattern());
            ctx.n_h[i] = db.cold_lookup_proj(&ctx.normal[i]);
            ctx.r_h[i] = db.cold_lookup_proj(&ctx.reflected[i]);
        }
        (self.value(&ctx), ctx)
    }

    fn advance(&self, parent: &Self::Ctx, _child: &State, m: Move) -> (u8, Self::Ctx) {
        let tm = transpose_move(m);
        let mut ctx = *parent;
        for i in 0..N {
            let db = self.dbs[i];
            let p = db.pattern();

            // Normal view.
            let (np, _) = parent.normal[i].apply(m);
            let n_idx = db.layout().rank(&np, p);
            let n_h = if n_idx == parent.n_idx[i] {
                parent.n_h[i]
            } else {
                db.diff_lookup(n_idx, parent.n_h[i])
            };
            ctx.normal[i] = np;
            ctx.n_idx[i] = n_idx;
            ctx.n_h[i] = n_h;

            // Reflected view (same logic, transpose_move(m)).
            let (rp, _) = parent.reflected[i].apply(tm);
            let r_idx = db.layout().rank(&rp, p);
            let r_h = if r_idx == parent.r_idx[i] {
                parent.r_h[i]
            } else {
                db.diff_lookup(r_idx, parent.r_h[i])
            };
            ctx.reflected[i] = rp;
            ctx.r_idx[i] = r_idx;
            ctx.r_h[i] = r_h;
        }
        (self.value(&ctx), ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::pdb::db::PatternDb;
    use crate::puzzle15::search::tests_util::bfs_distances;
    use crate::puzzle15::state::{State, GOAL};

    #[test]
    fn zpdb_heuristic_zero_on_goal() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        assert_eq!(ZpdbHeuristic::new(&zdb).h(&GOAL), 0);
    }

    /// Admissibility vs truth: zpdb(s) <= true_distance(s) over bfs_distances(10).
    #[test]
    fn zpdb_admissible_vs_truth_k3() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3]));
        let h = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(10);
        let mut violations = 0;
        for (raw, &td) in &truth {
            if h.h(&State(*raw)) > td {
                violations += 1;
            }
        }
        assert_eq!(violations, 0, "{} admissibility violations", violations);
    }

    #[test]
    fn zpdb_admissible_vs_truth_k4() {
        let zdb = ZPatternDb::build(Pattern::new(&[1, 2, 3, 4]));
        let h = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(10);
        let mut violations = 0;
        for (raw, &td) in &truth {
            if h.h(&State(*raw)) > td {
                violations += 1;
            }
        }
        assert_eq!(violations, 0, "{} admissibility violations", violations);
    }

    /// Tighter-or-equal than blank-agnostic: zpdb(s) >= additive.h(s) pointwise.
    #[test]
    fn zpdb_dominates_blank_agnostic_k3() {
        let pattern = Pattern::new(&[1, 2, 3]);
        let zdb = ZPatternDb::build(pattern);
        let pdb = PatternDb::build(pattern);
        let zh = ZpdbHeuristic::new(&zdb);
        let truth = bfs_distances(10);
        let mut strictly_greater = 0usize;
        let mut total = 0usize;
        for (raw, _) in &truth {
            let s = State(*raw);
            let z = zh.h(&s);
            let a = pdb.h(&s);
            assert!(z >= a, "zpdb {} < blank-agnostic {} for {:?}", z, a, raw);
            if z > a {
                strictly_greater += 1;
            }
            total += 1;
        }
        println!(
            "zpdb strictly tighter than blank-agnostic on {}/{} states ({:.1}%)",
            strictly_greater,
            total,
            100.0 * strictly_greater as f64 / total as f64
        );
    }

    /// Incremental advance must agree with a fresh root at every step.
    #[test]
    fn zpdb_inc_advance_matches_fresh_root_random_walk() {
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3])),
            ZPatternDb::build(Pattern::new(&[6, 7, 8])),
        ];
        let inc = ZpdbInc::new([&dbs[0], &dbs[1]]);

        let mut rng: u64 = 0xFEED_FACE_C0DE_BAD0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let (_, mut ctx) = IncHeuristic::root(&inc, &s);
        for i in 0..1500 {
            let opts: Vec<Move> = s.legal_moves().iter().collect();
            let m = opts[(next() as usize) % opts.len()];
            let ns = s.apply(m);
            let (h_adv, ctx_adv) = inc.advance(&ctx, &ns, m);
            let (h_fresh, _) = IncHeuristic::root(&inc, &ns);
            assert_eq!(h_adv, h_fresh, "advance diverged at step {} (state {:?})", i, ns.0);
            s = ns;
            ctx = ctx_adv;
        }
    }

    /// IDA* with the zero-aware heuristic returns optimal-length solutions.
    #[test]
    fn zpdb_inc_idastar_optimal_on_known_depths() {
        use crate::puzzle15::search::idastar_inc_with_stats;
        let dbs = [
            ZPatternDb::build(Pattern::new(&[1, 2, 3, 4])),
            ZPatternDb::build(Pattern::new(&[5, 6, 7, 8])),
        ];
        let inc = ZpdbInc::new([&dbs[0], &dbs[1]]);
        let truth = bfs_distances(12);
        // Sample ~20 states of varying known depth.
        let mut samples: Vec<(State, u8)> = truth
            .iter()
            .map(|(raw, &d)| (State(*raw), d))
            .filter(|(_, d)| *d >= 8)
            .collect();
        samples.sort_by_key(|(s, _)| s.0);
        samples.truncate(20);
        assert!(samples.len() >= 10, "not enough deep samples ({})", samples.len());
        for (s, true_d) in &samples {
            let (sol, _) = idastar_inc_with_stats(s, &inc);
            let sol = sol.expect("ZPDB IDA* found no solution");
            assert_eq!(sol.len() as u8, *true_d, "non-optimal for {:?}", s.0);
            let mut cur = *s;
            for &m in &sol {
                cur = cur.apply(m);
            }
            assert_eq!(cur, GOAL, "solution did not reach GOAL");
        }
    }
}
