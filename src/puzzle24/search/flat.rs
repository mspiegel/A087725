//! Flat (iterative) IDA\* — one hardcoded configuration, no recursion, no undo.
//!
//! **The** 24-puzzle search engine, specialised to the exact stack the `R`
//! lower-bound program runs: **cWD + move-DFA + neighbour-WD child pre-prune +
//! root σ-orbit split**, bounded lower-bound mode only.
//!
//! It began as a second engine beside a generic recursive one, which is why so
//! much below is phrased against "the generic engine". That engine has since
//! been deleted; what survives of it is [`idastar`](super::idastar)'s Copy-path
//! ladder, kept only for the ML/corridor tooling and the heuristic-correctness
//! tests, and never used to search here.
//!
//! # Why a second engine
//!
//! §8k names three ways to beat the settled frontier and the first is "a cheaper
//! cWD — paid on *every* node, so any node-identical speedup is pure win". §8l's
//! profile of the 2-tier splits as search loop 29% / ZPDB 40% / cWD 23% / glue
//! 8%; strip the ZPDB and the glue for a cWD-only run and the remainder
//! renormalises to roughly **56% search loop, 44% cWD**, so the loop itself is
//! the largest single component here.
//!
//! The mechanism is measured. `5b75923` won −2.0% by bundling
//! `search_inc_mut`'s 15 arguments behind one `&SearchInv`; the disassembly
//! showed a 384 B frame with **85 stack spill/reload instructions per node**,
//! cut to 288 B / 53 by the bundle. A flat loop removes the call frame
//! altogether — the recursion state lives in a depth-indexed arena, so there is
//! nothing to spill across a call boundary that no longer exists.
//!
//! # Structure
//!
//! Depth is bounded by the IDA\* threshold, so every array is allocated once at
//! construction and indexed by depth ([`MAX_DEPTH`]). Three regions, split by
//! *index* and by *size*:
//!
//! - [`FrameHot`] — the ~12 B of scalars read on every child, array-of-structs so
//!   depth `d` and `d+1` usually share one cache line.
//! - `board` — the ancestor boards, 32 B-aligned slots so a copy is two aligned
//!   register moves instead of a 16 + 8 + 1 sequence.
//! - `row` / `col` — the per-axis cWD state, indexed by `row_at[d]` / `col_at[d]`
//!   rather than by `d`, because a move touches exactly one axis and the other is
//!   **shared with an ancestor rather than copied**.
//!
//! That sharing is what makes the arena cheap: the per-node cost for the
//! untouched axis is one byte (the index), against the 20 B `CwdUndo` write the
//! generic engine performs.
//!
//! # Node identity
//!
//! Every choice here is required to reproduce the deleted recursive engine's
//! tree **exactly** — same nodes, same count, same next threshold. Nothing in
//! this module changes a pruning decision; it only evaluates the same predicates
//! more cheaply.
//!
//! Since that engine is gone, the gate is now [`super::flat_oracle`]: 180 cases
//! and ~3.7 × 10^7 nodes of its frozen answers, captured before removal. It
//! **cannot be regenerated** — a mismatch means this engine's tree changed, not
//! that the fixture is stale.

use super::cwd::{
    demand_col_line, demand_row_line, key_bit, pack, project, surcharge_from_curves, Cwd,
    CwdMerged, DEMAND_LUT_LEN, KEY_BLANK_BIT,
};
use super::idastar::{BoundedOutcome, SearchStats};
use super::move_dfa::MoveDfa;
use crate::puzzle24::state::{Move, MoveSet, State, GOAL, N_CELLS, W};
use crate::puzzle24::symmetry;

/// Depth ceiling: the 24-puzzle diameter is ≤ 205, and no IDA\* iteration ever
/// descends past its threshold, so this bounds every arena index.
const MAX_DEPTH: usize = 210;

/// Sentinel for "this move is illegal from this cell" in [`GEOM`].
const NO_CELL: u8 = u8::MAX;

// ------------------------------ compile-time tables ---------------------------

/// Per-`(blank, move)` move geometry. [`Cwd::make`](super::cwd::Cwd) recomputes
/// all of this at every node — `parent_blank = br*W + bc`, then `from/W`, `to/W`,
/// `from%W`, `to%W` — six divisions by 5 plus a multiply-add, each division
/// lowering to a multiply-high/shift/multiply/subtract on aarch64. The flat
/// engine already carries the blank cell, so one indexed load replaces the lot.
#[derive(Clone, Copy)]
struct Geom {
    /// Cell the moved tile comes from (equivalently, the blank's destination).
    from: u8,
    /// Row of `from`.
    rf: u8,
    /// Row of the blank before the move.
    rt: u8,
    /// Column of `from`.
    cf: u8,
    /// Column of the blank before the move.
    ct: u8,
    /// Whether this move slides a tile between rows (⇒ touches the row axis).
    vertical: bool,
}

const GEOM: [[Geom; 4]; N_CELLS] = {
    let z = Geom {
        from: NO_CELL,
        rf: 0,
        rt: 0,
        cf: 0,
        ct: 0,
        vertical: false,
    };
    let mut t = [[z; 4]; N_CELLS];
    let mut b = 0;
    while b < N_CELLS {
        let rt = (b / W) as u8;
        let ct = (b % W) as u8;
        if b / W > 0 {
            let f = b - W;
            t[b][Move::Up as usize] = Geom {
                from: f as u8,
                rf: (f / W) as u8,
                rt,
                cf: (f % W) as u8,
                ct,
                vertical: true,
            };
        }
        if b / W < W - 1 {
            let f = b + W;
            t[b][Move::Down as usize] = Geom {
                from: f as u8,
                rf: (f / W) as u8,
                rt,
                cf: (f % W) as u8,
                ct,
                vertical: true,
            };
        }
        if b % W > 0 {
            let f = b - 1;
            t[b][Move::Left as usize] = Geom {
                from: f as u8,
                rf: (f / W) as u8,
                rt,
                cf: (f % W) as u8,
                ct,
                vertical: false,
            };
        }
        if b % W < W - 1 {
            let f = b + 1;
            t[b][Move::Right as usize] = Geom {
                from: f as u8,
                rf: (f / W) as u8,
                rt,
                cf: (f % W) as u8,
                ct,
                vertical: false,
            };
        }
        b += 1;
    }
    t
};

/// A board holding each cell's **goal coordinates** rather than its tile number:
/// `(goal_row << 4) | goal_col` for a tile, and [`BLANK_CODE`] for the blank.
///
/// The engine never needs a tile's *identity*, only where it wants to be. Storing
/// the destination directly turns `GOAL_ROW[tile]` — a dependent L1 load sitting
/// at the head of the node's longest chain, right before the `KEY_DELTA` index
/// and the probe — into a shift on a byte already in a register. That is the same
/// lever G2 pulled for -2.4%: shorten the *start* of the chain.
///
/// The encoding is a bijection on the 24 tiles, since tile `t` has goal cell
/// `t-1` and `(t-1) < 24` covers every `(row, col)` except `(4,4)`. The blank
/// takes `0xFF`, whose nibbles are both `0xF` and therefore never equal any
/// `g ∈ 0..5` — which is what lets the demand predicate drop its `tile != 0`
/// term entirely.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Coded([u8; N_CELLS]);

/// Code for the blank. Deliberately not `0x44` (the blank's own goal cell), so
/// that neither nibble can match a real line index.
const BLANK_CODE: u8 = 0xFF;

/// `TILE_CODE[t]` = the goal-coordinate code of tile `t`; index 0 is the blank.
const TILE_CODE: [u8; N_CELLS] = {
    let mut t = [BLANK_CODE; N_CELLS];
    let mut i = 1;
    while i < N_CELLS {
        t[i] = (((i - 1) / W) as u8) << 4 | ((i - 1) % W) as u8;
        i += 1;
    }
    t
};

impl Coded {
    #[inline]
    fn encode(s: &State) -> Self {
        let mut a = [BLANK_CODE; N_CELLS];
        let mut i = 0;
        while i < N_CELLS {
            a[i] = TILE_CODE[s.0[i] as usize];
            i += 1;
        }
        Coded(a)
    }

    /// Inverse of [`encode`](Self::encode). Only *executed* by debug assertions
    /// and tests, which still want to talk in tile numbers — but it must exist
    /// in release too, because `debug_assert_eq!` type-checks its body even
    /// where it compiles the check away.
    #[inline]
    fn decode(&self) -> State {
        let mut a = [0u8; N_CELLS];
        for (i, &c) in self.0.iter().enumerate() {
            a[i] = if c == BLANK_CODE {
                0
            } else {
                ((c >> 4) as usize * W + (c & 0xF) as usize + 1) as u8
            };
        }
        State(a)
    }
}

/// `CAND[b][m]` = the moves legal from blank cell `b`, minus `m.inverse()` — the
/// candidate children of a node reached *by* `m`. Folding the immediate-undo
/// filter into the table means the inner loop never generates a move it will
/// immediately reject, and caps the iteration at **3 below the root**
/// (`legal_moves_at` gives 2/3/4 by corner/edge/interior; one always dies to the
/// inverse filter).
///
/// The inverse of move code `i` is `i ^ 1`, since `Up`=0/`Down`=1 and
/// `Left`=2/`Right`=3.
const CAND: [[MoveSet; 4]; N_CELLS] = {
    let mut t = [[MoveSet::empty(); 4]; N_CELLS];
    let mut b = 0;
    while b < N_CELLS {
        let mut legal = 0u8;
        if b / W > 0 {
            legal |= 1 << (Move::Up as u8);
        }
        if b / W < W - 1 {
            legal |= 1 << (Move::Down as u8);
        }
        if b % W > 0 {
            legal |= 1 << (Move::Left as u8);
        }
        if b % W < W - 1 {
            legal |= 1 << (Move::Right as u8);
        }
        let mut mi = 0usize;
        while mi < 4 {
            t[b][mi] = MoveSet(legal & !(1u8 << (mi ^ 1)));
            mi += 1;
        }
        b += 1;
    }
    t
};

/// The whole packed-key update for one move, as a single addend.
///
/// A slide moves one tile of goal-line `g` from physical line `rf` to line `rt`
/// and leaves the blank on `rf`. In the packed key that is three edits: decrement
/// the 3-bit field at `(rf, g)`, increment the one at `(rt, g)`, and overwrite the
/// blank field. All three fold into one `wrapping_add`, because no field can carry
/// into its neighbour:
///
/// - `M[rf][g] ≥ 1` before the decrement (there is a tile there to move) and
///   `M[rt][g] ≤ W-1` before the increment (`rt` is the blank's line, so it holds
///   one fewer tile), while a 3-bit field holds 0..=7 — no borrow, no carry.
/// - The blank field is the *top* field (bits 60–62, bit 63 unused), so its
///   `+rf − rt` cannot disturb anything above it.
/// - When `g == W-1` the cell pair is not encoded at all — the last column is
///   derived from the blank-line margins — so the cell part is zero, which also
///   folds away the `if g < W - 1` branch.
///
/// Replaces two `key_get_cell` reads, two `key_set_cell` writes, a `key_set_blank`
/// and a branch with one 1000 B table read and one add. Measured −2.7% search
/// time on `R` at exhaust-144, node-for-node; the `pack` drift assertion in
/// `build_child` pins the arithmetic against the field-insert form.
const KEY_DELTA: [[[u64; W]; W]; W] = {
    let mut t = [[[0u64; W]; W]; W];
    let mut rf = 0;
    while rf < W {
        let mut rt = 0;
        while rt < W {
            let mut g = 0;
            while g < W {
                let mut d: u64 = 0;
                if g < W - 1 {
                    d = (1u64 << key_bit(rt, g)).wrapping_sub(1u64 << key_bit(rf, g));
                }
                d = d
                    .wrapping_add((rf as u64) << KEY_BLANK_BIT)
                    .wrapping_sub((rt as u64) << KEY_BLANK_BIT);
                t[rf][rt][g] = d;
                g += 1;
            }
            rt += 1;
        }
        rf += 1;
    }
    t
};

/// `demand_row_line` without the divisions. `cwd.rs`'s key builder recomputes
/// `gp / W` and `gp % W` for each of the `W` cells — ten divisions by 5 per call,
/// each lowering to a multiply-high/shift/multiply/subtract on aarch64 — where
/// this engine already has every tile's goal line in a table. Identical key,
/// hence identical LUT entry, hence node-identical; the `debug_assert` pins that.
///
/// Measured −4.8% search time on `R` at exhaust-144, node-for-node.
#[inline]
fn demand_row_fast(lut: &[u8; DEMAND_LUT_LEN], s: &Coded, g: usize) -> u8 {
    let mut key = 0usize;
    for c in 0..W {
        let code = s.0[g * W + c] as usize;
        // Branchless: "is this cell a resident of its own line?" is essentially
        // random against the board, and CPU-counter sampling attributed a large
        // share of this engine's mispredicts to it. On a goal-coded board the
        // goal row *is* the high nibble, so this is a shift rather than a table
        // load — and the blank's `0xF` nibble can never equal `g`, which is what
        // retires the `tile != 0` term the tile-numbered version needed.
        let res = ((code >> 4) == g) as usize;
        key |= (res * ((code & 0xF) + 1)) << (3 * c);
    }
    let d = lut[key];
    debug_assert_eq!(
        d,
        demand_row_line(lut, &s.decode(), g),
        "fast row demand key drift"
    );
    d
}

/// `demand_col_line` without the divisions; see [`demand_row_fast`].
#[inline]
fn demand_col_fast(lut: &[u8; DEMAND_LUT_LEN], s: &Coded, g: usize) -> u8 {
    let mut key = 0usize;
    for r in 0..W {
        let code = s.0[r * W + g] as usize;
        let res = ((code & 0xF) == g) as usize;
        key |= (res * ((code >> 4) + 1)) << (3 * r);
    }
    let d = lut[key];
    debug_assert_eq!(
        d,
        demand_col_line(lut, &s.decode(), g),
        "fast col demand key drift"
    );
    d
}

/// Take the lowest-numbered move out of `set`. Lowest-bit-first consumption
/// reproduces `MoveSetIter`'s discriminant order exactly, which is what keeps the
/// child order — and hence the tree — identical to the generic engine.
#[inline(always)]
fn pop_lowest(set: &mut MoveSet) -> Move {
    let i = set.0.trailing_zeros() as usize;
    set.0 &= set.0 - 1;
    Move::ALL[i]
}

// --------------------------------- the arena ----------------------------------

/// The scalars read on every child. Kept small and array-of-structs so a frame
/// and its child sit in the same cache line; the board and the axis state live in
/// their own arrays (the first because 25 B would separate `d` from `d+1`, the
/// second because it is indexed by an ancestor's depth, not by `d`).
#[derive(Clone, Copy)]
#[repr(C)]
struct FrameHot {
    /// Unconsumed candidate children — the resume point. Bits are cleared as
    /// children are taken, so no separate cursor is needed.
    cand: MoveSet,
    /// Running minimum of the children's `f` — this engine's `min_next` fold.
    minf: u8,
    /// The blank's cell.
    blank: u8,
    /// `h` of this frame's board.
    h: u8,
    /// Move that produced this frame (the root's value is never read).
    mv: Move,
    /// Depth whose `row` slot is live for this frame (`≤ d`).
    row_at: u8,
    /// Depth whose `col` slot is live for this frame (`≤ d`).
    col_at: u8,
    /// Move-DFA state along this path.
    dfa: u32,
}

/// A board slot, padded so the copy is two aligned register moves. [`State`]
/// itself is unchanged (`[u8; 25]`, align 1); in a bare `Box<[State]>` elements
/// land at byte offsets `25·i` and most are unaligned.
#[derive(Clone, Copy)]
#[repr(align(32))]
struct BoardSlot(Coded);

/// One axis's cWD state. A move touches exactly one axis — the same disjointness
/// that makes `WD_row + WD_col` admissible — so the *other* axis's slot is shared
/// with an ancestor by index rather than copied.
///
/// Co-locating `key`/`dem`/`nbr`/`wd`/`surch` in one 32 B slot is the mechanism
/// `5496995` used for 2.9%: they are read together, and reducing the *number* of
/// misses is what pays where the hardware cannot help.
#[derive(Clone, Copy)]
struct AxisSlot {
    /// Packed WD-table key, maintained incrementally.
    key: u64,
    /// Per-goal-line escape demand.
    dem: [u8; W],
    /// Bit `g` set iff `1 <= dem[g] <= 4`, i.e. iff line `g` can contribute to
    /// the surcharge at all. Maintained wherever `dem` changes, which is only
    /// when the moved tile enters or leaves its own line, so it is close to free
    /// — and it copies with the slot like `dem` does.
    ///
    /// Measured shape on R (`demand-histogram`, exhaust-146): 51.26% of axis
    /// updates have NO demanded line, and 41.58% have exactly one, so the
    /// unguarded `W`-iteration loop was doing five range checks to find zero or
    /// one relevant line 99.6% of the time.
    dem_mask: u8,
    /// WD of each axis-neighbour, for the child pre-prune.
    nbr: [u8; 2 * W],
    wd: u8,
    surch: u8,
    /// Drift-check mirror of the contingency matrix. Debug builds only — the
    /// release path reads counts straight out of `key` via [`key_get_cell`].
    #[cfg(debug_assertions)]
    m: super::cwd::Matrix,
}

impl AxisSlot {
    #[inline(always)]
    fn term(&self) -> u8 {
        self.wd + self.surch
    }
}

/// The depth-indexed arena. Allocated once; every index is `< MAX_DEPTH`.
struct Arena {
    hot: Box<[FrameHot]>,
    board: Box<[BoardSlot]>,
    row: Box<[AxisSlot]>,
    col: Box<[AxisSlot]>,
}

impl Arena {
    fn new() -> Self {
        let blank_axis = AxisSlot {
            key: 0,
            dem: [0; W],
            dem_mask: 0,
            nbr: [255; 2 * W],
            wd: 0,
            surch: 0,
            #[cfg(debug_assertions)]
            m: [[0; W]; W],
        };
        let blank_frame = FrameHot {
            cand: MoveSet::empty(),
            minf: u8::MAX,
            blank: 0,
            h: 0,
            mv: Move::Up,
            row_at: 0,
            col_at: 0,
            dfa: 0,
        };
        Arena {
            hot: vec![blank_frame; MAX_DEPTH].into_boxed_slice(),
            board: vec![BoardSlot(Coded::encode(&GOAL)); MAX_DEPTH].into_boxed_slice(),
            row: vec![blank_axis; MAX_DEPTH].into_boxed_slice(),
            col: vec![blank_axis; MAX_DEPTH].into_boxed_slice(),
        }
    }

    /// `h` of the frame at `d`, read through the two axis indices.
    #[inline(always)]
    fn h_at(&self, d: usize) -> u8 {
        let f = &self.hot[d];
        debug_assert!(f.row_at as usize <= d && f.col_at as usize <= d);
        self.row[f.row_at as usize].term() + self.col[f.col_at as usize].term()
    }
}

// -------------------------------- the engine ----------------------------------

/// Bounded lower-bound IDA\* over cWD, with the move-DFA, the neighbour-WD child
/// pre-prune and (optionally) the root σ-orbit split.
///
/// Returns exactly what [`idastar_inc_bounded_with_stats`](super::idastar::idastar_inc_bounded_with_stats)
/// would for the same configuration, including the node count: an exhausted
/// threshold `b` proves `dist(start) ≥ b+1`, reported as
/// [`BoundedOutcome::ProvedAtLeast`].
///
/// `cwd` must carry the merged table (`data/cwd_single.bin` present at
/// construction). `orbit_split` is only sound on a σ-fixed board; the caller owns
/// that precondition, and it is debug-asserted here.
///
/// # Panics
///
/// If `cwd` has no merged table — the A\* reference path is deliberately not
/// reachable from this engine, since silently falling back to it would make a
/// throughput comparison meaningless.
pub fn flat_bounded(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    orbit_split: bool,
    max_bound: u8,
) -> (BoundedOutcome, SearchStats) {
    flat_bounded_telemetry(start, cwd, dfa, orbit_split, max_bound, |_, _, _| {})
}

/// [`flat_bounded_telemetry`] with a **node budget**: stop after `max_nodes` and
/// return [`BoundedOutcome::BudgetExhausted`].
///
/// This exists for A/B benchmarking at depth, not for proving anything. An
/// exhaust-146 run on `R` is ~18.2 B nodes and ~12 minutes; capping it at, say,
/// 3 B lets the 144 pass finish and then samples ~2.6 B nodes of the 146 pass in
/// ~100 s. Because the engine is node-identical across variants, the budget cuts
/// every arm at the *same* tree position, so the truncated runs stay directly
/// comparable — which is the whole point.
///
/// Pass `u64::MAX` for no budget; that dispatches to the same monomorphisation
/// [`flat_bounded`] uses, in which the budget check does not exist.
pub fn flat_bounded_budgeted<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    orbit_split: bool,
    max_bound: u8,
    max_nodes: u64,
    on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    flat_bounded_inner(start, cwd, dfa, orbit_split, max_bound, max_nodes, on_iter)
}

/// [`flat_bounded`] with a per-iteration callback, mirroring
/// [`idastar_inc_bounded_telemetry`](super::idastar::idastar_inc_bounded_telemetry)
/// so the ladder harnesses can report per-threshold nodes and wall time.
pub fn flat_bounded_telemetry<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    orbit_split: bool,
    max_bound: u8,
    on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    flat_bounded_inner(start, cwd, dfa, orbit_split, max_bound, u64::MAX, on_iter)
}

fn flat_bounded_inner<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    orbit_split: bool,
    max_bound: u8,
    budget: u64,
    mut on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    let merged = cwd
        .merged_table()
        .expect("flat_bounded needs the merged cWD table (data/cwd_single.bin)");
    debug_assert!(
        !orbit_split || symmetry::is_symmetric(start),
        "root orbit-split is only sound on a σ-fixed board"
    );

    let mut cache = ProbeCache::new();
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (BoundedOutcome::Solved(Vec::new()), stats);
    }

    let mut arena = Arena::new();
    let h0 = seed_root(&mut arena, start, cwd, merged, dfa, orbit_split);
    let mut bound = h0;

    loop {
        if bound > max_bound {
            return (BoundedOutcome::ProvedAtLeast(bound), stats);
        }
        stats.iterations += 1;
        let iter_start = std::time::Instant::now();
        // Re-seed: the previous iteration consumed the root's candidate set.
        seed_root(&mut arena, start, cwd, merged, dfa, orbit_split);
        let step = if budget == u64::MAX {
            run_iteration::<false>(
                &mut arena, &mut cache, cwd, merged, dfa, bound, &mut stats, budget,
            )
        } else {
            run_iteration::<true>(
                &mut arena, &mut cache, cwd, merged, dfa, bound, &mut stats, budget,
            )
        };
        match step {
            Step::BudgetOut => {
                #[cfg(feature = "probe-cache-stats")]
                {
                    let (h, m) = cache.stats();
                    eprintln!(
                        "    probe-cache @thr {bound} (truncated): cumulative {h} hits /                          {m} misses = {:.3}% hit",
                        h as f64 / (h + m).max(1) as f64 * 100.0
                    );
                }
                return (BoundedOutcome::BudgetExhausted(bound), stats);
            }
            Step::Found(path) => return (BoundedOutcome::Solved(path), stats),
            Step::Exhausted(next) => {
                if next == u8::MAX {
                    return (BoundedOutcome::Unsolvable, stats);
                }
                #[cfg(feature = "probe-cache-stats")]
                {
                    let (h, m) = cache.stats();
                    eprintln!(
                        "    probe-cache @thr {bound}: cumulative {h} hits / {m} misses                          = {:.3}% hit",
                        h as f64 / (h + m).max(1) as f64 * 100.0
                    );
                }
                on_iter(bound, &stats, iter_start.elapsed());
                bound = next;
            }
        }
    }
}

enum Step {
    Found(Vec<Move>),
    Exhausted(u8),
    /// The node budget ran out mid-iteration.
    BudgetOut,
}

/// Project `start` into both axes, probe the merged table once per axis, and lay
/// the result down as depth 0. Also seeds the DFA state and the root's candidate
/// set (where the σ-orbit filter applies — it is a `g == 0`-only rule, so it
/// belongs here rather than in the loop).
fn seed_root(
    arena: &mut Arena,
    start: &State,
    cwd: &Cwd,
    merged: &CwdMerged,
    dfa: &MoveDfa,
    orbit_split: bool,
) -> u8 {
    let (m_row, br, dem_row, m_col, bc, dem_col) = project(start);
    let key_row = pack(&m_row, br);
    let key_col = pack(&m_col, bc);
    let r = merged.get(&key_row).expect("root row state reachable");
    let c = merged.get(&key_col).expect("root col state reachable");

    arena.row[0] = AxisSlot {
        key: key_row,
        dem: dem_row,
        dem_mask: dem_mask_of(&dem_row),
        nbr: r.nbr_wd,
        wd: r.wd,
        surch: surcharge_from_curves(&r.curves, &dem_row),
        #[cfg(debug_assertions)]
        m: m_row,
    };
    arena.col[0] = AxisSlot {
        key: key_col,
        dem: dem_col,
        dem_mask: dem_mask_of(&dem_col),
        nbr: c.nbr_wd,
        wd: c.wd,
        surch: surcharge_from_curves(&c.curves, &dem_col),
        #[cfg(debug_assertions)]
        m: m_col,
    };
    arena.board[0] = BoardSlot(Coded::encode(start));

    let blank = start.blank_pos();
    let dfa0 = <MoveDfa as super::move_dfa::MovePruner>::root_state(dfa, blank);

    // The root has no incoming move, so its candidates are simply the legal
    // moves; the DFA still applies (the generic engine tests `is_pruned` at the
    // root too), and the orbit filter drops one representative per σ-orbit.
    let mut cand = MoveSet(State::legal_moves_at(blank).0 & !dfa.prune_mask(dfa0));
    if orbit_split {
        let mut kept = 0u8;
        for m in Move::ALL {
            if cand.contains(m) && symmetry::is_orbit_representative(m) {
                kept |= 1 << (m as u8);
            }
        }
        cand = MoveSet(kept);
    }

    arena.hot[0] = FrameHot {
        cand,
        minf: u8::MAX,
        blank,
        h: 0,
        mv: Move::Up,
        row_at: 0,
        col_at: 0,
        dfa: dfa0,
    };
    let h = arena.h_at(0);
    arena.hot[0].h = h;
    let _ = cwd;
    h
}

/// One IDA\* iteration at `bound`, as an explicit-cursor DFS over the arena.
///
/// The ordering here is load-bearing for node identity. The generic engine
/// increments `nodes` on *entry* and only then tests `f > bound`
/// (`idastar.rs:1324`, `:1340`), so a child that is immediately over bound **is**
/// counted, while one dropped by the pre-prune is **not** — it `continue`s before
/// recursing. Both are reproduced below.
// Same convention as `build_child` and the sibling engine's search functions:
// most of these are loop invariants threaded down from the driver.
#[allow(clippy::too_many_arguments)]
fn run_iteration<const BUDGETED: bool>(
    arena: &mut Arena,
    cache: &mut ProbeCache,
    cwd: &Cwd,
    merged: &CwdMerged,
    dfa: &MoveDfa,
    bound: u8,
    stats: &mut SearchStats,
    budget: u64,
) -> Step {
    let lut = cwd.demand_lut();
    let neighbor_prune = cwd.neighbor_prune_enabled();

    // The root is a node too.
    stats.nodes += 1;
    let h0 = arena.hot[0].h;
    if h0 > bound {
        return Step::Exhausted(h0);
    }
    if h0 == 0 {
        debug_assert_eq!(arena.board[0].0.decode(), GOAL);
        return Step::Found(Vec::new());
    }

    let mut d: usize = 0;
    // `cand` and `minf` are the only fields of the current frame the loop
    // *mutates*, and both are touched on every candidate child — a
    // read-modify-write pair each, through a bounds-checked index. Holding them
    // in locals makes those register operations; the arena copy is refreshed only
    // on descend and ascend. Every other field (`blank`, `dfa`, `row_at`,
    // `col_at`) is read-only at this depth, so `child_lb` and `build_child` keep
    // reading the arena and see the same values.
    let mut cand = arena.hot[0].cand;
    let mut minf = arena.hot[0].minf;
    loop {
        if cand.is_empty() {
            // Children exhausted: fold this subtree's minimum into the parent.
            if d == 0 {
                return Step::Exhausted(minf);
            }
            let sub = minf;
            d -= 1;
            cand = arena.hot[d].cand;
            minf = arena.hot[d].minf;
            if sub < minf {
                minf = sub;
            }
            continue;
        }

        let m = pop_lowest(&mut cand);
        let g_next = (d as u8) + 1;
        // The move geometry and the moved tile are both needed by the pre-prune
        // and again by the child build, which recomputed them — and 54.87% of
        // candidates survive the pre-prune and reach it. Compute once per
        // candidate and pass them down.
        let geom = GEOM[arena.hot[d].blank as usize][m as usize];
        let tile = arena.board[d].0 .0[geom.from as usize] as usize;

        // Pre-prune from the parent's cached neighbour-WD: no probe, and — like
        // the generic engine — no node counted.
        if neighbor_prune {
            if let Some(lb) = child_lb(arena, d, m, geom, tile) {
                let f = g_next.saturating_add(lb);
                if f > bound {
                    if f < minf {
                        minf = f;
                    }
                    continue;
                }
            }
        }

        build_child(arena, cache, d, m, geom, tile, merged, lut, dfa);
        stats.nodes += 1;
        // Compiles away entirely when `BUDGETED` is false, so the proof driver's
        // hot loop is byte-identical to a build without this feature.
        if BUDGETED && stats.nodes >= budget {
            return Step::BudgetOut;
        }

        let h = arena.hot[d + 1].h;
        let f = g_next.saturating_add(h);
        if f > bound {
            if f < minf {
                minf = f;
            }
            continue;
        }
        if h == 0 {
            // Exact for cWD: WD_row = 0 ⟺ every tile is in its goal row, WD_col
            // = 0 ⟺ every tile is in its goal column, and surcharges are ≥ 0, so
            // h = 0 ⟺ solved. (Not true of a k-tile PDB, which reads 0 whenever
            // its own pattern is home — this must not migrate to the generic
            // engine.) The assert catches any future change that clamps `h`.
            debug_assert_eq!(arena.board[d + 1].0.decode(), GOAL);
            let mut path = Vec::with_capacity(d + 1);
            for k in 1..=d + 1 {
                path.push(arena.hot[k].mv);
            }
            return Step::Found(path);
        }

        // Descend: park this frame's live values, then take the child's.
        arena.hot[d].cand = cand;
        arena.hot[d].minf = minf;
        let child_blank = arena.hot[d + 1].blank as usize;
        let child_dfa = arena.hot[d + 1].dfa;
        minf = u8::MAX;
        // The whole filter chain in one AND-NOT: legality and the immediate-undo
        // rule come from CAND, redundancy from the DFA's own 4-bit mask (which
        // numbers moves identically to MoveSet). Once per node, not once per
        // child.
        cand = MoveSet(CAND[child_blank][m as usize].0 & !dfa.prune_mask(child_dfa));
        d += 1;
        debug_assert!(d < MAX_DEPTH, "depth exceeded the threshold ceiling");
    }
}

/// Escape surcharge for one axis, driven by the demanded-line mask.
///
/// Same value as [`surcharge_from_curves`] over all `W` lines — this only skips
/// lines that provably cannot contribute, since `dem_mask` is exactly the set
/// with `1 <= dem[g] <= 4` and that is the only range the surcharge reacts to.
/// The empty case, over half of all calls at depth, returns without touching
/// `curves` at all.
#[inline(always)]
fn surcharge_masked(curves: &[u16; W], dem: &[u8; W], mut mask: u8) -> u8 {
    let mut best = 0u8;
    while mask != 0 {
        let g = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let d = dem[g] as u16;
        let s = 2 * ((curves[g] >> (4 * (d - 1))) & 0xF) as u8;
        if s > best {
            best = s;
        }
    }
    best
}

/// `dem_mask` for a freshly-projected demand vector (root only; the search
/// maintains the mask incrementally).
#[inline]
fn dem_mask_of(dem: &[u8; W]) -> u8 {
    let mut m = 0u8;
    for (g, &d) in dem.iter().enumerate() {
        if (1..=4).contains(&d) {
            m |= 1 << g;
        }
    }
    m
}

/// Shape of the cWD escape-demand vector, behind the `demand-histogram` feature.
///
/// [`surcharge_from_curves`] loops all `W` goal lines and range-checks each one,
/// on every axis update — and line-level sampling in the exhaust-146 regime puts
/// it at ~11.7% of samples, the single largest identifiable cost in the node.
/// The proposed fix is an early-out when no line is demanded. Whether that is
/// worth writing depends entirely on how often the demand vector is empty, which
/// is what this measures.
///
/// `surcharge_from_curves` only reacts to `1 <= d <= 4`, so "demanded" means
/// exactly that; `d == 0` and `d >= 5` contribute nothing.
#[cfg(feature = "demand-histogram")]
pub mod demand_histogram {
    use super::W;
    use std::sync::atomic::{AtomicU64, Ordering};

    static CALLS: AtomicU64 = AtomicU64::new(0);
    static NONZERO_SURCH: AtomicU64 = AtomicU64::new(0);
    /// `LINES[k]` = calls whose demand vector had exactly `k` demanded lines.
    static LINES: [AtomicU64; W + 1] = [const { AtomicU64::new(0) }; W + 1];

    #[inline]
    pub fn note(dem: &[u8; W], surch: u8) {
        let k = dem.iter().filter(|&&d| (1..=4).contains(&d)).count();
        CALLS.fetch_add(1, Ordering::Relaxed);
        LINES[k].fetch_add(1, Ordering::Relaxed);
        if surch > 0 {
            NONZERO_SURCH.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn report() -> String {
        let calls = CALLS.load(Ordering::Relaxed).max(1);
        let mut s = format!("demand-histogram: {calls} surcharge calls\n");
        for (k, c) in LINES.iter().enumerate() {
            let v = c.load(Ordering::Relaxed);
            s += &format!(
                "  {k} demanded line(s): {v:>15} ({:>6.2}%)\n",
                v as f64 / calls as f64 * 100.0
            );
        }
        let nz = NONZERO_SURCH.load(Ordering::Relaxed);
        s += &format!(
            "  surcharge > 0     : {nz:>15} ({:>6.2}%)\n",
            nz as f64 / calls as f64 * 100.0
        );
        let empty = LINES[0].load(Ordering::Relaxed);
        s += &format!(
            "  => an empty-demand early-out would skip {:.2}% of the loops",
            empty as f64 / calls as f64 * 100.0
        );
        s
    }
}

/// A direct-mapped memo in front of the merged-table probe.
///
/// # Why this exists, and why 64 K
///
/// A 1 K-entry version of this was built and **rejected at +5.8%** (exhaust-144).
/// The recorded cause was that its tag test became "a 77/23 mispredicting
/// branch" — which is simply its hit rate: 1 K entries memo 77.40% of probes.
///
/// Entry-level reuse is far better than that. Measured over 40 M probes in the
/// exhaust-146 regime (`puzzle24::probe_locality`), and essentially identical at
/// exhaust-144:
///
/// | entries | RAM | hit rate |
/// |--------:|----:|---------:|
/// | 1 K | 0.03 MB | 79.6% |
/// | 16 K | 0.5 MB | 96.3% |
/// | 64 K | 2.1 MB | 99.0% |
/// | **256 K** | **8.4 MB** | **99.5%** |
///
/// At these sizes the tag test is a 99/1 branch — predicted, not mispredicted —
/// and installs happen ~20× less often than at 1 K.
///
/// **256 K beat 64 K by 5.8% at exhaust-146, far more than that 0.55 pp of hit
/// rate can explain.** The table above comes from a fully-associative LRU
/// simulation, but this cache is *direct-mapped*, so it also suffers conflict
/// misses that LRU never sees — two hot keys sharing a slot thrash forever.
/// Quadrupling the slots cuts those 4×, which is the effect actually being
/// bought. Do not read the LRU plateau at 99.5% as the top of the curve; it
/// cannot see the thing that is improving.
///
/// The second reason is spatial. The hot set is tiny in bytes (~64 K entries ≈
/// 2.1 MB) but hashbrown scatters those 32 B entries across 4.43 GB, so half of
/// all probes touch 2,448 separate 16 KiB pages holding a couple of hot entries
/// each. That is why L2 TLB misses run ~0.40/node at depth. Gathering the hot
/// set into 2.1 MB of contiguous memory replaces tens of thousands of touched
/// pages with ~128.
///
/// Node identity is by construction: this is a pure memo of an immutable table
/// with an exact key match, so a hit returns precisely what the probe would.
/// # Sizing is thread-count dependent — READ THIS BEFORE ADDING PARALLELISM
///
/// This cache is per-search mutable state, so a parallel engine needs one **per
/// thread**, and they compete for a shared L2. On this machine
/// (`hw.perflevel0.*`): 8 P-cores in 2 clusters of 4, **16 MB L2 per cluster**,
/// so 8 threads means 4 threads sharing 16 MB — a per-thread budget of ~4 MB
/// that must also cover the hashbrown probes on a miss.
///
/// | entries | per thread | x4 threads/cluster | vs 16 MB L2 |
/// |--------:|-----------:|-------------------:|------------:|
/// | 64 K | 2.1 MB | 8.4 MB | 52% — fits |
/// | 256 K | 8.4 MB | 33.6 MB | **200% — thrashes** |
/// | 512 K | 16.8 MB | 67.2 MB | 400% |
///
/// 18 bits (256 K, 8.4 MB) is the measured optimum for the **sequential** engine
/// — it beat 64 K by 5.8% at exhaust-146. It is 2x over budget at 8 threads, so
/// when parallelism lands this must be re-sized, most likely back to 16 bits.
/// Re-run the A/B at the target thread count rather than inheriting this value.
const CACHE_BITS: u32 = 18;
const CACHE_LEN: usize = 1 << CACHE_BITS;

/// Tag and payload interleaved, 32 B aligned, so a hit touches exactly one line
/// and a slot never straddles two.
#[repr(align(32))]
#[derive(Clone, Copy)]
struct CacheSlot {
    /// The packed key, or 0 for "empty". Real keys are never 0 — every reachable
    /// contingency matrix distributes 24 tiles, so some field is nonzero.
    tag: u64,
    cell: super::cwd::CwdCell,
}

pub(crate) struct ProbeCache {
    slots: Box<[CacheSlot]>,
    #[cfg(feature = "probe-cache-stats")]
    hits: u64,
    #[cfg(feature = "probe-cache-stats")]
    misses: u64,
}

impl ProbeCache {
    fn new() -> Self {
        ProbeCache {
            slots: vec![
                CacheSlot {
                    tag: 0,
                    cell: super::cwd::CwdCell {
                        wd: 0,
                        curves: [0; W],
                        nbr_wd: [255; 2 * W],
                    },
                };
                CACHE_LEN
            ]
            .into_boxed_slice(),
            #[cfg(feature = "probe-cache-stats")]
            hits: 0,
            #[cfg(feature = "probe-cache-stats")]
            misses: 0,
        }
    }

    /// `(hits, misses)` so far. The LRU model cannot predict this: it is fully
    /// associative, so it sees only capacity misses, while a direct-mapped cache
    /// also thrashes when two hot keys share a slot.
    #[cfg(feature = "probe-cache-stats")]
    pub(crate) fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Index from a single multiply-shift. The keys are structured (packed 3-bit
    /// fields), so their low bits are a poor index directly; taking the high bits
    /// of a multiply spreads them, the same reason the table's own hash folds the
    /// high half down.
    #[inline(always)]
    fn slot_of(key: u64) -> usize {
        ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> (64 - CACHE_BITS)) as usize
    }

    /// Memoised probe. Returns exactly what `merged.get(&key)` would.
    #[inline(always)]
    fn get<'a>(&'a mut self, merged: &'a CwdMerged, key: u64) -> &'a super::cwd::CwdCell {
        debug_assert_ne!(key, 0, "key 0 collides with the empty-slot sentinel");
        let i = Self::slot_of(key);
        if self.slots[i].tag != key {
            let cell = *merged.get(&key).expect("state reachable");
            self.slots[i] = CacheSlot { tag: key, cell };
            #[cfg(feature = "probe-cache-stats")]
            {
                self.misses += 1;
            }
        } else {
            #[cfg(feature = "probe-cache-stats")]
            {
                self.hits += 1;
            }
        }
        debug_assert_eq!(
            self.slots[i].cell.wd,
            merged.get(&key).expect("state reachable").wd,
            "probe cache returned a different cell than the table"
        );
        &self.slots[i].cell
    }
}

/// Cheap admissible lower bound on the `h` of the child reached by `m`, read from
/// the parent's cached neighbour-WD — the port of
/// [`Cwd::child_h_lb`](super::cwd::Cwd). The moved axis contributes its cached
/// neighbour WD (the surcharge is dropped, which keeps the bound admissible) and
/// the untouched axis carries over whole.
#[inline]
fn child_lb(arena: &Arena, d: usize, m: Move, geom: Geom, tile: usize) -> Option<u8> {
    let f = &arena.hot[d];

    let (slot, other, g) = if geom.vertical {
        (
            &arena.row[f.row_at as usize],
            &arena.col[f.col_at as usize],
            tile >> 4,
        )
    } else {
        (
            &arena.col[f.col_at as usize],
            &arena.row[f.row_at as usize],
            tile & 0xF,
        )
    };
    // `dir` 0 = the blank's index along the axis decreases (Up / Left).
    let dir = usize::from(matches!(m, Move::Down | Move::Right));
    let w = slot.nbr[dir * W + g];
    if w == 255 {
        return None; // no cached neighbour (unreachable for a legal move)
    }
    Some(w.saturating_add(other.term()))
}

/// Materialise the child of `d` reached by `m` into depth `d+1`.
///
/// Port of [`Cwd::make`](super::cwd::Cwd), minus the undo record and minus the
/// contingency matrix: the two counts the incremental key update needs are read
/// back out of the key itself with [`key_get_cell`], which is a shift and a mask
/// on a value already in a register.
// Same convention as the sibling engine's search functions (`idastar.rs:1296`,
// `:1311`): three of these are loop invariants threaded down from the driver.
#[allow(clippy::too_many_arguments)]
fn build_child(
    arena: &mut Arena,
    cache: &mut ProbeCache,
    d: usize,
    m: Move,
    geom: Geom,
    tile: usize,
    merged: &CwdMerged,
    lut: &[u8; DEMAND_LUT_LEN],
    dfa: &MoveDfa,
) {
    let parent = arena.hot[d];
    debug_assert_ne!(geom.from, NO_CELL, "illegal move reached build_child");

    // Board: copy the parent and swap the two cells. Same operation as
    // `State::apply_at`, which the generic engine calls at `idastar.rs:1387`.
    let mut child = arena.board[d].0;
    child.0.swap(parent.blank as usize, geom.from as usize);
    arena.board[d + 1] = BoardSlot(child);
    debug_assert_eq!(tile, child.0[parent.blank as usize] as usize, "tile drift");

    let child_dfa = <MoveDfa as super::move_dfa::MovePruner>::advance(dfa, parent.dfa, m);

    let (row_at, col_at) = if geom.vertical {
        // Vertical slide: only the row axis changes; the column axis is shared.
        let g = tile >> 4;
        let (rf, rt) = (geom.rf as usize, geom.rt as usize);
        let src = &arena.row[parent.row_at as usize];
        let mut key = src.key;
        let mut dem = src.dem;
        #[cfg(debug_assertions)]
        let mut mat = src.m;
        #[cfg(debug_assertions)]
        {
            mat[rf][g] -= 1;
            mat[rt][g] += 1;
        }
        // The moved tile's own goal line is the only demand that can change, and
        // only when it enters or leaves its own physical line.
        let mut dem_mask = src.dem_mask;
        if g == rf || g == rt {
            let d = demand_row_fast(lut, &child, g);
            dem[g] = d;
            // One bit, at the only place `dem` can change.
            let bit = 1u8 << g;
            dem_mask = (dem_mask & !bit) | (u8::from((1..=4).contains(&d)) << g);
        }
        // Cell pair and blank field in one add — see `KEY_DELTA`.
        key = key.wrapping_add(KEY_DELTA[rf][rt][g]);
        #[cfg(debug_assertions)]
        debug_assert_eq!(key, pack(&mat, geom.rf), "key_row drift");

        let cell = cache.get(merged, key);
        #[cfg(feature = "probe-locality")]
        crate::puzzle24::probe_locality::note_probe(cell as *const _ as usize);
        arena.row[d + 1] = AxisSlot {
            key,
            dem,
            nbr: cell.nbr_wd,
            wd: cell.wd,
            dem_mask,
            surch: {
                let s = surcharge_masked(&cell.curves, &dem, dem_mask);
                debug_assert_eq!(
                    s,
                    surcharge_from_curves(&cell.curves, &dem),
                    "masked surcharge differs from the all-lines reference"
                );
                #[cfg(feature = "demand-histogram")]
                demand_histogram::note(&dem, s);
                s
            },
            #[cfg(debug_assertions)]
            m: mat,
        };
        ((d + 1) as u8, parent.col_at)
    } else {
        // Horizontal slide: only the column axis changes.
        let g = tile & 0xF;
        let (cf, ct) = (geom.cf as usize, geom.ct as usize);
        let src = &arena.col[parent.col_at as usize];
        let mut key = src.key;
        let mut dem = src.dem;
        #[cfg(debug_assertions)]
        let mut mat = src.m;
        #[cfg(debug_assertions)]
        {
            mat[cf][g] -= 1;
            mat[ct][g] += 1;
        }
        let mut dem_mask = src.dem_mask;
        if g == cf || g == ct {
            let d = demand_col_fast(lut, &child, g);
            dem[g] = d;
            let bit = 1u8 << g;
            dem_mask = (dem_mask & !bit) | (u8::from((1..=4).contains(&d)) << g);
        }
        key = key.wrapping_add(KEY_DELTA[cf][ct][g]);
        #[cfg(debug_assertions)]
        debug_assert_eq!(key, pack(&mat, geom.cf), "key_col drift");

        let cell = cache.get(merged, key);
        #[cfg(feature = "probe-locality")]
        crate::puzzle24::probe_locality::note_probe(cell as *const _ as usize);
        arena.col[d + 1] = AxisSlot {
            key,
            dem,
            nbr: cell.nbr_wd,
            wd: cell.wd,
            dem_mask,
            surch: {
                let s = surcharge_masked(&cell.curves, &dem, dem_mask);
                debug_assert_eq!(
                    s,
                    surcharge_from_curves(&cell.curves, &dem),
                    "masked surcharge differs from the all-lines reference"
                );
                #[cfg(feature = "demand-histogram")]
                demand_histogram::note(&dem, s);
                s
            },
            #[cfg(debug_assertions)]
            m: mat,
        };
        (parent.row_at, (d + 1) as u8)
    };

    arena.hot[d + 1] = FrameHot {
        cand: MoveSet::empty(),
        minf: u8::MAX,
        blank: geom.from,
        h: 0,
        mv: m,
        row_at,
        col_at,
        dfa: child_dfa,
    };
    let h = arena.h_at(d + 1);
    arena.hot[d + 1].h = h;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::move_dfa::MovePruner;

    // ------------------------- fast, table-free LUT checks ---------------------

    /// `CAND` replaces two filters the generic engine applies inline (legality and
    /// the immediate-undo rule). If it ever disagrees with them the tree shape
    /// changes silently, so pin it against the originals for every pair.
    #[test]
    fn cand_table_matches_legal_minus_inverse() {
        for b in 0..N_CELLS {
            let legal = State::legal_moves_at(b as u8);
            for last in Move::ALL {
                let got = CAND[b][last as usize];
                for m in Move::ALL {
                    let want = legal.contains(m) && m != last.inverse();
                    assert_eq!(
                        got.contains(m),
                        want,
                        "CAND[{b}][{last:?}] disagrees on {m:?}"
                    );
                }
            }
        }
    }

    /// A node never has four children: 2/3/4 legal by corner/edge/interior, and
    /// every non-root node loses one more to the inverse filter.
    #[test]
    fn cand_never_exceeds_three() {
        for b in 0..N_CELLS {
            for last in Move::ALL {
                assert!(
                    CAND[b][last as usize].len() <= 3,
                    "CAND[{b}][{last:?}] has {} children",
                    CAND[b][last as usize].len()
                );
            }
        }
    }

    /// `GEOM` replaces the six divisions by `W` that `Cwd::make` performs per
    /// node. Check it against `apply_at`, which is what the generic engine uses.
    #[test]
    fn geom_matches_apply_at() {
        for b in 0..N_CELLS {
            let legal = State::legal_moves_at(b as u8);
            let mut cells = GOAL.0;
            cells.swap(N_CELLS - 1, b);
            let s = State(cells);
            assert_eq!(s.blank_pos() as usize, b);
            for m in Move::ALL {
                let g = &GEOM[b][m as usize];
                if !legal.contains(m) {
                    assert_eq!(g.from, NO_CELL, "GEOM[{b}][{m:?}] should be illegal");
                    continue;
                }
                let (_, nb) = s.apply_at(m, b as u8);
                assert_eq!(g.from, nb, "GEOM[{b}][{m:?}].from");
                assert_eq!(g.rf as usize, g.from as usize / W);
                assert_eq!(g.cf as usize, g.from as usize % W);
                assert_eq!(g.rt as usize, b / W);
                assert_eq!(g.ct as usize, b % W);
                assert_eq!(g.vertical, matches!(m, Move::Up | Move::Down));
                // A move touches exactly one axis — the disjointness the whole
                // per-axis sharing scheme rests on.
                assert_eq!(g.vertical, g.cf == g.ct);
                assert_eq!(!g.vertical, g.rf == g.rt);
            }
        }
    }

    #[test]
    fn goal_code_matches_arithmetic_and_round_trips() {
        // The code's nibbles must be exactly the goal row/col the tile-numbered
        // version looked up in a table, since the engine now shifts them out of
        // the board byte instead.
        for t in 1..N_CELLS {
            let c = TILE_CODE[t];
            assert_eq!((c >> 4) as usize, (t - 1) / W, "goal row of tile {t}");
            assert_eq!((c & 0xF) as usize, (t - 1) % W, "goal col of tile {t}");
        }
        // The blank must not collide with any tile, and neither nibble may equal
        // a real line index — that is what lets the demand predicate drop its
        // `tile != 0` term.
        assert_eq!(TILE_CODE[0], BLANK_CODE);
        assert!((BLANK_CODE >> 4) as usize >= W && (BLANK_CODE & 0xF) as usize >= W);
        for t in 1..N_CELLS {
            assert_ne!(TILE_CODE[t], BLANK_CODE, "tile {t} collides with the blank");
        }
        // Injectivity, and encode/decode is the identity on real boards.
        let mut seen = std::collections::HashSet::new();
        for t in 0..N_CELLS {
            assert!(seen.insert(TILE_CODE[t]), "code repeats at tile {t}");
        }
        let mut x: u64 = 0x51ED_2704_9C31_A6B5;
        for _ in 0..2_000 {
            let mut a = [0u8; N_CELLS];
            for (i, c) in a.iter_mut().enumerate() {
                *c = i as u8;
            }
            for i in (1..N_CELLS).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                a.swap(i, (x % (i as u64 + 1)) as usize);
            }
            let s = State(a);
            assert_eq!(Coded::encode(&s).decode(), s, "round trip failed on {a:?}");
        }
    }

    /// The engine drops every DFA-pruned move with one `& !prune_mask`, instead of
    /// an `is_pruned` call per child. That is only valid because `mcode` numbers
    /// the moves exactly as `MoveSet` does — assert it across every state.
    #[test]
    fn prune_mask_agrees_with_is_pruned() {
        let dfa = MoveDfa::build_default();
        for st in 0..dfa.states() as u32 {
            let mask = dfa.prune_mask(st);
            for m in Move::ALL {
                assert_eq!(
                    (mask >> (m as u8)) & 1 == 1,
                    dfa.is_pruned(st, m),
                    "prune_mask disagrees at state {st}, {m:?}"
                );
            }
        }
    }

    /// The branchless demand key must equal the branched reference on every line
    /// of arbitrary boards.
    ///
    /// This is the right gate for changes to the demand path. `demand_*_fast` is
    /// reached only from `build_child`, so before this test the sole coverage was
    /// the 5-minute full-search differential — which is the wrong instrument for a
    /// value-preserving rewrite. The demand LUT is 32 KiB built in-process, so this
    /// needs no data files and runs in about a second.
    ///
    /// It earns its keep in **release** especially: `demand_*_fast`'s own
    /// `debug_assert` against the reference vanishes there, and this does not.
    #[test]
    fn demand_fast_matches_reference_on_random_boards() {
        let lut = super::super::cwd::build_demand_lut();
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..20_000 {
            let mut a = [0u8; N_CELLS];
            for (i, c) in a.iter_mut().enumerate() {
                *c = i as u8;
            }
            for i in (1..N_CELLS).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                a.swap(i, (x % (i as u64 + 1)) as usize);
            }
            let s = State(a);
            // The fast path takes a goal-coded board; the reference still takes
            // tile numbers, so the encoding is under test here too.
            let c = Coded::encode(&s);
            for g in 0..W {
                assert_eq!(
                    demand_row_fast(&lut, &c, g),
                    demand_row_line(&lut, &s, g),
                    "row demand differs on line {g} of {a:?}"
                );
                assert_eq!(
                    demand_col_fast(&lut, &c, g),
                    demand_col_line(&lut, &s, g),
                    "col demand differs on line {g} of {a:?}"
                );
            }
        }
    }

    /// The mask-driven surcharge must equal the all-lines reference for every
    /// `(curves, dem)` pair, and the mask must be exactly the set of lines the
    /// reference can react to.
    ///
    /// This is the gate for H3. The `debug_assert` inside `build_child` covers
    /// the same property but only fires in debug builds, where a search over the
    /// real tables is far too slow to run — so the equivalence is pinned here,
    /// table-free, instead.
    #[test]
    fn masked_surcharge_matches_all_lines_reference() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut nz = 0usize;
        for _ in 0..200_000 {
            let mut curves = [0u16; W];
            let mut dem = [0u8; W];
            for g in 0..W {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                curves[g] = (x >> 11) as u16;
                // Cover d = 0..=5 so the out-of-range values the reference
                // ignores (0 and >=5) are exercised, not just the live band.
                dem[g] = ((x >> 40) % 6) as u8;
            }
            let mask = dem_mask_of(&dem);
            for g in 0..W {
                assert_eq!(
                    mask >> g & 1 == 1,
                    (1..=4).contains(&dem[g]),
                    "mask bit {g} disagrees with the reference predicate"
                );
            }
            let want = surcharge_from_curves(&curves, &dem);
            let got = surcharge_masked(&curves, &dem, mask);
            assert_eq!(got, want, "curves {curves:?} dem {dem:?}");
            if want != 0 {
                nz += 1;
            }
        }
        assert!(nz > 1000, "test data never produced a nonzero surcharge");
    }

    /// Consuming a `MoveSet` lowest-bit-first must reproduce `MoveSetIter`'s
    /// order, which is what keeps child order — and the tree — identical.
    #[test]
    fn pop_lowest_matches_moveset_iter_order() {
        for bits in 0u8..16 {
            let set = MoveSet(bits);
            let want: Vec<Move> = set.iter().collect();
            let mut left = set;
            let mut got = Vec::new();
            while !left.is_empty() {
                got.push(pop_lowest(&mut left));
            }
            assert_eq!(got, want, "order differs for bits {bits:#06b}");
        }
    }

    // --------------------------- table-gated differential ----------------------

    /// The process-wide shared table (see `cwd::shared_merged_cwd`). Building a
    /// fresh `Cwd` per test cost ~4.4 GB each and made the parallel suite thrash.
    #[cfg(feature = "cwd-table-tests")]
    fn cwd_merged_or_skip() -> Option<&'static Cwd> {
        super::super::cwd::shared_merged_cwd()
    }

    /// **The gate.** The flat engine's tree, compared node-for-node against the
    /// frozen answers of the deleted recursive engine
    /// ([`super::flat_oracle`]). Node counts are compared exactly: they catch
    /// tree-shape divergence that comparing `h` values or final bounds would
    /// miss.
    ///
    /// The oracle covers ~3.7 × 10^7 nodes over 180 cases, the largest a single
    /// 10.4 M-node exhaustion. That sizing is deliberate and was arrived at the
    /// hard way — the original grid searched **1,570 nodes in total** and was
    /// nearly frozen in that state. Two properties of cWD make a naive grid
    /// vacuous:
    ///
    /// - **Parity.** cWD shares the true distance's parity, so `f` only takes
    ///   `h0, h0+2, h0+4, …`; a bound of `h0+1` is the same search as `h0`.
    /// - **`bound = h0` is a narrow corridor.** Each child has `h = h0 ± 1`, so
    ///   `f` is `h0` or `h0+2` and the latter is pruned immediately — only
    ///   strictly-`h`-decreasing paths survive, tens of nodes.
    ///
    /// Hence the real cases sit at `h0 + {4,6,8}` on 200-700 step walks, where
    /// cWD stops being exact and the search genuinely exhausts.
    ///
    /// A failure here means the flat engine's tree changed. The fixture is not
    /// refreshable — the engine that produced it is gone — so it is ground
    /// truth, never something to regenerate to make a test pass.
    #[cfg(feature = "cwd-table-tests")]
    #[test]
    fn flat_matches_frozen_recursive_oracle() {
        let Some(cwd) = cwd_merged_or_skip() else {
            return;
        };
        let dfa = MoveDfa::build_default();

        let mut checked = 0usize;
        let mut nodes_total = 0u64;
        for (i, c) in crate::puzzle24::search::flat_oracle::CASES
            .iter()
            .enumerate()
        {
            let s = State(c.board);
            let (fo, fs) = flat_bounded(&s, cwd, &dfa, c.orbit_split, c.bound);

            let ok = match &fo {
                BoundedOutcome::Solved(mv) => c.tag == 0 && mv.len() as u8 == c.val,
                BoundedOutcome::ProvedAtLeast(k) => c.tag == 1 && *k == c.val,
                BoundedOutcome::Unsolvable => c.tag == 2,
                // The oracle never runs with a budget; if this fires, the test
                // harness is wrong rather than the engine.
                BoundedOutcome::BudgetExhausted(_) => {
                    panic!("oracle case {i} hit a node budget — the fixture is run unbudgeted")
                }
            };
            assert!(
                ok,
                "case {i}: outcome differs (board {:?}, bound {}, orbit_split {}): \
                 flat {fo:?} vs frozen tag={} val={}",
                c.board, c.bound, c.orbit_split, c.tag, c.val
            );
            assert_eq!(
                fs.nodes, c.nodes,
                "case {i}: node count differs (board {:?}, bound {}, orbit_split {})",
                c.board, c.bound, c.orbit_split
            );
            assert_eq!(
                fs.iterations, c.iterations,
                "case {i}: iteration count differs (board {:?}, bound {}, orbit_split {})",
                c.board, c.bound, c.orbit_split
            );
            checked += 1;
            nodes_total += fs.nodes;
        }

        // Guard the fixture itself: a truncated or half-written oracle would make
        // every assertion above vacuous.
        assert_eq!(checked, 180, "oracle case count changed");
        assert!(
            nodes_total > 30_000_000,
            "oracle got weaker: only {nodes_total} nodes searched"
        );
    }

    /// The orbit split must actually reduce the tree. The frozen oracle records
    /// both settings at each bound, so this compares split against unsplit
    /// without needing the deleted engine.
    #[cfg(feature = "cwd-table-tests")]
    #[test]
    fn flat_orbit_split_reduces_nodes_against_oracle() {
        let Some(cwd) = cwd_merged_or_skip() else {
            return;
        };
        let dfa = MoveDfa::build_default();

        let r = State([
            0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
            1,
        ]);
        assert!(symmetry::is_symmetric(&r), "board R must be σ-symmetric");

        let mut pairs = 0usize;
        for c in crate::puzzle24::search::flat_oracle::CASES
            .iter()
            .filter(|c| c.orbit_split)
        {
            let s = State(c.board);
            assert!(symmetry::is_symmetric(&s), "orbit case must be σ-symmetric");
            let Some(off) = crate::puzzle24::search::flat_oracle::CASES
                .iter()
                .find(|o| !o.orbit_split && o.board == c.board && o.bound == c.bound)
            else {
                continue;
            };
            let (_, on_stats) = flat_bounded(&s, cwd, &dfa, true, c.bound);
            assert_eq!(on_stats.nodes, c.nodes, "split node count differs");
            assert!(
                c.nodes < off.nodes,
                "orbit split did not reduce nodes at bound {}: {} vs {}",
                c.bound,
                c.nodes,
                off.nodes
            );
            pairs += 1;
        }
        assert!(pairs >= 6, "expected orbit split pairs in the oracle");
    }

    /// The bounded-lower-bound contract, mirroring `tests/puzzle24_bounded.rs`.
    #[cfg(feature = "cwd-table-tests")]
    #[test]
    fn flat_bounded_outcome_contract() {
        let Some(cwd) = cwd_merged_or_skip() else {
            return;
        };
        let dfa = MoveDfa::build_default();

        // Solved at the goal itself, with an empty path and no search.
        let (o, s) = flat_bounded(&GOAL, cwd, &dfa, false, 10);
        assert_eq!(o, BoundedOutcome::Solved(Vec::new()));
        assert_eq!(s.nodes, 0);

        // Optimal depths come from the frozen oracle rather than from a second
        // engine or from this engine's own solve, so the contract below is
        // anchored to ground truth. Tier-1 oracle cases that report `Solved`
        // carry the optimum in `val`.
        let solved: Vec<(State, u8)> = {
            let mut seen: Vec<[u8; N_CELLS]> = Vec::new();
            crate::puzzle24::search::flat_oracle::CASES
                .iter()
                .filter(|c| c.tag == 0 && !c.orbit_split)
                .filter(|c| {
                    let fresh = !seen.contains(&c.board);
                    if fresh {
                        seen.push(c.board);
                    }
                    fresh
                })
                .take(4)
                .map(|c| (State(c.board), c.val))
                .collect()
        };
        assert_eq!(solved.len(), 4, "oracle must supply four solved boards");

        for (start, d) in solved {
            let start = &start;

            // bound == depth ⇒ Solved at exactly that length.
            match flat_bounded(start, cwd, &dfa, false, d).0 {
                BoundedOutcome::Solved(p) => assert_eq!(p.len() as u8, d),
                other => panic!("expected Solved at bound {d}, got {other:?}"),
            }

            // bound == depth-1 ⇒ ProvedAtLeast(depth): the exhaust is the proof.
            assert_eq!(
                flat_bounded(start, cwd, &dfa, false, d - 1).0,
                BoundedOutcome::ProvedAtLeast(d),
            );

            // bound < h0 ⇒ the bound is returned without visiting a single node.
            let h0 = match flat_bounded(start, cwd, &dfa, false, 0).0 {
                BoundedOutcome::ProvedAtLeast(k) => k,
                other => panic!("expected ProvedAtLeast, got {other:?}"),
            };
            let (o, s) = flat_bounded(start, cwd, &dfa, false, h0 - 1);
            assert_eq!(o, BoundedOutcome::ProvedAtLeast(h0));
            assert_eq!(s.nodes, 0, "no node should be visited below h0");
        }
    }
}
