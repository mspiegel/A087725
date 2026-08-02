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
    MergedBacking, DEMAND_LUT_LEN, KEY_BLANK_BIT,
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

/// Inverse of [`TILE_CODE`]: `TILE_OF_CODE[code]` = the tile number, for the k8
/// tier, which must talk to the PDB machinery in tile numbers while the engine's
/// boards are goal-coded. Codes are `(goal_row << 4) | goal_col`, so the table
/// is 256 entries with only the 24 real codes populated.
const TILE_OF_CODE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut k = 1;
    while k < N_CELLS {
        t[TILE_CODE[k] as usize] = k as u8;
        k += 1;
    }
    t
};

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
    /// k8 tier state, written only on consults; untouched (dead weight, ~64 KB)
    /// when the tier is off.
    k8: Box<[K8Slot]>,
    /// Last-Move tier: per-depth (row of tile 20, col of tile 24), written at
    /// the consult like the k8 slots (every entered node has a current entry).
    lmpos: Box<[[u8; 2]]>,
    /// LM front cache, allocated on first [`seed_lm`] (4 MB default); persists
    /// across units and thresholds like the worker [`ProbeCache`].
    lmcache: Option<Box<LmCache>>,
    /// LM2 tier: per-depth tracked lines
    /// [row20, col24, row15, row19, col19, col23], written at the consult.
    lm2pos: Box<[[u8; 6]]>,
    /// k6 tier state, written only on consults; untouched when the tier is off.
    k6slots: Box<[K6Slot]>,
    /// Positional k6 tier state and its front cache.
    k6pos: Box<[K6PosSlot]>,
    k6cache: Option<Box<K6Cache>>,
    /// LM2 combined front cache (single + pair values under one tag),
    /// allocated on first [`seed_lm2`].
    lm2cache: Option<Box<Lm2Cache>>,
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
        let blank_k8 = K8Slot {
            views: [[crate::puzzle24::pdb::ProjectedState::from_state(
                &GOAL,
                crate::puzzle24::pdb::Pattern(0),
            ); 3]; 2],
            h: [[0; 3]; 2],
        };
        Arena {
            hot: vec![blank_frame; MAX_DEPTH].into_boxed_slice(),
            board: vec![BoardSlot(Coded::encode(&GOAL)); MAX_DEPTH].into_boxed_slice(),
            row: vec![blank_axis; MAX_DEPTH].into_boxed_slice(),
            col: vec![blank_axis; MAX_DEPTH].into_boxed_slice(),
            k8: vec![blank_k8; MAX_DEPTH].into_boxed_slice(),
            lmpos: vec![[0u8; 2]; MAX_DEPTH].into_boxed_slice(),
            lmcache: None,
            lm2pos: vec![[0u8; 6]; MAX_DEPTH].into_boxed_slice(),
            k6pos: vec![K6PosSlot::default(); MAX_DEPTH].into_boxed_slice(),
            k6cache: None,
            k6slots: vec![
                K6Slot {
                    views: [[crate::puzzle24::pdb::ProjectedState::from_state(
                        &GOAL,
                        crate::puzzle24::pdb::Pattern(0),
                    ); 4]; 2],
                    h: [[0; 4]; 2],
                };
                MAX_DEPTH
            ]
            .into_boxed_slice(),
            lm2cache: None,
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

// ------------------------------ lazy k8 tier ---------------------------------

/// The three disjoint 8-tile zero-aware PDBs, mmap'd. Loaded only when the
/// `--zpdb8` flag asks for the 2-tier `max(cWD, k8)` stack.
pub struct K8Ctx {
    dbs: [crate::puzzle24::pdb::ZPatternDb; 3],
    /// `group_of[t]` = which of the three patterns holds tile `t` (1..=24).
    group_of: [u8; N_CELLS],
    /// The 8 tiles of each pattern, ascending (instrumentation: per-group MD).
    #[cfg(feature = "k8-probe-locality")]
    tiles: [[u8; 8]; 3],
}

impl K8Ctx {
    /// Load `pdb24_k8_{a,b,c}.zbin` from `dir` and verify the patterns are
    /// pairwise disjoint and cover all 24 tiles (Korf–Felner additivity).
    pub fn load_mmap(dir: &std::path::Path) -> Result<Self, String> {
        let names = ["pdb24_k8_a.zbin", "pdb24_k8_b.zbin", "pdb24_k8_c.zbin"];
        let mut dbs = Vec::with_capacity(3);
        for n in names {
            let path = dir.join(n);
            dbs.push(
                crate::puzzle24::pdb::ZPatternDb::load_mmap(&path)
                    .map_err(|e| format!("{}: {e:?}", path.display()))?,
            );
        }
        let dbs: [crate::puzzle24::pdb::ZPatternDb; 3] =
            dbs.try_into().map_err(|_| "expected 3 dbs".to_string())?;
        let mut union: u32 = 0;
        for db in &dbs {
            let p = db.pattern().0;
            if union & p != 0 {
                return Err("k8 patterns are not pairwise disjoint".into());
            }
            union |= p;
        }
        if union != ((1u32 << 25) - 2) {
            return Err(format!(
                "k8 patterns do not cover tiles 1..=24 (union {union:#x})"
            ));
        }
        let mut group_of = [0u8; N_CELLS];
        #[cfg(feature = "k8-probe-locality")]
        let mut tiles = [[0u8; 8]; 3];
        #[cfg(feature = "k8-probe-locality")]
        let mut tn = [0usize; 3];
        for (gi, db) in dbs.iter().enumerate() {
            for t in 1..N_CELLS {
                if db.pattern().contains(t as u8) {
                    group_of[t] = gi as u8;
                    #[cfg(feature = "k8-probe-locality")]
                    {
                        tiles[gi][tn[gi]] = t as u8;
                        tn[gi] += 1;
                    }
                }
            }
        }
        Ok(K8Ctx {
            dbs,
            group_of,
            #[cfg(feature = "k8-probe-locality")]
            tiles,
        })
    }
}

/// Per-depth k8 state: the three groups' projections in the normal and
/// σ-reflected views, plus each group's absolute `h`. ~306 B; copied whole from
/// the parent at each consult, so there is no sharing subtlety and no staleness.
///
/// The invariant mirroring the recursive lazy combiner: this slot is written for
/// a child **only after the child passes cWD's bound test** (the consult point).
/// Every *entered* node passed that test, so a consulting child always finds its
/// parent's slot current — single-step laziness with no deferral bookkeeping,
/// which is the entire benefit the recursive `LazyMaxInc` glue existed to buy.
#[derive(Clone, Copy)]
struct K8Slot {
    /// `[normal, reflected]` × 3 groups.
    views: [[crate::puzzle24::pdb::ProjectedState; 3]; 2],
    h: [[u8; 3]; 2],
}

/// Online locality instrumentation for the k8 `diff_lookup` stream.
///
/// Simulates the candidate **(idx → h) memo cache** directly — direct-mapped at
/// several sizes, keyed `idx * 3 + table` with the same multiply-shift index as
/// [`ProbeCache`] — and tracks the distinct 16 KiB pages touched per table
/// (131,072 one-bit entries per page on this machine). Sequential runs only.
#[cfg(feature = "k8-probe-locality")]
mod k8_locality {
    use std::sync::Mutex;

    const BITS: [u32; 6] = [14, 16, 18, 20, 22, 24];

    /// Set-associative variant at the same total capacity, LRU within the set
    /// (MRU-first ordering of the ways). Separates CONFLICT misses (which
    /// associativity fixes) from capacity/compulsory misses (which it cannot).
    struct Assoc {
        ways: usize,
        set_shift: u32,
        tags: Vec<u64>, // sets * ways, MRU first within a set
        hits: u64,
    }

    impl Assoc {
        fn new(total_bits: u32, ways: usize) -> Self {
            Assoc {
                ways,
                set_shift: 64 - (total_bits - ways.trailing_zeros()),
                tags: vec![0; 1usize << total_bits],
                hits: 0,
            }
        }
        fn probe(&mut self, key: u64, h: u64) {
            let set = (h >> self.set_shift) as usize * self.ways;
            let w = &mut self.tags[set..set + self.ways];
            if let Some(i) = w.iter().position(|&t| t == key) {
                self.hits += 1;
                w[..=i].rotate_right(1); // move to MRU
            } else {
                w.rotate_right(1);
                w[0] = key;
            }
        }
    }

    /// ~16 KB HyperLogLog (2^14 registers, ~0.8% error) for the distinct-idx
    /// count — the compulsory-miss floor no finite cache can beat.
    struct Hll {
        reg: Vec<u8>,
    }
    impl Hll {
        fn new() -> Self {
            Hll {
                reg: vec![0; 1 << 14],
            }
        }
        fn add(&mut self, h: u64) {
            let i = (h >> 50) as usize;
            let z = (h << 14).leading_zeros() as u8 + 1;
            if z > self.reg[i] {
                self.reg[i] = z;
            }
        }
        fn estimate(&self) -> f64 {
            let m = self.reg.len() as f64;
            let sum: f64 = self.reg.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
            let e = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
            if e <= 2.5 * m {
                let zeros = self.reg.iter().filter(|&&r| r == 0).count();
                if zeros > 0 {
                    return m * (m / zeros as f64).ln();
                }
            }
            e
        }
    }

    /// Readahead-value probe: page-fault stream analysis at one eviction
    /// horizon. A "fault" is a page whose last touch is older than `horizon`
    /// page-events (`u64::MAX` horizon = first-touch only). `coverage` counts
    /// faults with a recent prior fault within ±8 pages of the same table —
    /// the faults default cluster-fill readahead would have absorbed.
    struct FaultProbe {
        horizon: u64,
        faults: u64,
        cov4: u64,
        cov8: u64,
        /// |Δpage| to the nearest same-table fault in the ring:
        /// buckets [1, 2-4, 5-8, 9-64, >64/none]
        dist: [u64; 5],
        ring: [(u8, u32); 512],
        ring_pos: usize,
    }

    impl FaultProbe {
        fn new(horizon: u64) -> Self {
            FaultProbe {
                horizon,
                faults: 0,
                cov4: 0,
                cov8: 0,
                dist: [0; 5],
                ring: [(255, 0); 512],
                ring_pos: 0,
            }
        }
        fn on_fault(&mut self, table: u8, page: u32) {
            self.faults += 1;
            let mut best = u32::MAX;
            for &(t, p) in &self.ring {
                if t == table {
                    let d = page.abs_diff(p);
                    if d > 0 && d < best {
                        best = d;
                    }
                }
            }
            match best {
                1 => self.dist[0] += 1,
                2..=4 => self.dist[1] += 1,
                5..=8 => self.dist[2] += 1,
                9..=64 => self.dist[3] += 1,
                _ => self.dist[4] += 1,
            }
            if best <= 4 {
                self.cov4 += 1;
            }
            if best <= 8 {
                self.cov8 += 1;
            }
            self.ring[self.ring_pos] = (table, page);
            self.ring_pos = (self.ring_pos + 1) % self.ring.len();
        }
    }

    struct Sim {
        tags: Vec<Vec<u64>>, // per BITS entry: 1 << bits tag slots, 0 = empty
        hits: [u64; BITS.len()],
        assoc: Vec<(u32, Assoc)>, // (total_bits, sim) for 2- and 4-way
        per_table_dm20: [(u64, u64); 3], // (hits, probes) at 20 bits, per table
        table_tags20: Vec<Vec<u64>>,
        quarters: Vec<(u64, u64)>, // cumulative (hits@dm20agg, probes) checkpoints
        dm20_hits: u64,
        hll: Hll,
        probes: u64,
        per_table: [u64; 3],
        pages: [std::collections::HashSet<u32>; 3],
        // ---- readahead-value analysis ----
        /// last page probed per table, for consecutive-dedupe.
        last_page: [u32; 3],
        /// same-page streak stat: probes collapsed by the dedupe.
        page_events: u64,
        /// last-touch epoch per (table, page); 666,368 pages/table.
        last_touch: Vec<Vec<u64>>,
        fault_probes: Vec<FaultProbe>,
    }

    static SIM: Mutex<Option<Sim>> = Mutex::new(None);

    pub fn record(table: usize, idx: u64) {
        let mut g = SIM.lock().unwrap();
        let sim = g.get_or_insert_with(|| Sim {
            tags: BITS.iter().map(|b| vec![0u64; 1usize << b]).collect(),
            hits: [0; BITS.len()],
            assoc: vec![
                (18, Assoc::new(18, 2)),
                (18, Assoc::new(18, 4)),
                (22, Assoc::new(22, 2)),
                (22, Assoc::new(22, 4)),
            ],
            per_table_dm20: [(0, 0); 3],
            table_tags20: (0..3).map(|_| vec![0u64; 1 << 20]).collect(),
            quarters: Vec::new(),
            dm20_hits: 0,
            hll: Hll::new(),
            probes: 0,
            per_table: [0; 3],
            pages: Default::default(),
            last_page: [u32::MAX; 3],
            page_events: 0,
            last_touch: (0..3).map(|_| vec![0u64; 670_000]).collect(),
            fault_probes: vec![
                FaultProbe::new(50_000),
                FaultProbe::new(200_000),
                FaultProbe::new(1_000_000),
                FaultProbe::new(u64::MAX),
            ],
        });
        sim.probes += 1;
        sim.per_table[table] += 1;
        let page = (idx >> 17) as u32;
        sim.pages[table].insert(page);
        // Readahead-value analysis, on the page-event stream (consecutive
        // probes to the same page collapse — those never fault).
        if sim.last_page[table] != page {
            sim.last_page[table] = page;
            sim.page_events += 1;
            let epoch = sim.page_events;
            let last = sim.last_touch[table][page as usize];
            for fp in sim.fault_probes.iter_mut() {
                let is_fault = last == 0 || epoch - last > fp.horizon;
                if is_fault {
                    fp.on_fault(table as u8, page);
                }
            }
            sim.last_touch[table][page as usize] = epoch;
        }
        let key = idx * 3 + table as u64 + 1; // +1 so 0 stays "empty"
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        sim.hll.add(h);
        for (i, b) in BITS.iter().enumerate() {
            let slot = (h >> (64 - b)) as usize;
            if sim.tags[i][slot] == key {
                sim.hits[i] += 1;
                if *b == 20 {
                    sim.dm20_hits += 1;
                }
            } else {
                sim.tags[i][slot] = key;
            }
        }
        for (_, a) in sim.assoc.iter_mut() {
            a.probe(key, h);
        }
        // per-table direct-mapped at 20 bits (each table gets its own array)
        let pt = &mut sim.per_table_dm20[table];
        pt.1 += 1;
        let slot = (h >> 44) as usize; // 64 - 20
        let t = &mut sim.table_tags20[table][slot];
        if *t == key {
            pt.0 += 1;
        }
        *t = key;
        // checkpoint every 512M probes for the warm-up curve
        if sim.probes % (512 << 20) == 0 {
            let ck = (sim.dm20_hits, sim.probes);
            sim.quarters.push(ck);
        }
    }

    pub fn report() {
        let g = SIM.lock().unwrap();
        let Some(sim) = g.as_ref() else {
            eprintln!("k8-probe-locality: no probes recorded");
            return;
        };
        eprintln!("k8 probe locality (diff_lookup stream, sequential):");
        eprintln!(
            "  probes: {}  (a {}, b {}, c {})",
            sim.probes, sim.per_table[0], sim.per_table[1], sim.per_table[2]
        );
        for t in 0..3 {
            let pg = sim.pages[t].len();
            eprintln!(
                "  table {}: {} distinct 16KiB pages = {:.2} GB touched (of 10.9 GB)",
                (b'a' + t as u8) as char,
                pg,
                pg as f64 * 16384.0 / 1e9
            );
        }
        eprintln!("  direct-mapped (idx -> h) memo, 8 B/slot tags+val:");
        for (i, b) in BITS.iter().enumerate() {
            eprintln!(
                "    bits={b:2}  {:7.3}% hit  ({} misses, {:>4} MB/worker)",
                100.0 * sim.hits[i] as f64 / sim.probes as f64,
                sim.probes - sim.hits[i],
                (1u64 << b) * 8 / (1 << 20),
            );
        }
        eprintln!("  set-associative at equal capacity (LRU in set):");
        for (b, a) in &sim.assoc {
            eprintln!(
                "    bits={b:2} {}-way  {:7.3}% hit",
                a.ways,
                100.0 * a.hits as f64 / sim.probes as f64
            );
        }
        eprintln!("  per-table direct-mapped @20 bits:");
        for t in 0..3 {
            let (h, p) = sim.per_table_dm20[t];
            eprintln!(
                "    table {}  {:7.3}% hit  ({} probes)",
                (b'a' + t as u8) as char,
                100.0 * h as f64 / p.max(1) as f64,
                p
            );
        }
        let distinct = sim.hll.estimate();
        eprintln!(
            "  distinct idx (HLL ±0.8%): {:.0}  ->  mean reuse {:.1}x, compulsory floor {:.3}%",
            distinct,
            sim.probes as f64 / distinct,
            100.0 * distinct / sim.probes as f64
        );
        eprintln!(
            "  readahead value (page-event stream: {} events = {:.1}% of probes changed page):",
            sim.page_events,
            100.0 * sim.page_events as f64 / sim.probes as f64
        );
        eprintln!("    horizon      faults      cov±4    cov±8    Δ=1 | 2-4 | 5-8 | 9-64 | far");
        for fp in &sim.fault_probes {
            let h = if fp.horizon == u64::MAX {
                "first-touch".to_string()
            } else {
                format!("{:>7}", fp.horizon)
            };
            let pct = |x: u64| 100.0 * x as f64 / fp.faults.max(1) as f64;
            eprintln!(
                "    {h:>11}  {:>10}  {:6.2}%  {:6.2}%   {:4.1} | {:4.1} | {:4.1} | {:4.1} | {:4.1}",
                fp.faults,
                pct(fp.cov4),
                pct(fp.cov8),
                pct(fp.dist[0]),
                pct(fp.dist[1]),
                pct(fp.dist[2]),
                pct(fp.dist[3]),
                pct(fp.dist[4]),
            );
        }
        eprintln!("  dm@20 cumulative hit rate at 512M-probe checkpoints (warm-up curve):");
        let mut prev = (0u64, 0u64);
        for &(h, p) in &sim.quarters {
            eprintln!(
                "    at {:>5}M probes: cumulative {:6.3}%   window {:6.3}%",
                p / 1_000_000,
                100.0 * h as f64 / p as f64,
                100.0 * (h - prev.0) as f64 / (p - prev.1) as f64
            );
            prev = (h, p);
        }
    }
}

/// Print the k8 probe-locality report (see [`k8_locality`]).
#[cfg(feature = "k8-probe-locality")]
pub fn k8_locality_report() {
    k8_locality::report();
}

/// Partial-PDB ("winner set") feasibility instrumentation.
///
/// At every consult the engine already knows each group-view's true `h` (the
/// diff-chain maintains it) and can compute the group's Manhattan distance from
/// the same positions. This module accumulates, online:
///
/// - the **prune-survival curve**: for each surplus threshold `s`, would this
///   consult still have pruned if every group with `h − MD < s` were served MD
///   instead of its table value (i.e. only "winner" configs kept)?
/// - the **map hit-rate curve**: the surplus distribution of the two *changed*
///   group-views — the probes a sparse winner-map would actually receive.
///
/// Sequential runs only; counts, so valid under load.
#[cfg(feature = "k8-probe-locality")]
mod k8_surplus {
    use std::sync::Mutex;

    const SMAX: usize = 16;

    struct Sim {
        consults: u64,
        actual_prunes: u64,
        /// simulated prunes with only surplus>=s configs kept, s in 0..SMAX
        sim_prunes: [u64; SMAX],
        /// prunes with MD alone (s = infinity)
        md_prunes: u64,
        /// surplus histogram of the two changed group-views per consult
        changed_surplus: [u64; 64],
        /// surplus histogram over all six group-views (background)
        all_surplus: [u64; 64],
        /// (hk8 − h_cwd) advantage histograms: at pruning consults and at
        /// consults where the advantage did not suffice.
        adv_prune: [u64; 64],
        adv_noprune: [u64; 64],
        /// prunes retained under capped-surplus encoding h' = MD + 2·min(surplus/2, cap),
        /// for cap = 1, 2, 3.
        cap_retained: [u64; 3],
        /// absolute scores at pruning consults, bucketed by 4: hk8 and h_cwd.
        abs_hk8: [u64; 40],
        abs_hcwd: [u64; 40],
        /// slack = need − h_cwd over ALL consults — the gate-rate denominator.
        slack_hist: [u64; 32],
    }

    static SIM: Mutex<Option<Sim>> = Mutex::new(None);

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        h: &[[u8; 3]; 2],
        md: &[[u8; 3]; 2],
        gi: usize,
        gj: usize,
        need: u8,
        h_cwd: u8,
        hk8: u8,
    ) {
        let mut g = SIM.lock().unwrap();
        let sim = g.get_or_insert_with(|| Sim {
            consults: 0,
            actual_prunes: 0,
            sim_prunes: [0; SMAX],
            md_prunes: 0,
            changed_surplus: [0; 64],
            all_surplus: [0; 64],
            adv_prune: [0; 64],
            adv_noprune: [0; 64],
            cap_retained: [0; 3],
            abs_hk8: [0; 40],
            abs_hcwd: [0; 40],
            slack_hist: [0; 32],
        });
        sim.consults += 1;
        let sur = |v: usize, k: usize| (h[v][k] - md[v][k]) as usize; // admissible: h >= MD
        for v in 0..2 {
            for k in 0..3 {
                sim.all_surplus[sur(v, k).min(63)] += 1;
            }
        }
        sim.changed_surplus[sur(0, gi).min(63)] += 1;
        sim.changed_surplus[sur(1, gj).min(63)] += 1;

        let need = need as u16;
        for (s, slot) in sim.sim_prunes.iter_mut().enumerate() {
            let mut best = 0u16;
            for v in 0..2 {
                let mut t = 0u16;
                for k in 0..3 {
                    t += if sur(v, k) >= s { h[v][k] } else { md[v][k] } as u16;
                }
                best = best.max(t);
            }
            if best > need {
                *slot += 1;
            }
        }
        // s = 0 keeps everything == the actual consult outcome.
        sim.actual_prunes = sim.sim_prunes[0];
        sim.slack_hist[(need.saturating_sub(h_cwd as u16) as usize).min(31)] += 1;
        // Advantage histogram: hk8 − h_cwd, split by whether this consult pruned.
        let adv = hk8.saturating_sub(h_cwd) as usize;
        let pruned = hk8 as u16 > need;
        if pruned {
            sim.adv_prune[adv.min(63)] += 1;
            sim.abs_hk8[(hk8 as usize / 4).min(39)] += 1;
            sim.abs_hcwd[(h_cwd as usize / 4).min(39)] += 1;
            // Capped-surplus retention: would this prune survive the O(1)-cold
            // encoding h' = MD + 2·min(surplus/2, cap)?
            for (ci, cap) in [1u16, 2, 3].iter().enumerate() {
                let mut best = 0u16;
                for v in 0..2 {
                    let t: u16 = (0..3)
                        .map(|k| {
                            let sur2 = ((h[v][k] - md[v][k]) / 2) as u16;
                            md[v][k] as u16 + 2 * sur2.min(*cap)
                        })
                        .sum();
                    best = best.max(t);
                }
                if best > need {
                    sim.cap_retained[ci] += 1;
                }
            }
        } else {
            sim.adv_noprune[adv.min(63)] += 1;
        }
        let mut md_best = 0u16;
        for v in 0..2 {
            let t: u16 = (0..3).map(|k| md[v][k] as u16).sum();
            md_best = md_best.max(t);
        }
        if md_best > need {
            sim.md_prunes += 1;
        }
    }

    pub(super) fn report() {
        let g = SIM.lock().unwrap();
        let Some(sim) = g.as_ref() else {
            return;
        };
        eprintln!("k8 partial-PDB (winner-set) feasibility:");
        eprintln!(
            "  consults {}   actual k8 prunes {} ({:.3}%)   MD-only prunes {} ({:.1}% of actual)",
            sim.consults,
            sim.actual_prunes,
            100.0 * sim.actual_prunes as f64 / sim.consults as f64,
            sim.md_prunes,
            100.0 * sim.md_prunes as f64 / sim.actual_prunes.max(1) as f64,
        );
        let changed_total: u64 = sim.changed_surplus.iter().sum();
        eprintln!("    s   prunes-kept   map-hit-rate (changed probes with surplus >= s)");
        let mut tail: u64 = changed_total;
        for s in 0..SMAX {
            if s > 0 {
                tail -= sim.changed_surplus[s - 1];
            }
            eprintln!(
                "   {s:2}     {:6.2}%        {:6.2}%",
                100.0 * sim.sim_prunes[s] as f64 / sim.actual_prunes.max(1) as f64,
                100.0 * tail as f64 / changed_total.max(1) as f64,
            );
        }
        eprintln!("  advantage (hk8 − h_cwd) at PRUNING consults [0..15,16+]:");
        let dump = |hist: &[u64; 64]| {
            let n: u64 = hist.iter().sum::<u64>().max(1);
            eprint!("   ");
            for v in hist[..16].iter() {
                eprint!(" {:.2}", 100.0 * *v as f64 / n as f64);
            }
            eprintln!(
                " {:.2}   (n={n})",
                100.0 * hist[16..].iter().sum::<u64>() as f64 / n as f64
            );
        };
        dump(&sim.adv_prune);
        eprintln!("  advantage at NON-pruning consults [0..15,16+]:");
        dump(&sim.adv_noprune);
        eprintln!("  slack (need − h_cwd) over ALL consults [0..15,16+] (%):");
        {
            let n: u64 = sim.slack_hist.iter().sum::<u64>().max(1);
            eprint!("   ");
            for v in sim.slack_hist[..16].iter() {
                eprint!(" {:.2}", 100.0 * *v as f64 / n as f64);
            }
            eprintln!(
                " {:.2}",
                100.0 * sim.slack_hist[16..].iter().sum::<u64>() as f64 / n as f64
            );
        }
        eprintln!("  ABSOLUTE hk8 at pruning consults (buckets of 4, % — first nonzero..last):");
        let dump_abs = |hist: &[u64; 40]| {
            let n: u64 = hist.iter().sum::<u64>().max(1);
            let lo = hist.iter().position(|&v| v > 0).unwrap_or(0);
            let hi = hist.iter().rposition(|&v| v > 0).unwrap_or(0);
            eprint!("   ");
            for (i, v) in hist.iter().enumerate().take(hi + 1).skip(lo) {
                eprint!(" {}:{:.1}", i * 4, 100.0 * *v as f64 / n as f64);
            }
            eprintln!();
        };
        dump_abs(&sim.abs_hk8);
        eprintln!("  ABSOLUTE h_cwd at the same consults:");
        dump_abs(&sim.abs_hcwd);
        eprintln!("  capped-surplus prune retention (h' = MD + 2*min(surplus/2, cap)):");
        for (ci, cap) in [1, 2, 3].iter().enumerate() {
            eprintln!(
                "    cap +{}: {:6.2}% of prunes retained",
                2 * cap,
                100.0 * sim.cap_retained[ci] as f64 / sim.actual_prunes.max(1) as f64
            );
        }
        eprint!("  all-view surplus histogram (0..15,16+):");
        let hi: u64 = sim.all_surplus[16..].iter().sum();
        for v in sim.all_surplus[..16].iter() {
            eprint!(" {v}");
        }
        eprintln!(" {hi}");
    }
}

/// Print the k8 partial-PDB feasibility report.
#[cfg(feature = "k8-probe-locality")]
pub fn k8_surplus_report() {
    k8_surplus::report();
}

/// Prune-config **harvest**: which configurations participate in successful
/// prunes, how many distinct ones there are, and whether a set harvested at
/// one threshold / time-slice covers the prunes of the next.
///
/// Phase A = the first bound seen (full 144 iteration): harvest every
/// load-bearing config (surplus ≥ 2 at a pruning consult) into a Bloom set.
/// Phase B = the next bound: the first `SPLIT_AT` prune events harvest a
/// second set; all later events are held out and replayed against both sets —
/// an event is "reproduced" if serving map-hits their `h` and misses their MD
/// still clears the prune bound. Count-based; sequential runs only.
#[cfg(feature = "k8-probe-locality")]
mod k8_harvest {
    use std::sync::Mutex;

    const SPLIT_AT: u64 = 50_000_000;

    struct Hll {
        reg: Vec<u8>,
    }
    impl Hll {
        fn new() -> Self {
            Hll {
                reg: vec![0; 1 << 14],
            }
        }
        fn add(&mut self, h: u64) {
            let i = (h >> 50) as usize;
            let z = (h << 14).leading_zeros() as u8 + 1;
            if z > self.reg[i] {
                self.reg[i] = z;
            }
        }
        fn estimate(&self) -> f64 {
            let m = self.reg.len() as f64;
            let sum: f64 = self.reg.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
            let e = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
            if e <= 2.5 * m {
                let zeros = self.reg.iter().filter(|&&r| r == 0).count();
                if zeros > 0 {
                    return m * (m / zeros as f64).ln();
                }
            }
            e
        }
    }

    /// 512 MB Bloom filter (2^32 bits, 4 double-hashed probes): for even 300 M
    /// inserts the false-positive rate is well under 1%, and false positives
    /// only *overstate* coverage — flagged in the report.
    struct Bloom {
        bits: Vec<u64>,
    }
    impl Bloom {
        fn new() -> Self {
            Bloom {
                bits: vec![0u64; 1 << 26],
            }
        }
        #[inline]
        fn idx(key: u64, i: u64) -> usize {
            let h1 = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let h2 = key.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) | 1;
            (h1.wrapping_add(i.wrapping_mul(h2)) >> 32) as usize
        }
        fn insert(&mut self, key: u64) {
            for i in 0..4 {
                let b = Self::idx(key, i);
                self.bits[b >> 6] |= 1u64 << (b & 63);
            }
        }
        fn contains(&self, key: u64) -> bool {
            (0..4).all(|i| {
                let b = Self::idx(key, i);
                self.bits[b >> 6] & (1u64 << (b & 63)) != 0
            })
        }
    }

    struct Sim {
        first_bound: u8,
        a_events: u64,
        a_inserts: u64,
        a_hll: Hll,
        a_bloom: Bloom,
        b_events: u64,
        b_inserts: u64,
        b_hll_half: Hll,
        b_bloom: Bloom,
        b_hll_all: Hll,
        test_events: u64,
        covered_by_a: u64,
        covered_by_b: u64,
        cfg_total: u64,
        cfg_in_a: u64,
        lb_hist: [u64; 7],
        view_win: [u64; 3],
        depth_hist: [u64; 16],
    }

    static SIM: Mutex<Option<Sim>> = Mutex::new(None);

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        h: &[[u8; 3]; 2],
        md: &[[u8; 3]; 2],
        keys: &[[u64; 3]; 2],
        need: u8,
        bound: u8,
        depth: u8,
    ) {
        let mut g = SIM.lock().unwrap();
        let sim = g.get_or_insert_with(|| Sim {
            first_bound: bound,
            a_events: 0,
            a_inserts: 0,
            a_hll: Hll::new(),
            a_bloom: Bloom::new(),
            b_events: 0,
            b_inserts: 0,
            b_hll_half: Hll::new(),
            b_bloom: Bloom::new(),
            b_hll_all: Hll::new(),
            test_events: 0,
            covered_by_a: 0,
            covered_by_b: 0,
            cfg_total: 0,
            cfg_in_a: 0,
            lb_hist: [0; 7],
            view_win: [0; 3],
            depth_hist: [0; 16],
        });

        // Features.
        let mut lb = 0usize;
        for v in 0..2 {
            for k in 0..3 {
                if h[v][k] > md[v][k] {
                    lb += 1;
                }
            }
        }
        sim.lb_hist[lb] += 1;
        let need16 = need as u16;
        let sum = |v: usize| (0..3).map(|k| h[v][k] as u16).sum::<u16>();
        let (p0, p1) = (sum(0) > need16, sum(1) > need16);
        sim.view_win[if p0 && p1 {
            2
        } else if p0 {
            0
        } else {
            1
        }] += 1;
        sim.depth_hist[((depth / 4) as usize).min(15)] += 1;

        let khash = |key: u64| key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let phase_a = bound == sim.first_bound;
        if phase_a {
            sim.a_events += 1;
            for v in 0..2 {
                for k in 0..3 {
                    if h[v][k] > md[v][k] {
                        sim.a_bloom.insert(keys[v][k]);
                        sim.a_hll.add(khash(keys[v][k]));
                        sim.a_inserts += 1;
                    }
                }
            }
        } else {
            sim.b_events += 1;
            for v in 0..2 {
                for k in 0..3 {
                    if h[v][k] > md[v][k] {
                        sim.b_hll_all.add(khash(keys[v][k]));
                    }
                }
            }
            if sim.b_events <= SPLIT_AT {
                for v in 0..2 {
                    for k in 0..3 {
                        if h[v][k] > md[v][k] {
                            sim.b_bloom.insert(keys[v][k]);
                            sim.b_hll_half.add(khash(keys[v][k]));
                            sim.b_inserts += 1;
                        }
                    }
                }
            } else {
                sim.test_events += 1;
                let served = |bloom: &Bloom| -> u16 {
                    (0..2)
                        .map(|v| {
                            (0..3)
                                .map(|k| {
                                    if h[v][k] == md[v][k] || bloom.contains(keys[v][k]) {
                                        h[v][k] as u16
                                    } else {
                                        md[v][k] as u16
                                    }
                                })
                                .sum::<u16>()
                        })
                        .max()
                        .unwrap()
                };
                // borrow dance: read blooms via raw refs before mutating counters
                let (ca, cb) = {
                    let sa = served(&sim.a_bloom) > need16;
                    let sb = served(&sim.b_bloom) > need16;
                    (sa, sb)
                };
                if ca {
                    sim.covered_by_a += 1;
                }
                if cb {
                    sim.covered_by_b += 1;
                }
                for v in 0..2 {
                    for k in 0..3 {
                        if h[v][k] > md[v][k] {
                            sim.cfg_total += 1;
                            if sim.a_bloom.contains(keys[v][k]) {
                                sim.cfg_in_a += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    pub(super) fn report() {
        let g = SIM.lock().unwrap();
        let Some(sim) = g.as_ref() else {
            return;
        };
        eprintln!("k8 prune-config harvest:");
        eprintln!(
            "  phase A (bound {}): {} prune events, {} load-bearing inserts, distinct ~{:.2e} (reuse {:.1}x)",
            sim.first_bound,
            sim.a_events,
            sim.a_inserts,
            sim.a_hll.estimate(),
            sim.a_inserts as f64 / sim.a_hll.estimate().max(1.0),
        );
        eprintln!(
            "  phase B: {} prune events; harvest half {} inserts, distinct ~{:.2e}; all-B distinct ~{:.2e}",
            sim.b_events,
            sim.b_inserts,
            sim.b_hll_half.estimate(),
            sim.b_hll_all.estimate(),
        );
        eprintln!(
            "  held-out coverage ({} events; Bloom fp inflates these slightly):",
            sim.test_events
        );
        eprintln!(
            "    by phase-A set (prev threshold): {:6.2}% events reproduced; {:.2}% of load-bearing configs present",
            100.0 * sim.covered_by_a as f64 / sim.test_events.max(1) as f64,
            100.0 * sim.cfg_in_a as f64 / sim.cfg_total.max(1) as f64,
        );
        eprintln!(
            "    by B-first-half set (same threshold): {:6.2}% events reproduced",
            100.0 * sim.covered_by_b as f64 / sim.test_events.max(1) as f64,
        );
        eprint!("  load-bearing group-views per prune event [0..6]:");
        for v in &sim.lb_hist {
            eprint!(" {v}");
        }
        eprintln!();
        eprintln!(
            "  winning view: normal {} reflected {} both {}",
            sim.view_win[0], sim.view_win[1], sim.view_win[2]
        );
        eprint!("  prune depth (buckets of 4 plies):");
        for v in &sim.depth_hist {
            eprint!(" {v}");
        }
        eprintln!();
    }
}

/// Print the harvest report.
#[cfg(feature = "k8-probe-locality")]
pub fn k8_harvest_report() {
    k8_harvest::report();
}

/// Tile-level structure of pruning configurations, against a 1/64 background
/// sample of non-pruning consults. The payload is the surplus × linear-conflict
/// cross-tab: if the table's edge over Manhattan is mostly LC-shaped, a cheap
/// computable-per-group refinement exists; if surplus appears without conflicts,
/// the edge is deeper interaction structure no simple formula captures.
#[cfg(feature = "k8-probe-locality")]
mod k8_struct {
    use std::sync::Mutex;

    #[derive(Default)]
    struct Feat {
        configs: u64,
        blank: [u64; 25],
        displaced: [u64; 9],
        lc: [u64; 4],
        group: [u64; 3],
    }

    #[derive(Default)]
    struct Sim {
        pruner: Feat,
        background: Feat,
        /// pruners only: surplus bucket {2,4,6+} × lc pairs {0,1,2,3+}
        joint: [[u64; 4]; 3],
        bg_tick: u64,
        /// tile-position distributions in the BOARD frame (reflected views
        /// mapped back through σ/τ): `[tile-1][cell]` counts for three
        /// populations — P: load-bearing at pruning consults; N: load-bearing
        /// at non-pruning consults (sampled 1/64); B: all configs at
        /// non-pruning consults (sampled 1/64).
        tc_p: Vec<[u64; 25]>,
        tc_n: Vec<[u64; 25]>,
        tc_b: Vec<[u64; 25]>,
    }

    static SIM: Mutex<Option<Sim>> = Mutex::new(None);

    /// Per-config features from the projection: displaced count, per-line
    /// linear-conflict pairs (rows + cols), given the view-correct blank.
    fn features(proj: &crate::puzzle24::pdb::ProjectedState, tiles: &[u8; 8]) -> (u32, u32) {
        let mut displaced = 0u32;
        let mut lc = 0u32;
        let pos: Vec<(u8, u8)> = tiles.iter().map(|&t| (proj.pos_of(t), t - 1)).collect();
        for &(p, g) in &pos {
            if p != g {
                displaced += 1;
            }
        }
        for i in 0..8 {
            for j in i + 1..8 {
                let (pi, gi) = pos[i];
                let (pj, gj) = pos[j];
                // row conflict: same current row == both goal rows, reversed order
                if pi / 5 == pj / 5 && gi / 5 == pi / 5 && gj / 5 == pi / 5 {
                    let inv = (pi % 5 < pj % 5) != (gi % 5 < gj % 5);
                    if inv {
                        lc += 1;
                    }
                }
                // column conflict
                if pi % 5 == pj % 5 && gi % 5 == pi % 5 && gj % 5 == pi % 5 {
                    let inv = (pi / 5 < pj / 5) != (gi / 5 < gj / 5);
                    if inv {
                        lc += 1;
                    }
                }
            }
        }
        (displaced, lc)
    }

    fn tally(
        f: &mut Feat,
        proj: &crate::puzzle24::pdb::ProjectedState,
        tiles: &[u8; 8],
        blank: usize,
        group: usize,
    ) -> u32 {
        let (displaced, lc) = features(proj, tiles);
        f.configs += 1;
        f.blank[blank] += 1;
        f.displaced[displaced.min(8) as usize] += 1;
        f.lc[lc.min(3) as usize] += 1;
        f.group[group] += 1;
        lc
    }

    /// Accumulate one group-view's 8 tile positions into a `[tile-1][cell]`
    /// matrix, mapping reflected-view (v=1) labels/cells back to the board
    /// frame through τ/σ.
    fn tally_tc(
        m: &mut [[u64; 25]],
        proj: &crate::puzzle24::pdb::ProjectedState,
        tiles: &[u8; 8],
        v: usize,
    ) {
        use crate::puzzle24::symmetry::{SIGMA, TAU};
        for &t in tiles {
            let c = proj.pos_of(t) as usize;
            let (bt, bc) = if v == 0 {
                (t as usize, c)
            } else {
                (TAU[t as usize] as usize, SIGMA[c] as usize)
            };
            m[bt - 1][bc] += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        slot_views: &[[crate::puzzle24::pdb::ProjectedState; 3]; 2],
        h: &[[u8; 3]; 2],
        md: &[[u8; 3]; 2],
        tiles: &[[u8; 8]; 3],
        n: usize,
        rn: usize,
        prune: bool,
    ) {
        let mut g = SIM.lock().unwrap();
        let sim = g.get_or_insert_with(Sim::default);
        if sim.tc_p.is_empty() {
            sim.tc_p = vec![[0u64; 25]; 24];
            sim.tc_n = vec![[0u64; 25]; 24];
            sim.tc_b = vec![[0u64; 25]; 24];
        }
        if prune {
            for v in 0..2 {
                let blank = if v == 0 { n } else { rn };
                for k in 0..3 {
                    if h[v][k] > md[v][k] {
                        let lc = tally(&mut sim.pruner, &slot_views[v][k], &tiles[k], blank, k);
                        let sb = (((h[v][k] - md[v][k]) / 2).min(3) as usize).saturating_sub(1);
                        sim.joint[sb][lc.min(3) as usize] += 1;
                        tally_tc(&mut sim.tc_p, &slot_views[v][k], &tiles[k], v);
                    }
                }
            }
        } else {
            sim.bg_tick += 1;
            if sim.bg_tick % 64 == 0 {
                for v in 0..2 {
                    let blank = if v == 0 { n } else { rn };
                    for k in 0..3 {
                        tally(&mut sim.background, &slot_views[v][k], &tiles[k], blank, k);
                        tally_tc(&mut sim.tc_b, &slot_views[v][k], &tiles[k], v);
                        if h[v][k] > md[v][k] {
                            tally_tc(&mut sim.tc_n, &slot_views[v][k], &tiles[k], v);
                        }
                    }
                }
            }
        }
    }

    pub(super) fn report() {
        let g = SIM.lock().unwrap();
        let Some(sim) = g.as_ref() else {
            return;
        };
        let pct = |f: &Feat, c: u64| 100.0 * c as f64 / f.configs.max(1) as f64;
        eprintln!("k8 pruner tile-structure (load-bearing configs vs 1/64 background):");
        for (name, f) in [("pruners", &sim.pruner), ("background", &sim.background)] {
            eprintln!("  {name}: {} configs", f.configs);
            eprint!("    displaced-of-8:");
            for d in &f.displaced {
                eprint!(" {:.1}", pct(f, *d));
            }
            eprintln!();
            eprint!("    lc-pairs [0,1,2,3+]:");
            for d in &f.lc {
                eprint!(" {:.1}", pct(f, *d));
            }
            eprintln!();
            eprint!("    group a/b/c:");
            for d in &f.group {
                eprint!(" {:.1}", pct(f, *d));
            }
            eprintln!();
            eprintln!("    blank heat (rows of 5, %):");
            for r in 0..5 {
                eprint!("     ");
                for c in 0..5 {
                    eprint!(" {:5.1}", pct(f, f.blank[r * 5 + c]));
                }
                eprintln!();
            }
        }
        eprintln!(
            "  pruners: surplus x lc-pairs cross-tab (rows: surplus 2/4/6+; cols: lc 0,1,2,3+; %):"
        );
        let tot: u64 = sim.joint.iter().flatten().sum();
        for row in &sim.joint {
            eprint!("   ");
            for cell in row {
                eprint!(" {:6.2}", 100.0 * *cell as f64 / tot.max(1) as f64);
            }
            eprintln!();
        }
        // Raw tile-position matrices, one line per (population, tile), for
        // offline analysis: "TC <pop> <tile>: c0 c1 ... c24".
        for (tag, m) in [("P", &sim.tc_p), ("N", &sim.tc_n), ("B", &sim.tc_b)] {
            for (ti, row) in m.iter().enumerate() {
                eprint!("TC {tag} {}:", ti + 1);
                for c in row {
                    eprint!(" {c}");
                }
                eprintln!();
            }
        }
    }
}

/// Print the tile-structure report.
#[cfg(feature = "k8-probe-locality")]
pub fn k8_struct_report() {
    k8_struct::report();
}

/// Autopsy sampling: every Nth prune event dumps the full board + per-view
/// group values, for offline reconstruction of the abstract plans.
#[cfg(feature = "k8-probe-locality")]
static AUTOPSY_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Manhattan distance of one group-view's pattern tiles (goal cell of tile `t`
/// is `t − 1`; holds for the reflected view too, since σ fixes GOAL).
#[cfg(feature = "k8-probe-locality")]
fn k8_group_md(proj: &crate::puzzle24::pdb::ProjectedState, tiles: &[u8; 8]) -> u8 {
    let mut md = 0u16;
    for &t in tiles {
        let p = proj.pos_of(t) as u16;
        let g = (t - 1) as u16;
        md += (p / 5).abs_diff(g / 5) + (p % 5).abs_diff(g % 5);
    }
    md.min(255) as u8
}

/// Seed depth-0 k8 state from `board`: 6 projections + 6 cold lookups (O(h)
/// each — this is the once-per-subtree cost, never the per-node cost). Returns
/// `max` of the two views' additive sums.
fn seed_k8(arena: &mut Arena, board: &State, ctx: &K8Ctx) -> u8 {
    let rs = symmetry::reflect(board);
    let slot = &mut arena.k8[0];
    let (mut s0, mut s1) = (0u16, 0u16);
    for i in 0..3 {
        let db = &ctx.dbs[i];
        slot.views[0][i] = crate::puzzle24::pdb::ProjectedState::from_state(board, db.pattern());
        slot.views[1][i] = crate::puzzle24::pdb::ProjectedState::from_state(&rs, db.pattern());
        slot.h[0][i] = db.cold_lookup(board);
        slot.h[1][i] = db.cold_lookup(&rs);
        s0 += slot.h[0][i] as u16;
        s1 += slot.h[1][i] as u16;
    }
    s0.max(s1).min(255) as u8
}

/// Advance the k8 state across move `m` from depth `d` to `d+1` and return the
/// child's `h_k8 = max(Σ normal, Σ reflected)`.
///
/// Exactly one group is cost-1 per view (the 3 patterns partition the 24 tiles),
/// and by the projected-edge law the old `ZpdbInc` documented, **cost-0 slides
/// leave the index and `h` unchanged** — so the four cost-0 group-views are not
/// slid at all. Their stored blank goes stale; it is healed by
/// [`ProjectedState::set_blank_pos`] the next time that group-view is cost-1,
/// which is sound because `rank` reads pattern-tile positions, the blank, and
/// `cells` only at occupied pattern cells — all maintained. Net cost per
/// consult: one slot copy + **2 slides + 2 ranks + 2 `diff_lookup`s** (the
/// recorded recursive figure, probes/make = 2.000, §8i).
///
/// The reflected view needs no `transpose_move`: its blank/target cells are the
/// σ-images of the board's, and its cost-1 group is the σ-relabelled tile's.
#[inline]
fn k8_child(arena: &mut Arena, ctx: &K8Ctx, d: usize, geom: Geom, tile: usize) -> u8 {
    let (lo, hi) = arena.k8.split_at_mut(d + 1);
    let parent = &lo[d];
    let child = &mut hi[0];
    *child = *parent;
    let b = arena.hot[d].blank as usize; // pre-move blank cell
    let n = geom.from as usize; // the moved tile's cell = blank destination
                                // `tile` arrives goal-coded (the engine's boards are `Coded`); the PDB
                                // machinery talks tile numbers.
    let tile = TILE_OF_CODE[tile] as usize;

    // Normal view.
    let gi = ctx.group_of[tile] as usize;
    {
        let db = &ctx.dbs[gi];
        let v = &mut child.views[0][gi];
        v.set_blank_pos(b as u8);
        let cost = v.apply_in_place_at(b, n);
        debug_assert_eq!(cost, 1, "moved tile must be in its own group's pattern");
        let idx = db.layout().rank(v, db.pattern());
        #[cfg(feature = "k8-probe-locality")]
        k8_locality::record(gi, idx);
        child.h[0][gi] = db.diff_lookup(idx, parent.h[0][gi]);
    }

    // Reflected view: σ-image cells, σ-relabelled tile.
    let rt = symmetry::TAU[tile] as usize;
    let gj = ctx.group_of[rt] as usize;
    {
        let db = &ctx.dbs[gj];
        let rb = symmetry::SIGMA[b] as usize;
        let rn = symmetry::SIGMA[n] as usize;
        let v = &mut child.views[1][gj];
        v.set_blank_pos(rb as u8);
        let cost = v.apply_in_place_at(rb, rn);
        debug_assert_eq!(cost, 1, "σ-image tile must be in its own group's pattern");
        let idx = db.layout().rank(v, db.pattern());
        #[cfg(feature = "k8-probe-locality")]
        k8_locality::record(gj, idx);
        child.h[1][gj] = db.diff_lookup(idx, parent.h[1][gj]);
    }

    let s0 = child.h[0][0] as u16 + child.h[0][1] as u16 + child.h[0][2] as u16;
    let s1 = child.h[1][0] as u16 + child.h[1][1] as u16 + child.h[1][2] as u16;
    s0.max(s1).min(255) as u8
}

// ------------------------- last-move (cwd-lm) tier ----------------------------

/// Direct-mapped front cache for the LM table: axis key → the four queryable
/// branch values (lines 0–3; line 4 is the free-ride degeneracy and is never
/// looked up). One miss fills all four lines from a single map probe, so a
/// row-key miss also warms the same key's future queries.
///
/// Sized by measured priors, not a fresh sim: LM probes are axis keys — the
/// exact population the cWD [`ProbeCache`] serves at a measured in-situ
/// 98.9–99.7% across 64 K–256 K entries — and the LM stream is the hotter
/// survivor subset of it. Default 18 bits (16 B slots → 4 MB/worker, half the
/// cWD cache); `FLAT_LM_CACHE_BITS` overrides for sweeps (read once, at
/// allocation).
struct LmCache {
    slots: Box<[LmSlot]>,
    shift: u32,
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct LmSlot {
    /// Axis key, or 0 for empty (a real key always has nonzero matrix bits).
    tag: u64,
    vals: [u8; 4],
}

#[cfg(feature = "probe-cache-stats")]
static LM_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "probe-cache-stats")]
static LM_CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Print the LM front cache's aggregate hit/miss counters (all workers).
#[cfg(feature = "probe-cache-stats")]
pub fn lm_cache_stats_report() {
    let h = LM_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let m = LM_CACHE_MISSES.load(std::sync::atomic::Ordering::Relaxed);
    if h + m > 0 {
        eprintln!(
            "    lm cache: {h} hits / {m} misses = {:.3}% hit",
            100.0 * h as f64 / (h + m) as f64
        );
    }
}

fn lm_cache_bits() -> u32 {
    static BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| {
        std::env::var("FLAT_LM_CACHE_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(18)
    })
}

impl LmCache {
    fn new() -> Self {
        let bits = lm_cache_bits();
        assert!((8..=24).contains(&bits), "implausible LM cache size");
        LmCache {
            slots: vec![
                LmSlot {
                    tag: 0,
                    vals: [0xFF; 4]
                };
                1usize << bits
            ]
            .into_boxed_slice(),
            shift: 64 - bits,
        }
    }

    /// The four branch values for `key`, via the cache.
    #[inline(always)]
    fn get(&mut self, lm: &super::cwd_lm::CwdLmMm, key: u64) -> [u8; 4] {
        let i = ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> self.shift) as usize;
        if self.slots[i].tag != key {
            let mut vals = [0xFFu8; 4];
            if let Some((s, _)) = lm.probe(key) {
                vals.copy_from_slice(s);
            }
            self.slots[i] = LmSlot { tag: key, vals };
            #[cfg(feature = "probe-cache-stats")]
            LM_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            #[cfg(feature = "probe-cache-stats")]
            LM_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.slots[i].vals
    }
}

/// Seed depth-0 last-move positions from the board.
fn seed_lm(arena: &mut Arena, board: &State) {
    arena
        .lmcache
        .get_or_insert_with(|| Box::new(LmCache::new()));
    let p20 = board.0.iter().position(|&t| t == 20).unwrap();
    let p24 = board.0.iter().position(|&t| t == 24).unwrap();
    arena.lmpos[0] = [(p20 / W_LM) as u8, (p24 % W_LM) as u8];
}

const W_LM: usize = 5;

/// Price the last-move branches for the child at depth `d+1` and return its
/// effective heuristic `max(h_cwd, min(branch20, branch24))`.
///
/// branch20 = refined row distance (tile 20 obligated to cross row 3→4 and
/// return) + the child's FULL column term (wd + surcharge — sound to combine
/// because vertical and horizontal moves are disjoint move sets); branch24 is
/// the mirror. A branch degenerates to `h_cwd` when its tile is already in
/// line 4 (free ride) or the table has no entry (invalid/unreached state).
#[inline]
fn lm_child(
    arena: &mut Arena,
    lm: &super::cwd_lm::CwdLmMm,
    d: usize,
    tile_decoded: usize,
    h_cwd: u8,
) -> u8 {
    // Update the child's tracked positions: the moved tile lands on the
    // parent's blank cell.
    let mut lp = arena.lmpos[d];
    let bcell = arena.hot[d].blank as usize;
    if tile_decoded == 20 {
        lp[0] = (bcell / W_LM) as u8;
    } else if tile_decoded == 24 {
        lp[1] = (bcell % W_LM) as u8;
    }
    arena.lmpos[d + 1] = lp;
    debug_assert_eq!(
        {
            let b = arena.board[d + 1].0.decode();
            let p20 = b.0.iter().position(|&t| t == 20).unwrap();
            let p24 = b.0.iter().position(|&t| t == 24).unwrap();
            [(p20 / W_LM) as u8, (p24 % W_LM) as u8]
        },
        lp,
        "incremental lmpos drifted from the board"
    );

    let f = &arena.hot[d + 1];
    let rslot = &arena.row[f.row_at as usize];
    let cslot = &arena.col[f.col_at as usize];
    let rterm = rslot.term();
    let cterm = cslot.term();
    let (rkey, ckey) = (rslot.key, cslot.key);
    let cache = arena
        .lmcache
        .as_mut()
        .expect("seed_lm allocates the cache")
        .as_mut();

    let b20 = if lp[0] >= 4 {
        h_cwd
    } else {
        match cache.get(lm, rkey)[lp[0] as usize] {
            0xFF => h_cwd,
            d20 => d20.saturating_add(cterm),
        }
    };
    let b24 = if lp[1] >= 4 {
        h_cwd
    } else {
        match cache.get(lm, ckey)[lp[1] as usize] {
            0xFF => h_cwd,
            d24 => d24.saturating_add(rterm),
        }
    };
    h_cwd.max(b20.min(b24))
}

// ----------------------- last-two-moves (cwd-lm2) tier ------------------------

/// Load the last-two-moves mmap artifact (`data/cwd_lm_mm.bin`) — a
/// page-table operation, not a parse; pages fault in on demand.
pub fn load_cwd_lm_mm(path: &std::path::Path) -> std::io::Result<super::cwd_lm::CwdLmMm> {
    super::cwd_lm::CwdLmMm::load(path)
}

/// Combined front cache for the LM2 tier: one tag per axis key serving both
/// the four single-tracked line values AND the 25 pair values, filled from
/// two map probes on a miss. The LM2 tier probes only this cache, leaving
/// [`LmCache`] (and the `--lm` path's layout) untouched.
struct Lm2Cache {
    slots: Box<[Lm2Slot]>,
    shift: u32,
}

#[repr(align(32))]
#[derive(Clone, Copy)]
struct Lm2Slot {
    /// Axis key, or 0 for empty (a real key always has nonzero matrix bits).
    tag: u64,
    single: [u8; 4],
    /// Only the queryable pair placements — la ∈ 0..4 × lb ∈ 0..3, packed
    /// `la*3 + lb` (la = 4 and lb ≥ 3 are degenerate branches the caller
    /// sentinels before the probe). 24 real bytes → a 32-byte slot, four
    /// per 128-byte line.
    pair: [u8; 12],
}

#[cfg(feature = "probe-cache-stats")]
static LM2_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "probe-cache-stats")]
static LM2_CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Print the LM2 front cache's aggregate hit/miss counters (all workers).
#[cfg(feature = "probe-cache-stats")]
pub fn lm2_cache_stats_report() {
    let h = LM2_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let m = LM2_CACHE_MISSES.load(std::sync::atomic::Ordering::Relaxed);
    if h + m > 0 {
        eprintln!(
            "    lm2 cache: {h} hits / {m} misses = {:.3}% hit",
            100.0 * h as f64 / (h + m) as f64
        );
    }
}

/// Consult-slack histogram: `need − h_cwd` at every LM2 consult, clamped to
/// 15. Slack ≥ 7 consults are provably unable to prune (branch gain is
/// capped at +6 by the excursion-splice bound), so this measures what a
/// slack gate could skip.
#[cfg(feature = "probe-cache-stats")]
static LM2_SLACK_HIST: [std::sync::atomic::AtomicU64; 16] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// Print the LM2 consult-slack histogram and reset it (per-threshold use).
#[cfg(feature = "probe-cache-stats")]
pub fn lm2_slack_report_reset() {
    let counts: Vec<u64> = LM2_SLACK_HIST
        .iter()
        .map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed))
        .collect();
    let tot: u64 = counts.iter().sum();
    if tot == 0 {
        return;
    }
    let gateable: u64 = counts[7..].iter().sum();
    eprintln!(
        "    lm2 consult slack: total {tot}; >=7 (gateable) {gateable} ({:.3}%); hist {:?}",
        100.0 * gateable as f64 / tot as f64,
        counts
    );
}

fn lm2_cache_bits() -> u32 {
    static BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| {
        std::env::var("FLAT_LM2_CACHE_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(19)
    })
}

impl Lm2Cache {
    fn new() -> Self {
        let bits = lm2_cache_bits();
        assert!((8..=24).contains(&bits), "implausible LM2 cache size");
        Lm2Cache {
            slots: vec![
                Lm2Slot {
                    tag: 0,
                    single: [0xFF; 4],
                    pair: [0xFF; 12]
                };
                1usize << bits
            ]
            .into_boxed_slice(),
            shift: 64 - bits,
        }
    }

    /// The three values a branch compose needs from `key`'s slot: two
    /// single-tracked lines (`0xFF`-sentinel indices ≥ 4 return `0xFF`) and
    /// one pair placement (sentinel index ≥ 25). Extracting inside the probe
    /// keeps the 29-byte slot payload out of the caller's stack — the
    /// by-value tuple return provably survived inlining (two q-register
    /// copy pairs per probe in the disassembly).
    #[inline(always)]
    fn get3(
        &mut self,
        mm: &super::cwd_lm::CwdLmMm,
        key: u64,
        s1: u8,
        s2: u8,
        pidx: u8,
    ) -> (u8, u8, u8) {
        let i = ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> self.shift) as usize;
        if self.slots[i].tag != key {
            let mut single = [0xFFu8; 4];
            let mut pair = [0xFFu8; 12];
            if let Some((s, p)) = mm.probe(key) {
                single.copy_from_slice(s);
                pair.copy_from_slice(p);
            }
            self.slots[i] = Lm2Slot {
                tag: key,
                single,
                pair,
            };
            #[cfg(feature = "probe-cache-stats")]
            LM2_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            #[cfg(feature = "probe-cache-stats")]
            LM2_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let sl = &self.slots[i];
        let a = if s1 < 4 { sl.single[s1 as usize] } else { 0xFF };
        let b = if s2 < 4 { sl.single[s2 as usize] } else { 0xFF };
        let p = if pidx < 12 {
            sl.pair[pidx as usize]
        } else {
            0xFF
        };
        (a, b, p)
    }
}

/// Seed depth-0 tracked lines for the LM2 tier.
fn seed_lm2(arena: &mut Arena, board: &State) {
    arena
        .lm2cache
        .get_or_insert_with(|| Box::new(Lm2Cache::new()));
    let pos = |t: u8| board.0.iter().position(|&x| x == t).unwrap();
    let (p20, p24, p15, p19, p23) = (pos(20), pos(24), pos(15), pos(19), pos(23));
    arena.lm2pos[0] = [
        (p20 / W_LM) as u8,
        (p24 % W_LM) as u8,
        (p15 / W_LM) as u8,
        (p19 / W_LM) as u8,
        (p19 % W_LM) as u8,
        (p23 % W_LM) as u8,
    ];
}

/// Price the four last-two-moves endgame branches for the child at depth
/// `d+1` and return `max(h_cwd, min(A, B, C, D))`.
///
/// Every optimal solution's final two moves pin the blank's suffix to one of
/// 14→19→24 (A: row pair 20+15), 18→19→24 (B: row 20 + col 19),
/// 18→23→24 (C: row 19 + col 24), 22→23→24 (D: col pair 24+23). Tile 19 is
/// type-3 in both axes, so B/C read the same single-tracked table as LM. A
/// tile past its crossing line (or a missing table entry) degrades that
/// constraint to the axis's baseline term, so a fully degenerate branch
/// equals `h_cwd`. Like the LM tier, inadmissible only at the goal board
/// itself (obligations presume ≥ 2 remaining moves), which the exhaust
/// regime never consults; the two pre-goal states are safe because their
/// in-line-4 tile degenerates one branch to `h_cwd` = d*.
#[inline]
fn lm2_child(
    arena: &mut Arena,
    mm: &super::cwd_lm::CwdLmMm,
    d: usize,
    tile_decoded: usize,
    h_cwd: u8,
) -> u8 {
    // Update the child's tracked lines: the moved tile lands on the parent's
    // blank cell.
    let mut lp = arena.lm2pos[d];
    let bcell = arena.hot[d].blank as usize;
    match tile_decoded {
        20 => lp[0] = (bcell / W_LM) as u8,
        24 => lp[1] = (bcell % W_LM) as u8,
        15 => lp[2] = (bcell / W_LM) as u8,
        19 => {
            lp[3] = (bcell / W_LM) as u8;
            lp[4] = (bcell % W_LM) as u8;
        }
        23 => lp[5] = (bcell % W_LM) as u8,
        _ => {}
    }
    arena.lm2pos[d + 1] = lp;
    debug_assert_eq!(
        {
            let b = arena.board[d + 1].0.decode();
            let pos = |t: u8| b.0.iter().position(|&x| x == t).unwrap();
            [
                (pos(20) / W_LM) as u8,
                (pos(24) % W_LM) as u8,
                (pos(15) / W_LM) as u8,
                (pos(19) / W_LM) as u8,
                (pos(19) % W_LM) as u8,
                (pos(23) % W_LM) as u8,
            ]
        },
        lp,
        "incremental lm2pos drifted from the board"
    );

    let f = &arena.hot[d + 1];
    let rslot = &arena.row[f.row_at as usize];
    let cslot = &arena.col[f.col_at as usize];
    let rterm = rslot.term();
    let cterm = cslot.term();
    let (rkey, ckey) = (rslot.key, cslot.key);
    let cache = arena
        .lm2cache
        .as_mut()
        .expect("seed_lm2 allocates the cache")
        .as_mut();
    let pri = if lp[0] < 4 && lp[2] < 3 {
        lp[0] * 3 + lp[2]
    } else {
        0xFF
    };
    let pci = if lp[1] < 4 && lp[5] < 3 {
        lp[1] * 3 + lp[5]
    } else {
        0xFF
    };
    let (r20, r19, pr) = cache.get3(mm, rkey, lp[0], lp[3], pri);
    let (c24, c19, pc) = cache.get3(mm, ckey, lp[1], lp[4], pci);
    let or_ = |v: u8, fb: u8| if v != 0xFF { v } else { fb };
    let r20f = or_(r20, rterm);
    let c24f = or_(c24, cterm);
    let ba = or_(pr, r20f).saturating_add(cterm);
    let bb = r20f.saturating_add(or_(c19, cterm));
    let bc = or_(r19, rterm).saturating_add(c24f);
    let bd = rterm.saturating_add(or_(pc, c24f));
    h_cwd.max(ba.min(bb).min(bc).min(bd))
}

/// k6 consult-stream locality (feature `k6-locality`, instrumented runs only).
///
/// Answers two questions from one budgeted run, before any positional map is
/// built: (1) does a value-memo cache in front of the tier pay — the k8 memo
/// lost at every size with an 89% hit rate, but k6's per-group abstraction is
/// far coarser, so its reuse may be in a different regime; (2) what footprint
/// would the positional layout's address stream actually touch (distinct
/// 128-byte lines and 16 KiB pages), which is the TLB/page risk of trading the
/// 88 MB ranked family for a 3.05 GB map.
#[cfg(feature = "k6-locality")]
pub mod k6_locality {
    use std::sync::Mutex;

    const SIZES: [u32; 5] = [14, 16, 18, 20, 22];
    const PAGES_PER_MAP: usize = 46_592; // 763 MB / 16 KiB
    const LINES_PER_MAP: usize = 5_963_000; // 763 MB / 128 B

    struct Sim {
        tags: Vec<Vec<u64>>,
        hits: [u64; SIZES.len()],
        probes: u64,
        same_line_as_prev: u64,
        prev_line: u64,
        pages: Vec<u64>,
        lines: Vec<u64>,
    }

    static SIM: Mutex<Option<Sim>> = Mutex::new(None);

    /// Record one probe: `group`, the view's blank cell, and the six-tile
    /// positional index the positional layout would address.
    pub fn record(group: usize, blank: u8, tiles_idx: u32) {
        let mut g = SIM.lock().unwrap();
        let s = g.get_or_insert_with(|| Sim {
            tags: SIZES.iter().map(|b| vec![u64::MAX; 1usize << b]).collect(),
            hits: [0; SIZES.len()],
            probes: 0,
            same_line_as_prev: 0,
            prev_line: u64::MAX,
            pages: vec![0u64; (4 * PAGES_PER_MAP).div_ceil(64)],
            lines: vec![0u64; (4 * LINES_PER_MAP).div_ceil(64)],
        });
        let key = ((group as u64) << 40) | ((blank as u64) << 32) | tiles_idx as u64;
        for (i, &bits) in SIZES.iter().enumerate() {
            let slot = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize;
            if s.tags[i][slot] == key {
                s.hits[i] += 1;
            } else {
                s.tags[i][slot] = key;
            }
        }
        // hypothetical positional address
        let bit = blank as u64 * super::super::k6pos::STRIDE_BITS + tiles_idx as u64;
        let byte = bit >> 3;
        let line = group as u64 * LINES_PER_MAP as u64 + byte / 128;
        let page = group as u64 * PAGES_PER_MAP as u64 + byte / 16384;
        if line == s.prev_line {
            s.same_line_as_prev += 1;
        }
        s.prev_line = line;
        s.lines[(line / 64) as usize] |= 1 << (line % 64);
        s.pages[(page / 64) as usize] |= 1 << (page % 64);
        s.probes += 1;
    }

    pub fn report() {
        let g = SIM.lock().unwrap();
        let Some(s) = g.as_ref() else { return };
        eprintln!("=== k6 consult-stream locality ({} probes) ===", s.probes);
        for (i, &bits) in SIZES.iter().enumerate() {
            eprintln!(
                "    memo cache 2^{bits:<2} ({:>5} MB): {:.3}% hit",
                (1usize << bits) * 16 / 1_048_576,
                100.0 * s.hits[i] as f64 / s.probes.max(1) as f64
            );
        }
        let lines: u64 = s.lines.iter().map(|w| w.count_ones() as u64).sum();
        let pages: u64 = s.pages.iter().map(|w| w.count_ones() as u64).sum();
        eprintln!(
            "    positional footprint: {lines} distinct 128 B lines ({:.1} MB), {pages} distinct 16 KiB pages ({:.1} MB); consecutive-probe same-line {:.2}%",
            lines as f64 * 128.0 / 1e6,
            pages as f64 * 16384.0 / 1e6,
            100.0 * s.same_line_as_prev as f64 / s.probes.max(1) as f64
        );
    }
}

// ------------------------------ lazy k6 tier ----------------------------------

/// The 6-6-6-6 zPDB family (`pdb24_{a..d}.zbin`, 88 MB total, RAM-resident):
/// the cheap cascade tier. Consulted cold at LM2-survivors — decode the child
/// board, 4 additive lookups per σ-view, max of the two views. Measured on the
/// unconditioned survivor sample: 42.0% standalone frontier bind, 9.37%
/// beyond LM2 (advantages up to +8).
pub struct K6Ctx {
    dbs: [crate::puzzle24::pdb::ZPatternDb; 4],
    /// `group_of[t]` = which of the four patterns holds tile `t` (1..=24).
    group_of: [u8; N_CELLS],
}

impl K6Ctx {
    /// Load `pdb24_{a,b,c,d}.zbin` from `dir`; verify the patterns are
    /// pairwise disjoint and cover tiles 1..=24.
    pub fn load_mmap(dir: &std::path::Path) -> Result<Self, String> {
        let names = [
            "pdb24_a.zbin",
            "pdb24_b.zbin",
            "pdb24_c.zbin",
            "pdb24_d.zbin",
        ];
        let mut dbs = Vec::with_capacity(4);
        for n in names {
            dbs.push(
                crate::puzzle24::pdb::ZPatternDb::load_mmap(&dir.join(n))
                    .map_err(|e| format!("{n}: {e:?}"))?,
            );
        }
        let mut cover = 0u32;
        for db in &dbs {
            if cover & db.pattern().0 != 0 {
                return Err("k6 patterns overlap".into());
            }
            cover |= db.pattern().0;
        }
        if cover != 0x01FF_FFFE {
            return Err("k6 patterns must cover tiles 1..=24".into());
        }
        let dbs: [crate::puzzle24::pdb::ZPatternDb; 4] = dbs
            .try_into()
            .map_err(|_| "expected four dbs".to_string())?;
        let mut group_of = [0u8; N_CELLS];
        for (i, db) in dbs.iter().enumerate() {
            for tile in db.pattern().iter() {
                group_of[tile as usize] = i as u8;
            }
        }
        Ok(K6Ctx { dbs, group_of })
    }

    /// `max` over σ-views of the four-group additive sum, cold lookups.
    /// Seed-time only ([`seed_k6v`] uses per-group lookups directly); kept
    /// for tests and any future cold-path caller.
    #[allow(dead_code)]
    #[inline]
    fn h(&self, board: &State) -> u8 {
        let rs = symmetry::reflect(board);
        let (mut s0, mut s1) = (0u16, 0u16);
        for db in &self.dbs {
            s0 += db.cold_lookup(board) as u16;
            s1 += db.cold_lookup(&rs) as u16;
        }
        s0.max(s1).min(255) as u8
    }
}

/// Which k6 backing the tier consults. The discriminant is constant for a
/// run, so the match below is perfectly predicted; keeping both lets the
/// ranked and positional paths be A/B'd against each other node-identically.
pub enum K6Tier {
    /// `pdb24_{a..d}.zbin` addressed by `ZpdbLayout::rank` (88 MB).
    Ranked(K6Ctx),
    /// `k6pos_{a..d}.bin` addressed by tile positions (2.9 GB), behind a
    /// front cache.
    Pos(super::k6pos::K6PosCtx),
}

/// Positional k6 state: the six-tile index per group per σ-view, plus the
/// carried absolute values. 40 bytes against the ranked slot's eight
/// `ProjectedState`s — the 16% of tier cost the profile attributed to
/// `memmove`.
#[derive(Clone, Copy, Default)]
struct K6PosSlot {
    idx: [[u32; 4]; 2],
    h: [[u8; 4]; 2],
}

/// Direct-mapped front cache for the positional maps, keyed on the exact
/// abstract state `(group, blank, six-tile index)` — which determines the
/// value, so a hit is exact and skips the map entirely.
///
/// Sized from the measured consult stream (feature `k6-locality`, 49.7 M
/// probes): 92.1% hit at 2^18 entries (4 MB), 95.5% at 2^20, 96.9% at 2^22.
/// Default 18 bits keeps the cache L2-resident, which is the difference from
/// the k8 value-memo that lost at every size despite an 89% hit rate.
/// `FLAT_K6_CACHE_BITS` overrides.
struct K6Cache {
    slots: Box<[K6CacheSlot]>,
    shift: u32,
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct K6CacheSlot {
    /// `group << 40 | blank << 32 | tiles_idx`; 0 = empty (a real key has a
    /// nonzero tiles index, since the six cells are distinct).
    tag: u64,
    val: u8,
}

#[cfg(feature = "probe-cache-stats")]
static K6_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "probe-cache-stats")]
static K6_CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Print the k6 front cache's aggregate hit/miss counters (all workers).
#[cfg(feature = "probe-cache-stats")]
pub fn k6_cache_stats_report() {
    let h = K6_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let m = K6_CACHE_MISSES.load(std::sync::atomic::Ordering::Relaxed);
    if h + m > 0 {
        eprintln!(
            "    k6 cache: {h} hits / {m} misses = {:.3}% hit",
            100.0 * h as f64 / (h + m) as f64
        );
    }
}

impl K6Cache {
    fn new() -> Self {
        let bits: u32 = std::env::var("FLAT_K6_CACHE_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(18);
        assert!((8..=24).contains(&bits), "implausible k6 cache size");
        K6Cache {
            slots: vec![K6CacheSlot { tag: 0, val: 0 }; 1usize << bits].into_boxed_slice(),
            shift: 64 - bits,
        }
    }

    #[inline(always)]
    fn get(
        &mut self,
        ctx: &super::k6pos::K6PosCtx,
        group: usize,
        tiles_idx: u32,
        blank: u8,
        parent_h: u8,
    ) -> u8 {
        let key = ((group as u64) << 40) | ((blank as u64) << 32) | tiles_idx as u64;
        let i = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        if self.slots[i].tag == key {
            #[cfg(feature = "probe-cache-stats")]
            K6_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return self.slots[i].val;
        }
        let v = ctx.probe(group, tiles_idx, blank, parent_h);
        self.slots[i] = K6CacheSlot { tag: key, val: v };
        #[cfg(feature = "probe-cache-stats")]
        K6_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    }
}

/// Seed depth-0 positional k6 state (cold — once per subtree).
fn seed_k6pos(arena: &mut Arena, board: &State, ctx: &super::k6pos::K6PosCtx) -> u8 {
    arena
        .k6cache
        .get_or_insert_with(|| Box::new(K6Cache::new()));
    let (idx, h, h0) = ctx.seed(board);
    arena.k6pos[0] = K6PosSlot { idx, h };
    h0
}

/// Advance the positional k6 state across the move and return `h_k6`.
/// One multiply-add per σ-view updates the moved tile's group index; the
/// blank digit is applied inside the probe.
#[inline]
fn k6pos_child(
    arena: &mut Arena,
    ctx: &super::k6pos::K6PosCtx,
    d: usize,
    geom: Geom,
    tile: usize,
) -> u8 {
    let b = arena.hot[d].blank as i64; // pre-move blank = the tile's destination
    let n = geom.from as i64; // the tile's cell = the new blank
    let tile = TILE_OF_CODE[tile] as usize;
    let (lo, hi) = arena.k6pos.split_at_mut(d + 1);
    let parent = &lo[d];
    let child = &mut hi[0];
    *child = *parent;

    let gi = ctx.group_of[tile] as usize;
    child.idx[0][gi] = (child.idx[0][gi] as i64 + (b - n) * ctx.coeff[tile] as i64) as u32;
    let rt = symmetry::TAU[tile] as usize;
    let gj = ctx.group_of[rt] as usize;
    let (rb, rn) = (
        symmetry::SIGMA[b as usize] as i64,
        symmetry::SIGMA[n as usize] as i64,
    );
    child.idx[1][gj] = (child.idx[1][gj] as i64 + (rb - rn) * ctx.coeff[rt] as i64) as u32;

    let (i0, i1) = (child.idx[0][gi], child.idx[1][gj]);
    let (p0, p1) = (parent.h[0][gi], parent.h[1][gj]);
    let cache = arena
        .k6cache
        .as_mut()
        .expect("seed_k6pos allocates the cache")
        .as_mut();
    let v0 = cache.get(ctx, gi, i0, n as u8, p0);
    let v1 = cache.get(ctx, gj, i1, rn as u8, p1);
    let child = &mut arena.k6pos[d + 1];
    child.h[0][gi] = v0;
    child.h[1][gj] = v1;

    let s0 = child.h[0].iter().map(|&x| x as u16).sum::<u16>();
    let s1 = child.h[1].iter().map(|&x| x as u16).sum::<u16>();
    let out = s0.max(s1).min(255) as u8;
    debug_assert_eq!(
        out,
        ctx.cold_h(&arena.board[d + 1].0.decode()),
        "incremental positional k6 drifted from cold lookups"
    );
    out
}

/// k6 tier per-depth state: the four pattern projections in both σ-views and
/// their table values. Written only on consults, same discipline as [`K8Slot`].
#[derive(Clone, Copy)]
struct K6Slot {
    views: [[crate::puzzle24::pdb::ProjectedState; 4]; 2],
    h: [[u8; 4]; 2],
}

/// Seed depth-0 k6 state from the board (cold lookups — seeds are rare).
fn seed_k6v(arena: &mut Arena, board: &State, ctx: &K6Ctx) -> u8 {
    let rs = symmetry::reflect(board);
    let slot = &mut arena.k6slots[0];
    let (mut s0, mut s1) = (0u16, 0u16);
    for i in 0..4 {
        let db = &ctx.dbs[i];
        slot.views[0][i] = crate::puzzle24::pdb::ProjectedState::from_state(board, db.pattern());
        slot.views[1][i] = crate::puzzle24::pdb::ProjectedState::from_state(&rs, db.pattern());
        slot.h[0][i] = db.cold_lookup(board);
        slot.h[1][i] = db.cold_lookup(&rs);
        s0 += slot.h[0][i] as u16;
        s1 += slot.h[1][i] as u16;
    }
    s0.max(s1).min(255) as u8
}

/// The positional layout's six-tile index for a projected view (instrument
/// only: the ranked path never needs it).
#[cfg(feature = "k6-locality")]
fn k6_tiles_index(
    v: &crate::puzzle24::pdb::ProjectedState,
    pattern: crate::puzzle24::pdb::Pattern,
) -> u32 {
    let mut idx = 0u64;
    for (j, t) in pattern.iter().enumerate() {
        idx += v.pos_of(t) as u64 * [9_765_625u64, 390_625, 15_625, 625, 25, 1][j];
    }
    idx as u32
}

/// Advance the k6 state across move `m` from depth `d` to `d+1` and return the
/// child's `h_k6 = max(Σ normal, Σ reflected)`. Exact mirror of [`k8_child`]:
/// one group is cost-1 per view, the other three are untouched.
#[inline]
fn k6_child(arena: &mut Arena, ctx: &K6Ctx, d: usize, geom: Geom, tile: usize) -> u8 {
    let (lo, hi) = arena.k6slots.split_at_mut(d + 1);
    let parent = &lo[d];
    let child = &mut hi[0];
    *child = *parent;
    let b = arena.hot[d].blank as usize;
    let n = geom.from as usize;
    let tile = TILE_OF_CODE[tile] as usize;

    let gi = ctx.group_of[tile] as usize;
    {
        let db = &ctx.dbs[gi];
        let v = &mut child.views[0][gi];
        v.set_blank_pos(b as u8);
        let cost = v.apply_in_place_at(b, n);
        debug_assert_eq!(cost, 1, "moved tile must be in its own group's pattern");
        let idx = db.layout().rank(v, db.pattern());
        #[cfg(feature = "k6-locality")]
        k6_locality::record(gi, v.blank_pos(), k6_tiles_index(v, db.pattern()));
        child.h[0][gi] = db.diff_lookup(idx, parent.h[0][gi]);
    }

    let rt = symmetry::TAU[tile] as usize;
    let gj = ctx.group_of[rt] as usize;
    {
        let db = &ctx.dbs[gj];
        let rb = symmetry::SIGMA[b] as usize;
        let rn = symmetry::SIGMA[n] as usize;
        let v = &mut child.views[1][gj];
        v.set_blank_pos(rb as u8);
        let cost = v.apply_in_place_at(rb, rn);
        debug_assert_eq!(cost, 1, "σ-image tile must be in its own group's pattern");
        let idx = db.layout().rank(v, db.pattern());
        #[cfg(feature = "k6-locality")]
        k6_locality::record(gj, v.blank_pos(), k6_tiles_index(v, db.pattern()));
        child.h[1][gj] = db.diff_lookup(idx, parent.h[1][gj]);
    }

    let s0 = child.h[0].iter().map(|&x| x as u16).sum::<u16>();
    let s1 = child.h[1].iter().map(|&x| x as u16).sum::<u16>();
    let out = s0.max(s1).min(255) as u8;
    debug_assert_eq!(
        out,
        ctx.h(&arena.board[d + 1].0.decode()),
        "incremental k6 drifted from cold lookups"
    );
    out
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

/// [`flat_bounded_telemetry`] with the Last-Move tier (`cwd-lm`):
/// `h = max(cWD, min(branch20, branch24))`, priced lazily at cWD-survivors.
#[allow(clippy::too_many_arguments)]
pub fn flat_bounded_lm_telemetry<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    lm: &super::cwd_lm::CwdLmMm,
    orbit_split: bool,
    max_bound: u8,
    max_nodes: u64,
    on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    flat_bounded_inner_k8(
        start,
        cwd,
        dfa,
        None,
        Some(lm),
        None,
        None,
        orbit_split,
        max_bound,
        max_nodes,
        on_iter,
    )
}

/// [`flat_bounded_telemetry`] with the last-two-moves tier (`cwd-lm2`):
/// `h = max(cWD, min(A, B, C, D))` over the four endgame branches, priced
/// lazily at cWD-survivors. Needs BOTH tables — the pair table serves
/// branches A/D, the single table serves B/C and the pair fallbacks.
#[allow(clippy::too_many_arguments)]
pub fn flat_bounded_lm2_telemetry<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    lm2: &super::cwd_lm::CwdLmMm,
    k6: Option<&K6Tier>,
    orbit_split: bool,
    max_bound: u8,
    max_nodes: u64,
    on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    flat_bounded_inner_k8(
        start,
        cwd,
        dfa,
        None,
        None,
        Some(lm2),
        k6,
        orbit_split,
        max_bound,
        max_nodes,
        on_iter,
    )
}

/// [`flat_bounded_telemetry`] with the lazy k8 tier: `max(cWD, k8)` consulted at
/// cWD-survivors, node-identical to the recorded 2-tier trees (§8j:
/// 269,180,930 at exhaust-144; 8,808,311,484 at exhaust-146).
#[allow(clippy::too_many_arguments)]
pub fn flat_bounded_k8_telemetry<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    k8: &K8Ctx,
    orbit_split: bool,
    max_bound: u8,
    max_nodes: u64,
    on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    flat_bounded_inner_k8(
        start,
        cwd,
        dfa,
        Some(k8),
        None,
        None,
        None,
        orbit_split,
        max_bound,
        max_nodes,
        on_iter,
    )
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
    flat_bounded_inner_k8(
        start,
        cwd,
        dfa,
        None,
        None,
        None,
        None,
        orbit_split,
        max_bound,
        max_nodes,
        on_iter,
    )
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
    flat_bounded_inner_k8(
        start,
        cwd,
        dfa,
        None,
        None,
        None,
        None,
        orbit_split,
        max_bound,
        u64::MAX,
        on_iter,
    )
}

#[allow(clippy::too_many_arguments)]
fn flat_bounded_inner_k8<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    k8: Option<&K8Ctx>,
    lm: Option<&super::cwd_lm::CwdLmMm>,
    lm2: Option<&super::cwd_lm::CwdLmMm>,
    k6t: Option<&K6Tier>,
    orbit_split: bool,
    max_bound: u8,
    budget: u64,
    mut on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    let merged = cwd
        .backing()
        .expect("flat_bounded needs the merged cWD table (data/cwd_mm.bin or cwd_single.bin)");
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
    let mut h0 = seed_root(&mut arena, start, cwd, merged, dfa, orbit_split);
    if let Some(ctx) = k8 {
        h0 = h0.max(seed_k8(&mut arena, start, ctx));
    }
    if lm2.is_some() {
        seed_lm2(&mut arena, start);
    } else if lm.is_some() {
        seed_lm(&mut arena, start);
    }
    if let Some(c) = k6t {
        h0 = h0.max(match c {
            K6Tier::Ranked(c) => seed_k6v(&mut arena, start, c),
            K6Tier::Pos(c) => seed_k6pos(&mut arena, start, c),
        });
    }
    let mut bound = h0;

    loop {
        if bound > max_bound {
            return (BoundedOutcome::ProvedAtLeast(bound), stats);
        }
        stats.iterations += 1;
        let iter_start = std::time::Instant::now();
        // Re-seed: the previous iteration consumed the root's candidate set.
        seed_root(&mut arena, start, cwd, merged, dfa, orbit_split);
        let step = match (
            budget == u64::MAX,
            k8.is_some(),
            lm2.is_some(),
            lm.is_some(),
            k6t.is_some(),
        ) {
            (true, false, false, false, _) => run_iteration::<false, false, false, false, false>(
                &mut arena, &mut cache, cwd, merged, dfa, None, None, None, None, bound,
                &mut stats, budget,
            ),
            (false, false, false, false, _) => run_iteration::<true, false, false, false, false>(
                &mut arena, &mut cache, cwd, merged, dfa, None, None, None, None, bound,
                &mut stats, budget,
            ),
            (true, true, _, _, _) => run_iteration::<false, true, false, false, false>(
                &mut arena, &mut cache, cwd, merged, dfa, k8, None, None, None, bound, &mut stats,
                budget,
            ),
            (false, true, _, _, _) => run_iteration::<true, true, false, false, false>(
                &mut arena, &mut cache, cwd, merged, dfa, k8, None, None, None, bound, &mut stats,
                budget,
            ),
            (true, false, true, _, false) => run_iteration::<false, false, false, true, false>(
                &mut arena, &mut cache, cwd, merged, dfa, None, lm, lm2, None, bound, &mut stats,
                budget,
            ),
            (false, false, true, _, false) => run_iteration::<true, false, false, true, false>(
                &mut arena, &mut cache, cwd, merged, dfa, None, lm, lm2, None, bound, &mut stats,
                budget,
            ),
            (true, false, true, _, true) => run_iteration::<false, false, false, true, true>(
                &mut arena, &mut cache, cwd, merged, dfa, None, lm, lm2, k6t, bound, &mut stats,
                budget,
            ),
            (false, false, true, _, true) => run_iteration::<true, false, false, true, true>(
                &mut arena, &mut cache, cwd, merged, dfa, None, lm, lm2, k6t, bound, &mut stats,
                budget,
            ),
            (true, false, false, true, _) => run_iteration::<false, false, true, false, false>(
                &mut arena, &mut cache, cwd, merged, dfa, None, lm, None, None, bound, &mut stats,
                budget,
            ),
            (false, false, false, true, _) => run_iteration::<true, false, true, false, false>(
                &mut arena, &mut cache, cwd, merged, dfa, None, lm, None, None, bound, &mut stats,
                budget,
            ),
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

// ============================ parallel driver =================================

/// Grow the frontier to about this many subtree roots before going parallel.
///
/// Far more than the core count, because subtree sizes vary by orders of
/// magnitude and rayon's work-stealing needs slack to balance them. The deleted
/// recursive driver used the same figure for the same reason.
#[cfg(feature = "parallel")]
const SPLIT_TARGET: usize = 4096;

/// Probe-cache size for a *worker*, in bits. Same as the sequential
/// [`CACHE_BITS`], and that is not a coincidence — see below.
///
/// This was 16 bits (2.1 MB) on the theory that per-thread caches must share an
/// L2: 4 P-cores share 16 MB here, so 8 threads at 18 bits want 67 MB of a 32 MB
/// total. **Measured at exhaust-146, W=8, that reasoning is wrong** — 18 bits wins
/// by 5.7% over 16 despite overflowing L2 several times over:
///
/// | bits | per thread | hit rate | misses  | ns/node | span    |
/// |------|------------|----------|---------|---------|---------|
/// | 16   | 2.1 MB     | 96.026%  | 706.1 M | 43.07   | 96.77 s |
/// | 17   | 4.2 MB     | 97.617%  | 423.4 M | 42.14   | 94.73 s |
/// | 18   | 8.4 MB     | 98.592%  | 250.1 M | 40.96   | 91.52 s |
/// | 19   | 16.8 MB    | 99.265%  | 130.6 M | 40.20   | 91.96 s |
///
/// What the cache buys is not L2 residency but *not probing the 4.4 GB merged
/// table*: a miss is a SwissTable lookup that TLB-misses into multi-GB territory,
/// which costs far more than an L2 miss on the cache itself. So the figure to
/// minimise is misses, and the L2 budget is not the binding constraint.
///
/// 18 also happens to be where eight per-worker caches together cover the same
/// ~250 M misses one sequential 256 K cache does (249.7 M measured), which drops
/// the parallel driver's per-node contention from +13.3% to +7.7% over sequential.
/// 19 bits lowers ns/node further but ties on wall clock for 2x the memory, so 18
/// is the knee.
///
/// Re-measure at exhaust-146 or deeper if the thread count or machine changes;
/// exhaust-144 cannot see this (there the same sweep spans only 1.2%, because
/// misses at that depth are 4.2x rarer per node and land in cache rather than
/// DRAM).
#[cfg(feature = "parallel")]
const WORKER_CACHE_BITS: u32 = 18;

/// Both tuning constants, overridable from the environment so that a sweep costs
/// runs rather than rebuilds. Gated behind the profile feature, so the default
/// build reads no environment and keeps the constants.
#[cfg(feature = "parallel-profile")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "parallel-profile")]
fn split_target() -> usize {
    env_usize("FLAT_SPLIT_TARGET", SPLIT_TARGET)
}

#[cfg(all(feature = "parallel", not(feature = "parallel-profile")))]
fn split_target() -> usize {
    SPLIT_TARGET
}

#[cfg(feature = "parallel-profile")]
fn worker_cache_bits() -> u32 {
    env_usize("FLAT_WORKER_CACHE_BITS", WORKER_CACHE_BITS as usize) as u32
}

#[cfg(all(feature = "parallel", not(feature = "parallel-profile")))]
fn worker_cache_bits() -> u32 {
    WORKER_CACHE_BITS
}

/// One subtree-root work unit: enough to reseed a worker's arena, plus the
/// move-prefix that reaches it so a worker finding the goal can return a full
/// path from `start`.
#[cfg(feature = "parallel")]
struct Unit {
    board: State,
    /// Depth of this node below the true root. A worker searches with
    /// `bound - g`, which is why `run_iteration` needs no depth offset.
    g: u8,
    /// Move-DFA state, and the move that reached this node. Both are *carried*,
    /// never recomputed: re-seeding the DFA here would forget the prefix and
    /// admit sequences the sequential engine prunes.
    dfa: u32,
    last: Move,
    prefix: Vec<Move>,
    h: u8,
}

#[cfg(feature = "parallel")]
thread_local! {
    /// One arena and one probe cache per **rayon worker thread**, reused across every
    /// unit that thread draws and across thresholds.
    ///
    /// Building these inside the `map` closure — i.e. once per work unit — is what
    /// cost the parallel driver 20% per node against the sequential engine at equal
    /// thread count. The probe cache is direct-mapped and warms over millions of
    /// probes, but a unit averages only ~100 K nodes, so a per-unit cache never
    /// warms: measured at exhaust-144 the hit rate was 95.15% against the sequential
    /// engine's 99.665%, a 14.5x increase in misses into the 4.4 GB table. Those are
    /// *compulsory* misses, so enlarging the cache cannot help — going 64 K -> 1 M
    /// moved the hit rate 0.47 pp while re-zeroing pushed setup from 1.3% to 15.2% of
    /// worker time.
    ///
    /// A thread-local rather than rayon's `map_init`, because `map_init` re-runs its
    /// initialiser once per producer-split leaf, which is not once per thread.
    ///
    /// Reuse is sound: [`seed_subtree`] rewrites depth 0 in full and
    /// [`run_iteration`] only reads arena depths it has written, which is already why
    /// the sequential engine can reuse one arena across thresholds. The cache is
    /// transparent by construction — it returns exactly what `merged.get` returns.
    static WORKER: std::cell::RefCell<Option<(Arena, ProbeCache)>> =
        const { std::cell::RefCell::new(None) };
}

/// [`flat_bounded`] with each threshold iteration parallelised by
/// **tree-splitting + work-stealing**.
///
/// A cheap sequential expansion grows a frontier of ~[`SPLIT_TARGET`] subtree
/// roots, then rayon runs the *unmodified* sequential loop on each subtree and the
/// results are reduced: SUM the stats, MIN the next threshold, first `Solved`
/// wins.
///
/// # Why this is sound
///
/// Exhausting a threshold in parallel still proves `dist ≥ next`, because MIN is
/// order-independent and every node with `f ≤ bound` is visited by exactly one
/// worker. IDA\* stays memory-light — a worker holds one arena (~21 KB) plus its
/// probe cache — so this does not blow up the way parallel A\* does.
///
/// # Node identity
///
/// The total node count must equal the sequential engine's exactly, which
/// constrains the accounting. Each node is counted **once**: on pop when it is
/// expanded during the split, immediately when it is built and found over-bound
/// (it is never popped), and by its own worker's root-count when it survives into
/// the frontier. The split reuses [`child_lb`] and [`build_child`] through a
/// scratch arena rather than reimplementing them, so its pruning decisions cannot
/// drift from the sequential engine's.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn flat_bounded_parallel<F>(
    start: &State,
    cwd: &Cwd,
    dfa: &MoveDfa,
    k8: Option<&K8Ctx>,
    lm: Option<&super::cwd_lm::CwdLmMm>,
    lm2: Option<&super::cwd_lm::CwdLmMm>,
    k6t: Option<&K6Tier>,
    orbit_split: bool,
    max_bound: u8,
    mut on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    use rayon::prelude::*;
    use std::collections::VecDeque;

    let merged = cwd
        .backing()
        .expect("flat_bounded needs the merged cWD table (data/cwd_mm.bin or cwd_single.bin)");
    let lut = cwd.demand_lut();
    let neighbor_prune = cwd.neighbor_prune_enabled();
    debug_assert!(
        !orbit_split || symmetry::is_symmetric(start),
        "root orbit-split is only sound on a σ-fixed board"
    );

    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (BoundedOutcome::Solved(Vec::new()), stats);
    }

    // Scratch arena/cache for the sequential split. Small cache: the split visits
    // only a few thousand nodes, so a big one would be cold anyway.
    let mut split_arena = Arena::new();
    let mut split_cache = ProbeCache::with_bits(12);

    let mut h0 = seed_root(&mut split_arena, start, cwd, merged, dfa, orbit_split);
    if let Some(ctx) = k8 {
        h0 = h0.max(seed_k8(&mut split_arena, start, ctx));
    }
    if lm.is_some() {
        seed_lm(&mut split_arena, start);
    }
    let mut bound = h0;

    loop {
        if bound > max_bound {
            return (BoundedOutcome::ProvedAtLeast(bound), stats);
        }
        stats.iterations += 1;
        let iter_start = std::time::Instant::now();

        // ---- split: sequential expansion into a frontier of subtree roots ----
        let mut frontier: VecDeque<Unit> = VecDeque::new();
        frontier.push_back(Unit {
            board: *start,
            g: 0,
            dfa: <MoveDfa as super::move_dfa::MovePruner>::root_state(dfa, start.blank_pos()),
            last: Move::Up, // unused at the root; `seed_root` supplies its cand set
            prefix: Vec::new(),
            h: h0,
        });
        let mut min_f = u8::MAX;
        let mut found: Option<Vec<Move>> = None;

        while frontier.len() < split_target() {
            let Some(u) = frontier.pop_front() else { break };
            stats.nodes += 1; // counted here because it is expanded, not searched
            if u.h > bound {
                if u.h < min_f {
                    min_f = u.h;
                }
                continue;
            }
            if u.h == 0 {
                found = Some(u.prefix);
                break;
            }
            // Reseed the scratch arena at this node. The root keeps `seed_root`'s
            // candidate set, which carries the σ-orbit filter.
            if u.g == 0 {
                seed_root(&mut split_arena, &u.board, cwd, merged, dfa, orbit_split);
            } else {
                seed_subtree(&mut split_arena, &u.board, u.dfa, u.last, merged, dfa);
            }
            if let Some(ctx) = k8 {
                seed_k8(&mut split_arena, &u.board, ctx);
            }
            if lm2.is_some() {
                seed_lm2(&mut split_arena, &u.board);
            } else if lm.is_some() {
                seed_lm(&mut split_arena, &u.board);
            }
            if let Some(c) = k6t {
                match c {
                    K6Tier::Ranked(c) => seed_k6v(&mut split_arena, &u.board, c),
                    K6Tier::Pos(c) => seed_k6pos(&mut split_arena, &u.board, c),
                };
            }
            let mut cand = split_arena.hot[0].cand;
            let g_next = u.g + 1;
            while !cand.is_empty() {
                let m = pop_lowest(&mut cand);
                let geom = GEOM[split_arena.hot[0].blank as usize][m as usize];
                let tile = split_arena.board[0].0 .0[geom.from as usize] as usize;
                if neighbor_prune {
                    if let Some(lb) = child_lb(&split_arena, 0, m, geom, tile) {
                        let f = g_next.saturating_add(lb);
                        if f > bound {
                            if f < min_f {
                                min_f = f;
                            }
                            continue; // pre-pruned: not counted, as sequentially
                        }
                    }
                }
                build_child(
                    &mut split_arena,
                    &mut split_cache,
                    0,
                    m,
                    geom,
                    tile,
                    merged,
                    lut,
                    dfa,
                );
                let h = split_arena.hot[1].h;
                let f = g_next.saturating_add(h);
                if f > bound {
                    stats.nodes += 1; // over-bound: counted, never popped
                    if f < min_f {
                        min_f = f;
                    }
                    continue;
                }
                if let Some(lm2t) = lm2 {
                    let t = TILE_OF_CODE[tile] as usize;
                    let heff = lm2_child(&mut split_arena, lm2t, 0, t, h);
                    if heff > h {
                        let f2 = g_next.saturating_add(heff);
                        if f2 > bound {
                            stats.nodes += 1; // counted, same as the cWD over-bound arm
                            if f2 < min_f {
                                min_f = f2;
                            }
                            continue;
                        }
                    }
                }
                if let Some(k6c) = k6t {
                    let hk6 = match k6c {
                        K6Tier::Ranked(c) => k6_child(&mut split_arena, c, 0, geom, tile),
                        K6Tier::Pos(c) => k6pos_child(&mut split_arena, c, 0, geom, tile),
                    };
                    if hk6 > h {
                        let f2 = g_next.saturating_add(hk6);
                        if f2 > bound {
                            stats.nodes += 1; // counted, same as the cWD over-bound arm
                            if f2 < min_f {
                                min_f = f2;
                            }
                            continue;
                        }
                    }
                } else if let Some(lmt) = lm {
                    let t = TILE_OF_CODE[tile] as usize;
                    let heff = lm_child(&mut split_arena, lmt, 0, t, h);
                    if heff > h {
                        let f2 = g_next.saturating_add(heff);
                        if f2 > bound {
                            stats.nodes += 1; // counted, same as the cWD over-bound arm
                            if f2 < min_f {
                                min_f = f2;
                            }
                            continue;
                        }
                    }
                }
                // Lazy k8 consult at cWD-survivors — must mirror run_iteration
                // exactly or the split's tree diverges from the sequential one.
                if let Some(ctx) = k8 {
                    let hk8 = k8_child(&mut split_arena, ctx, 0, geom, tile);
                    if hk8 > h {
                        let f2 = g_next.saturating_add(hk8);
                        if f2 > bound {
                            stats.nodes += 1; // counted, same as the cWD over-bound arm
                            if f2 < min_f {
                                min_f = f2;
                            }
                            continue;
                        }
                    }
                }
                let mut prefix = u.prefix.clone();
                prefix.push(m);
                frontier.push_back(Unit {
                    board: split_arena.board[1].0.decode(),
                    g: g_next,
                    dfa: split_arena.hot[1].dfa,
                    last: m,
                    prefix,
                    h,
                });
            }
        }

        if let Some(path) = found {
            return (BoundedOutcome::Solved(path), stats);
        }
        if frontier.is_empty() {
            // The whole threshold fitted inside the split.
            if min_f == u8::MAX {
                return (BoundedOutcome::Unsolvable, stats);
            }
            on_iter(bound, &stats, iter_start.elapsed());
            bound = min_f;
            continue;
        }

        // ---- parallel: the sequential loop on each subtree, then reduce ----
        enum Wr {
            Found(Vec<Move>),
            Bound(u8),
        }
        let units: Vec<Unit> = frontier.into_iter().collect();
        let _split_span = iter_start.elapsed();
        let _par_start = std::time::Instant::now();
        let results: Vec<(Wr, SearchStats, UnitProf)> = units
            .par_iter()
            .map(|u| {
                let unit_start = std::time::Instant::now();
                WORKER.with(|slot| {
                    let mut slot = slot.borrow_mut();
                    let (arena, cache) = slot.get_or_insert_with(|| {
                        (Arena::new(), ProbeCache::with_bits(worker_cache_bits()))
                    });
                    #[cfg(feature = "probe-cache-stats")]
                    let c0 = cache.stats();
                    let mut st = SearchStats::default();
                    seed_subtree(arena, &u.board, u.dfa, u.last, merged, dfa);
                    if let Some(ctx) = k8 {
                        seed_k8(arena, &u.board, ctx);
                    }
                    if lm2.is_some() {
                        seed_lm2(arena, &u.board);
                    } else if lm.is_some() {
                        seed_lm(arena, &u.board);
                    }
                    if let Some(c) = k6t {
                        match c {
                            K6Tier::Ranked(c) => seed_k6v(arena, &u.board, c),
                            K6Tier::Pos(c) => seed_k6pos(arena, &u.board, c),
                        };
                    }
                    let _setup_ns = unit_start.elapsed().as_nanos() as u64;
                    // f = g + d + h <= bound  <=>  d + h <= bound - g, so the
                    // worker searches its own subtree with a reduced threshold and
                    // needs no notion of the depth it sits at.
                    let reduced = bound - u.g;
                    let step = if k8.is_some() {
                        run_iteration::<false, true, false, false, false>(
                            arena,
                            cache,
                            cwd,
                            merged,
                            dfa,
                            k8,
                            None,
                            None,
                            None,
                            reduced,
                            &mut st,
                            u64::MAX,
                        )
                    } else if lm2.is_some() && k6t.is_some() {
                        run_iteration::<false, false, false, true, true>(
                            arena,
                            cache,
                            cwd,
                            merged,
                            dfa,
                            None,
                            lm,
                            lm2,
                            k6t,
                            reduced,
                            &mut st,
                            u64::MAX,
                        )
                    } else if lm2.is_some() {
                        run_iteration::<false, false, false, true, false>(
                            arena,
                            cache,
                            cwd,
                            merged,
                            dfa,
                            None,
                            lm,
                            lm2,
                            None,
                            reduced,
                            &mut st,
                            u64::MAX,
                        )
                    } else if lm.is_some() {
                        run_iteration::<false, false, true, false, false>(
                            arena,
                            cache,
                            cwd,
                            merged,
                            dfa,
                            None,
                            lm,
                            None,
                            None,
                            reduced,
                            &mut st,
                            u64::MAX,
                        )
                    } else {
                        run_iteration::<false, false, false, false, false>(
                            arena,
                            cache,
                            cwd,
                            merged,
                            dfa,
                            None,
                            None,
                            None,
                            None,
                            reduced,
                            &mut st,
                            u64::MAX,
                        )
                    };
                    let wr = match step {
                        Step::Found(mut path) => {
                            let mut full = u.prefix.clone();
                            full.append(&mut path);
                            Wr::Found(full)
                        }
                        // Back into absolute terms.
                        Step::Exhausted(n) => Wr::Bound(n.saturating_add(u.g)),
                        Step::BudgetOut => unreachable!("the parallel driver sets no budget"),
                    };
                    (
                        wr,
                        st,
                        UnitProf {
                            ns: unit_start.elapsed().as_nanos() as u64,
                            setup_ns: _setup_ns,
                            thread: rayon::current_thread_index().unwrap_or(usize::MAX),
                            // Delta, not cumulative: the cache now outlives the unit.
                            #[cfg(feature = "probe-cache-stats")]
                            cache: {
                                let c1 = cache.stats();
                                (c1.0 - c0.0, c1.1 - c0.1)
                            },
                        },
                    )
                })
            })
            .collect();
        let _par_span = _par_start.elapsed();

        let mut found: Option<Vec<Move>> = None;
        let mut _prof: Vec<(UnitProf, u64)> = Vec::with_capacity(results.len());
        for (wr, st, p) in results {
            _prof.push((p, st.nodes));
            stats.nodes += st.nodes;
            match wr {
                Wr::Found(p) => {
                    if found.is_none() {
                        found = Some(p);
                    }
                }
                Wr::Bound(n) => {
                    if n < min_f {
                        min_f = n;
                    }
                }
            }
        }
        if let Some(p) = found {
            return (BoundedOutcome::Solved(p), stats);
        }
        if min_f == u8::MAX {
            return (BoundedOutcome::Unsolvable, stats);
        }
        #[cfg(feature = "parallel-profile")]
        report_parallel_profile(bound, &_prof, _split_span, _par_span);
        on_iter(bound, &stats, iter_start.elapsed());
        bound = min_f;
    }
}

/// What one work unit cost, and which rayon thread paid it.
///
/// `thread` exists because the busy fraction is **blind to core heterogeneity**:
/// this machine has 8 P-cores and 4 E-cores, rayon asks the OS for threads
/// without pinning, and a thread parked on an E-core still reads as busy while
/// its nodes cost 2-3x more. Per-thread ns/node is what exposes that.
#[cfg(feature = "parallel")]
#[cfg_attr(not(feature = "parallel-profile"), allow(dead_code))]
struct UnitProf {
    ns: u64,
    /// Arena + probe-cache allocation and [`seed_subtree`], i.e. everything
    /// before the search loop starts. Charged to worker busy time, so it inflates
    /// `c_p` without touching the busy fraction.
    setup_ns: u64,
    thread: usize,
    /// Probe-cache hits/misses for this unit's own cache. The cache is built
    /// inside the closure, so these expose whether it ever warms up.
    #[cfg(feature = "probe-cache-stats")]
    cache: (u64, u64),
}

/// Makespan of a greedy longest-processing-time-first schedule of `desc` (unit
/// times, **already sorted descending**) onto `w` machines.
///
/// This is the reference point that separates the two halves of the imbalance
/// question. `busy / w` is the makespan a perfectly divisible workload would
/// reach; LPT is the best a *good* scheduler can do given that a unit is
/// indivisible — a worker that draws a huge subtree runs it to completion. So
/// `LPT - busy/w` is the cost of the split's granularity (fix: more units, or a
/// second split level) and `actual - LPT` is scheduling loss (steal latency,
/// pool spin-up, the serial reduce).
///
/// LPT is a 4/3-approximation, so it is an upper bound on the optimal makespan,
/// not the optimum itself; with thousands of units the gap is negligible.
#[cfg(feature = "parallel-profile")]
fn lpt_makespan(desc: &[u64], w: usize) -> u64 {
    let mut load = vec![0u64; w.max(1)];
    for &t in desc {
        // w <= 16 here, so a linear scan for the least-loaded machine is fine.
        let i = load
            .iter()
            .enumerate()
            .min_by_key(|&(_, l)| *l)
            .map(|(i, _)| i)
            .unwrap();
        load[i] += t;
    }
    load.into_iter().max().unwrap_or(0)
}

/// Decompose one threshold's parallel efficiency and print it to stderr.
///
/// `prof` is one `(wall_ns, nodes)` pair per work unit. The identity being
/// instrumented is
///
/// ```text
///   speedup / W  =  B  x  (c_1 / c_p)
/// ```
///
/// with `B` the busy fraction (`busy / (W * span)`) and `c_p` the worker's
/// ns/node. The two factors have unrelated fixes — `B` is the split policy and
/// the scheduler, `c_p` is memory contention, probe-cache size and clock — so a
/// single speedup number cannot tell you which to attack.
#[cfg(feature = "parallel-profile")]
fn report_parallel_profile(
    bound: u8,
    prof: &[(UnitProf, u64)],
    split: std::time::Duration,
    span: std::time::Duration,
) {
    let w = rayon::current_num_threads();
    let busy: u64 = prof.iter().map(|p| p.0.ns).sum();
    let setup: u64 = prof.iter().map(|p| p.0.setup_ns).sum();
    let nodes: u64 = prof.iter().map(|p| p.1).sum();
    let span_ns = span.as_nanos().max(1) as u64;
    let secs = |ns: u64| ns as f64 / 1e9;

    let mut desc: Vec<u64> = prof.iter().map(|p| p.0.ns).collect();
    desc.sort_unstable_by(|a, b| b.cmp(a));
    let n = desc.len();
    let max_unit = desc[0];
    let top_w: u64 = desc.iter().take(w).sum();

    let div_bound = busy / w as u64;
    let lpt = lpt_makespan(&desc, w);
    let busy_frac = busy as f64 / (w as u64 * span_ns) as f64;
    let ns_per_node = if nodes > 0 {
        busy as f64 / nodes as f64
    } else {
        0.0
    };

    eprintln!("[par-profile] bound={bound} W={w} units={n}");
    eprintln!(
        "  split (serial) : {:.2}s   parallel span: {:.2}s",
        split.as_secs_f64(),
        secs(span_ns)
    );
    eprintln!(
        "  worker busy    : {:.2}s  ->  busy fraction {:.1}%",
        secs(busy),
        100.0 * busy_frac
    );
    eprintln!(
        "  per-node cost  : {ns_per_node:.2} ns/node   ({nodes} nodes, unit setup {:.2}s = {:.1}% of busy)",
        secs(setup),
        100.0 * setup as f64 / busy as f64
    );
    eprintln!(
        "  unit wall time : max {:.2}s ({:.1}% of span)  p50 {:.3}s  p90 {:.3}s  p99 {:.3}s  top-{w} sum {:.2}s",
        secs(max_unit),
        100.0 * max_unit as f64 / span_ns as f64,
        secs(desc[n / 2]),
        secs(desc[n / 10]),
        secs(desc[n / 100]),
        secs(top_w),
    );
    eprintln!(
        "  makespan bounds: work/W {:.2}s | max-unit {:.2}s | LPT {:.2}s | actual {:.2}s",
        secs(div_bound),
        secs(max_unit),
        secs(lpt),
        secs(span_ns)
    );
    eprintln!(
        "  attribution    : granularity {:+.2}s ({:+.1}%)   scheduling {:+.2}s ({:+.1}%)",
        secs(lpt.saturating_sub(div_bound)),
        100.0 * lpt.saturating_sub(div_bound) as f64 / span_ns as f64,
        secs(span_ns.saturating_sub(lpt)),
        100.0 * span_ns.saturating_sub(lpt) as f64 / span_ns as f64,
    );

    #[cfg(feature = "probe-cache-stats")]
    {
        let h: u64 = prof.iter().map(|p| p.0.cache.0).sum();
        let m: u64 = prof.iter().map(|p| p.0.cache.1).sum();
        eprintln!(
            "  probe cache    : {h} hits / {m} misses = {:.3}% hit  ({:.0} misses/unit)",
            100.0 * h as f64 / (h + m).max(1) as f64,
            m as f64 / n as f64
        );
    }

    // Per-thread ns/node. A spread here is core heterogeneity (E-core placement)
    // or L2-cluster asymmetry, neither of which the busy fraction can see.
    let mut per: Vec<(u64, u64, u64)> = vec![(0, 0, 0); w + 1]; // (ns, nodes, units)
    for (p, nd) in prof {
        let i = if p.thread < w { p.thread } else { w };
        per[i].0 += p.ns;
        per[i].1 += nd;
        per[i].2 += 1;
    }
    eprint!("  per-thread     :");
    for (i, &(ns, nd, u)) in per.iter().enumerate() {
        if u == 0 {
            continue;
        }
        let tag = if i == w {
            "?".to_string()
        } else {
            i.to_string()
        };
        eprint!(
            " t{tag}={:.1}ns/n({:.0}s,{u}u)",
            if nd > 0 { ns as f64 / nd as f64 } else { 0.0 },
            secs(ns)
        );
    }
    eprintln!();
}

/// Lay a **subtree root** down at arena depth 0.
///
/// Generalises [`seed_root`] for the parallel driver. The cWD axis state is
/// recomputed from the board — one `project` plus two probes — rather than
/// threaded through the frontier, which keeps a work unit small; at a few
/// thousand units that cost is nothing.
///
/// What must NOT be recomputed is the move-history state. `dfa_state` and `last`
/// are carried from the split, because re-seeding the DFA at a subtree root would
/// forget the prefix that reached it and admit move sequences the sequential
/// engine prunes — changing the tree, which is the one thing this may not do.
#[cfg(feature = "parallel")]
fn seed_subtree(
    arena: &mut Arena,
    board: &State,
    dfa_state: u32,
    last: Move,
    merged: MergedBacking,
    dfa: &MoveDfa,
) -> u8 {
    seed_axes(arena, board, merged);
    let blank = board.blank_pos();
    arena.hot[0] = FrameHot {
        cand: MoveSet(CAND[blank as usize][last as usize].0 & !dfa.prune_mask(dfa_state)),
        minf: u8::MAX,
        blank,
        h: 0,
        mv: last,
        row_at: 0,
        col_at: 0,
        dfa: dfa_state,
    };
    let h = arena.h_at(0);
    arena.hot[0].h = h;
    h
}

/// Project `board` into both axes and lay the result down at arena depth 0.
/// Shared by [`seed_root`] and [`seed_subtree`].
fn seed_axes(arena: &mut Arena, board: &State, merged: MergedBacking) {
    let (m_row, br, dem_row, m_col, bc, dem_col) = project(board);
    let key_row = pack(&m_row, br);
    let key_col = pack(&m_col, bc);
    let r = merged.cell(key_row).expect("row state reachable");
    let c = merged.cell(key_col).expect("col state reachable");
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
    arena.board[0] = BoardSlot(Coded::encode(board));
}

/// Project `start` into both axes, probe the merged table once per axis, and lay
/// the result down as depth 0. Also seeds the DFA state and the root's candidate
/// set (where the σ-orbit filter applies — it is a `g == 0`-only rule, so it
/// belongs here rather than in the loop).
fn seed_root(
    arena: &mut Arena,
    start: &State,
    cwd: &Cwd,
    merged: MergedBacking,
    dfa: &MoveDfa,
    orbit_split: bool,
) -> u8 {
    seed_axes(arena, start, merged);

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
#[allow(clippy::too_many_arguments)]
fn run_iteration<
    const BUDGETED: bool,
    const K8: bool,
    const LM: bool,
    const LM2: bool,
    const K6: bool,
>(
    arena: &mut Arena,
    cache: &mut ProbeCache,
    cwd: &Cwd,
    merged: MergedBacking,
    dfa: &MoveDfa,
    k8ctx: Option<&K8Ctx>,
    lmctx: Option<&super::cwd_lm::CwdLmMm>,
    lm2ctx: Option<&super::cwd_lm::CwdLmMm>,
    k6ctx: Option<&K6Tier>,
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

        // The lazy k8 tier: consulted exactly at cWD-survivors (the recorded
        // recursive convention, so the tree matches §8j's node counts). The
        // child is already counted; a k8 prune folds its full f and moves on.
        // Compiles away entirely when `K8` is false.
        // Unconditioned survivor sampling (stats build + FLAT_SURVIVOR_DUMP=N):
        // every Nth child that survives cWD, before any tier consults — the
        // population all heuristic upgrades compete to prune, free of the
        // k8-prune-event conditioning of the autopsy sample.
        #[cfg(feature = "probe-cache-stats")]
        {
            static DUMP_EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let every = *DUMP_EVERY.get_or_init(|| {
                std::env::var("FLAT_SURVIVOR_DUMP")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            });
            if every > 0 {
                static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let t = TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if t % every == 0 {
                    let b = arena.board[d + 1].0.decode();
                    eprintln!(
                        "SURVIVOR bound={bound} g={g_next} h={h} slack={} board={:?}",
                        bound - g_next - h,
                        b.0
                    );
                }
            }
        }
        if LM2 {
            #[cfg(feature = "probe-cache-stats")]
            LM2_SLACK_HIST[((bound - g_next - h) as usize).min(15)]
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = TILE_OF_CODE[tile] as usize;
            let heff = lm2_child(arena, lm2ctx.unwrap(), d, t, h);
            if heff > h {
                let f2 = g_next.saturating_add(heff);
                if f2 > bound {
                    if f2 < minf {
                        minf = f2;
                    }
                    continue;
                }
            }
        } else if LM {
            let t = TILE_OF_CODE[tile] as usize;
            let heff = lm_child(arena, lmctx.unwrap(), d, t, h);
            if heff > h {
                let f2 = g_next.saturating_add(heff);
                if f2 > bound {
                    if f2 < minf {
                        minf = f2;
                    }
                    continue;
                }
            }
        }
        // The k6 tier: incremental consult at LM2-survivors (reached only
        // when neither cWD nor LM2 pruned the child). Slot copy + 2 slides +
        // 2 ranks + 2 diff lookups into the RAM-resident 88 MB family.
        if K6 {
            let hk6 = match k6ctx.unwrap() {
                K6Tier::Ranked(c) => k6_child(arena, c, d, geom, tile),
                K6Tier::Pos(c) => k6pos_child(arena, c, d, geom, tile),
            };
            if hk6 > h {
                let f2 = g_next.saturating_add(hk6);
                if f2 > bound {
                    if f2 < minf {
                        minf = f2;
                    }
                    continue;
                }
            }
        }
        if K8 {
            let hk8 = k8_child(arena, k8ctx.unwrap(), d, geom, tile);
            #[cfg(feature = "k8-probe-locality")]
            {
                let ctx = k8ctx.unwrap();
                let t = TILE_OF_CODE[tile] as usize;
                let gi = ctx.group_of[t] as usize;
                let gj = ctx.group_of[symmetry::TAU[t] as usize] as usize;
                let slot = &arena.k8[d + 1];
                let mut md = [[0u8; 3]; 2];
                for v in 0..2 {
                    for k in 0..3 {
                        md[v][k] = k8_group_md(&slot.views[v][k], &ctx.tiles[k]);
                    }
                }
                // f2 = g_next + H > bound  <=>  H > bound - g_next
                k8_surplus::record(&slot.h, &md, gi, gj, bound - g_next, h, hk8);
                k8_struct::record(
                    &slot.views,
                    &slot.h,
                    &md,
                    &ctx.tiles,
                    geom.from as usize,
                    symmetry::SIGMA[geom.from as usize] as usize,
                    hk8 > bound - g_next,
                );
                // Harvest only actual prune events (hk8 > need).
                if hk8 > bound - g_next {
                    // ~500 samples over a 1.5 B-node budgeted run.
                    let t = AUTOPSY_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if t % 340_000 == 0 {
                        let b = arena.board[d + 1].0.decode();
                        eprintln!(
                            "AUTOPSY bound={bound} g={g_next} hcwd={h} hk8={hk8} h={:?} md={:?} board={:?}",
                            slot.h, md, b.0
                        );
                    }
                    let n = geom.from as usize;
                    let mut keys = [[0u64; 3]; 2];
                    for v in 0..2 {
                        let blank = if v == 0 {
                            n
                        } else {
                            symmetry::SIGMA[n] as usize
                        };
                        for k in 0..3 {
                            let mut key = 0u64;
                            for (j, &t) in ctx.tiles[k].iter().enumerate() {
                                key |= (slot.views[v][k].pos_of(t) as u64) << (5 * j);
                            }
                            keys[v][k] = key | (blank as u64) << 40 | (k as u64) << 45;
                        }
                    }
                    k8_harvest::record(&slot.h, &md, &keys, bound - g_next, bound, g_next);
                }
            }
            if hk8 > h {
                let f2 = g_next.saturating_add(hk8);
                if f2 > bound {
                    if f2 < minf {
                        minf = f2;
                    }
                    continue;
                }
            }
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
    /// `64 - bits`, so the index is the high `bits` of one multiply. Held as a
    /// field rather than a const so the parallel driver can size worker caches
    /// independently of the sequential default, and so a sweep costs runs rather
    /// than rebuilds — see [`WORKER_CACHE_BITS`].
    shift: u32,
    #[cfg(feature = "probe-cache-stats")]
    hits: u64,
    #[cfg(feature = "probe-cache-stats")]
    misses: u64,
}

impl ProbeCache {
    fn new() -> Self {
        Self::with_bits(CACHE_BITS)
    }

    /// A cache of `1 << bits` slots. Sequential and per-worker caches are both
    /// [`CACHE_BITS`]/[`WORKER_CACHE_BITS`] = 18; the note there records why the
    /// worker cache is *not* sized down to fit a shared L2.
    fn with_bits(bits: u32) -> Self {
        assert!((8..=24).contains(&bits), "implausible probe-cache size");
        ProbeCache {
            shift: 64 - bits,
            slots: vec![
                CacheSlot {
                    tag: 0,
                    cell: super::cwd::CwdCell {
                        wd: 0,
                        curves: [0; W],
                        nbr_wd: [255; 2 * W],
                    },
                };
                1usize << bits
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
    fn slot_of(&self, key: u64) -> usize {
        ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> self.shift) as usize
    }

    /// Memoised probe. Returns exactly what `merged.get(&key)` would.
    #[inline(always)]
    fn get<'a>(&'a mut self, merged: MergedBacking, key: u64) -> &'a super::cwd::CwdCell {
        debug_assert_ne!(key, 0, "key 0 collides with the empty-slot sentinel");
        let i = self.slot_of(key);
        if self.slots[i].tag != key {
            let cell = merged.cell(key).expect("state reachable");
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
            merged.cell(key).expect("state reachable").wd,
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
    merged: MergedBacking,
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

    /// k6 (pdb24_{a..d}, 6-6-6-6) and k7 (pdb24_k7_{a..d}, 7-7-7-3) zPDB
    /// families vs the production baseline and LM2 on the unconditioned
    /// survivor sample: the cheap-tier candidates' frontier reach.
    ///
    ///   cargo test --release k6k7_certification_survivor_sample -- --ignored --nocapture
    #[test]
    #[ignore = "needs the artifacts, the k6/k7 zPDBs and data/survivors_146.txt"]
    fn k6k7_certification_survivor_sample() {
        use crate::puzzle24::pdb::ZPatternDb;
        let cwd = Cwd::mm_only(std::path::Path::new("data/cwd_mm.bin")).expect("cwd_mm.bin");
        let backing = cwd.backing().unwrap();
        let mm = super::load_cwd_lm_mm(std::path::Path::new("data/cwd_lm_mm.bin")).expect("lm mm");
        let load_family = |names: &[&str]| -> Vec<ZPatternDb> {
            names
                .iter()
                .map(|n| {
                    ZPatternDb::load_mmap(std::path::Path::new(&format!("data/{n}.zbin")))
                        .unwrap_or_else(|e| panic!("{n}: {e:?}"))
                })
                .collect()
        };
        let k6 = load_family(&["pdb24_a", "pdb24_b", "pdb24_c", "pdb24_d"]);
        let k7 = load_family(&["pdb24_k7_a", "pdb24_k7_b", "pdb24_k7_c", "pdb24_k7_d"]);
        for (name, fam) in [("k6", &k6), ("k7", &k7)] {
            let mut cover = 0u32;
            for db in fam.iter() {
                let tiles: Vec<u8> = db.pattern().iter().collect();
                eprintln!("{name} pattern: {tiles:?}");
                assert_eq!(cover & db.pattern().0, 0, "{name} patterns overlap");
                cover |= db.pattern().0;
            }
            assert_eq!(cover, 0x01FF_FFFE, "{name} must cover tiles 1..=24");
        }
        let text = std::fs::read_to_string("data/survivors_146.txt").expect("survivor sample");
        let (mut nb, mut c6, mut c7, mut cl) = (0u64, 0u64, 0u64, 0u64);
        let (mut c6single, mut only6single) = (0u64, 0u64);
        let (mut only6, mut only7, mut u7l) = (0u64, 0u64, 0u64);
        let (mut adv6, mut adv7) = ([0u64; 9], [0u64; 9]);
        for line in text.lines().filter(|l| l.contains("board=[")) {
            let seg = &line[line.find("board=[").unwrap() + 7..];
            let seg = &seg[..seg.find(']').unwrap()];
            let vals: Vec<u8> = seg.split(',').map(|w| w.trim().parse().unwrap()).collect();
            if vals.len() != 25 {
                continue;
            }
            let mut cells = [0u8; 25];
            cells.copy_from_slice(&vals);
            let s = State(cells);
            let (mr, br, dr, mc, bc, dc) = project(&s);
            let (rkey, ckey) = (pack(&mr, br), pack(&mc, bc));
            let rc = backing.cell(rkey).unwrap();
            let cc = backing.cell(ckey).unwrap();
            let rterm = rc.wd + surcharge_from_curves(&rc.curves, &dr);
            let cterm = cc.wd + surcharge_from_curves(&cc.curves, &dc);
            let h0p = rterm as u32 + cterm as u32;
            nb += 1;
            let rs = symmetry::reflect(&s);
            let hfam = |fam: &[ZPatternDb]| -> (u32, u32) {
                let (mut s0, mut s1) = (0u32, 0u32);
                for db in fam {
                    s0 += db.cold_lookup(&s) as u32;
                    s1 += db.cold_lookup(&rs) as u32;
                }
                (s0.max(s1), s0)
            };
            let ((h6, h6single), (h7, _)) = (hfam(&k6), hfam(&k7));
            // LM2 compose
            let pos = |t: u8| s.0.iter().position(|&x| x == t).unwrap();
            let lp = [
                (pos(20) / 5) as u8,
                (pos(24) % 5) as u8,
                (pos(15) / 5) as u8,
                (pos(19) / 5) as u8,
                (pos(19) % 5) as u8,
                (pos(23) % 5) as u8,
            ];
            let probe = |key: u64| -> ([u8; 4], [u8; 12]) {
                let (mut s4, mut p12) = ([0xFFu8; 4], [0xFFu8; 12]);
                if let Some((a, b)) = mm.probe(key) {
                    s4.copy_from_slice(a);
                    p12.copy_from_slice(b);
                }
                (s4, p12)
            };
            let (vr, pr_all) = probe(rkey);
            let (vc, pc_all) = probe(ckey);
            let sv = |v: &[u8; 4], l: u8| if l < 4 { v[l as usize] } else { 0xFF };
            let or_ = |v: u8, fb: u8| if v != 0xFF { v } else { fb };
            let pr = if lp[0] < 4 && lp[2] < 3 {
                pr_all[(lp[0] * 3 + lp[2]) as usize]
            } else {
                0xFF
            };
            let pc = if lp[1] < 4 && lp[5] < 3 {
                pc_all[(lp[1] * 3 + lp[5]) as usize]
            } else {
                0xFF
            };
            let r20f = or_(sv(&vr, lp[0]), rterm);
            let c24f = or_(sv(&vc, lp[1]), cterm);
            let ba = or_(pr, r20f) as u32 + cterm as u32;
            let bb = r20f as u32 + or_(sv(&vc, lp[4]), cterm) as u32;
            let bcv = or_(sv(&vr, lp[3]), rterm) as u32 + c24f as u32;
            let bd = rterm as u32 + or_(pc, c24f) as u32;
            let lm2 = h0p.max(ba.min(bb).min(bcv).min(bd));
            let l = lm2 >= h0p + 2;
            let k6c = h6 >= h0p + 2;
            let k7c = h7 >= h0p + 2;
            if l {
                cl += 1;
            }
            if k6c {
                c6 += 1;
                adv6[((h6 - h0p) as usize).min(8)] += 1;
                if !l {
                    only6 += 1;
                }
            }
            // single-view variant: halves the tier's probe count
            if h6single >= h0p + 2 {
                c6single += 1;
                if !l {
                    only6single += 1;
                }
            }
            if k7c {
                c7 += 1;
                adv7[((h7 - h0p) as usize).min(8)] += 1;
                if !l {
                    only7 += 1;
                }
            }
            if k7c || l {
                u7l += 1;
            }
        }
        eprintln!(
            "k6 single-view (normal σ only): certK6 {c6single} ({:.1}%), k6-only-beyond-LM2 {only6single} ({:.2}%) — vs both-views {only6} ({:.2}%)",
            100.0 * c6single as f64 / nb.max(1) as f64,
            100.0 * only6single as f64 / nb.max(1) as f64,
            100.0 * only6 as f64 / nb.max(1) as f64,
        );
        eprintln!(
            "k6k7: {nb} boards; certL {cl} ({:.1}%); certK6 {c6} ({:.1}%) k6-only {only6} ({:.2}%); certK7 {c7} ({:.1}%) k7-only {only7} ({:.2}%); union(k7,L) {u7l} ({:.1}%); adv6 {:?} adv7 {:?}",
            100.0 * cl as f64 / nb.max(1) as f64,
            100.0 * c6 as f64 / nb.max(1) as f64,
            100.0 * only6 as f64 / nb.max(1) as f64,
            100.0 * c7 as f64 / nb.max(1) as f64,
            100.0 * only7 as f64 / nb.max(1) as f64,
            100.0 * u7l as f64 / nb.max(1) as f64,
            &adv6[2..],
            &adv7[2..]
        );
    }

    /// Per-group k8 surplus breakdown on the unconditioned survivor sample,
    /// overall and restricted to the k8-only-beyond-LM2 boards. Re-examines
    /// the earlier "group b carries 52% of surplus" finding (which was
    /// measured on k8-conditioned prune events against an MD baseline).
    ///
    ///   cargo test --release k8_group_breakdown_survivor_sample -- --ignored --nocapture
    #[test]
    #[ignore = "needs the artifacts, the three zPDBs and data/survivors_146.txt"]
    fn k8_group_breakdown_survivor_sample() {
        let cwd = Cwd::mm_only(std::path::Path::new("data/cwd_mm.bin")).expect("cwd_mm.bin");
        let backing = cwd.backing().unwrap();
        let mm = super::load_cwd_lm_mm(std::path::Path::new("data/cwd_lm_mm.bin")).expect("lm mm");
        let ctx = K8Ctx::load_mmap(std::path::Path::new("data")).expect("zpdbs");
        for (i, db) in ctx.dbs.iter().enumerate() {
            let tiles: Vec<u8> = db.pattern().iter().collect();
            eprintln!("group {}: tiles {:?}", (b'a' + i as u8) as char, tiles);
        }
        let text = std::fs::read_to_string("data/survivors_146.txt").expect("survivor sample");
        let md = |t: u8, p: usize| -> u32 {
            let g = (t - 1) as usize;
            ((g / 5).abs_diff(p / 5) + (g % 5).abs_diff(p % 5)) as u32
        };
        // [population][group]: 0 = all, 1 = certK, 2 = k8-only
        let mut sur = [[0u64; 3]; 3];
        let mut counts = [0u64; 3];
        let mut crit = [0u64; 8];
        for line in text.lines().filter(|l| l.contains("board=[")) {
            let seg = &line[line.find("board=[").unwrap() + 7..];
            let seg = &seg[..seg.find(']').unwrap()];
            let vals: Vec<u8> = seg.split(',').map(|w| w.trim().parse().unwrap()).collect();
            if vals.len() != 25 {
                continue;
            }
            let mut cells = [0u8; 25];
            cells.copy_from_slice(&vals);
            let s = State(cells);
            let (mr, br, dr, mc, bc, dc) = project(&s);
            let (rkey, ckey) = (pack(&mr, br), pack(&mc, bc));
            let rc = backing.cell(rkey).unwrap();
            let cc = backing.cell(ckey).unwrap();
            let rterm = rc.wd + surcharge_from_curves(&rc.curves, &dr);
            let cterm = cc.wd + surcharge_from_curves(&cc.curves, &dc);
            let h0p = rterm as u32 + cterm as u32;
            let pos = |t: u8| s.0.iter().position(|&x| x == t).unwrap();
            let lp = [
                (pos(20) / 5) as u8,
                (pos(24) % 5) as u8,
                (pos(15) / 5) as u8,
                (pos(19) / 5) as u8,
                (pos(19) % 5) as u8,
                (pos(23) % 5) as u8,
            ];
            let probe = |key: u64| -> ([u8; 4], [u8; 12]) {
                let (mut s4, mut p12) = ([0xFFu8; 4], [0xFFu8; 12]);
                if let Some((a, b)) = mm.probe(key) {
                    s4.copy_from_slice(a);
                    p12.copy_from_slice(b);
                }
                (s4, p12)
            };
            let (vr, pr_all) = probe(rkey);
            let (vc, pc_all) = probe(ckey);
            let sv = |v: &[u8; 4], l: u8| if l < 4 { v[l as usize] } else { 0xFF };
            let or_ = |v: u8, fb: u8| if v != 0xFF { v } else { fb };
            let pr = if lp[0] < 4 && lp[2] < 3 {
                pr_all[(lp[0] * 3 + lp[2]) as usize]
            } else {
                0xFF
            };
            let pc = if lp[1] < 4 && lp[5] < 3 {
                pc_all[(lp[1] * 3 + lp[5]) as usize]
            } else {
                0xFF
            };
            let r20f = or_(sv(&vr, lp[0]), rterm);
            let c24f = or_(sv(&vc, lp[1]), cterm);
            let ba = or_(pr, r20f) as u32 + cterm as u32;
            let bb = r20f as u32 + or_(sv(&vc, lp[4]), cterm) as u32;
            let bcv = or_(sv(&vr, lp[3]), rterm) as u32 + c24f as u32;
            let bd = rterm as u32 + or_(pc, c24f) as u32;
            let lm2 = h0p.max(ba.min(bb).min(bcv).min(bd));

            let rs = symmetry::reflect(&s);
            let (mut s0, mut s1) = (0u32, 0u32);
            let mut g0 = [0u32; 3];
            let mut g1 = [0u32; 3];
            for (i, db) in ctx.dbs.iter().enumerate() {
                g0[i] = db.cold_lookup(&s) as u32;
                g1[i] = db.cold_lookup(&rs) as u32;
                s0 += g0[i];
                s1 += g1[i];
            }
            let (hk8, gv, view_board) = if s0 >= s1 {
                (s0, g0, &s)
            } else {
                (s1, g1, &rs)
            };
            let mut gsur = [0u32; 3];
            for (i, db) in ctx.dbs.iter().enumerate() {
                let mut m = 0u32;
                for t in db.pattern().iter() {
                    let p = view_board.0.iter().position(|&x| x == t).unwrap();
                    m += md(t, p);
                }
                gsur[i] = gv[i].saturating_sub(m);
            }
            let k = hk8 >= h0p + 2;
            let l = lm2 >= h0p + 2;
            let mut pops: Vec<usize> = vec![0];
            if k {
                pops.push(1);
            }
            if k && !l {
                pops.push(2);
                // criticality: replace group i's value with its MD in BOTH
                // views, re-max, ask if certification survives
                let mut crit_mask = 0usize;
                for i in 0..3 {
                    let (mut m0, mut m1) = (0u32, 0u32);
                    for t in ctx.dbs[i].pattern().iter() {
                        let p0 = s.0.iter().position(|&x| x == t).unwrap();
                        let p1 = rs.0.iter().position(|&x| x == t).unwrap();
                        m0 += md(t, p0);
                        m1 += md(t, p1);
                    }
                    let h0v = s0 - g0[i] + m0;
                    let h1v = s1 - g1[i] + m1;
                    if h0v.max(h1v) < h0p + 2 {
                        crit_mask |= 1 << i;
                    }
                }
                crit[crit_mask] += 1;
            }
            for &p_ in &pops {
                counts[p_] += 1;
                for i in 0..3 {
                    sur[p_][i] += gsur[i] as u64;
                }
            }
        }
        eprintln!(
            "k8-only criticality (mask a|b<<1|c<<2): none {} | a {} | b {} | ab {} | c {} | ac {} | bc {} | abc {}",
            crit[0], crit[1], crit[2], crit[3], crit[4], crit[5], crit[6], crit[7]
        );
        for (p_, name) in [(0usize, "all"), (1, "certK"), (2, "k8-only")] {
            let tot: u64 = sur[p_].iter().sum();
            eprintln!(
                "{name} ({} boards): group surpluses a {} ({:.1}%), b {} ({:.1}%), c {} ({:.1}%)",
                counts[p_],
                sur[p_][0],
                100.0 * sur[p_][0] as f64 / tot.max(1) as f64,
                sur[p_][1],
                100.0 * sur[p_][1] as f64 / tot.max(1) as f64,
                sur[p_][2],
                100.0 * sur[p_][2] as f64 / tot.max(1) as f64,
            );
        }
    }

    /// k8 certification on the unconditioned survivor sample: per board,
    /// production baseline h0p, the LM2 bound (from the mmap artifacts) and
    /// h_k8 (three zPDBs, both σ-views); reports the union table that pins
    /// the cascade's node floor. Cross-check: certL must reproduce the
    /// harness ovlunion figure (336/555).
    ///
    ///   cargo test --release k8_certification_survivor_sample -- --ignored --nocapture
    #[test]
    #[ignore = "needs data/cwd_mm.bin, data/cwd_lm_mm.bin, the three 32.8 GB zPDBs and data/survivors_146.txt"]
    fn k8_certification_survivor_sample() {
        let cwd = Cwd::mm_only(std::path::Path::new("data/cwd_mm.bin")).expect("cwd_mm.bin");
        let backing = cwd.backing().unwrap();
        let mm = super::load_cwd_lm_mm(std::path::Path::new("data/cwd_lm_mm.bin")).expect("lm mm");
        let ctx = K8Ctx::load_mmap(std::path::Path::new("data")).expect("zpdbs");
        let text = std::fs::read_to_string("data/survivors_146.txt").expect("survivor sample");

        let (mut nb, mut ck, mut cl, mut both, mut konly, mut lonly, mut neither) =
            (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        let mut adv_hist = [0u64; 9];
        for line in text.lines().filter(|l| l.contains("board=[")) {
            let seg = &line[line.find("board=[").unwrap() + 7..];
            let seg = &seg[..seg.find(']').unwrap()];
            let vals: Vec<u8> = seg.split(',').map(|w| w.trim().parse().unwrap()).collect();
            if vals.len() != 25 {
                continue;
            }
            let mut cells = [0u8; 25];
            cells.copy_from_slice(&vals);
            let s = State(cells);
            nb += 1;

            // production baseline: per-axis wd + single-line-max surcharge
            let (mr, br, dr, mc, bc, dc) = project(&s);
            let (rkey, ckey) = (pack(&mr, br), pack(&mc, bc));
            let rc = backing.cell(rkey).expect("row reachable");
            let cc = backing.cell(ckey).expect("col reachable");
            let rterm = rc.wd + surcharge_from_curves(&rc.curves, &dr);
            let cterm = cc.wd + surcharge_from_curves(&cc.curves, &dc);
            let h0p = rterm as u32 + cterm as u32;

            // LM2 bound, exactly the engine's compose
            let pos = |t: u8| s.0.iter().position(|&x| x == t).unwrap();
            let lp = [
                (pos(20) / 5) as u8,
                (pos(24) % 5) as u8,
                (pos(15) / 5) as u8,
                (pos(19) / 5) as u8,
                (pos(19) % 5) as u8,
                (pos(23) % 5) as u8,
            ];
            let probe = |key: u64| -> ([u8; 4], [u8; 12]) {
                let (mut s4, mut p12) = ([0xFFu8; 4], [0xFFu8; 12]);
                if let Some((a, b)) = mm.probe(key) {
                    s4.copy_from_slice(a);
                    p12.copy_from_slice(b);
                }
                (s4, p12)
            };
            let (vr, pr_all) = probe(rkey);
            let (vc, pc_all) = probe(ckey);
            let sv = |v: &[u8; 4], l: u8| if l < 4 { v[l as usize] } else { 0xFF };
            let or_ = |v: u8, fb: u8| if v != 0xFF { v } else { fb };
            let pr = if lp[0] < 4 && lp[2] < 3 {
                pr_all[(lp[0] * 3 + lp[2]) as usize]
            } else {
                0xFF
            };
            let pc = if lp[1] < 4 && lp[5] < 3 {
                pc_all[(lp[1] * 3 + lp[5]) as usize]
            } else {
                0xFF
            };
            let r20f = or_(sv(&vr, lp[0]), rterm);
            let c24f = or_(sv(&vc, lp[1]), cterm);
            let ba = or_(pr, r20f) as u32 + cterm as u32;
            let bb = r20f as u32 + or_(sv(&vc, lp[4]), cterm) as u32;
            let bcv = or_(sv(&vr, lp[3]), rterm) as u32 + c24f as u32;
            let bd = rterm as u32 + or_(pc, c24f) as u32;
            let lm2 = h0p.max(ba.min(bb).min(bcv).min(bd));

            // h_k8: three additive zPDBs, both σ-views
            let rs = symmetry::reflect(&s);
            let (mut s0, mut s1) = (0u32, 0u32);
            for db in &ctx.dbs {
                s0 += db.cold_lookup(&s) as u32;
                s1 += db.cold_lookup(&rs) as u32;
            }
            let hk8 = s0.max(s1);

            let k = hk8 >= h0p + 2;
            let l = lm2 >= h0p + 2;
            if k {
                ck += 1;
                adv_hist[(hk8 - h0p).min(8) as usize] += 1;
            }
            if l {
                cl += 1;
            }
            match (k, l) {
                (true, true) => both += 1,
                (true, false) => konly += 1,
                (false, true) => lonly += 1,
                (false, false) => neither += 1,
            }
        }
        eprintln!(
            "k8cert: {nb} boards; certK {ck} ({:.1}%), certL {cl} ({:.1}%); both {both}, k8-only {konly} ({:.1}%), lm2-only {lonly}, neither {neither}; k8 advantage hist (>=2) {:?}",
            100.0 * ck as f64 / nb.max(1) as f64,
            100.0 * cl as f64 / nb.max(1) as f64,
            100.0 * konly as f64 / nb.max(1) as f64,
            &adv_hist[2..]
        );
        assert!(nb == 555, "sample size changed");
    }

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
    /// Prune-event autopsy: reconstruct the winning view's abstract group plans
    /// for dumped prune boards and extract the AWAY-moves (steps that increase a
    /// tile's Manhattan-to-goal) — the concrete carriers of the surplus, i.e.
    /// the "+2 over cWD". Reads `AUTOPSY_FILE` (lines from the instrumented
    /// run). Greedy descent picks one optimal plan among many; classification
    /// is over that canonical plan (caveat noted in the report).
    ///
    ///   AUTOPSY_FILE=... cargo test --release --features k8-probe-locality \
    ///       k8_prune_autopsy -- --ignored --nocapture
    #[test]
    #[ignore = "needs the 32.8 GB k8 tables and an AUTOPSY_FILE dump"]
    fn k8_prune_autopsy() {
        use crate::puzzle24::pdb::ProjectedState;
        use crate::puzzle24::symmetry;
        let path = std::env::var("AUTOPSY_FILE").expect("set AUTOPSY_FILE");
        let text = std::fs::read_to_string(&path).expect("read dump");
        let ctx = K8Ctx::load_mmap(std::path::Path::new("data")).expect("k8 tables");
        let nums = |seg: &str| -> Vec<i64> {
            seg.chars()
                .map(|c| {
                    if c.is_ascii_digit() || c == '-' {
                        c
                    } else {
                        ' '
                    }
                })
                .collect::<String>()
                .split_whitespace()
                .map(|w| w.parse().unwrap())
                .collect()
        };
        // aggregates: away-move counts by board-frame (tile, direction), and
        // by tile alone; direction: 0=up,1=down,2=left,3=right (board frame).
        let mut by_tile = [0u32; 25];
        let mut by_dir = [0u32; 4];
        let mut by_cell_from = [0u32; 25];
        let mut boards = 0;
        let mut plans = 0;
        let mut aways = 0;
        for (bi, line) in text
            .lines()
            .filter(|l| l.starts_with("AUTOPSY"))
            .enumerate()
        {
            let hseg = &line[line.find("h=[").unwrap()..line.find(" md=").unwrap()];
            let mseg = &line[line.find("md=[").unwrap()..line.find(" board=").unwrap()];
            let bseg = &line[line.find("board=[").unwrap()..];
            let hv = nums(hseg);
            let mv = nums(mseg);
            let bv = nums(bseg);
            assert_eq!(hv.len(), 6);
            assert_eq!(bv.len(), 25);
            let mut cells = [0u8; 25];
            for (i, &x) in bv.iter().enumerate() {
                cells[i] = x as u8;
            }
            let board = State(cells);
            let rboard = symmetry::reflect(&board);
            boards += 1;
            // winning view = larger sum
            let s0: i64 = hv[..3].iter().sum();
            let s1: i64 = hv[3..].iter().sum();
            let v = usize::from(s1 > s0);
            let vb = if v == 0 { board } else { rboard };
            for k in 0..3 {
                let h0 = hv[v * 3 + k] as u8;
                let md0 = mv[v * 3 + k] as u8;
                if h0 <= md0 + 1 {
                    continue; // no surplus to explain
                }
                let db = &ctx.dbs[k];
                let pattern = db.pattern();
                let tiles: Vec<u8> = (1u8..25).filter(|&t| pattern.contains(t)).collect();
                let mut cur = ProjectedState::from_state(&vb, pattern);
                let mut h_cur = h0;
                let mut succ: Vec<ProjectedState> = Vec::new();
                let mut away_this = 0u32;
                let mut steps = 0u32;
                while h_cur > 0 && steps < 200 {
                    steps += 1;
                    succ.clear();
                    crate::puzzle24::pdb::zbuild::gen_moves(db.layout(), &cur, &mut succ);
                    let mut advanced = false;
                    for ns in succ.iter() {
                        let idx = db.layout().rank(ns, pattern);
                        let nh = db.diff_lookup(idx, h_cur);
                        if nh + 1 == h_cur {
                            // which tile moved?
                            for &t in &tiles {
                                let (a, b) = (cur.pos_of(t), ns.pos_of(t));
                                if a != b {
                                    let (gr, gc) = (((t - 1) / 5) as i32, ((t - 1) % 5) as i32);
                                    let d_old =
                                        ((a / 5) as i32 - gr).abs() + ((a % 5) as i32 - gc).abs();
                                    let d_new =
                                        ((b / 5) as i32 - gr).abs() + ((b % 5) as i32 - gc).abs();
                                    if d_new > d_old {
                                        // AWAY move — map to board frame
                                        let (bt, ba, bb) = if v == 0 {
                                            (t, a, b)
                                        } else {
                                            (
                                                symmetry::TAU[t as usize],
                                                symmetry::SIGMA[a as usize],
                                                symmetry::SIGMA[b as usize],
                                            )
                                        };
                                        let dir = if bb == ba + 5 {
                                            1
                                        } else if ba == bb + 5 {
                                            0
                                        } else if bb == ba + 1 {
                                            3
                                        } else {
                                            2
                                        };
                                        by_tile[bt as usize] += 1;
                                        by_dir[dir] += 1;
                                        by_cell_from[ba as usize] += 1;
                                        aways += 1;
                                        away_this += 1;
                                        if bi < 8 {
                                            println!(
                                                "  board {bi} view {v} grp {k}: AWAY tile {bt} {ba}->{bb} (step {}/{h0}, blank@{})",
                                                h0 - h_cur + 1,
                                                if v == 0 { cur.blank_pos() } else { symmetry::SIGMA[cur.blank_pos() as usize] }
                                            );
                                        }
                                    }
                                    break;
                                }
                            }
                            cur = *ns;
                            h_cur = nh;
                            advanced = true;
                            break;
                        }
                    }
                    assert!(advanced, "descent stuck at h={h_cur} (board {bi} grp {k})");
                }
                plans += 1;
                let expect = ((h0 - md0) / 2) as u32;
                assert_eq!(
                    away_this, expect,
                    "away-count {away_this} != surplus/2 {expect} (board {bi} grp {k})",
                );
            }
        }
        println!("=== autopsy aggregate: {boards} boards, {plans} plans, {aways} away-moves ===");
        println!("away-moves by board-frame tile:");
        for t in 1..25 {
            if by_tile[t] > 0 {
                println!("  tile {t:2}: {}", by_tile[t]);
            }
        }
        println!("by direction u/d/l/r: {by_dir:?}");
        println!("by from-cell:");
        for c in 0..25 {
            if by_cell_from[c] > 0 {
                println!("  cell {c:2} (r{},c{}): {}", c / 5, c % 5, by_cell_from[c]);
            }
        }
    }

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

    /// **The parallel gate.** The tree-splitting driver must reproduce the frozen
    /// oracle exactly — same outcomes, same node counts, same iteration counts.
    ///
    /// Node identity is the correctness argument for the parallel driver — but it
    /// holds only for an **exhausted** threshold, and the distinction is not a
    /// technicality:
    ///
    /// * **Exhausted** (`ProvedAtLeast`): every node with `f ≤ bound` is visited by
    ///   exactly one worker whatever the order, so the count is order-independent
    ///   and must match to the node. Any drift in the split's pruning or
    ///   accounting shows up here.
    /// * **Solving** (`Solved`): the search short-circuits on the goal. The
    ///   sequential engine is depth-first and returns the moment it hits `h == 0`,
    ///   never building shallower siblings; the split expands breadth-first and
    ///   builds nodes the DFS never reached. Counts legitimately differ — only the
    ///   solution *length* is invariant, and that is what optimality means. The
    ///   deleted recursive driver documented the same caveat.
    ///
    /// This matters for the `R` program not at all, since it is exhaust-only, and
    /// the two production checks are both exhausts: 422,379,806 at exhaust-144 and
    /// 18,189,473,636 at exhaust-146, both reproduced exactly.
    ///
    /// Most oracle cases are small enough to finish inside the split, which
    /// exercises the "threshold fitted inside the split" path a large board never
    /// reaches.
    #[cfg(all(feature = "cwd-table-tests", feature = "parallel"))]
    #[test]
    fn flat_parallel_matches_frozen_oracle() {
        let Some(cwd) = cwd_merged_or_skip() else {
            return;
        };
        let dfa = MoveDfa::build_default();
        let mut checked = 0usize;
        let mut exhausts = 0usize;
        for (i, c) in crate::puzzle24::search::flat_oracle::CASES
            .iter()
            .enumerate()
        {
            let s = State(c.board);
            let (po, ps) =
                flat_bounded_parallel(&s, cwd, &dfa, c.orbit_split, c.bound, |_, _, _| {});
            let ok = match &po {
                BoundedOutcome::Solved(mv) => c.tag == 0 && mv.len() as u8 == c.val,
                BoundedOutcome::ProvedAtLeast(k) => c.tag == 1 && *k == c.val,
                BoundedOutcome::Unsolvable => c.tag == 2,
                BoundedOutcome::BudgetExhausted(_) => {
                    panic!("the parallel driver sets no budget")
                }
            };
            assert!(
                ok,
                "case {i}: parallel outcome differs: {po:?} vs frozen tag={} val={}",
                c.tag, c.val
            );
            // Counts are comparable only on an exhausted threshold (see above): a
            // solving run short-circuits, and DFS versus split-BFS reach the goal
            // after different amounts of work.
            if c.tag == 1 {
                assert_eq!(
                    ps.nodes, c.nodes,
                    "case {i}: parallel node count differs on an EXHAUST \
                     (board {:?}, bound {})",
                    c.board, c.bound
                );
                assert_eq!(
                    ps.iterations, c.iterations,
                    "case {i}: iteration count differs"
                );
                exhausts += 1;
            }
            checked += 1;
        }
        assert_eq!(checked, 180, "oracle case count changed");
        // Guard the guard: if the oracle ever lost its exhaust cases, the
        // node-identity assertion above would become vacuous.
        assert!(
            exhausts >= 100,
            "expected ~117 exhausted-threshold cases, got {exhausts}"
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
