//! Single-demanded-line joint tables: the last-two-moves obligations priced
//! **together with** "at least `d` escapes of line `g`", for one line at a
//! time — the closed-table form of the joint LM2 heuristic.
//!
//! Measured basis (2026-08-06): the full joint form prunes 780 slack-0
//! survivor consults (1.38× fewer nodes than `--lm2` at threshold 144; 545
//! of the 780 beyond both table-LM2 and the k8 tier); restricting demands to
//! one line at a time retains 778/780 (`data/lm2j1l_survivors148.txt`). So a
//! closed table keyed `(WD key, line g, demand d, tracked variant)` captures
//! essentially the whole gain with a pure probe — no per-node search.
//!
//! **Layered build.** For a fixed line `g`, let `D_r(x)` be the min abstract
//! moves from product state `x` to the discharged goal making ≥ `r` escapes
//! of `g` (a forward move is a `g`-escape iff a goal-type-`g` token departs
//! physical line `g`; in the reversed-edge loop variables below that is
//! `t == g && b == g`). Any optimal path's first edge is either a `g`-escape
//! (rest needs `r−1`) or not (rest still needs `r`), so `D_r` is computed
//! from the complete `D_{r-1}` by seeding every reversed escape edge with
//! `D_{r-1}(S) + 1` and then relaxing **only non-escape** edges with a
//! multi-source dial. `(goal, r ≥ 1)` is non-terminal; no edge changes `r`
//! by more than one; extra escapes are dominated (`D_r ≥ D_{r-1}`).
//!
//! Two port hazards this module exists to get right (vs `build_cwd_lm2`'s
//! plain BFS): the 0xFF-only write guard becomes a **min-write** (a state
//! seeded at depth 10 but propagation-reachable at 6 must take 6 — the
//! plain guard would ship an inadmissible overestimate), and termination
//! must wait for both an empty frontier **and** exhausted seed depths.
//!
//! Storage: 2-bit fields `min((D − base)/2, 3)` where
//! `base(key, g, d) = wd + 2·curve_nibble(g, d)` from the merged cWD cell —
//! deltas are even (blank-line parity is fixed by the key) and `D ≥ base`
//! (obligations only add constraints), so the field covers advantages
//! 0/+2/+4/+6-saturating and saturation clamps down (admissible). Invalid
//! placements carry no code: validity comes from the zero-demand
//! `cwd_lm_mm` values the consult already reads.

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::OnceLock;

use super::cwd::{pack, unpack, CwdMm};
use crate::puzzle24::state::W;

/// Tracked token types (mirrors `cwd_lm.rs`): A = goal-line 3, B = goal-line 2.
const TT: usize = 3;
const TB: usize = 2;

/// Number of WD-reachable keys (known from the wd24 build).
const N_WD_KEYS: usize = 65_650_495;

/// Demand configs per line: `d ∈ 1..=4`.
const ND: usize = 4;

/// Queryable variants per `(g, d)` config: 4 single-A lines (`la ∈ 0..4`),
/// 3 single-B lines (`lb ∈ 0..3`), 12 pair placements (`la·3 + lb`).
const NV: usize = 19;

/// 2-bit fields per key: 20 configs × 19 variants + 3 single-B zero-demand.
const NFIELDS: usize = W * ND * NV + 3;

/// Payload bytes per key (2-bit packed).
pub const LM1L_PAYLOAD: usize = NFIELDS.div_ceil(4); // 96

/// Field index for `(g, d, variant)`; `v`: 0..4 = single-A `la`, 4..7 =
/// single-B `lb`, 7..19 = pair `la·3 + lb`.
#[inline]
pub fn field_of(g: usize, d: usize, v: usize) -> usize {
    debug_assert!(g < W && (1..=ND).contains(&d) && v < NV);
    (g * ND + (d - 1)) * NV + v
}

/// Field index for the single-B zero-demand values (`lb ∈ 0..3`).
#[inline]
pub fn field_b0(lb: usize) -> usize {
    debug_assert!(lb < 3);
    W * ND * NV + lb
}

/// Read a 2-bit field from a payload.
#[inline]
pub fn read_field(payload: &[u8], idx: usize) -> u8 {
    (payload[idx >> 2] >> ((idx & 3) * 2)) & 3
}

// ------------------------------ shared infra ---------------------------------

/// One reversed edge of a key `S`: the forward move `P → S` slides a type-`t`
/// token from line `b = blank(S)` into line `f = blank(P) = b ± 1`.
/// Packed `pidx:27 | t:3 | dir:1 | valid:1`.
#[derive(Clone, Copy)]
struct Edge(u32);

impl Edge {
    const INVALID: Edge = Edge(0);
    #[inline]
    fn new(pidx: usize, t: usize, dir: usize) -> Edge {
        Edge((pidx as u32) | ((t as u32) << 27) | ((dir as u32) << 30) | (1 << 31))
    }
    #[inline]
    fn valid(self) -> bool {
        self.0 >> 31 == 1
    }
    #[inline]
    fn pidx(self) -> usize {
        (self.0 & 0x07FF_FFFF) as usize
    }
    #[inline]
    fn t(self) -> usize {
        ((self.0 >> 27) & 7) as usize
    }
    #[inline]
    fn f(self, b: usize) -> usize {
        if (self.0 >> 30) & 1 == 1 {
            b + 1
        } else {
            b - 1
        }
    }
}

/// Max reversed edges per key: 2 directions × 5 token types.
const MAX_EDGES: usize = 2 * W;

/// One-time indexed infrastructure shared by every BFS pass: the key
/// enumeration, per-key blank line, per-key saturating type-2/3 line counts
/// (2 bits per line, cap 3 — validity checks only need ≥1/≥2), the flat
/// reversed-edge adjacency, and the per-key `wd`/surcharge-curve bases for
/// delta extraction.
pub struct Infra {
    keys: Vec<u64>,
    blank: Vec<u8>,
    /// `cnt3[idx] >> (2*line) & 3` = min(type-3 tokens in `line`, 3).
    cnt3: Vec<u16>,
    cnt2: Vec<u16>,
    adj: Vec<Edge>, // n × MAX_EDGES
    wd: Vec<u8>,
    curves: Vec<[u16; W]>,
}

impl Infra {
    #[inline]
    fn c3(&self, idx: usize, line: usize) -> u16 {
        (self.cnt3[idx] >> (2 * line)) & 3
    }
    #[inline]
    fn c2(&self, idx: usize, line: usize) -> u16 {
        (self.cnt2[idx] >> (2 * line)) & 3
    }
    /// `wd + 2·nibble(g, d)` — the single-line-constrained base value.
    #[inline]
    fn base(&self, idx: usize, g: usize, d: usize) -> u32 {
        self.wd[idx] as u32 + 2 * ((self.curves[idx][g] >> (4 * (d - 1))) & 0xF) as u32
    }
}

fn threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8)
}

/// Run `f(chunk_start, chunk_end)` over `0..n` on all worker threads.
fn par_ranges(n: usize, f: impl Fn(usize, usize) + Sync) {
    let nt = threads();
    let chunk = n.div_ceil(nt);
    std::thread::scope(|s| {
        for w in 0..nt {
            let lo = w * chunk;
            let hi = ((w + 1) * chunk).min(n);
            if lo >= hi {
                continue;
            }
            let f = &f;
            s.spawn(move || f(lo, hi));
        }
    });
}

/// Build the shared infrastructure. `cwd_mm` supplies `wd` and the surcharge
/// curves per key (delta bases). ~4 GB resident, a few minutes.
pub fn build_infra(goal_key: u64, cwd_mm: &CwdMm) -> Infra {
    let t0 = std::time::Instant::now();
    // Key enumeration — identical to build_cwd_lm2's.
    let mut keys: Vec<u64> = Vec::with_capacity(N_WD_KEYS);
    let mut index: std::collections::HashMap<u64, u32> =
        std::collections::HashMap::with_capacity(N_WD_KEYS);
    keys.push(goal_key);
    index.insert(goal_key, 0);
    let mut qi = 0usize;
    while qi < keys.len() {
        let key = keys[qi];
        qi += 1;
        let (m, blank) = unpack(key);
        let b = blank as usize;
        for f in [b.wrapping_sub(1), b + 1] {
            if f >= W {
                continue;
            }
            for t in 0..W {
                if m[f][t] == 0 {
                    continue;
                }
                let mut m2 = m;
                m2[f][t] -= 1;
                m2[b][t] += 1;
                let pkey = pack(&m2, f as u8);
                if let std::collections::hash_map::Entry::Vacant(e) = index.entry(pkey) {
                    e.insert(keys.len() as u32);
                    keys.push(pkey);
                }
            }
        }
    }
    let n = keys.len();
    assert_eq!(n, N_WD_KEYS, "base WD key count mismatch");
    eprintln!("  lm1l infra: {n} keys enumerated in {:.0?}", t0.elapsed());

    let mut blank = vec![0u8; n];
    let mut cnt3 = vec![0u16; n];
    let mut cnt2 = vec![0u16; n];
    let mut adj = vec![Edge::INVALID; n * MAX_EDGES];
    let mut wd = vec![0u8; n];
    let mut curves = vec![[0u16; W]; n];
    {
        // SAFETY: each worker writes a disjoint index range of every array.
        struct Ptrs {
            blank: *mut u8,
            cnt3: *mut u16,
            cnt2: *mut u16,
            adj: *mut Edge,
            wd: *mut u8,
            curves: *mut [u16; W],
        }
        unsafe impl Sync for Ptrs {}
        let p = Ptrs {
            blank: blank.as_mut_ptr(),
            cnt3: cnt3.as_mut_ptr(),
            cnt2: cnt2.as_mut_ptr(),
            adj: adj.as_mut_ptr(),
            wd: wd.as_mut_ptr(),
            curves: curves.as_mut_ptr(),
        };
        let keys_ref = &keys;
        let index_ref = &index;
        let p = &p;
        par_ranges(n, |lo, hi| {
            for idx in lo..hi {
                let key = keys_ref[idx];
                let (m, bl) = unpack(key);
                let b = bl as usize;
                let (mut c3, mut c2) = (0u16, 0u16);
                for line in 0..W {
                    c3 |= (m[line][TT].min(3) as u16) << (2 * line);
                    c2 |= (m[line][TB].min(3) as u16) << (2 * line);
                }
                let cell = cwd_mm
                    .probe_cell(key)
                    .expect("every reachable key is in cwd_mm");
                unsafe {
                    *p.blank.add(idx) = bl;
                    *p.cnt3.add(idx) = c3;
                    *p.cnt2.add(idx) = c2;
                    *p.wd.add(idx) = cell.wd;
                    *p.curves.add(idx) = cell.curves;
                }
                for (dir, f) in [b.wrapping_sub(1), b + 1].into_iter().enumerate() {
                    if f >= W {
                        continue;
                    }
                    for t in 0..W {
                        if m[f][t] == 0 {
                            continue;
                        }
                        let mut m2 = m;
                        m2[f][t] -= 1;
                        m2[b][t] += 1;
                        let pidx = index_ref[&pack(&m2, f as u8)] as usize;
                        unsafe {
                            *p.adj.add(idx * MAX_EDGES + dir * W + t) = Edge::new(pidx, t, dir);
                        }
                    }
                }
            }
        });
    }
    drop(index);
    eprintln!(
        "  lm1l infra: adjacency + counts + bases in {:.0?}",
        t0.elapsed()
    );
    Infra {
        keys,
        blank,
        cnt3,
        cnt2,
        adj,
        wd,
        curves,
    }
}

// ------------------------------ layer engine ---------------------------------

/// Code space of a tracked-token product layer. `NC` codes per key; `preds`
/// enumerates the predecessor codes of one reversed edge, mirroring
/// `build_cwd_lm2`'s cases (i)/(ii)/(iii) with matrix lookups replaced by
/// the precomputed saturating counts of the predecessor key.
trait Space: Sync {
    const NC: usize;
    /// The fully-discharged code seeded at the goal key.
    fn goal_code() -> usize;
    #[allow(clippy::too_many_arguments)]
    fn preds(
        &self,
        infra: &Infra,
        pidx: usize,
        t: usize,
        b: usize,
        f: usize,
        code: usize,
        push: &mut impl FnMut(usize),
    );
}

/// Pair space: `(la, ca, lb, cb)` → 100 codes, `ca·50 + cb·25 + la·5 + lb`.
struct PairSpace;

impl Space for PairSpace {
    const NC: usize = 100;
    fn goal_code() -> usize {
        50 + 25 + TT * W + TB // (la=3, ca=true, lb=2, cb=true)
    }
    #[inline]
    fn preds(
        &self,
        infra: &Infra,
        pidx: usize,
        t: usize,
        b: usize,
        f: usize,
        cd: usize,
        push: &mut impl FnMut(usize),
    ) {
        let (ca, cb) = (cd >= 50, (cd % 50) >= 25);
        let (la, lb) = ((cd % 25) / W, cd % W);
        let code_of = |la: usize, ca: bool, lb: usize, cb: bool| {
            (ca as usize) * 50 + (cb as usize) * 25 + la * W + lb
        };
        // (i) moved token is neither tracked one
        let ok_a = if t == TT && la == b {
            infra.c3(pidx, b) >= 2
        } else {
            infra.c3(pidx, la) >= 1
        };
        let ok_b = if t == TB && lb == b {
            infra.c2(pidx, b) >= 2
        } else {
            infra.c2(pidx, lb) >= 1
        };
        if ok_a && ok_b {
            push(code_of(la, ca, lb, cb));
        }
        // (ii) moved token IS tracked A
        if t == TT && la == f {
            let crossing = b == 3 && f == 4;
            if crossing {
                if ca {
                    push(code_of(b, true, lb, cb));
                    push(code_of(b, false, lb, cb));
                }
            } else {
                push(code_of(b, ca, lb, cb));
            }
        }
        // (iii) moved token IS tracked B
        if t == TB && lb == f {
            let crossing = b == 2 && f == 3;
            if crossing {
                if cb {
                    push(code_of(la, ca, b, true));
                    push(code_of(la, ca, b, false));
                }
            } else {
                push(code_of(la, ca, b, cb));
            }
        }
    }
}

/// Single-tracked space for token type `T` (obligation: cross `T → T+1`
/// once, end in line `T`): `(la, ca)` → 10 codes, `ca·5 + la`.
struct SingleSpace<const T: usize>;

impl<const T: usize> Space for SingleSpace<T> {
    const NC: usize = 10;
    fn goal_code() -> usize {
        W + T // (la = T, ca = true)
    }
    #[inline]
    fn preds(
        &self,
        infra: &Infra,
        pidx: usize,
        t: usize,
        b: usize,
        f: usize,
        cd: usize,
        push: &mut impl FnMut(usize),
    ) {
        let (ca, la) = (cd >= W, cd % W);
        let cnt = |line: usize| {
            if T == TT {
                infra.c3(pidx, line)
            } else {
                infra.c2(pidx, line)
            }
        };
        // (i) moved token is not the tracked one
        let ok = if t == T && la == b {
            cnt(b) >= 2
        } else {
            cnt(la) >= 1
        };
        if ok {
            push((ca as usize) * W + la);
        }
        // (ii) moved token IS the tracked one
        if t == T && la == f {
            let crossing = b == T && f == T + 1;
            if crossing {
                if ca {
                    push(W + b);
                    push(b);
                }
            } else {
                push((ca as usize) * W + b);
            }
        }
    }
}

/// Atomic view of a `u8` distance array.
/// SAFETY: `AtomicU8` has the same layout as `u8`; the caller holds the
/// unique owner and all access during the parallel phase goes through this.
#[inline]
fn atomic_u8(v: &mut [u8]) -> &[AtomicU8] {
    unsafe { std::slice::from_raw_parts(v.as_mut_ptr() as *const AtomicU8, v.len()) }
}

/// CAS min-write: lower `dist[si]` to `v` if improving, and mark the key in
/// `bm`. 0xFF (UNSEEN) is greater than any depth, so it needs no special
/// case. Races are benign: the loser reloads and exits unless still
/// improving; the double bitmap OR is idempotent.
#[inline]
fn min_write(dist: &[AtomicU8], bm: &[AtomicU64], v: u8, kidx: usize, si: usize) {
    let a = &dist[si];
    let mut cur = a.load(Relaxed);
    while v < cur {
        match a.compare_exchange_weak(cur, v, Relaxed, Relaxed) {
            Ok(_) => {
                bm[kidx >> 6].fetch_or(1 << (kidx & 63), Relaxed);
                return;
            }
            Err(c) => cur = c,
        }
    }
}

/// Lazily-allocated per-depth seed bitmaps. A single flat bitmap would be a
/// correctness bug: a key's bit is consumed the round it is scanned, and
/// codes seeded at deeper depths would never be re-frontiered.
struct SeedBms {
    bms: Vec<OnceLock<Box<[AtomicU64]>>>,
    words: usize,
}

impl SeedBms {
    fn new(words: usize) -> SeedBms {
        SeedBms {
            bms: (0..256).map(|_| OnceLock::new()).collect(),
            words,
        }
    }
    #[inline]
    fn get(&self, depth: u8) -> &[AtomicU64] {
        self.bms[depth as usize].get_or_init(|| {
            (0..self.words)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }
    #[inline]
    fn peek(&self, depth: u8) -> Option<&[AtomicU64]> {
        self.bms[depth as usize].get().map(|b| &b[..])
    }
}

/// One relaxation round at depth `d`: scan every key in `cur` for codes at
/// exactly `d`, relax their reversed edges (optionally excluding the
/// `g`-escape edges) with min-writes of `d + 1` into `next`. Returns the
/// number of (key, code) states processed.
fn round<S: Space>(
    space: &S,
    infra: &Infra,
    dist: &[AtomicU8],
    cur: &[u64],
    next: &[AtomicU64],
    d: u8,
    skip_escapes_of: Option<usize>,
) -> u64 {
    let words = cur.len();
    let processed = AtomicU64::new(0);
    par_ranges(words, |lo, hi| {
        let mut local = 0u64;
        let mut active = [0u8; 100];
        for w in lo..hi {
            let mut bits = cur[w];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let idx = w * 64 + bit;
                if idx >= infra.keys.len() {
                    continue;
                }
                let base = idx * S::NC;
                let mut na = 0usize;
                for c in 0..S::NC {
                    if dist[base + c].load(Relaxed) == d {
                        active[na] = c as u8;
                        na += 1;
                    }
                }
                if na == 0 {
                    continue;
                }
                local += na as u64;
                let b = infra.blank[idx] as usize;
                let d1 = d.checked_add(1).expect("depth overflow vs 0xFF sentinel");
                for e in &infra.adj[idx * MAX_EDGES..(idx + 1) * MAX_EDGES] {
                    if !e.valid() {
                        continue;
                    }
                    let (pidx, t) = (e.pidx(), e.t());
                    if skip_escapes_of == Some(t) && b == t {
                        continue; // escape edge: enters only as a seed
                    }
                    let f = e.f(b);
                    for &c in &active[..na] {
                        space.preds(infra, pidx, t, b, f, c as usize, &mut |pc| {
                            min_write(dist, next, d1, pidx, pidx * S::NC + pc);
                        });
                    }
                }
            }
        }
        processed.fetch_add(local, Relaxed);
    });
    processed.into_inner()
}

/// Plain single-source backward BFS (the `r = 0` scaffolding): goal code at
/// key 0, no seeds, all edges relaxed.
fn build_r0<S: Space>(space: &S, infra: &Infra, dist: &mut [u8]) {
    let t0 = std::time::Instant::now();
    dist.fill(0xFF);
    dist[S::goal_code()] = 0;
    let n = infra.keys.len();
    let words = n.div_ceil(64);
    let mut cur = vec![0u64; words];
    cur[0] = 1;
    let next: Vec<AtomicU64> = (0..words).map(|_| AtomicU64::new(0)).collect();
    let da = atomic_u8(dist);
    let mut d: u8 = 0;
    loop {
        let processed = round(space, infra, da, &cur, &next, d, None);
        if processed == 0 && next.iter().all(|w| w.load(Relaxed) == 0) {
            break;
        }
        for (c, nx) in cur.iter_mut().zip(next.iter()) {
            *c = nx.swap(0, Relaxed);
        }
        d += 1;
    }
    eprintln!(
        "    r0 ({} codes/key): depth {d} in {:.0?}",
        S::NC,
        t0.elapsed()
    );
}

/// One demand layer: `dist` becomes `D_{g,r}` from complete `prev = D_{g,r-1}`.
fn build_layer<S: Space>(space: &S, infra: &Infra, g: usize, prev: &[u8], dist: &mut [u8]) {
    let t0 = std::time::Instant::now();
    dist.fill(0xFF);
    let n = infra.keys.len();
    let words = n.div_ceil(64);
    let da = atomic_u8(dist);
    let seeds = SeedBms::new(words);
    let max_seed = AtomicU8::new(0);

    // Seed sweep: keys with blank == g, escape edges (t == g), every live
    // code of `prev`, full predecessor case logic.
    par_ranges(n, |lo, hi| {
        for idx in lo..hi {
            if infra.blank[idx] as usize != g {
                continue;
            }
            let base = idx * S::NC;
            let b = g;
            for e in &infra.adj[idx * MAX_EDGES..(idx + 1) * MAX_EDGES] {
                if !e.valid() || e.t() != g {
                    continue;
                }
                let (pidx, t) = (e.pidx(), e.t());
                let f = e.f(b);
                for c in 0..S::NC {
                    let pv = prev[base + c];
                    if pv == 0xFF {
                        continue;
                    }
                    let v = pv.checked_add(1).expect("seed depth overflow");
                    max_seed.fetch_max(v, Relaxed);
                    space.preds(infra, pidx, t, b, f, c, &mut |pc| {
                        min_write(da, seeds.get(v), v, pidx, pidx * S::NC + pc);
                    });
                }
            }
        }
    });

    // Multi-source dial: relax non-escape edges only; a round's frontier is
    // the previous round's writes merged with this depth's seeds. Terminate
    // only when the frontier is empty AND no seed depth remains.
    let mut cur = vec![0u64; words];
    let next: Vec<AtomicU64> = (0..words).map(|_| AtomicU64::new(0)).collect();
    let mut max_depth = max_seed.into_inner();
    let mut d: u8 = 0;
    loop {
        let mut any = false;
        for (w, c) in cur.iter_mut().enumerate() {
            *c = next[w].swap(0, Relaxed);
            if let Some(bm) = seeds.peek(d) {
                *c |= bm[w].load(Relaxed);
            }
            any |= *c != 0;
        }
        if any {
            let processed = round(space, infra, da, &cur, &next, d, Some(g));
            if processed > 0 {
                max_depth = max_depth.max(d.checked_add(1).expect("depth overflow"));
            }
        }
        if d >= max_depth {
            break;
        }
        d += 1;
    }
    eprintln!(
        "    layer g={g}: max depth {max_depth} in {:.0?}",
        t0.elapsed()
    );
}

// ------------------------------ composition ----------------------------------

/// Inputs for one axis of the branch composition: the artifact payload, the
/// merged cell's `wd`/curves (delta bases), the demand vector, the
/// production term, and the zero-demand floors for this axis's variant
/// slots (`[single_final, single_19, pair]`, `or_`-chained like
/// `lm2_child`).
pub(crate) struct AxisIn<'a> {
    pub payload: &'a [u8],
    pub wd: u8,
    pub curves: [u16; W],
    pub dem: [u8; W],
    pub term: u8,
    pub floors: [u8; 3],
}

/// The four-branch single-demanded-line joint value:
/// `max(h_cwd, min(A, B, C, D))`. `lp` is the tracked-line vector
/// `[row20, col24, row15, row19, col19, col23]`. Shared by the engine's
/// consult path and the survivor gate so the two cannot diverge.
pub(crate) fn compose(row: &AxisIn, col: &AxisIn, lp: &[u8; 6], h_cwd: u8) -> u8 {
    let axis_val = |a: &AxisIn, v: usize, floor: u8| -> u8 {
        let mut best = floor;
        for g in 0..W {
            let dg = a.dem[g] as usize;
            if !(1..=4).contains(&dg) {
                continue;
            }
            let nib = ((a.curves[g] >> (4 * (dg - 1))) & 0xF) as u32;
            let lift = a.wd as u32 + 2 * nib + 2 * read_field(a.payload, field_of(g, dg, v)) as u32;
            best = best.max(lift.min(u8::MAX as u32) as u8);
        }
        best
    };
    let b0 = |a: &AxisIn, lb: usize| -> u8 {
        (a.wd as u32 + 2 * read_field(a.payload, field_b0(lb)) as u32).min(u8::MAX as u32) as u8
    };

    let va20 = if lp[0] <= 3 {
        axis_val(row, lp[0] as usize, row.floors[0])
    } else {
        row.term
    };
    let va19r = if lp[3] <= 3 {
        axis_val(row, lp[3] as usize, row.floors[1])
    } else {
        row.term
    };
    let pair_r = if lp[0] <= 3 && lp[2] <= 2 {
        axis_val(row, 7 + lp[0] as usize * 3 + lp[2] as usize, row.floors[2])
    } else if lp[2] <= 2 {
        // tb-only: the placement no other table covers.
        let fl = row.floors[2].max(b0(row, lp[2] as usize));
        axis_val(row, 4 + lp[2] as usize, fl)
    } else {
        va20
    };
    let vc24 = if lp[1] <= 3 {
        axis_val(col, lp[1] as usize, col.floors[0])
    } else {
        col.term
    };
    let vc19 = if lp[4] <= 3 {
        axis_val(col, lp[4] as usize, col.floors[1])
    } else {
        col.term
    };
    let pair_c = if lp[1] <= 3 && lp[5] <= 2 {
        axis_val(col, 7 + lp[1] as usize * 3 + lp[5] as usize, col.floors[2])
    } else if lp[5] <= 2 {
        let fl = col.floors[2].max(b0(col, lp[5] as usize));
        axis_val(col, 4 + lp[5] as usize, fl)
    } else {
        vc24
    };

    let ba = pair_r.saturating_add(col.term);
    let bb = va20.saturating_add(vc19);
    let bc = va19r.saturating_add(vc24);
    let bd = row.term.saturating_add(pair_c);
    h_cwd.max(ba.min(bb).min(bc).min(bd))
}

// ------------------------------ extraction -----------------------------------

/// Extract one `(g, d)` layer's queryable codes as 2-bit `min(delta/2, 3)`
/// fields into `out[idx * stride + off ..]`, `delta = D − base(g, d)`.
/// `codes` maps variant slot → code index; `D == 0xFF` (invalid placement)
/// writes 0 — the consult masks invalidity via the zero-demand tables.
#[allow(clippy::too_many_arguments)]
fn extract_layer(
    infra: &Infra,
    dist: &[u8],
    nc: usize,
    g: usize,
    d: usize,
    codes: &[usize],
    out: &mut [u8],
    stride: usize,
    bit_off: usize,
) {
    struct P(*mut u8);
    unsafe impl Sync for P {}
    let p = P(out.as_mut_ptr());
    let p = &p;
    let n = infra.keys.len();
    par_ranges(n, |lo, hi| {
        for idx in lo..hi {
            for (slot, &c) in codes.iter().enumerate() {
                let dv = dist[idx * nc + c];
                let field = if dv == 0xFF {
                    0
                } else {
                    let base = if d == 0 {
                        infra.wd[idx] as u32
                    } else {
                        infra.base(idx, g, d)
                    };
                    let delta = (dv as u32).checked_sub(base).unwrap_or_else(|| {
                        panic!("D < base at idx {idx} g {g} d {d} code {c}: {dv} < {base}")
                    });
                    assert!(delta % 2 == 0, "odd delta {delta} at idx {idx}");
                    ((delta / 2).min(3)) as u8
                };
                let bit = bit_off + slot * 2;
                let byte = idx * stride + bit / 8;
                let sh = bit % 8;
                // SAFETY: disjoint per-idx byte ranges across workers.
                unsafe {
                    let bp = p.0.add(byte);
                    *bp = (*bp & !(3 << sh)) | (field << sh);
                }
            }
        }
    });
}

// ------------------------------ the artifact ---------------------------------

const LM1L_SLOT: usize = 112; // 8 key + 96 payload + 8 pad
const LM1L_HEADER: usize = 16;

/// Fastrange home slot over the Fibonacci-mixed key (see `cwd::cwdm_home`).
#[inline]
fn lm1l_home(key: u64, slots: usize) -> usize {
    let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (((h as u128) * (slots as u128)) >> 64) as usize
}

/// The single-demanded-line joint table, mmap'd: magic `"CWJ1"`, dense 0.70
/// load, 112-byte slots (`key 8 + payload 96 + pad 8`), key 0 = empty
/// sentinel. Payload layout: 2-bit fields indexed by [`field_of`] /
/// [`field_b0`].
pub struct CwdLm1lMm {
    map: memmap2::Mmap,
    slots: usize,
}

impl CwdLm1lMm {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        // SAFETY: write-once-then-immutable build artifact.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        assert_eq!(&map[..4], b"CWJ1", "bad cwd_lm1l magic");
        let slots = u32::from_le_bytes(map[4..8].try_into().unwrap()) as usize;
        assert_eq!(
            map.len(),
            LM1L_HEADER + slots * LM1L_SLOT,
            "cwd_lm1l artifact has the wrong size for its header geometry"
        );
        // Eager pre-touch, one byte per 16 KiB page (Apple Silicon page
        // size), so demand-paging faults land at load time.
        let mut sum = 0u64;
        for off in (0..map.len()).step_by(16384) {
            sum = sum.wrapping_add(map[off] as u64);
        }
        std::hint::black_box(sum);
        Ok(CwdLm1lMm { map, slots })
    }

    /// One probe: the 96-byte 2-bit-field payload for `key`, `None` if
    /// unknown.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<&[u8]> {
        let mut i = lm1l_home(key, self.slots);
        loop {
            let off = LM1L_HEADER + i * LM1L_SLOT;
            let k = u64::from_le_bytes(self.map[off..off + 8].try_into().unwrap());
            if k == key {
                return Some(&self.map[off + 8..off + 8 + LM1L_PAYLOAD]);
            }
            if k == 0 {
                return None;
            }
            i += 1;
            if i == self.slots {
                i = 0;
            }
        }
    }
}

// ------------------------------ orchestration --------------------------------

/// Per-key payload stride during accumulation (same 96 bytes as the
/// artifact's payload).
const ACC_STRIDE: usize = LM1L_PAYLOAD;

/// Build everything: infra, the three tracked families (pair, single-A,
/// single-B) × (r=0 scaffolding + 20 demand layers each), accumulate the
/// 2-bit payload per key, and write the artifact. Budget: ~18 GB RAM peak,
/// ~7 GB transient disk (the pair `D₀` spill), hours of wall — see the
/// build test's ignore string.
pub fn build_cwd_lm1l(goal_key: u64, cwd_mm: &CwdMm, out_path: &Path) -> std::io::Result<()> {
    let t0 = std::time::Instant::now();
    let infra = build_infra(goal_key, cwd_mm);
    let n = infra.keys.len();
    let mut acc = vec![0u8; n * ACC_STRIDE];

    // Variant code lists (uncrossed blocks).
    let pair_codes: Vec<usize> = (0..4)
        .flat_map(|la| (0..3).map(move |lb| la * W + lb))
        .collect();
    let single_codes: Vec<usize> = (0..4).collect(); // la ∈ 0..4, uncrossed
    let single_b_codes: Vec<usize> = (0..3).collect();

    // --- singles (cheap: 10 codes/key) ---
    {
        let mut prev = vec![0u8; n * 10];
        let mut dist = vec![0u8; n * 10];
        // Single-A: r0 + layers. r0 values are already shipped in cwd_lm_mm;
        // only the layers' deltas are stored here.
        build_r0(&SingleSpace::<TT>, &infra, &mut prev);
        for g in 0..W {
            if g > 0 {
                // prev holds D_{g-1, 4}; rebuild D_0 (cheap for singles).
                build_r0(&SingleSpace::<TT>, &infra, &mut prev);
            }
            for d in 1..=ND {
                build_layer(&SingleSpace::<TT>, &infra, g, &prev, &mut dist);
                extract_layer(
                    &infra,
                    &dist,
                    10,
                    g,
                    d,
                    &single_codes,
                    &mut acc,
                    ACC_STRIDE,
                    field_of(g, d, 0) * 2,
                );
                std::mem::swap(&mut prev, &mut dist);
            }
        }
        // Single-B: r0 (stored — the "missing type-2 table") + layers.
        build_r0(&SingleSpace::<TB>, &infra, &mut prev);
        extract_layer(
            &infra,
            &prev,
            10,
            0,
            0,
            &single_b_codes,
            &mut acc,
            ACC_STRIDE,
            field_b0(0) * 2,
        );
        for g in 0..W {
            if g > 0 {
                build_r0(&SingleSpace::<TB>, &infra, &mut prev);
            }
            for d in 1..=ND {
                build_layer(&SingleSpace::<TB>, &infra, g, &prev, &mut dist);
                extract_layer(
                    &infra,
                    &dist,
                    10,
                    g,
                    d,
                    &single_b_codes,
                    &mut acc,
                    ACC_STRIDE,
                    field_of(g, d, 4) * 2,
                );
                std::mem::swap(&mut prev, &mut dist);
            }
        }
        eprintln!("  lm1l: singles done in {:.0?}", t0.elapsed());
    }

    // --- pairs (the long pole: 100 codes/key, D₀ spilled to disk) ---
    {
        let spill = out_path.with_extension("d0.tmp");
        let mut prev = vec![0u8; n * 100];
        let mut dist = vec![0u8; n * 100];
        build_r0(&PairSpace, &infra, &mut prev);
        std::fs::write(&spill, &prev)?;
        for g in 0..W {
            if g > 0 {
                let bytes = std::fs::read(&spill)?;
                prev.copy_from_slice(&bytes);
            }
            for d in 1..=ND {
                build_layer(&PairSpace, &infra, g, &prev, &mut dist);
                extract_layer(
                    &infra,
                    &dist,
                    100,
                    g,
                    d,
                    &pair_codes,
                    &mut acc,
                    ACC_STRIDE,
                    field_of(g, d, 7) * 2,
                );
                std::mem::swap(&mut prev, &mut dist);
            }
            eprintln!("  lm1l: pair line g={g} done in {:.0?}", t0.elapsed());
        }
        std::fs::remove_file(&spill)?;
    }

    // --- assemble the hash-slot artifact ---
    let slots = super::cwd::dense_slots(n);
    let mut buf = vec![0u8; LM1L_HEADER + slots * LM1L_SLOT];
    buf[..4].copy_from_slice(b"CWJ1");
    buf[4..8].copy_from_slice(&(slots as u32).to_le_bytes());
    buf[8..16].copy_from_slice(&(n as u64).to_le_bytes());
    for (idx, &k) in infra.keys.iter().enumerate() {
        assert_ne!(k, 0, "key 0 is the empty sentinel");
        let mut i = lm1l_home(k, slots);
        loop {
            let off = LM1L_HEADER + i * LM1L_SLOT;
            let cur = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            if cur == 0 {
                buf[off..off + 8].copy_from_slice(&k.to_le_bytes());
                buf[off + 8..off + 8 + ACC_STRIDE]
                    .copy_from_slice(&acc[idx * ACC_STRIDE..(idx + 1) * ACC_STRIDE]);
                break;
            }
            assert_ne!(cur, k, "duplicate key");
            i += 1;
            if i == slots {
                i = 0;
            }
        }
    }
    std::fs::write(out_path, &buf)?;
    eprintln!(
        "  lm1l: artifact {} ({} slots, {:.1} GB) in {:.0?}",
        out_path.display(),
        slots,
        buf.len() as f64 / 1e9,
        t0.elapsed()
    );
    Ok(())
}

/// The full build: three tracked families × (r0 + 20 demand layers), 2-bit
/// payload assembly, artifact write, reload, and validation against the
/// uncapped A\* oracle (hours, ~18 GB peak, ~7 GB transient disk; the
/// artifact is ~10.5 GB). If `out` already exists the build is skipped and
/// only the validation runs. Panics on any gate failure; run by
/// `build_cwd_artifacts lm1l`.
pub fn build_cwd_lm1l_artifact(cwd_mm_bin: &Path, out: &Path) {
    use crate::puzzle24::search::cwd::{goal_key, MergedBacking};
    use crate::puzzle24::search::cwd_lm_joint::reference_1l;

    let mm = CwdMm::load(cwd_mm_bin).expect("cwd_mm.bin");
    let gk = goal_key();
    if out.exists() {
        eprintln!("artifact present — skipping build, validating only");
    } else {
        build_cwd_lm1l(gk, &mm, out).expect("build");
    }

    let t = CwdLm1lMm::load(out).expect("reload");
    // Goal sanity: single-B r0 at the goal is 4 (base wd = 0 → field 2).
    let gp = t.probe(gk).expect("goal key present");
    assert_eq!(read_field(gp, field_b0(TB)), 2, "goal single-B r0 field");
    // Absent-key negative (key 1 is not a reachable contingency).
    assert!(t.probe(1).is_none());

    // Sampled oracle validation across all variant kinds.
    let backing = MergedBacking::Mm(&mm);
    let infra = build_infra(gk, &mm); // rebuilt for keys/bases (cheap vs build)
    let mut checked = 0u64;
    let mut skipped = 0u64;
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    while checked < 2_000 {
        let idx = (next() % infra.keys.len() as u64) as usize;
        let key = infra.keys[idx];
        let g = (next() % 5) as usize;
        let d = (next() % 4 + 1) as u8;
        let v = (next() % NV as u64) as usize;
        let (ta, tb) = if v < 4 {
            (Some(v as u8), None)
        } else if v < 7 {
            (None, Some((v - 4) as u8))
        } else {
            (Some(((v - 7) / 3) as u8), Some(((v - 7) % 3) as u8))
        };
        // Skip invalid placements: the oracle tracks phantoms, the
        // builder stores 0, and production never queries them.
        let ok_a = ta.is_none_or(|la| infra.c3(idx, la as usize) >= 1);
        let ok_b = tb.is_none_or(|lb| infra.c2(idx, lb as usize) >= 1);
        let payload = t.probe(key).expect("key present");
        let field = read_field(payload, field_of(g, d as usize, v));
        if !ok_a || !ok_b {
            assert_eq!(field, 0, "invalid placement must store 0");
            continue;
        }
        let Some(dv) = reference_1l(backing, gk, key, g, d, ta, tb) else {
            skipped += 1; // oracle pop budget — counted, never silent
            continue;
        };
        let base = infra.base(idx, g, d as usize);
        assert!(dv as u32 >= base, "oracle below base at key {key:#x}");
        let expect = (((dv as u32 - base) / 2).min(3)) as u8;
        assert_eq!(
            field, expect,
            "field mismatch key {key:#x} g {g} d {d} v {v} (D {dv}, base {base})"
        );
        checked += 1;
    }
    assert!(
        skipped * 20 < checked,
        "oracle budget-outs exceed 5%: {skipped} vs {checked}"
    );
    eprintln!(
        "build: {checked} sampled fields match the A* oracle ({skipped} budget-outs skipped)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::cwd::{goal_key, MergedBacking};
    use crate::puzzle24::search::cwd_lm::CwdLmMm;
    use crate::puzzle24::search::cwd_lm_joint::reference_1l;

    /// Fast pre-build gate for the layered builder: the single-tracked r=0
    /// pass must reproduce the shipped `cwd_lm_mm` single values, and one
    /// demand layer must match the uncapped A\* oracle on sampled keys.
    ///
    ///   cargo test --release lm1l_smoke_singles -- --ignored --nocapture
    #[test]
    #[ignore = "needs data/cwd_mm.bin + data/cwd_lm_mm.bin; ~10-20 min, ~5 GB peak"]
    fn lm1l_smoke_singles() {
        let mm = CwdMm::load(std::path::Path::new("data/cwd_mm.bin")).expect("cwd_mm.bin");
        let lm_mm = CwdLmMm::load(std::path::Path::new("data/cwd_lm_mm.bin")).expect("lm mm");
        let gk = goal_key();
        let infra = build_infra(gk, &mm);
        let n = infra.keys.len();

        // Single-A r0 must equal the shipped cwd_lm values (structural
        // validation of the enumeration, adjacency, counts, and case logic).
        let mut r0 = vec![0u8; n * 10];
        build_r0(&SingleSpace::<TT>, &infra, &mut r0);
        assert_eq!(r0[TT], 2, "goal: tracked A must leave line 3 and return");
        let mut checked = 0u64;
        for idx in (0..n).step_by(997) {
            let (single, _) = lm_mm.probe(infra.keys[idx]).expect("key in lm_mm");
            for la in 0..4 {
                let ours = r0[idx * 10 + la];
                assert_eq!(
                    ours, single[la],
                    "single-A r0 mismatch at idx {idx} la {la}"
                );
                checked += 1;
            }
        }
        eprintln!("smoke: single-A r0 matches cwd_lm_mm on {checked} sampled values");

        // Single-B r0 goal sanity (the mirror of cwd_lm's 2-move assert).
        let mut r0b = vec![0u8; n * 10];
        build_r0(&SingleSpace::<TB>, &infra, &mut r0b);
        // 4, not 2: the blank starts two lines away (4 → 3 → 2 and back).
        assert_eq!(r0b[TB], 4, "goal: tracked B round trip via the blank walk");

        // One demand layer vs the uncapped A* oracle on sampled keys.
        let (g, d) = (2usize, 1usize);
        let mut layer = vec![0u8; n * 10];
        build_layer(&SingleSpace::<TT>, &infra, g, &r0, &mut layer);
        let backing = MergedBacking::Mm(&mm);
        let mut cmp = 0u64;
        for idx in (0..n).step_by(1_500_017) {
            let key = infra.keys[idx];
            for la in 0..4u8 {
                let ours = layer[idx * 10 + la as usize];
                // The A* oracle does not validate placements (production
                // always tracks a real tile); an absent token is exactly the
                // builder's 0xFF.
                if infra.c3(idx, la as usize) == 0 {
                    assert_eq!(ours, 0xFF, "invalid placement must be 0xFF at idx {idx}");
                    continue;
                }
                let Some(v) = reference_1l(backing, gk, key, g, d as u8, Some(la), None) else {
                    continue; // oracle pop budget; smoke coverage is sampled anyway
                };
                assert_eq!(
                    ours, v,
                    "layer mismatch at idx {idx} la {la} (key {key:#x})"
                );
                let base = infra.base(idx, g, d);
                assert!(ours as u32 >= base && (ours as u32 - base) % 2 == 0);
                cmp += 1;
            }
        }
        eprintln!("smoke: layer (g={g},d={d}) matches the A* oracle on {cmp} sampled values");
    }

    /// Gate 3: the production table composition must reproduce the lm2j1l
    /// probe's prune set on survivors_148 — 778 at slack 0 (the single-line
    /// restriction's measured retention of the full joint's 780).
    ///
    ///   cargo test --release lm1l_survivor_gate -- --ignored --nocapture
    #[test]
    #[ignore = "needs data/cwd_mm.bin, data/cwd_lm_mm.bin, data/cwd_lm1l_mm.bin, data/survivors_148.txt"]
    fn lm1l_survivor_gate() {
        use crate::puzzle24::search::cwd::{project, surcharge_from_curves};
        let mm = CwdMm::load(std::path::Path::new("data/cwd_mm.bin")).expect("cwd_mm.bin");
        let lm_mm = CwdLmMm::load(std::path::Path::new("data/cwd_lm_mm.bin")).expect("lm mm");
        let t = CwdLm1lMm::load(std::path::Path::new("data/cwd_lm1l_mm.bin")).expect("lm1l mm");
        let text = std::fs::read_to_string("data/survivors_148.txt").expect("survivors");
        let mut prunes = [0u64; 3]; // slack 0 / 2 / 4
        let mut n = 0u64;
        for line in text.lines().filter(|l| l.contains("board=[")) {
            let field = |k: &str| -> u32 {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(k))
                    .and_then(|v| v.parse().ok())
                    .unwrap()
            };
            let (slack, hf) = (field("slack="), field("h="));
            let seg = &line[line.find("board=[").unwrap() + 7..];
            let seg = &seg[..seg.find(']').unwrap()];
            let vals: Vec<u8> = seg.split(',').map(|w| w.trim().parse().unwrap()).collect();
            let mut cells = [0u8; 25];
            cells.copy_from_slice(&vals);
            let s = crate::puzzle24::state::State(cells);
            n += 1;

            let (mr, br, dr, mc, bc, dc) = project(&s);
            let (rkey, ckey) = (pack(&mr, br), pack(&mc, bc));
            let rc = mm.probe_cell(rkey).expect("row reachable");
            let cc = mm.probe_cell(ckey).expect("col reachable");
            let rterm = rc.wd + surcharge_from_curves(&rc.curves, &dr);
            let cterm = cc.wd + surcharge_from_curves(&cc.curves, &dc);
            let h_cwd = rterm.saturating_add(cterm);
            assert_eq!(h_cwd as u32, hf, "baseline drift on {line}");
            let pos = |t: u8| s.0.iter().position(|&x| x == t).unwrap();
            let lp = [
                (pos(20) / W) as u8,
                (pos(24) % W) as u8,
                (pos(15) / W) as u8,
                (pos(19) / W) as u8,
                (pos(19) % W) as u8,
                (pos(23) % W) as u8,
            ];
            let or_ = |v: u8, fb: u8| if v != 0xFF { v } else { fb };
            let sv = |single: &[u8], l: u8| if l < 4 { single[l as usize] } else { 0xFF };
            let pv = |pair: &[u8], la: u8, lb: u8| {
                if la < 4 && lb < 3 {
                    pair[(la * 3 + lb) as usize]
                } else {
                    0xFF
                }
            };
            let (rsg, rpr) = lm_mm.probe(rkey).expect("row in lm_mm");
            let (csg, cpr) = lm_mm.probe(ckey).expect("col in lm_mm");
            let r20f = or_(sv(rsg, lp[0]), rterm);
            let c24f = or_(sv(csg, lp[1]), cterm);
            let rp = t.probe(rkey).expect("row in lm1l");
            let cp = t.probe(ckey).expect("col in lm1l");
            let row = AxisIn {
                payload: rp,
                wd: rc.wd,
                curves: rc.curves,
                dem: dr,
                term: rterm,
                floors: [
                    r20f,
                    or_(sv(rsg, lp[3]), rterm),
                    or_(pv(rpr, lp[0], lp[2]), r20f),
                ],
            };
            let col = AxisIn {
                payload: cp,
                wd: cc.wd,
                curves: cc.curves,
                dem: dc,
                term: cterm,
                floors: [
                    c24f,
                    or_(sv(csg, lp[4]), cterm),
                    or_(pv(cpr, lp[1], lp[5]), c24f),
                ],
            };
            let hj = compose(&row, &col, &lp, h_cwd);
            if hj as u32 > hf + slack {
                prunes[(slack / 2) as usize] += 1;
            }
        }
        eprintln!(
            "lm1l survivor gate: n={n}; prunes slack0/2/4 = {}/{}/{}",
            prunes[0], prunes[1], prunes[2]
        );
        assert_eq!(n, 3452, "sample size changed");
        assert_eq!(
            prunes[0], 778,
            "slack-0 prune set must match the lm2j1l probe"
        );
    }
}
