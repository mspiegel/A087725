//! Last-Move Escape-Constrained Walking Distance — the refined table.
//!
//! Extends the WD abstraction by tracking ONE type-3 token (tile 20 on the row
//! axis, tile 24 on the column axis) as an individual: its current line, and
//! whether it has yet made the 3→4 crossing that the last-move argument forces
//! (the blank's final entry into cell 24 ejects tile 20 or tile 24, so one of
//! them must visit line 4 and return home). The refined distance
//! `D(key, tile_line)` is the cost of an abstract plan that reaches the WD
//! goal with the tracked tile home (line 3) having crossed 3→4 at least once.
//!
//! `D ≥ WD` always; the branch value `D_row(key_r, row20) + WD-side col` (and
//! symmetrically for tile 24) lower-bounds solutions taking that branch, and
//! `max(cWD, min(branch20, branch24))` is admissible because every solution
//! takes a branch. Measured on 507 real prune-event boards from the 146
//! workload: this precomputable form certifies 43% of the +2s the k8 tables
//! prove (the joint-with-demands form reaches 66% but is not precomputable).
//!
//! Built by backward BFS from the goal over the product graph
//! (WD key × tile line × crossed flag), ~656 M states. The base WD move graph
//! is closed under move inversion, so reversed edges reuse the forward move
//! enumeration; only the monotone crossed flag needs directional handling.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering::Relaxed};

use crate::puzzle24::state::W;

/// Tracked token type: goal-line 3 (tiles 20 / 24).
const TT: usize = 3;

use super::cwd::{pack, unpack};

/// The refined table: `key → D(key, line)` for the crossing-not-yet-made
/// layer, `0xFF` = line unreachable for the tracked tile in this key (no
/// type-3 token in that line).
pub struct CwdLm {
    map: HashMap<u64, [u8; W]>,
}

impl CwdLm {
    /// Refined distance from `(key, tile_line)`; `None` if the state is
    /// invalid (no type-3 token in `tile_line`).
    #[inline]
    pub fn get(&self, key: u64, tile_line: usize) -> Option<u8> {
        let v = self.map.get(&key)?[tile_line];
        (v != 0xFF).then_some(v)
    }

    /// All five line values for `key` (0xFF = invalid line), or `None` if the
    /// key is unknown. One map probe; used by the engine's front cache to fill
    /// every line at once.
    #[inline]
    pub fn get_all(&self, key: u64) -> Option<&[u8; 5]> {
        self.map.get(&key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        w.write_all(b"CWLM")?;
        w.write_all(&(self.map.len() as u64).to_le_bytes())?;
        // Sorted-key order: byte-deterministic file, reproducible SHA pin.
        let mut keys: Vec<u64> = self.map.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            w.write_all(&k.to_le_bytes())?;
            w.write_all(&self.map[&k])?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut r = BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"CWLM", "bad cwd_lm magic");
        let mut n8 = [0u8; 8];
        r.read_exact(&mut n8)?;
        let n = u64::from_le_bytes(n8) as usize;
        let mut map = HashMap::with_capacity(n);
        let mut buf = [0u8; 13];
        for _ in 0..n {
            r.read_exact(&mut buf)?;
            let k = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let mut v = [0u8; W];
            v.copy_from_slice(&buf[8..13]);
            map.insert(k, v);
        }
        Ok(CwdLm { map })
    }
}

/// Build the refined table by BFS from the goal over reversed edges.
///
/// State `(key, line, crossed)`; distances are to the goal
/// `(goal_key, line = 3, crossed = true)`. The stored table is the
/// `crossed = false` layer (queries always start there); the `true` layer is
/// scaffolding.
pub fn build_cwd_lm(goal_key: u64) -> CwdLm {
    // dist[key] = [[u8; W]; 2] indexed [crossed][line]
    let mut dist: HashMap<u64, [[u8; W]; 2]> = HashMap::new();
    let unreach = [[0xFFu8; W]; 2];

    let mut frontier: Vec<(u64, u8, bool)> = Vec::new();
    let mut next: Vec<(u64, u8, bool)> = Vec::new();
    dist.entry(goal_key).or_insert(unreach)[1][TT] = 0;
    frontier.push((goal_key, TT as u8, true));

    let mut depth: u8 = 0;
    while !frontier.is_empty() {
        depth = depth.checked_add(1).expect("depth overflow");
        for &(key, line, crossed) in &frontier {
            // Enumerate base moves applicable to `key` — each is the inverse
            // of a forward move INTO this state. Forward move inverted:
            // token `t` moved from row `b` (this state's blank) into row `f`;
            // predecessor has blank at `f` and the token back in `b`... which
            // is exactly the standard move enumeration on `key` itself.
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
                    // Forward edge P → S moved token `t` from row `b` to row
                    // `f`?? No: P has blank at f; forward move takes token t
                    // from row b (P's adjacent) into f... Concretely: in P the
                    // token count of t at row b is m2[b][t] (≥1), and the
                    // forward move (t, from = b → to = f) yields S. Crossing
                    // iff t == TT, from-row b == 3 and to-row f == 4.
                    let crossing = t == TT && b == 3 && f == 4;
                    // Predecessor tile-line candidates:
                    // (i) tile not the moved token: lineP = line, crossedP =
                    //     crossed; invalid if that would require the moved
                    //     token to be the tile (t == TT && line-in-P == b and
                    //     no other type-3 token in P's row b): check count.
                    // (ii) tile IS the moved token: requires t == TT and
                    //     line == f (it arrived at f); lineP = b;
                    //     crossed = crossedP || crossing.
                    // (i)
                    {
                        let line_p = line as usize;
                        // In P, is the configuration consistent? The tile sits
                        // in line_p; P must have a type-3 token there.
                        let cnt = m2[line_p][TT];
                        let ok = if t == TT && line_p == b {
                            cnt >= 2 // moved token was a DIFFERENT type-3
                        } else {
                            cnt >= 1
                        };
                        if ok {
                            let e = dist.entry(pkey).or_insert(unreach);
                            let slot = &mut e[crossed as usize][line_p];
                            if *slot == 0xFF {
                                *slot = depth;
                                next.push((pkey, line, crossed));
                            }
                        }
                    }
                    // (ii)
                    if t == TT && line as usize == f {
                        let line_p = b;
                        if m2[line_p][TT] >= 1 {
                            let preds: &[bool] = if crossing {
                                if crossed {
                                    &[true, false]
                                } else {
                                    &[]
                                }
                            } else {
                                &[crossed]
                            };
                            for &cp in preds {
                                let e = dist.entry(pkey).or_insert(unreach);
                                let slot = &mut e[cp as usize][line_p];
                                if *slot == 0xFF {
                                    *slot = depth;
                                    next.push((pkey, line_p as u8, cp));
                                }
                            }
                        }
                    }
                }
            }
        }
        frontier.clear();
        std::mem::swap(&mut frontier, &mut next);
        if depth % 10 == 0 {
            eprintln!("  cwd_lm BFS depth {depth}: {} keys reached", dist.len());
        }
    }

    // Extract the crossed = false layer.
    let mut map = HashMap::with_capacity(dist.len());
    for (k, v) in dist {
        map.insert(k, v[0]);
    }
    CwdLm { map }
}

/// Tracked type for the penultimate-move obligation: goal-line 2 (tiles 15 /
/// 23 — the tiles settled by the move before last in the endgame branches
/// 14→19→24 and 22→23→24).
const TB: usize = 2;

/// The last-TWO-moves pair table: `key → D2(key, la, lb)` where `la` is the
/// tracked type-3 token's line (obligation: cross 3→4 once, end in line 3)
/// and `lb` the tracked type-2 token's line (cross 2→3 once, end in line 2).
/// Stored layer is (crossed=false, crossed=false); `0xFF` = no such token
/// placement in this key. Serves endgame branches A (row pair 20+15) and D
/// (col pair 24+23); branches B/C reuse the single-tracked [`CwdLm`].
pub struct CwdLm2 {
    map: HashMap<u64, [u8; 25]>,
}

impl CwdLm2 {
    /// Pair-refined distance from `(key, la, lb)`; `None` if the placement is
    /// invalid (missing token of either tracked type in its line).
    #[inline]
    pub fn get(&self, key: u64, la: usize, lb: usize) -> Option<u8> {
        let v = self.map.get(&key)?[la * W + lb];
        (v != 0xFF).then_some(v)
    }

    /// All 25 `(la, lb)` values for `key` (0xFF = invalid placement), or
    /// `None` if the key is unknown. One map probe, for the front cache.
    #[inline]
    pub fn get_all(&self, key: u64) -> Option<&[u8; 25]> {
        self.map.get(&key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        w.write_all(b"CWL2")?;
        w.write_all(&(self.map.len() as u64).to_le_bytes())?;
        // Sorted-key order: byte-deterministic file, reproducible SHA pin.
        let mut keys: Vec<u64> = self.map.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            w.write_all(&k.to_le_bytes())?;
            w.write_all(&self.map[&k])?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut r = BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"CWL2", "bad cwd_lm2 magic");
        let mut n8 = [0u8; 8];
        r.read_exact(&mut n8)?;
        let n = u64::from_le_bytes(n8) as usize;
        let mut map = HashMap::with_capacity(n);
        let mut buf = [0u8; 33];
        for _ in 0..n {
            r.read_exact(&mut buf)?;
            let k = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let mut v = [0u8; 25];
            v.copy_from_slice(&buf[8..33]);
            map.insert(k, v);
        }
        Ok(CwdLm2 { map })
    }
}

/// Number of WD-reachable keys (known from the wd24 build).
const N_WD_KEYS: usize = 65_650_495;

#[inline]
fn code_of(la: usize, ca: bool, lb: usize, cb: bool) -> usize {
    (ca as usize) * 50 + (cb as usize) * 25 + la * W + lb
}

/// Record predecessor `(pidx, pcode)` at depth `d1` through the layer's atomic
/// views. Two threads may race on one slot, but both store the SAME `d1`
/// (first discovery within the layer), so the outcome is deterministic; the
/// atomics only make the benign race defined behavior.
#[inline]
fn set_pred(dist: &[AtomicU8], nextbm: &[AtomicU64], d1: u8, pidx: usize, pcode: usize) {
    let s = &dist[pidx * 100 + pcode];
    if s.load(Relaxed) == 0xFF {
        s.store(d1, Relaxed);
        nextbm[pidx >> 6].fetch_or(1u64 << (pidx & 63), Relaxed);
    }
}

/// Build the pair table by BFS from the goal over reversed edges of the
/// product graph (WD key × A-line × A-crossed × B-line × B-crossed),
/// ~6.57 B states. Same edge inversion as [`build_cwd_lm`]; the two tracked
/// tokens have distinct types (3 and 2), so their interpretation cases never
/// interact on one edge. Frontier is a bitmap over key indices; per active
/// key the base-edge work (unpack/pack/index probe) is shared across all
/// 100 tracked codes.
pub fn build_cwd_lm2(goal_key: u64) -> CwdLm2 {
    // Base key enumeration: plain BFS over the WD graph (self-inverse moves).
    let t0 = std::time::Instant::now();
    let mut keys: Vec<u64> = Vec::with_capacity(N_WD_KEYS);
    let mut index: HashMap<u64, u32> = HashMap::with_capacity(N_WD_KEYS);
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
    eprintln!(
        "  cwd_lm2: {n} base keys enumerated in {:.0?}",
        t0.elapsed()
    );
    assert_eq!(n, N_WD_KEYS, "base WD key count mismatch");

    let mut dist = vec![0xFFu8; n * 100];
    let words = n.div_ceil(64);
    let mut cur = vec![0u64; words];
    let mut next = vec![0u64; words];
    dist[code_of(TT, true, TB, true)] = 0; // goal is idx 0
    cur[0] |= 1;

    // Layer expansion fans out over all cores: a shared cursor hands out
    // word-chunks of the frontier bitmap; layer writes go through atomic views
    // of `dist` / `next`. Within one layer every write to a slot stores the
    // SAME value (d+1, first discovery), so racing writers are benign and the
    // content is deterministic for any thread count; layers are barriers, and
    // the sorted-key save already makes the file bytes canonical.
    let nt = std::thread::available_parallelism().map_or(1, |p| p.get());
    const CHUNK: usize = 2048;
    let (keys_r, index_r) = (&keys, &index);
    let mut d: u8 = 0;
    let mut total_set: u64 = 1;
    loop {
        // SAFETY: AtomicU8 / AtomicU64 have the same size, alignment and bit
        // validity as u8 / u64, and the plain views are untouched while the
        // atomic views are live (worker threads are scoped below).
        let dist_a: &[AtomicU8] =
            unsafe { std::slice::from_raw_parts(dist.as_mut_ptr().cast(), dist.len()) };
        let next_a: &[AtomicU64] =
            unsafe { std::slice::from_raw_parts(next.as_mut_ptr().cast(), next.len()) };
        let cur_r: &[u64] = &cur;
        let cursor = AtomicUsize::new(0);
        let d1 = d.checked_add(1).expect("depth overflow");
        let layer: u64 = std::thread::scope(|s| {
            let handles: Vec<_> = (0..nt)
                .map(|_| {
                    s.spawn(|| {
                        let mut local: u64 = 0;
                        loop {
                            let w0 = cursor.fetch_add(CHUNK, Relaxed);
                            if w0 >= words {
                                break;
                            }
                            for w in w0..(w0 + CHUNK).min(words) {
                                let mut bits = cur_r[w];
                                while bits != 0 {
                                    let idx = (w << 6) + bits.trailing_zeros() as usize;
                                    bits &= bits - 1;
                                    let base = idx * 100;
                                    let mut active = [0u8; 100];
                                    let mut na = 0usize;
                                    for c in 0..100 {
                                        if dist_a[base + c].load(Relaxed) == d {
                                            active[na] = c as u8;
                                            na += 1;
                                        }
                                    }
                                    if na == 0 {
                                        continue;
                                    }
                                    let key = keys_r[idx];
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
                                            let pidx = index_r[&pkey] as usize;
                                            // forward move: token t, line b → line f
                                            let crossing_a = t == TT && b == 3 && f == 4;
                                            let crossing_b = t == TB && b == 2 && f == 3;
                                            for &cd in &active[..na] {
                                                let cd = cd as usize;
                                                let (ca, cb) = (cd >= 50, (cd % 50) >= 25);
                                                let (la, lb) = ((cd % 25) / W, cd % W);
                                                // (i) moved token is neither tracked one
                                                let ok_a = if t == TT && la == b {
                                                    m2[b][TT] >= 2
                                                } else {
                                                    m2[la][TT] >= 1
                                                };
                                                let ok_b = if t == TB && lb == b {
                                                    m2[b][TB] >= 2
                                                } else {
                                                    m2[lb][TB] >= 1
                                                };
                                                if ok_a && ok_b {
                                                    set_pred(
                                                        dist_a,
                                                        next_a,
                                                        d1,
                                                        pidx,
                                                        code_of(la, ca, lb, cb),
                                                    );
                                                }
                                                // (ii) moved token IS tracked A (type 3)
                                                if t == TT && la == f {
                                                    let preds: &[bool] = if crossing_a {
                                                        if ca {
                                                            &[true, false]
                                                        } else {
                                                            &[]
                                                        }
                                                    } else {
                                                        &[ca]
                                                    };
                                                    for &cap in preds {
                                                        set_pred(
                                                            dist_a,
                                                            next_a,
                                                            d1,
                                                            pidx,
                                                            code_of(b, cap, lb, cb),
                                                        );
                                                    }
                                                }
                                                // (iii) moved token IS tracked B (type 2)
                                                if t == TB && lb == f {
                                                    let preds: &[bool] = if crossing_b {
                                                        if cb {
                                                            &[true, false]
                                                        } else {
                                                            &[]
                                                        }
                                                    } else {
                                                        &[cb]
                                                    };
                                                    for &cbp in preds {
                                                        set_pred(
                                                            dist_a,
                                                            next_a,
                                                            d1,
                                                            pidx,
                                                            code_of(la, ca, b, cbp),
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    local += na as u64;
                                }
                            }
                        }
                        local
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        if layer == 0 {
            break;
        }
        total_set += layer;
        if d % 10 == 0 {
            eprintln!(
                "  cwd_lm2 BFS depth {d}: layer {layer}, total {total_set} states, {:.0?}",
                t0.elapsed()
            );
        }
        cur.copy_from_slice(&next);
        next.fill(0);
        d += 1;
    }
    eprintln!(
        "  cwd_lm2 BFS done at depth {d}: {total_set} states in {:.0?}",
        t0.elapsed()
    );

    // Extract the (crossed=false, crossed=false) layer: codes 0..25.
    let mut map = HashMap::with_capacity(n);
    for (idx, &k) in keys.iter().enumerate() {
        let mut v = [0u8; 25];
        v.copy_from_slice(&dist[idx * 100..idx * 100 + 25]);
        map.insert(k, v);
    }
    CwdLm2 { map }
}

// ------------------- combined mmap artifact (LM2 tier) -------------------

/// Hash-table geometry of the mmap artifact (`magic "CWMN"`): 32-byte slots
/// (key 8 + single 4 + pair 12 + pad), linear probing at 0.70 load (slot
/// count in header field \[4..8\]), key 0 = empty (the zero key decodes to
/// an impossible token matrix, so no real key collides with the sentinel).
/// Mirrors `cwd.rs`'s CWDN geometry; probes are always for reachable axis
/// keys, so only the successful-search chain length (~2.1 slots) matters.
const LMM_SLOT: usize = 32;
const LMM_HEADER: usize = 16;

/// Fastrange home slot over the Fibonacci-mixed key (see `cwd::cwdm_home`).
#[inline]
fn lmm_home(key: u64, slots: usize) -> usize {
    let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (((h as u128) * (slots as u128)) >> 64) as usize
}

/// The LM2 tier's zero-parse backing store: both tables' queryable values in
/// one open-addressing file, mmap'd so "loading" is a page-table operation
/// and a front-cache miss costs one linear-probe scan instead of two
/// HashMap probes. Slot payload mirrors the engine's Lm2Slot: 4 single-
/// tracked line values + the 12 queryable pair placements (`la*3 + lb`).
pub struct CwdLmMm {
    map: memmap2::Mmap,
    /// Slot count of this artifact's geometry (header field \[4..8\]).
    slots: usize,
}

impl CwdLmMm {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        // SAFETY: write-once-then-immutable build artifact.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        assert_eq!(
            &map[..4],
            b"CWMN",
            "bad cwd_lm_mm magic (legacy CWMM artifacts are no longer supported; \
             rebuild with build_cwd_lm_mm_artifact)"
        );
        let slots = u32::from_le_bytes(map[4..8].try_into().unwrap()) as usize;
        assert_eq!(
            map.len(),
            LMM_HEADER + slots * LMM_SLOT,
            "cwd_lm_mm artifact has the wrong size for its header geometry"
        );
        // Eager pre-touch: one byte per 16 KiB page (Apple Silicon page size),
        // sequential so readahead batches it. Moves all demand-paging faults
        // into load time instead of scattering them through the first search
        // thresholds (measured +1.2 s @144, +1.5 s @146 without this).
        let mut sum = 0u64;
        for off in (0..map.len()).step_by(16384) {
            sum = sum.wrapping_add(map[off] as u64);
        }
        std::hint::black_box(sum);
        Ok(CwdLmMm { map, slots })
    }

    /// One probe: `(single[0..4], pair[0..12])` for `key`, `None` if unknown.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<(&[u8], &[u8])> {
        let mut i = lmm_home(key, self.slots);
        loop {
            let off = LMM_HEADER + i * LMM_SLOT;
            let k = u64::from_le_bytes(self.map[off..off + 8].try_into().unwrap());
            if k == key {
                return Some((&self.map[off + 8..off + 12], &self.map[off + 12..off + 24]));
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

/// Build the mmap artifact from the two canonical tables. Writes the dense
/// geometry (`magic "CWMN"`, 0.70 load) directly — building and compaction
/// are one step.
pub fn build_cwd_lm_mm(lm: &CwdLm, lm2: &CwdLm2, path: &Path) -> std::io::Result<()> {
    let slots = super::cwd::dense_slots(lm2.map.len());
    let mut buf = vec![0u8; LMM_HEADER + slots * LMM_SLOT];
    buf[..4].copy_from_slice(b"CWMN");
    buf[4..8].copy_from_slice(
        &u32::try_from(slots)
            .expect("slot count fits u32")
            .to_le_bytes(),
    );
    buf[8..16].copy_from_slice(&(lm2.map.len() as u64).to_le_bytes());
    // Insert in sorted-key order so the artifact is byte-deterministic (the
    // open-addressed layout places colliding keys in insertion order, and
    // HashMap iteration order is process-random). Same recipe as build_wd24's
    // key-sorted layout; makes the SHA-256 pin reproducible across rebuilds.
    let mut keys: Vec<u64> = lm2.map.keys().copied().collect();
    keys.sort_unstable();
    let (mut occupied, mut max_chain) = (0u64, 0u32);
    for &k in &keys {
        let pv = &lm2.map[&k];
        assert_ne!(k, 0, "key 0 is the empty sentinel");
        let sv = lm.get_all(k).expect("key present in lm2 but not in lm");
        let mut i = lmm_home(k, slots);
        let mut chain = 1u32;
        loop {
            let off = LMM_HEADER + i * LMM_SLOT;
            let cur = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            if cur == 0 {
                buf[off..off + 8].copy_from_slice(&k.to_le_bytes());
                buf[off + 8..off + 12].copy_from_slice(&sv[..4]);
                for la in 0..4 {
                    for lb in 0..3 {
                        buf[off + 12 + la * 3 + lb] = pv[la * W + lb];
                    }
                }
                break;
            }
            assert_ne!(cur, k, "duplicate key");
            i += 1;
            if i == slots {
                i = 0;
            }
            chain += 1;
        }
        occupied += 1;
        max_chain = max_chain.max(chain);
    }
    eprintln!("  cwd_lm_mm: {occupied} keys placed, longest probe chain {max_chain}");
    std::fs::write(path, &buf)
}

/// Build the Last-Move refined table and save it to `out` (~66 M keys,
/// minutes, ~3 GB peak). Panics on the goal-sanity gate; run by
/// `build_cwd_artifacts lm`.
pub fn build_cwd_lm_table(out: &Path) {
    let gk = super::cwd::goal_key();
    let t0 = std::time::Instant::now();
    let lm = build_cwd_lm(gk);
    eprintln!("built {} keys in {:.0?}", lm.len(), t0.elapsed());
    // Sanity: from the goal, the tracked tile must leave line 3 and
    // return — exactly 2 abstract moves.
    assert_eq!(lm.get(gk, 3), Some(2), "D(goal, line 3) must be 2");
    lm.save(out).expect("save");
    eprintln!("saved {}", out.display());
}

/// Build the last-two-moves pair table and save it to `out` (~6.6 B product
/// states, tens of minutes, ~9 GB peak). If the single table at `lm_bin`
/// exists, gates pair ≥ single dominance on a sample. Run by
/// `build_cwd_artifacts lm2`.
pub fn build_cwd_lm2_table(lm_bin: &Path, out: &Path) {
    let gk = super::cwd::goal_key();
    let t0 = std::time::Instant::now();
    let lm2 = build_cwd_lm2(gk);
    eprintln!("built {} keys in {:.0?}", lm2.len(), t0.elapsed());
    // Sanity: from the goal, each tracked token must make its excursion
    // and return — two independent 2-move round trips.
    assert_eq!(lm2.get(gk, 3, 2), Some(4), "D2(goal, 3, 2) must be 4");
    // Pair dominates single on every placement where both are defined.
    if let Ok(lm) = CwdLm::load(lm_bin) {
        let mut checked = 0u64;
        for (&k, v2) in lm2.map.iter().take(200_000) {
            if let Some(v1) = lm.get_all(k) {
                for la in 0..W {
                    for lb in 0..W {
                        let (a, b) = (v2[la * W + lb], v1[la]);
                        if a != 0xFF && b != 0xFF {
                            assert!(a >= b, "pair < single at key {k:#x} ({la},{lb})");
                            checked += 1;
                        }
                    }
                }
            }
        }
        eprintln!("pair>=single dominance checked on {checked} placements");
    }
    lm2.save(out).expect("save");
    eprintln!("saved {}", out.display());
}

/// Build the combined mmap artifact at `out` from the two canonical tables
/// (~4 GiB written) and verify every key round-trips. Run by
/// `build_cwd_artifacts lm-mm`.
pub fn build_cwd_lm_mm_artifact(lm_bin: &Path, lm2_bin: &Path, out: &Path) {
    let t0 = std::time::Instant::now();
    let lm = CwdLm::load(lm_bin).expect("cwd_lm.bin");
    let lm2 = CwdLm2::load(lm2_bin).expect("cwd_lm2.bin");
    eprintln!("tables loaded in {:.0?}", t0.elapsed());
    build_cwd_lm_mm(&lm, &lm2, out).expect("build");
    let mm = CwdLmMm::load(out).expect("load");
    let mut n = 0u64;
    for (&k, pv) in &lm2.map {
        let (s, p) = mm.probe(k).expect("every key must probe");
        assert_eq!(s, &lm.get_all(k).unwrap()[..4], "single mismatch at {k:#x}");
        for la in 0..4 {
            for lb in 0..3 {
                assert_eq!(p[la * 3 + lb], pv[la * W + lb], "pair mismatch at {k:#x}");
            }
        }
        n += 1;
    }
    assert!(!lm2.map.contains_key(&1));
    assert!(mm.probe(1).is_none(), "absent key must return None");
    eprintln!("verified {n} keys round-trip in {:.0?} total", t0.elapsed());
}
