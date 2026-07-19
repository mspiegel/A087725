//! The 1-bit ZPDB codec (Clausecker & Reinefeld, SOCS 2019).
//!
//! Stores **bit 1** of the absolute distance `h` — `(h >> 1) & 1` — per
//! `(m, p, r)` entry, packed 8 to a byte. Bit 0 (parity) is recovered from
//! the bipartite structure of the abstract ZPDB graph: every edge moves one
//! pattern tile by ±1 (horizontal) or ±5 (vertical) cells — both odd offsets
//! — so the parity of the sum of pattern-tile cell indices flips on every
//! edge. Stored XOR'd with the goal-shape's parity in
//! [`ZpdbLayout::shape_parity`], so
//!
//! ```text
//! parity(h(state)) == shape_parity[shape_of(state)]
//! ```
//!
//! at every reachable entry (the goal's `h = 0` is even, and induction
//! propagates the invariant along every edge). Combining that bit with the
//! stored bit yields `h mod 4`, which the differential step `diff_lookup`
//! turns into the exact `h` along any search path (`Δh = ±1`).
//!
//! See `docs/zpdb-codec-spec.md` §3.

use super::zpdb::ZpdbLayout;

/// Pack a byte-distance array into 1-bit form. Output byte `i / 8`, bit
/// `i % 8` = `(dist[i] >> 1) & 1`. Output length is `(dist.len() + 7) / 8`;
/// for a `dist.len()` not a multiple of 8 the unused high bits of the last
/// byte are zero.
///
/// For built ZPDBs every entry is reachable, so `dist[i] != UNVISITED`; the
/// `debug_assert` catches a malformed input.
pub fn pack_bits(dist: &[u8]) -> Vec<u8> {
    let n = dist.len();
    let mut packed = vec![0u8; n.div_ceil(8)];
    for (i, &d) in dist.iter().enumerate() {
        debug_assert!(d != u8::MAX, "ZPDB has UNVISITED at index {i}");
        let bit = (d >> 1) & 1;
        packed[i / 8] |= bit << (i % 8);
    }
    packed
}

/// Compact a 2-bit-per-state working array (from the frontier-free build) into
/// the 1-bit codec table **in place**. Each 2-bit field is `0b1<bit1>` for a
/// visited state, so its low bit is exactly the stored codec bit `(h >> 1) & 1`
/// (identical to [`pack_bits`]). `buf` holds `(total + 3) / 4` bytes on entry;
/// on return its first `(total + 7) / 8` bytes are the packed table and it is
/// truncated to that length.
///
/// The forward pass is safe in place: output byte `w` reads only input bytes
/// `2w` and `2w+1` and writes byte `w ≤ 2w`, and byte `w` is never read again by
/// any later output byte (which start at input byte `2(w+1) > w`).
pub fn pack1_from_2bit_inplace(buf: &mut Vec<u8>, total: usize) {
    let out_len = total.div_ceil(8);
    debug_assert!(buf.len() >= total.div_ceil(4), "2-bit buffer too small");
    for w in 0..out_len {
        let mut byte = 0u8;
        for b in 0..8usize {
            let i = w * 8 + b;
            if i >= total {
                break;
            }
            let field = (buf[i / 4] >> (2 * (i % 4))) & 0b11;
            byte |= (field & 1) << b;
        }
        buf[w] = byte;
    }
    buf.truncate(out_len);
}

/// Diagnostic inverse of [`pack_bits`] *up to the stored bit*: returns `len`
/// values each in `{0, 2}` (the stored bit 1 shifted into value position).
/// Used by tests and the round-trip check; hot-path code uses
/// [`lookup_bit`] / [`diff_lookup`] directly.
pub fn unpack_to_bit_values(packed: &[u8], len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = ((packed[i / 8] >> (i % 8)) & 1) << 1;
    }
    out
}

/// Stored bit at `idx`, shifted into value position: returns `0` or `2`.
#[inline]
pub fn lookup_bit(packed: &[u8], idx: u64) -> u8 {
    let i = idx as usize;
    ((packed[i / 8] >> (i % 8)) & 1) << 1
}

/// Differential lookup: given the current state's absolute `old_h` and the
/// neighbor's index `idx`, returns the neighbor's absolute `h` — always
/// `old_h + 1` or `old_h - 1` (the quotient graph is bipartite with
/// `Δh = ±1`).
///
/// Formula (Clausecker & Reinefeld §3):
///
/// ```text
/// next_h = old_h + 1 - ((entry ^ old_h ^ (old_h << 1)) & 2)
/// ```
///
/// where `entry = lookup_bit(packed, idx)`. Derivation: `old_h + 1` and
/// `old_h - 1` differ in bit 1; the neighbor's stored bit selects between
/// them, and `old_h << 1` injects bit 0 (parity) for the carry/borrow. The
/// `(old << 1)` cast to `u32` is to avoid a u8 overflow at large `old_h`;
/// only bits 0-1 matter for the final result.
#[inline]
pub fn diff_lookup(packed: &[u8], idx: u64, old_h: u8) -> u8 {
    let entry = lookup_bit(packed, idx) as u32;
    let oh = old_h as u32;
    (oh + 1 - ((entry ^ oh ^ (oh << 1)) & 2)) as u8
}

/// Parity of a permutation given its Lehmer rank over `k` items: `0` (even)
/// or `1` (odd). Walk the Lehmer digits MSB→LSB; the inversion count is the
/// sum of digits, and parity is its sum mod 2.
///
/// Exposed as a numeric utility; the bipartite parity invariant the 1-bit
/// codec relies on is [`ZpdbLayout::shape_parity`], **not** this — moving a
/// pattern tile vertically can pass other pattern tiles, so perm parity is
/// not a clean per-edge invariant.
pub fn perm_parity(rank: u64, k: usize) -> u8 {
    if k <= 1 {
        return 0;
    }
    // fact = (k-1)! initially, then drops to (k-2)!, (k-3)!, ..., 0! = 1.
    let mut fact: u64 = (1..k as u64).product();
    let mut r = rank;
    let mut par: u8 = 0;
    for i in 0..k {
        let digit = r / fact;
        r %= fact;
        par ^= (digit as u8) & 1;
        if i + 1 < k {
            fact /= (k - 1 - i) as u64;
        }
    }
    par
}

/// Decompose a global ZPDB index into `(shape_idx, perm_rank, region)`.
///
/// The cohort layout is `cohort_base[shape] + perm_rank * regions(shape) + r`,
/// so the inverse is a binary search for the cohort followed by a divmod.
pub fn unrank_into_cohort(layout: &ZpdbLayout, idx: u64) -> (usize, u64, u8) {
    let base = layout.cohort_base();
    let s = match base.binary_search(&idx) {
        Ok(s) => s,
        Err(s) => s - 1,
    };
    let off = idx - base[s];
    let count = layout.region_counts()[s] as u64;
    let p = off / count;
    let r = (off % count) as u8;
    (s, p, r)
}

/// Parity bit 0 of `h(idx)` derived from the index alone — the bipartite
/// invariant of the abstract ZPDB graph. Equals `dist[idx] & 1` at every
/// reachable entry. O(log C(25, k)) via [`unrank_into_cohort`] +
/// [`ZpdbLayout::shape_parity`].
#[inline]
pub fn parity_of_h(layout: &ZpdbLayout, idx: u64) -> u8 {
    let (s, _, _) = unrank_into_cohort(layout, idx);
    layout.shape_parity()[s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::pdb::pattern::Pattern;
    use crate::puzzle24::pdb::zbuild::{build_zpdb, gen_moves};

    /// Round-trip the bit-1 values: pack → unpack returns each `(d >> 1) & 1`
    /// shifted into value position.
    #[test]
    fn pack_unpack_round_trip_on_hand_built_array() {
        // Deliberately span all four h mod 4 classes; lengths /8 and /8±1.
        let dist: Vec<u8> = (0u8..73).collect();
        let packed = pack_bits(&dist);
        let recovered = unpack_to_bit_values(&packed, dist.len());
        for (i, &d) in dist.iter().enumerate() {
            let want = ((d >> 1) & 1) << 1;
            assert_eq!(recovered[i], want, "round-trip differs at i={i} (d={d})");
            assert_eq!(
                lookup_bit(&packed, i as u64),
                want,
                "lookup_bit differs at i={i}"
            );
        }
    }

    #[test]
    fn pack_length_pads_partial_byte() {
        // 13 entries → 2 bytes (ceil 13/8).
        let packed = pack_bits(&[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]);
        assert_eq!(packed.len(), 2);
        // Bit at index 12 (d=0 → 0) sits in byte 1 bit 4.
        assert_eq!(packed[1] >> 4 & 1, 0);
    }

    /// Hand-verified rows from the formula derivation: for every `(old_h,
    /// next_h)` pair with `|next_h - old_h| = 1` and `old_h, next_h ≤ 5`,
    /// `diff_lookup` must reconstruct `next_h` from the encoded bit 1 of
    /// `next_h` and the absolute `old_h`.
    #[test]
    fn diff_lookup_reconstructs_neighbor_h_on_unit_edges() {
        // packed[0] bit 0 holds the stored bit 1 of next_h.
        for (old_h, next_h) in [
            (0u8, 1u8),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
        ] {
            // Build a single-entry packed array whose bit-1 encodes next_h.
            let packed = pack_bits(&[next_h]);
            let got = diff_lookup(&packed, 0, old_h);
            assert_eq!(
                got,
                next_h,
                "diff_lookup({}→?, bit={}) = {} (wanted {})",
                old_h,
                (next_h >> 1) & 1,
                got,
                next_h
            );
        }
    }

    /// For every valid `(old_h, next_h)` with `|Δ| = 1`, the formula must
    /// reconstruct `next_h` from the encoded bit-1-of-`next_h` and `old_h` —
    /// across the full bit-1 cycle of length 4 in `h mod 4` (so the
    /// `(old_h << 1)` carry/borrow injection is exercised on both edges of
    /// each h-mod-4 boundary).
    ///
    /// We do **not** test arbitrary `(old_h, bit)` combinations: bipartite
    /// neighbors only ever differ by ±1, so a state at `old_h = 0` has no
    /// valid neighbor at `h = -1`, and feeding the corresponding bit pattern
    /// to the formula is meaningless (it underflows by design).
    #[test]
    fn diff_lookup_always_lands_one_step_away() {
        for old_h in 0u8..60 {
            for &delta in &[1i8, -1i8] {
                if delta == -1 && old_h == 0 {
                    continue;
                }
                let next_h = (old_h as i16 + delta as i16) as u8;
                let bit = (next_h >> 1) & 1;
                let packed = [bit];
                let got = diff_lookup(&packed, 0, old_h);
                assert_eq!(
                    got, next_h,
                    "diff_lookup(old_h={old_h}, Δ={delta}) = {got} (wanted {next_h})"
                );
            }
        }
    }

    /// `perm_parity(rank, k)` must match the inversion-count parity for every
    /// rank, for small `k`. Cross-checks the closed-form helper against the
    /// definition.
    #[test]
    fn perm_parity_matches_inversion_count_small_k() {
        for k in 1usize..=5 {
            let kfact: u64 = (1..=k as u64).product();
            for rank in 0..kfact {
                // Unrank → permutation.
                let mut digits = vec![0usize; k];
                let mut r = rank;
                let mut f: u64 = (1..k as u64).product();
                for i in 0..k {
                    digits[i] = (r / f) as usize;
                    r %= f;
                    if i + 1 < k {
                        f /= (k - 1 - i) as u64;
                    }
                }
                let mut remaining: Vec<u8> = (0..k as u8).collect();
                let mut perm = Vec::with_capacity(k);
                for d in digits {
                    perm.push(remaining.remove(d));
                }
                // Inversion count over `perm`.
                let mut inv = 0u32;
                for i in 0..k {
                    for j in (i + 1)..k {
                        if perm[i] > perm[j] {
                            inv += 1;
                        }
                    }
                }
                assert_eq!(
                    perm_parity(rank, k),
                    (inv & 1) as u8,
                    "k={k} rank={rank} perm={perm:?} inv={inv}"
                );
            }
        }
    }

    #[test]
    fn perm_parity_trivial_k01() {
        assert_eq!(perm_parity(0, 0), 0);
        assert_eq!(perm_parity(0, 1), 0);
    }

    /// `unrank_into_cohort` is the inverse of the rank composition: starting
    /// from every visited `(m, p, r)`, the recomposed index must equal the
    /// original. We piggy-back on a tiny built ZPDB for the indices.
    #[test]
    fn unrank_into_cohort_inverts_rank_composition_k2() {
        let pattern = Pattern::new(&[1, 2]);
        let (dist, layout) = build_zpdb(pattern);
        for (idx, _) in dist.iter().enumerate() {
            let idx = idx as u64;
            let (s, p, r) = unrank_into_cohort(&layout, idx);
            let recomposed =
                layout.cohort_base()[s] + p * layout.region_counts()[s] as u64 + r as u64;
            assert_eq!(recomposed, idx, "unrank/rank not inverse at idx={idx}");
        }
    }

    /// End-to-end on a real (small) ZPDB: every neighbor pair `(s, ns)` must
    /// satisfy `diff_lookup(packed, rank(ns), dist[rank(s)]) == dist[rank(ns)]`.
    /// This is the codec's single most important correctness invariant — the
    /// differential reconstruction matches the absolute distance.
    #[test]
    fn diff_lookup_matches_dist_over_every_edge_k2() {
        assert_diff_lookup_matches_dist(Pattern::new(&[1, 2]));
    }

    #[test]
    fn diff_lookup_matches_dist_over_every_edge_k3() {
        assert_diff_lookup_matches_dist(Pattern::new(&[1, 7, 13]));
    }

    /// The bipartite parity invariant: at every reachable entry,
    /// `parity_of_h(idx) == dist[idx] & 1` — the bit the 1-bit codec does
    /// NOT store, recovered from the index via the shape's cell-sum parity.
    #[test]
    fn parity_of_h_matches_dist_parity_at_every_entry_k3() {
        let pattern = Pattern::new(&[1, 7, 13]);
        let (dist, layout) = build_zpdb(pattern);
        for (idx, &d) in dist.iter().enumerate() {
            assert_ne!(d, u8::MAX, "unvisited entry at idx={idx}");
            let par = parity_of_h(&layout, idx as u64);
            assert_eq!(
                par,
                d & 1,
                "parity_of_h({}) = {} but dist = {} (parity {})",
                idx,
                par,
                d,
                d & 1
            );
        }
    }

    // --- helpers --------------------------------------------------------

    fn assert_diff_lookup_matches_dist(pattern: Pattern) {
        let (dist, layout) = build_zpdb(pattern);
        let packed = pack_bits(&dist);

        // BFS over the abstract ZPDB graph from the goal projection so we
        // enumerate reachable states once each, recovering the
        // `idx → ProjectedState` mapping the codec doesn't otherwise expose.
        use super::super::pattern::ProjectedState;
        use super::super::zpdb::{regions, OCCUPIED};
        let goal = ProjectedState::goal(pattern);
        let mut occ = 0u32;
        for (c, &v) in goal.cells.iter().enumerate() {
            if v != 0 && v != crate::puzzle24::pdb::pattern::ANON {
                occ |= 1u32 << c;
            }
        }
        let (count, labels) = regions(occ);
        let mut base = [crate::puzzle24::pdb::pattern::ANON; 25];
        for (c, &v) in goal.cells.iter().enumerate() {
            if v != 0 && v != crate::puzzle24::pdb::pattern::ANON {
                base[c] = v;
            }
        }
        let mut seeds: Vec<ProjectedState> = Vec::new();
        let mut rep = vec![usize::MAX; count as usize];
        for (c, &l) in labels.iter().enumerate() {
            if l != OCCUPIED && rep[l as usize] == usize::MAX {
                rep[l as usize] = c;
            }
        }
        for &rc in &rep {
            let mut nc = base;
            nc[rc] = 0;
            seeds.push(ProjectedState::from_projection(nc));
        }

        let mut visited = vec![false; dist.len()];
        let mut frontier: Vec<ProjectedState> = Vec::new();
        for s in &seeds {
            let i = layout.rank(s, pattern) as usize;
            if !visited[i] {
                visited[i] = true;
                frontier.push(*s);
            }
        }
        let mut succ: Vec<ProjectedState> = Vec::with_capacity(16);
        while let Some(s) = frontier.pop() {
            let s_idx = layout.rank(&s, pattern);
            let s_h = dist[s_idx as usize];
            succ.clear();
            gen_moves(&layout, &s, &mut succ);
            for ns in succ.drain(..) {
                let n_idx = layout.rank(&ns, pattern);
                let n_h = dist[n_idx as usize];
                // The codec's defining invariant.
                let got = diff_lookup(&packed, n_idx, s_h);
                assert_eq!(
                    got, n_h,
                    "diff_lookup mismatch: s_idx={s_idx} s_h={s_h} → n_idx={n_idx} got {got} want {n_h}"
                );
                // The bipartite Δ=±1 property — already implied by
                // diff_lookup landing on n_h, but assert it explicitly so a
                // future divergence here surfaces clearly.
                let delta = n_h as i32 - s_h as i32;
                assert!(
                    delta == 1 || delta == -1,
                    "non-bipartite step: s_h={s_h} n_h={n_h}"
                );
                if !visited[n_idx as usize] {
                    visited[n_idx as usize] = true;
                    frontier.push(ns);
                }
            }
        }
        // Sanity: BFS covered every entry the build BFS covered.
        for (i, &v) in visited.iter().enumerate() {
            assert!(v, "BFS missed reachable entry {i} (build reached it)");
        }
    }
}
