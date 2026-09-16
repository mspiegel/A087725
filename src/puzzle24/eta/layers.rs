//! Exact breadth-first layers `V_k = { v : d(v, GOAL) = k }` for small `k`.
//!
//! The state graph is bipartite (each move changes the blank's colour), so a
//! layer has no internal edges and `V_{k+1} = N(V_k) \ V_{k−1}`: only two
//! layers are held, as sorted, deduplicated `Vec<u128>` of packed boards.
//! Memory is ~16 B per state plus the raw neighbour list, which puts k ≈ 22
//! (6.9×10⁷ states) comfortably inside 32 GB.
//!
//! Layer sizes are checked against OEIS A090031, the 24-puzzle's sphere sizes
//! with the blank starting in a corner.

use crate::puzzle24::state::{State, N_CELLS, W};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// OEIS A090031: number of 24-puzzle states at exactly `k` moves from a
/// corner-blank goal, `k = 0..=30` (as tabulated in Clausecker, ZIB Report
/// 20-17, App. B).
pub const A090031: [u64; 31] = [
    1,
    2,
    4,
    10,
    26,
    64,
    159,
    366,
    862,
    1_904,
    4_538,
    10_238,
    24_098,
    53_186,
    123_435,
    268_416,
    616_374,
    1_326_882,
    3_021_126,
    6_438_828,
    14_524_718,
    30_633_586,
    68_513_713,
    143_106_496,
    317_305_688,
    656_178_756,
    1_442_068_376,
    2_951_523_620,
    6_427_133_737,
    13_014_920_506,
    28_070_588_413,
];

/// Pack a board as 5 bits per cell, cell 0 in the low bits. A bijection on
/// boards (125 bits), cheaper than [`rank`](crate::puzzle24::rank) and
/// order-agnostic, which is all sorting and deduplication need.
#[inline]
pub fn pack(s: &State) -> u128 {
    let mut k = 0u128;
    for (i, &t) in s.0.iter().enumerate() {
        k |= (t as u128) << (5 * i);
    }
    k
}

/// Inverse of [`pack`].
#[inline]
pub fn unpack(k: u128) -> State {
    let mut c = [0u8; N_CELLS];
    for (i, ci) in c.iter_mut().enumerate() {
        *ci = ((k >> (5 * i)) & 31) as u8;
    }
    State(c)
}

#[inline]
fn blank_of(k: u128) -> usize {
    (0..N_CELLS)
        .find(|&i| (k >> (5 * i)) & 31 == 0)
        .expect("packed board has no blank")
}

/// The 2–4 neighbours of a packed board, as `(array, count)`.
#[inline]
pub fn neighbours(k: u128) -> ([u128; 4], usize) {
    let b = blank_of(k);
    let (r, c) = (b / W, b % W);
    let mut out = [0u128; 4];
    let mut n = 0;
    let mut push = |nb: usize| {
        let tile = (k >> (5 * nb)) & 31;
        // Move `tile` from cell nb into the blank cell b; nb becomes the blank.
        out[n] = (k & !(31u128 << (5 * nb))) | (tile << (5 * b));
        n += 1;
    };
    if r > 0 {
        push(b - W);
    }
    if r + 1 < W {
        push(b + W);
    }
    if c > 0 {
        push(b - 1);
    }
    if c + 1 < W {
        push(b + 1);
    }
    (out, n)
}

/// Remove from sorted `next` every element of sorted `prev`.
fn subtract_sorted(next: &mut Vec<u128>, prev: &[u128]) {
    let mut j = 0;
    next.retain(|x| {
        while j < prev.len() && prev[j] < *x {
            j += 1;
        }
        !(j < prev.len() && prev[j] == *x)
    });
}

fn expand(cur: &[u128]) -> Vec<u128> {
    #[cfg(feature = "parallel")]
    let mut raw: Vec<u128> = cur
        .par_iter()
        .flat_map_iter(|&k| {
            let (nb, n) = neighbours(k);
            nb.into_iter().take(n)
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let mut raw: Vec<u128> = cur
        .iter()
        .flat_map(|&k| {
            let (nb, n) = neighbours(k);
            nb.into_iter().take(n)
        })
        .collect();
    #[cfg(feature = "parallel")]
    raw.par_sort_unstable();
    #[cfg(not(feature = "parallel"))]
    raw.sort_unstable();
    raw.dedup();
    raw
}

/// Generate layers `V_0..=V_max_k` in order, calling `f(k, layer)` on each
/// sorted layer of packed boards.
pub fn for_each_layer(max_k: usize, mut f: impl FnMut(usize, &[u128])) {
    let mut prev: Vec<u128> = Vec::new();
    let mut cur = vec![pack(&crate::puzzle24::state::GOAL)];
    f(0, &cur);
    for k in 1..=max_k {
        let mut next = expand(&cur);
        subtract_sorted(&mut next, &prev);
        f(k, &next);
        prev = std::mem::replace(&mut cur, next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::state::{Move, GOAL};

    #[test]
    fn pack_roundtrip_and_neighbours_match_apply() {
        let mut s = GOAL;
        for m in [Move::Up, Move::Left, Move::Left, Move::Down, Move::Left] {
            s = s.apply(m);
        }
        assert_eq!(unpack(pack(&s)), s);
        let (nb, n) = neighbours(pack(&s));
        let mut got: Vec<u128> = nb[..n].to_vec();
        let mut want: Vec<u128> = s.legal_moves().iter().map(|m| pack(&s.apply(m))).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn layers_match_a090031_to_depth_14() {
        let mut sizes = Vec::new();
        for_each_layer(14, |_, layer| sizes.push(layer.len() as u64));
        assert_eq!(sizes[..], A090031[..=14]);
    }

    #[test]
    #[ignore = "BFS to depth 20 (1.5e7 states, 1.1 GB peak); <1 s in release, minutes in debug; run with --release -- --ignored"]
    fn layers_match_a090031_to_depth_20() {
        let mut sizes = Vec::new();
        for_each_layer(20, |_, layer| sizes.push(layer.len() as u64));
        assert_eq!(sizes[..], A090031[..=20]);
    }
}
