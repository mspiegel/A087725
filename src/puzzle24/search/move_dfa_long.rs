//! Move-pruning DFA for windows beyond [`MoveDfa`]'s 13-move limit.
//!
//! Same pruning predicate as [`MoveDfa`]: a move is pruned when some recent
//! suffix of length `6..=W+1`, read from the blank cell it started at, reaches a
//! board that a strictly shorter or equal-length lexicographically smaller
//! sequence also reaches. [`MoveDfa`] packs sequences into `u32` with a 4-bit
//! length, which caps `W` at 13; this type packs them into `u64` (6-bit length,
//! 29 moves) and so supports `W ≤ 28`, memory permitting.
//!
//! It is a separate type rather than a widened [`MoveDfa`] because the engine's
//! checkpoint fingerprints hash [`MoveDfa`] state ids, which must not change.
//! [`LongMoveDfa::build`] yields an automaton equivalent to [`MoveDfa::build`]
//! at every window both support (tested by a product walk at `W = 11, 12`, and
//! `13` ignored); state numbering differs.
//!
//! # Construction
//!
//! 1. **Dominated sequences**, per start blank (`collect_blank`). A
//!    breadth-first search over inverse-pruned sequences in move-code order
//!    extends only canonical sequences, so the first sequence to reach a board
//!    is its shortest, lexicographically smallest one, and every later sequence
//!    reaching it is dominated. A canonical sequence is a shortest path, and
//!    each move changes the blank's colour, so a child of a depth-`j−1`
//!    sequence lies at distance `j` or `j−2`: duplicates are found against the
//!    current layer (a hash set) and layer `j−2` (a sorted array) alone.
//!    Frontier entries are packed sequences; boards are replayed from the start.
//! 2. **Minimal dominated sequences** (`minimal_words`): those with no
//!    dominated proper suffix. A dominated sequence never has a dominated
//!    proper prefix (the search does not extend dominated sequences), so every
//!    dominated sequence ends in a minimal one and the pruning language is
//!    unchanged. At `W = 15` this keeps 13% of the sequences.
//! 3. **Aho–Corasick automaton** (`build_trie`) over the minimal sequences:
//!    one trie per start blank, transitions filled in breadth-first order
//!    through failure links, with no hashing.
//! 4. **Moore minimization** (`minimize`).
//!
//! [`LongMoveDfa::build`] then asserts that *every* dominated sequence, not only
//! the minimal ones, is pruned on its final move.
//!
//! Measured need (records/eta24_yield.txt, Results 4–5): for the random walks
//! of the η sampler, a rule covering sequences of 15 moves (`W = 14`) removes
//! about a fifth of the walks that `W = 11` lets through to rejection.
//!
//! [`MoveDfa`]: super::move_dfa::MoveDfa
//! [`MoveDfa::build`]: super::move_dfa::MoveDfa::build

use std::collections::{HashMap, HashSet};

use super::move_dfa::MovePruner;
use crate::puzzle24::state::{Move, State, N_CELLS, W as WIDTH};

const N: usize = N_CELLS;
const INVALID: u32 = u32::MAX;

/// Largest supported window: sequences of `W + 1 ≤ 29` moves fit the packing.
pub const MAX_WINDOW: u8 = 28;

/// Shortest dominated sequence (half of a 2×2 rotation).
const MIN_DOMINATED: usize = 6;

/// Layer-size growth assumed for the first expansion when reserving capacity;
/// later layers use the measured growth of the previous one.
const INITIAL_LAYER_GROWTH: f64 = 3.0;

/// Headroom on the reserved capacity for the next layer.
const LAYER_RESERVE_SLACK: f64 = 1.05;

// ------------------------------ packed sequences ------------------------------

/// A move sequence packed into `u64`: move `i` in bits `2i..2i+2`, length in
/// bits `58..64`.
type Seq = u64;
const LEN_SHIFT: u32 = 58;
const MOVES_MASK: u64 = (1u64 << LEN_SHIFT) - 1;
const EMPTY: Seq = 0;

#[inline]
fn seq_len(s: Seq) -> usize {
    (s >> LEN_SHIFT) as usize
}

#[inline]
fn seq_move(s: Seq, i: usize) -> usize {
    ((s >> (2 * i)) & 3) as usize
}

#[inline]
fn seq_push(s: Seq, mc: usize) -> Seq {
    let l = seq_len(s);
    debug_assert!(l < (LEN_SHIFT / 2) as usize, "sequence overflow");
    (s & MOVES_MASK) | ((mc as u64) << (2 * l)) | (((l + 1) as u64) << LEN_SHIFT)
}

/// The last `l` moves of `s`.
#[inline]
fn seq_suffix(s: Seq, l: usize) -> Seq {
    let k = seq_len(s);
    let moves = ((s & MOVES_MASK) >> (2 * (k - l))) & ((1u64 << (2 * l)) - 1);
    moves | ((l as u64) << LEN_SHIFT)
}

/// Move code of `m`, numbered as [`Move`] (`Up = 0 … Right = 3`).
#[inline]
fn mcode(m: Move) -> usize {
    m as usize
}

/// Blank cell after move code `mc` from `b` (the move must be legal).
#[inline]
fn step_blank(b: u8, mc: usize) -> u8 {
    let w = WIDTH as u8;
    match mc {
        0 => b - w,
        1 => b + w,
        2 => b - 1,
        _ => b + 1,
    }
}

/// Blank cells along `s` from `bs`: `out[i]` is the blank after `i` moves.
fn blank_path(bs: u8, s: Seq, out: &mut Vec<u8>) {
    out.clear();
    out.push(bs);
    let mut b = bs;
    for i in 0..seq_len(s) {
        b = step_blank(b, seq_move(s, i));
        out.push(b);
    }
}

/// A board with distinct tiles and the blank at `b`.
fn generic(b: usize) -> State {
    let mut c = [0u8; N];
    let mut t = 1u8;
    for (i, ci) in c.iter_mut().enumerate() {
        if i != b {
            *ci = t;
            t += 1;
        }
    }
    State(c)
}

fn bkey(s: &State) -> u128 {
    let mut k = 0u128;
    for (i, &t) in s.0.iter().enumerate() {
        k |= (t as u128) << (5 * i);
    }
    k
}

/// Board and blank after playing `s` from `start` (blank at `blank`).
fn replay(start: &State, blank: u8, s: Seq) -> (State, u8) {
    let (mut board, mut b) = (*start, blank);
    for i in 0..seq_len(s) {
        (board, b) = board.apply_at(Move::ALL[seq_move(s, i)], b);
    }
    (board, b)
}

// ------------------------------- the automaton --------------------------------

/// Compiled, minimized move-pruning DFA for window `W ≤ 28`.
pub struct LongMoveDfa {
    trans: Vec<[u32; 4]>,
    prune: Vec<u8>,
    start: [u32; N],
    window: u8,
}

impl MovePruner for LongMoveDfa {
    type St = u32;
    #[inline(always)]
    fn root_state(&self, blank: u8) -> u32 {
        self.start[blank as usize]
    }
    #[inline(always)]
    fn is_pruned(&self, st: u32, m: Move) -> bool {
        (self.prune[st as usize] >> mcode(m)) & 1 == 1
    }
    #[inline(always)]
    fn advance(&self, st: u32, m: Move) -> u32 {
        let nx = self.trans[st as usize][mcode(m)];
        debug_assert_ne!(nx, INVALID, "advanced along an illegal move");
        nx
    }
}

impl LongMoveDfa {
    /// Number of states (after minimization).
    pub fn states(&self) -> usize {
        self.trans.len()
    }

    /// Runtime table footprint in bytes (`trans` + `prune`).
    pub fn table_bytes(&self) -> usize {
        self.trans.len() * (4 * 4 + 1)
    }

    /// The history window this automaton was built for.
    pub fn window(&self) -> u8 {
        self.window
    }

    /// Build and minimize the DFA for window `w` (dominated sequences up to
    /// length `w + 1`), asserting that every dominated sequence is caught.
    pub fn build(w: u8) -> LongMoveDfa {
        assert!(
            (MIN_DOMINATED as u8..=MAX_WINDOW).contains(&w),
            "window {w} outside {MIN_DOMINATED}..={MAX_WINDOW}"
        );
        let dominated = collect_dominated(w);
        let words = minimal_words(&dominated);
        let raw = build_trie(w, &words);
        drop(words);
        let dfa = minimize(&raw);
        drop(raw);
        let (caught, total) = dfa.count_caught(&dominated);
        assert_eq!(
            caught, total,
            "move-pruning DFA dropped a prune ({caught}/{total}) — refusing to use an unsound pruner"
        );
        dfa
    }

    /// Feed every dominated sequence through the DFA from its start blank; each
    /// must be pruned on its final move. Returns `(caught, total)`.
    fn count_caught(&self, dominated: &[Vec<Seq>]) -> (u64, u64) {
        let (mut total, mut caught) = (0u64, 0u64);
        for (b, dom) in dominated.iter().enumerate() {
            for &s in dom {
                total += 1;
                let len = seq_len(s);
                let mut st = self.start[b];
                for i in 0..len {
                    let mc = seq_move(s, i);
                    if i + 1 == len {
                        if (self.prune[st as usize] >> mc) & 1 == 1 {
                            caught += 1;
                        }
                    } else {
                        st = self.trans[st as usize][mc];
                        if st == INVALID {
                            break;
                        }
                    }
                }
            }
        }
        (caught, total)
    }
}

// --------------------------- construction internals ---------------------------

/// Buffers reused across start blanks, so each blank does not re-grow them.
#[derive(Default)]
struct Scratch {
    /// Canonical boards of the layer being built.
    layer: HashSet<u128>,
    /// Canonical boards of the previous layer, sorted.
    prev1: Vec<u128>,
    /// Canonical boards two layers back, sorted.
    prev2: Vec<u128>,
    /// Canonical sequences of the previous layer, in generation order.
    front: Vec<Seq>,
    /// Canonical sequences of the layer being built, in generation order.
    next: Vec<Seq>,
}

/// Dominated sequences for every start blank, each sorted.
fn collect_dominated(w: u8) -> Vec<Vec<Seq>> {
    let mut scratch = Scratch::default();
    (0..N).map(|b| collect_blank(w, b, &mut scratch)).collect()
}

/// Dominated sequences of length `1..=w+1` from start blank `b`, sorted. See
/// the module docs for why layers `j` and `j−2` suffice for duplicate checks.
/// The frontier stays in generation order, so the canonical sequence for each
/// board is the same as a search holding every board ever seen.
fn collect_blank(w: u8, b: usize, sc: &mut Scratch) -> Vec<Seq> {
    let s0 = generic(b);
    sc.layer.clear();
    sc.prev1.clear();
    sc.prev2.clear();
    sc.front.clear();
    sc.next.clear();
    sc.prev1.push(bkey(&s0));
    sc.front.push(EMPTY);
    let mut dom: Vec<Seq> = Vec::new();
    let mut growth = INITIAL_LAYER_GROWTH;
    for _ in 1..=w + 1 {
        let want = (sc.front.len() as f64 * growth * LAYER_RESERVE_SLACK) as usize + 16;
        sc.layer.reserve(want);
        sc.next.reserve_exact(want);
        for &seq in &sc.front {
            let (s, blank) = replay(&s0, b as u8, seq);
            let len = seq_len(seq);
            let last = (len > 0).then(|| seq_move(seq, len - 1));
            for m in State::legal_moves_at(blank).iter() {
                let mc = mcode(m);
                if last == Some(mc ^ 1) {
                    continue;
                }
                let (child, _) = s.apply_at(m, blank);
                let key = bkey(&child);
                let cseq = seq_push(seq, mc);
                if sc.prev2.binary_search(&key).is_ok() || !sc.layer.insert(key) {
                    dom.push(cseq);
                } else {
                    sc.next.push(cseq);
                }
            }
        }
        growth = sc.next.len() as f64 / sc.front.len().max(1) as f64;
        std::mem::swap(&mut sc.prev1, &mut sc.prev2);
        sc.prev1.clear();
        sc.prev1.extend(sc.layer.drain());
        sc.prev1.sort_unstable();
        std::mem::swap(&mut sc.front, &mut sc.next);
        sc.next.clear();
    }
    dom.sort_unstable();
    dom.shrink_to_fit();
    dom
}

/// The dominated sequences with no dominated proper suffix, per start blank,
/// sorted. `dominated` must be sorted per blank.
fn minimal_words(dominated: &[Vec<Seq>]) -> Vec<Vec<Seq>> {
    let mut blanks = Vec::with_capacity(MAX_WINDOW as usize + 2);
    dominated
        .iter()
        .enumerate()
        .map(|(b, dom)| {
            let mut out: Vec<Seq> = dom
                .iter()
                .copied()
                .filter(|&s| {
                    let k = seq_len(s);
                    blank_path(b as u8, s, &mut blanks);
                    !(MIN_DOMINATED..k).any(|l| {
                        dominated[blanks[k - l] as usize]
                            .binary_search(&seq_suffix(s, l))
                            .is_ok()
                    })
                })
                .collect();
            out.shrink_to_fit();
            out
        })
        .collect()
}

/// Un-minimized DFA: the Aho–Corasick automaton over `words` (per start blank).
///
/// Nodes are `(start blank, proper prefix of a word)`, with one root per blank.
/// A word's final move is a forbidden edge at its parent node (no word is a
/// prefix of another from the same blank, since the search never extends a
/// dominated sequence). `fail(v)` is the longest proper suffix of `v`'s string
/// that is a node, in the trie of that suffix's own start blank; for a depth-1
/// node it is the root of the blank reached. Filling rows in breadth-first
/// order makes `fail(v)`'s row complete when `v` is processed:
///
/// - `pruned(v, m)` — `m` is forbidden at `v` or anywhere on its failure chain;
/// - `delta(v, m)` — `child(v, m)` if present, else `delta(fail(v), m)`, and
///   from a root without that child, the root of the blank reached.
///
/// A prefix of a minimal word contains no dominated window, so no node both has
/// a child on `m` and prunes `m` (asserted).
fn build_trie(w: u8, words: &[Vec<Seq>]) -> LongMoveDfa {
    let mut trans: Vec<[u32; 4]> = Vec::new();
    let mut child_mask: Vec<u8> = Vec::new();
    let mut forbid: Vec<u8> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut start = [INVALID; N];
    for (b, sb) in start.iter_mut().enumerate() {
        *sb = trans.len() as u32;
        trans.push([INVALID; 4]);
        child_mask.push(0);
        forbid.push(0);
        cur.push(b as u8);
    }
    for (b, ws) in words.iter().enumerate() {
        for &word in ws {
            let len = seq_len(word);
            let mut v = start[b] as usize;
            for i in 0..len - 1 {
                let mc = seq_move(word, i);
                if (child_mask[v] >> mc) & 1 == 1 {
                    v = trans[v][mc] as usize;
                } else {
                    let id = trans.len();
                    trans.push([INVALID; 4]);
                    child_mask.push(0);
                    forbid.push(0);
                    cur.push(step_blank(cur[v], mc));
                    trans[v][mc] = id as u32;
                    child_mask[v] |= 1 << mc;
                    v = id;
                }
            }
            forbid[v] |= 1 << seq_move(word, len - 1);
        }
    }

    let n = trans.len();
    let mut fail: Vec<u32> = vec![INVALID; n];
    let mut prune: Vec<u8> = vec![0; n];
    let mut queue: Vec<u32> = start.to_vec();
    let mut qi = 0;
    while qi < queue.len() {
        let v = queue[qi] as usize;
        qi += 1;
        let f = fail[v];
        prune[v] = forbid[v] | if f == INVALID { 0 } else { prune[f as usize] };
        for m in State::legal_moves_at(cur[v]).iter() {
            let mc = mcode(m);
            if (prune[v] >> mc) & 1 == 1 {
                assert_eq!(
                    (child_mask[v] >> mc) & 1,
                    0,
                    "a word prefix ends in a dominated sequence"
                );
                continue;
            }
            let via_fail = if f == INVALID {
                start[step_blank(cur[v], mc) as usize]
            } else {
                trans[f as usize][mc]
            };
            assert_ne!(
                via_fail, INVALID,
                "failure row prunes a move the node allows"
            );
            if (child_mask[v] >> mc) & 1 == 1 {
                fail[trans[v][mc] as usize] = via_fail;
                queue.push(trans[v][mc]);
            } else {
                trans[v][mc] = via_fail;
            }
        }
    }
    assert_eq!(queue.len(), n, "trie node unreachable from the roots");
    LongMoveDfa {
        trans,
        prune,
        start,
        window: w,
    }
}

/// Moore partition refinement: merges states with identical future prune
/// behaviour on every input, preserving every prune decision.
#[allow(clippy::needless_range_loop)] // indexes several parallel arrays by one index
fn minimize(dfa: &LongMoveDfa) -> LongMoveDfa {
    let n = dfa.trans.len();
    const DEAD: u32 = u32::MAX;
    let mut class: Vec<u32> = vec![0; n];
    {
        let mut sig: HashMap<u8, u32> = HashMap::new();
        for s in 0..n {
            let next = sig.len() as u32;
            class[s] = *sig.entry(dfa.prune[s]).or_insert(next);
        }
    }
    let mut nclasses = class.iter().copied().collect::<HashSet<u32>>().len();
    loop {
        let mut sig2id: HashMap<(u32, [u32; 4]), u32> = HashMap::new();
        let mut newclass = vec![0u32; n];
        for s in 0..n {
            let tc = |mc: usize| {
                let t = dfa.trans[s][mc];
                if t == INVALID {
                    DEAD
                } else {
                    class[t as usize]
                }
            };
            let key = (class[s], [tc(0), tc(1), tc(2), tc(3)]);
            let next = sig2id.len() as u32;
            newclass[s] = *sig2id.entry(key).or_insert(next);
        }
        let cnt = sig2id.len();
        class = newclass;
        if cnt == nclasses {
            break;
        }
        nclasses = cnt;
    }
    let mut rep: Vec<u32> = vec![INVALID; nclasses];
    for s in 0..n {
        let c = class[s] as usize;
        if rep[c] == INVALID {
            rep[c] = s as u32;
        }
    }
    let mut trans = vec![[INVALID; 4]; nclasses];
    let mut prune = vec![0u8; nclasses];
    for c in 0..nclasses {
        let r = rep[c] as usize;
        prune[c] = dfa.prune[r];
        for mc in 0..4 {
            let t = dfa.trans[r][mc];
            trans[c][mc] = if t == INVALID {
                INVALID
            } else {
                class[t as usize]
            };
        }
    }
    let mut start = [INVALID; N];
    for b in 0..N {
        start[b] = class[dfa.start[b] as usize];
    }
    LongMoveDfa {
        trans,
        prune,
        start,
        window: dfa.window,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::move_dfa::{MoveDfa, DEFAULT_WINDOW};

    /// Proper prefix of length `l`.
    fn seq_prefix(s: Seq, l: usize) -> Seq {
        (s & ((1u64 << (2 * l)) - 1)) | ((l as u64) << LEN_SHIFT)
    }

    /// `a` precedes `b` in search order: shorter, or equal length and
    /// lexicographically smaller with the first move most significant.
    fn precedes(a: Seq, b: Seq) -> bool {
        let (la, lb) = (seq_len(a), seq_len(b));
        if la != lb {
            return la < lb;
        }
        (0..la)
            .map(|i| (seq_move(a, i), seq_move(b, i)))
            .find(|(x, y)| x != y)
            .is_some_and(|(x, y)| x < y)
    }

    /// Reference search holding every board ever seen: for start blank `b`,
    /// returns the canonical sequence of each board and the dominated sequences.
    fn reference_search(w: u8, b: usize) -> (HashMap<u128, Seq>, Vec<Seq>) {
        let s0 = generic(b);
        let mut canon: HashMap<u128, Seq> = HashMap::new();
        canon.insert(bkey(&s0), EMPTY);
        let mut dom = Vec::new();
        let mut frontier: Vec<(State, u8, Seq)> = vec![(s0, b as u8, EMPTY)];
        for _ in 1..=w + 1 {
            let mut next = Vec::new();
            for &(s, blank, seq) in &frontier {
                let len = seq_len(seq);
                let last = (len > 0).then(|| seq_move(seq, len - 1));
                for m in State::legal_moves_at(blank).iter() {
                    let mc = mcode(m);
                    if last == Some(mc ^ 1) {
                        continue;
                    }
                    let (child, cb) = s.apply_at(m, blank);
                    let cseq = seq_push(seq, mc);
                    match canon.entry(bkey(&child)) {
                        std::collections::hash_map::Entry::Occupied(_) => dom.push(cseq),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(cseq);
                            next.push((child, cb, cseq));
                        }
                    }
                }
            }
            frontier = next;
        }
        dom.sort_unstable();
        (canon, dom)
    }

    /// Product walk from every start blank: equal prune masks and consistent
    /// transitions everywhere reachable, and equal minimal sizes.
    fn assert_equivalent(short: &MoveDfa, long: &LongMoveDfa) {
        assert_eq!(short.states(), long.states(), "minimal state counts differ");
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut stack: Vec<(u32, u32, u8)> = (0..N as u8)
            .map(|b| (short.root_state(b), long.root_state(b), b))
            .collect();
        while let Some((a, l, blank)) = stack.pop() {
            if !seen.insert((a, l)) {
                continue;
            }
            assert_eq!(
                short.prune_mask(a),
                long.prune[l as usize],
                "prune masks differ"
            );
            for m in State::legal_moves_at(blank).iter() {
                if !short.is_pruned(a, m) {
                    stack.push((
                        short.advance(a, m),
                        long.advance(l, m),
                        step_blank(blank, mcode(m)),
                    ));
                }
            }
        }
        assert!(seen.len() >= short.states());
    }

    #[test]
    fn packing_roundtrips() {
        let mut s = EMPTY;
        let codes = [
            3, 0, 2, 2, 1, 0, 3, 1, 2, 0, 3, 3, 1, 2, 0, 1, 3, 2, 0, 0, 1, 2, 3, 3, 0, 1, 2, 3, 1,
        ];
        for &c in &codes {
            s = seq_push(s, c);
        }
        assert_eq!(seq_len(s), 29);
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(seq_move(s, i), c);
        }
        let suf = seq_suffix(s, 5);
        assert_eq!(seq_len(suf), 5);
        for i in 0..5 {
            assert_eq!(seq_move(suf, i), codes[24 + i]);
        }
        let pre = seq_prefix(s, 7);
        assert_eq!(seq_len(pre), 7);
        for i in 0..7 {
            assert_eq!(seq_move(pre, i), codes[i]);
        }
    }

    /// The layered search (layers j and j−2 only, replayed boards) finds exactly
    /// the dominated sequences of a search that keeps every board.
    #[test]
    fn layered_search_matches_a_search_keeping_every_board() {
        let mut scratch = Scratch::default();
        for w in [7u8, 10] {
            let mut total = 0;
            for b in 0..N {
                let got = collect_blank(w, b, &mut scratch);
                let (_, want) = reference_search(w, b);
                assert_eq!(got, want, "w={w} blank={b}");
                total += got.len();
            }
            assert!(total > 0, "w={w}: no dominated sequences");
        }
    }

    /// Every dominated sequence reaches the same board as its board's canonical
    /// sequence, which precedes it in search order.
    #[test]
    fn dominated_sequences_are_duplicates_of_an_earlier_canonical() {
        let w = 10;
        let mut scratch = Scratch::default();
        for b in 0..N {
            let s0 = generic(b);
            let (canon, _) = reference_search(w, b);
            for s in collect_blank(w, b, &mut scratch) {
                let key = bkey(&replay(&s0, b as u8, s).0);
                let c = canon[&key];
                assert!(precedes(c, s), "blank {b}: canonical does not precede");
                assert_eq!(bkey(&replay(&s0, b as u8, c).0), key);
            }
        }
    }

    /// Minimal words have no dominated proper suffix, and every other dominated
    /// sequence ends in a minimal word — so both sets prune the same strings.
    #[test]
    fn minimal_words_preserve_the_pruning_language() {
        let w = 12;
        let dominated = collect_dominated(w);
        let words = minimal_words(&dominated);
        let mut blanks = Vec::new();
        let mut kept = 0;
        for b in 0..N {
            for &s in &dominated[b] {
                let k = seq_len(s);
                blank_path(b as u8, s, &mut blanks);
                let is_word = words[b].binary_search(&s).is_ok();
                let dominated_suffix = (MIN_DOMINATED..k).any(|l| {
                    dominated[blanks[k - l] as usize]
                        .binary_search(&seq_suffix(s, l))
                        .is_ok()
                });
                assert_eq!(is_word, !dominated_suffix, "blank {b}");
                let ends_in_word = (MIN_DOMINATED..=k).any(|l| {
                    words[blanks[k - l] as usize]
                        .binary_search(&seq_suffix(s, l))
                        .is_ok()
                });
                assert!(
                    ends_in_word,
                    "blank {b}: dominated sequence has no minimal suffix"
                );
                kept += is_word as usize;
            }
            assert!(words[b]
                .iter()
                .all(|s| dominated[b].binary_search(s).is_ok()));
        }
        let total: usize = dominated.iter().map(Vec::len).sum();
        assert!(kept > 0 && kept < total, "kept {kept} of {total}");
    }

    /// The raw automaton's states are the tries' nodes: one root per blank plus
    /// every distinct proper prefix of a minimal word.
    #[test]
    fn trie_nodes_are_roots_plus_proper_prefixes() {
        let w = 12;
        let words = minimal_words(&collect_dominated(w));
        let prefixes: usize = words
            .iter()
            .map(|ws| {
                ws.iter()
                    .flat_map(|&s| (1..seq_len(s)).map(move |l| seq_prefix(s, l)))
                    .collect::<HashSet<Seq>>()
                    .len()
            })
            .sum();
        let raw = build_trie(w, &words);
        assert_eq!(raw.states(), N + prefixes);
    }

    #[test]
    fn equivalent_to_move_dfa_at_default_window() {
        assert_equivalent(
            &MoveDfa::build_default(),
            &LongMoveDfa::build(DEFAULT_WINDOW),
        );
    }

    #[test]
    fn equivalent_to_move_dfa_at_window_12() {
        assert_equivalent(&MoveDfa::build(12), &LongMoveDfa::build(12));
    }

    #[test]
    #[ignore = "MoveDfa::build(13) is slow in debug; ~5 s in release; run with --release -- --ignored"]
    fn equivalent_to_move_dfa_at_window_13() {
        assert_equivalent(&MoveDfa::build(13), &LongMoveDfa::build(13));
    }

    #[test]
    fn build_is_deterministic() {
        let a = LongMoveDfa::build(DEFAULT_WINDOW);
        let b = LongMoveDfa::build(DEFAULT_WINDOW);
        assert_eq!(a.trans, b.trans);
        assert_eq!(a.prune, b.prune);
        assert_eq!(a.start, b.start);
    }
}
