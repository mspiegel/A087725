//! Move-pruning DFA for windows beyond [`MoveDfa`]'s 13-move limit.
//!
//! Same construction and same pruning predicate as [`MoveDfa`]: a move is
//! pruned when some recent suffix of length `6..=W+1`, read from the blank cell
//! it started at, reaches a board that a strictly shorter or equal-length
//! lexicographically smaller sequence also reaches. [`MoveDfa`] packs sequences
//! into `u32` with a 4-bit length, which caps `W` at 13; this type packs them
//! into `u64` (6-bit length, 29 moves) and so supports `W ≤ 28`, memory
//! permitting.
//!
//! It is a separate type rather than a widened [`MoveDfa`] because the engine's
//! checkpoint fingerprints hash [`MoveDfa`] state ids, which must not change.
//! [`LongMoveDfa::build`] at `W = 11` yields an automaton equivalent to
//! [`MoveDfa::build_default`] (tested by a product walk over both).
//!
//! Measured need (records/eta24_yield.txt, Result 4): for the random walks of
//! the η sampler, a rule covering sequences of 15 moves (`W = 14`) removes about
//! a fifth of the walks that `W = 11` lets through to rejection, while
//! `W = 12, 13` remove under 5%.
//!
//! [`MoveDfa`]: super::move_dfa::MoveDfa
//! [`MoveDfa::build_default`]: super::move_dfa::MoveDfa::build_default

use std::collections::{HashMap, HashSet};

use super::move_dfa::MovePruner;
use crate::puzzle24::state::{Move, State, N_CELLS, W as WIDTH};

const N: usize = N_CELLS;
const INVALID: u32 = u32::MAX;

/// Largest supported window: sequences of `W + 1 ≤ 29` moves fit the packing.
pub const MAX_WINDOW: u8 = 28;

/// Shortest dominated sequence (half of a 2×2 rotation).
const MIN_DOMINATED: usize = 6;

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
        let (dominated, prefix) = build_tables(w);
        let raw = build_raw(w, &dominated, &prefix);
        let dfa = minimize(&raw);
        let (caught, total) = dfa.count_caught(&dominated);
        assert_eq!(
            caught, total,
            "move-pruning DFA dropped a prune ({caught}/{total}) — refusing to use an unsound pruner"
        );
        dfa
    }

    /// Feed every dominated sequence through the DFA from its start blank; each
    /// must be pruned on its final move. Returns `(caught, total)`.
    fn count_caught(&self, dominated: &[HashSet<Seq>]) -> (u64, u64) {
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

/// Per start blank: `dominated[b]` = dominated sequences; `prefix[b]` = proper
/// prefixes of dominated sequences. A breadth-first search over inverse-pruned
/// sequences in move-code order extends only canonical sequences, so the first
/// sequence to reach a board is its shortest, lexicographically smallest one and
/// every later sequence reaching it is dominated.
fn build_tables(w: u8) -> (Vec<HashSet<Seq>>, Vec<HashSet<Seq>>) {
    let mut dominated = Vec::with_capacity(N);
    let mut prefix = Vec::with_capacity(N);
    for b in 0..N {
        let s0 = generic(b);
        let mut canon: HashSet<u128> = HashSet::new();
        canon.insert(bkey(&s0));
        let mut dom: HashSet<Seq> = HashSet::new();
        let mut pfx: HashSet<Seq> = HashSet::new();
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
                    if canon.insert(bkey(&child)) {
                        next.push((child, cb, cseq));
                    } else {
                        dom.insert(cseq);
                        for l in 1..seq_len(cseq) {
                            pfx.insert(
                                (cseq & ((1u64 << (2 * l)) - 1)) | ((l as u64) << LEN_SHIFT),
                            );
                        }
                    }
                }
            }
            frontier = next;
        }
        dominated.push(dom);
        prefix.push(pfx);
    }
    (dominated, prefix)
}

/// Build the un-minimized DFA by BFS over `(start blank, history)` states with
/// the Aho–Corasick collapse: the history kept is the longest suffix of the
/// move string that is a proper prefix of some dominated sequence.
fn build_raw(w: u8, dominated: &[HashSet<Seq>], prefix: &[HashSet<Seq>]) -> LongMoveDfa {
    let wl = w as usize;
    let mut ids: HashMap<(u8, Seq), u32> = HashMap::new();
    let mut hists: Vec<(u8, Seq)> = Vec::new();
    let mut start = [INVALID; N];
    let mut queue: Vec<u32> = Vec::new();
    let mut intern = |bs: u8, h: Seq, hists: &mut Vec<(u8, Seq)>| -> (u32, bool) {
        if let Some(&id) = ids.get(&(bs, h)) {
            return (id, false);
        }
        let id = hists.len() as u32;
        ids.insert((bs, h), id);
        hists.push((bs, h));
        (id, true)
    };
    for (b, sb) in start.iter_mut().enumerate() {
        let (id, _) = intern(b as u8, EMPTY, &mut hists);
        *sb = id;
        queue.push(id);
    }
    let mut trans: Vec<[u32; 4]> = Vec::new();
    let mut prune: Vec<u8> = Vec::new();
    let mut blanks: Vec<u8> = Vec::with_capacity(MAX_WINDOW as usize + 2);

    let mut qi = 0;
    while qi < queue.len() {
        let sid = queue[qi] as usize;
        qi += 1;
        let (bs, hist) = hists[sid];
        if trans.len() <= sid {
            trans.resize(sid + 1, [INVALID; 4]);
            prune.resize(sid + 1, 0);
        }
        blank_path(bs, hist, &mut blanks);
        let cur = *blanks.last().expect("path has a start");
        for m in State::legal_moves_at(cur).iter() {
            let mc = mcode(m);
            let full = seq_push(hist, mc);
            let k = seq_len(full);
            blanks.push(step_blank(cur, mc));
            // blanks[cut] is the blank where the suffix of length k - cut starts.
            let pruned = (MIN_DOMINATED..=k)
                .any(|l| dominated[blanks[k - l] as usize].contains(&seq_suffix(full, l)));
            if pruned {
                prune[sid] |= 1 << mc;
            } else {
                let (bs2, hist2) = (1..=k.min(wl))
                    .rev()
                    .find_map(|l| {
                        let bs2 = blanks[k - l];
                        let suf = seq_suffix(full, l);
                        prefix[bs2 as usize].contains(&suf).then_some((bs2, suf))
                    })
                    .unwrap_or((blanks[k], EMPTY));
                let (nid, fresh) = intern(bs2, hist2, &mut hists);
                trans[sid][mc] = nid;
                if fresh {
                    queue.push(nid);
                }
            }
            blanks.pop();
        }
    }
    trans.resize(hists.len(), [INVALID; 4]);
    prune.resize(hists.len(), 0);
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
    use crate::puzzle24::search::move_dfa::MoveDfa;

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
    }

    /// At the default window the two builders must define the same pruning
    /// language: a product walk from every start blank finds equal prune masks
    /// and consistent transitions, and both minimal automata have equal size.
    #[test]
    fn equivalent_to_move_dfa_at_default_window() {
        let short = MoveDfa::build_default();
        let long = LongMoveDfa::build(super::super::move_dfa::DEFAULT_WINDOW);
        assert_eq!(short.states(), long.states());
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut stack: Vec<(u32, u32, u8)> = Vec::new();
        for b in 0..N as u8 {
            stack.push((short.root_state(b), long.root_state(b), b));
        }
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
                if short.is_pruned(a, m) {
                    continue;
                }
                stack.push((
                    short.advance(a, m),
                    long.advance(l, m),
                    step_blank(blank, mcode(m)),
                ));
            }
        }
        assert!(seen.len() >= short.states());
    }
}
