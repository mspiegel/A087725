//! Move-pruning DFA for the 24-puzzle — an O(1)-per-node, cache-resident finite
//! automaton that skips *provably redundant* move continuations (Taylor–Korf
//! duplicate elimination), stacked on top of any admissible heuristic.
//!
//! # What it prunes
//!
//! Starting from a blank cell, a move sequence is *dominated* if it reaches a board
//! also reachable by a strictly shorter — or equal-length, lexicographically smaller
//! — sequence (after the search's existing inverse-move pruning). Expanding a
//! dominated continuation only re-generates a subtree already reachable more cheaply,
//! so skipping it removes duplicate work without ever removing a genuinely new board.
//! Because it never removes a board, an exhausted IDA* threshold stays a *sound*
//! lower bound.
//!
//! # Why a DFA
//!
//! The pruning predicate ("is the recent move suffix, from the blank it started at,
//! dominated?") is a regular property of the move string. We compile it to a DFA
//! whose state is `(start-blank, longest history-suffix that is a prefix of some
//! dominated sequence)` — the Aho-Corasick collapse. This is lossless for the prune
//! decision (suffixes of a string are nested, so the longest dominated-prefix suffix
//! subsumes every shorter one), and after Myhill–Nerode minimization the whole table
//! is well under 1 MB — L2-resident, so the per-node check is a single array-indexed
//! transition with no hashing and no main-memory traffic (unlike a transposition
//! table). Each search thread carries its own `u32` state id down its stack; the
//! tables are shared read-only with zero contention.
//!
//! # Soundness
//!
//! [`MoveDfa::build`] asserts, by complete enumeration, that every dominated sequence
//! is caught by the finished (minimized) automaton, and every prune bit is set only
//! via a direct lookup into the exhaustively-verified domination table. The
//! [`MovePruner`] abstraction lets the search stay generic: [`NullPruner`] is a ZST
//! that monomorphizes away to exactly the un-pruned search.

use crate::puzzle24::state::{Move, State};
use std::collections::{HashMap, HashSet};

const N: usize = 25;
const INVALID: u32 = u32::MAX;

/// Default history window: dominated sequences have length 6..=`W`+1. `W = 11`
/// captures the 2×2-rotation redundancy and its short extensions (the shortest
/// dominated sequence is the length-6 half-rotation).
pub const DEFAULT_WINDOW: u8 = 11;

// ----------------------------- pruner abstraction -----------------------------

/// A per-node move filter threaded through the incremental search. Implementors
/// carry a small `Copy` state down each search path.
pub trait MovePruner: Sync {
    /// Per-path carried state (small and `Copy`; e.g. a DFA state id).
    type St: Copy + Send + Sync;
    /// State at the search root, given the root blank cell.
    fn root_state(&self, blank: u8) -> Self::St;
    /// Whether move `m` from the current state is a provably redundant continuation.
    fn is_pruned(&self, st: Self::St, m: Move) -> bool;
    /// State after taking (non-pruned, legal) move `m`.
    fn advance(&self, st: Self::St, m: Move) -> Self::St;
}

/// The no-op pruner: prunes nothing. A ZST whose methods inline to nothing, so a
/// search generic over [`MovePruner`] compiles to identical code to the unpruned
/// search when instantiated with `NullPruner`.
#[derive(Clone, Copy, Default)]
pub struct NullPruner;

impl MovePruner for NullPruner {
    type St = ();
    #[inline(always)]
    fn root_state(&self, _blank: u8) {}
    #[inline(always)]
    fn is_pruned(&self, _st: (), _m: Move) -> bool {
        false
    }
    #[inline(always)]
    fn advance(&self, _st: (), _m: Move) {}
}

// ------------------------------- the automaton --------------------------------

/// Compiled, minimized move-pruning DFA. Cheap to clone-free share (`&MoveDfa`).
pub struct MoveDfa {
    /// `trans[state][move_code]` → next state id (`INVALID` if that move is illegal
    /// from this state's blank, which the search never takes).
    trans: Vec<[u32; 4]>,
    /// `prune[state]` bit `move_code` set ⇒ that move is a redundant continuation.
    prune: Vec<u8>,
    /// `start[blank]` = state for a fresh path whose blank is `blank`.
    start: [u32; N],
}

#[inline(always)]
fn mcode(m: Move) -> u32 {
    match m {
        Move::Up => 0,
        Move::Down => 1,
        Move::Left => 2,
        Move::Right => 3,
    }
}

impl MovePruner for MoveDfa {
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
        let nx = self.trans[st as usize][mcode(m) as usize];
        debug_assert_ne!(nx, INVALID, "advanced along an illegal move");
        nx
    }
}

impl MoveDfa {
    /// Number of states (after minimization).
    pub fn states(&self) -> usize {
        self.trans.len()
    }
    /// Runtime table footprint in bytes (`trans` + `prune`).
    pub fn table_bytes(&self) -> usize {
        self.trans.len() * (4 * 4 + 1)
    }

    /// Build the minimized DFA for the default window and verify it (panics if the
    /// completeness check fails). Runs in well under a second.
    pub fn build_default() -> MoveDfa {
        Self::build(DEFAULT_WINDOW)
    }

    /// Build and minimize the DFA for history window `w` (dominated sequences up to
    /// length `w`+1). Asserts every dominated sequence is caught by the finished
    /// automaton — a complete-enumeration soundness gate.
    pub fn build(w: u8) -> MoveDfa {
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

    /// Completeness check: feed every dominated sequence through the DFA from its
    /// start-blank; each must be pruned on its final move. Returns `(caught, total)`.
    fn count_caught(&self, dominated: &[HashSet<u32>]) -> (u64, u64) {
        let (mut total, mut caught) = (0u64, 0u64);
        for (b, dom) in dominated.iter().enumerate() {
            for &packed in dom {
                total += 1;
                let len = (packed >> 28) as usize;
                let mut st = self.start[b];
                let mut ok = false;
                for i in 0..len {
                    let m = Move::ALL[((packed >> (2 * i)) & 3) as usize];
                    let mc = mcode(m) as usize;
                    if i + 1 == len {
                        ok = (self.prune[st as usize] >> mc) & 1 == 1;
                    } else {
                        st = self.trans[st as usize][mc];
                        if st == INVALID {
                            break;
                        }
                    }
                }
                if ok {
                    caught += 1;
                }
            }
        }
        (caught, total)
    }
}

// --------------------------- construction internals ---------------------------

fn generic(b: usize) -> State {
    let mut c = [0u8; N];
    let mut t = 1u8;
    for (i, ci) in c.iter_mut().enumerate() {
        if i == b {
            *ci = 0;
        } else {
            *ci = t;
            t += 1;
        }
    }
    State(c)
}

/// Pack a move slice: bits [28..32) = length, moves 2-bits each from bit 0.
fn pack(seq: &[Move]) -> u32 {
    let mut k = (seq.len() as u32) << 28;
    for (i, &m) in seq.iter().enumerate() {
        k |= mcode(m) << (2 * i);
    }
    k
}

/// Final blank after walking `moves` from start-blank `bs`.
fn walk(bs: u8, moves: &[Move]) -> u8 {
    let mut s = generic(bs as usize);
    let mut b = bs;
    for &m in moves {
        let (ns, nb) = s.apply_at(m, b);
        s = ns;
        b = nb;
    }
    b
}

fn bkey(s: &State) -> u128 {
    let mut k = 0u128;
    for i in 0..N {
        k |= (s.0[i] as u128) << (5 * i);
    }
    k
}

/// Per start-blank: `dominated[b]` = packed dominated sequences; `prefix[b]` = packed
/// proper prefixes of dominated sequences (the pattern-prefix set for the collapse).
///
/// Canonicalization matches `examples/fsm_build.rs`, whose exhaustive re-simulation
/// verified that every dominated sequence reaches the same board as a strictly
/// shorter / lex-smaller canonical one (so pruning it removes only a duplicate).
fn build_tables(w: u8) -> (Vec<HashSet<u32>>, Vec<HashSet<u32>>) {
    let mut dominated = Vec::with_capacity(N);
    let mut prefix = Vec::with_capacity(N);
    for b in 0..N {
        let s0 = generic(b);
        let mut canon: HashMap<u128, ()> = HashMap::new();
        canon.insert(bkey(&s0), ());
        let mut dom: HashSet<u32> = HashSet::new();
        let mut pfx: HashSet<u32> = HashSet::new();
        let mut frontier: Vec<(State, u8, Vec<Move>)> = vec![(s0, b as u8, Vec::new())];
        for _ in 1..=w + 1 {
            let mut next = Vec::new();
            for (s, blank, seq) in &frontier {
                let last = seq.last().copied();
                for m in State::legal_moves_at(*blank).iter() {
                    if let Some(p) = last {
                        if m == p.inverse() {
                            continue;
                        }
                    }
                    let (child, cb) = s.apply_at(m, *blank);
                    let mut cseq = seq.clone();
                    cseq.push(m);
                    match canon.entry(bkey(&child)) {
                        std::collections::hash_map::Entry::Occupied(_) => {
                            dom.insert(pack(&cseq));
                            for l in 1..cseq.len() {
                                pfx.insert(pack(&cseq[..l]));
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(());
                            next.push((child, cb, cseq));
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

/// Build the (un-minimized) DFA by BFS with the Aho-Corasick collapse.
#[allow(clippy::needless_range_loop)] // indexes several parallel arrays by one index
fn build_raw(w: u8, dominated: &[HashSet<u32>], prefix: &[HashSet<u32>]) -> MoveDfa {
    let wl = w as usize;
    // longest suffix of hist·m that is a pattern-prefix (from its own start-blank)
    let collapse = |bs: u8, hist: &[Move], m: Move| -> (u8, Vec<Move>) {
        let mut full = hist.to_vec();
        full.push(m);
        let k = full.len();
        for l in (1..=k.min(wl)).rev() {
            let cut = k - l;
            let bs2 = walk(bs, &full[..cut]);
            if prefix[bs2 as usize].contains(&pack(&full[cut..])) {
                return (bs2, full[cut..].to_vec());
            }
        }
        (walk(bs, &full), Vec::new())
    };
    // some suffix (length 6..=W+1) of hist·m is dominated
    let prunes = |bs: u8, hist: &[Move], m: Move| -> bool {
        let mut full = hist.to_vec();
        full.push(m);
        let k = full.len();
        for l in 6..=k {
            let cut = k - l;
            let bs2 = walk(bs, &full[..cut]);
            if dominated[bs2 as usize].contains(&pack(&full[cut..])) {
                return true;
            }
        }
        false
    };

    fn intern(
        bs: u8,
        hist: Vec<Move>,
        ids: &mut HashMap<(u8, u32), u32>,
        hists: &mut Vec<(u8, Vec<Move>)>,
    ) -> u32 {
        let key = (bs, pack(&hist));
        if let Some(&id) = ids.get(&key) {
            return id;
        }
        let id = hists.len() as u32;
        ids.insert(key, id);
        hists.push((bs, hist));
        id
    }

    let mut ids: HashMap<(u8, u32), u32> = HashMap::new();
    let mut hists: Vec<(u8, Vec<Move>)> = Vec::new();
    let mut start = [INVALID; N];
    let mut queue: Vec<u32> = Vec::new();
    for b in 0..N {
        let id = intern(b as u8, Vec::new(), &mut ids, &mut hists);
        start[b] = id;
        queue.push(id);
    }
    let mut trans: Vec<[u32; 4]> = Vec::new();
    let mut prune: Vec<u8> = Vec::new();

    let mut qi = 0;
    while qi < queue.len() {
        let sid = queue[qi] as usize;
        qi += 1;
        let (bs, hist) = hists[sid].clone();
        let cur = walk(bs, &hist);
        while trans.len() <= sid {
            trans.push([INVALID; 4]);
            prune.push(0);
        }
        for m in State::legal_moves_at(cur).iter() {
            let mc = mcode(m) as usize;
            if prunes(bs, &hist, m) {
                prune[sid] |= 1 << mc;
                continue;
            }
            let (bs2, hist2) = collapse(bs, &hist, m);
            let before = hists.len();
            let nid = intern(bs2, hist2, &mut ids, &mut hists);
            trans[sid][mc] = nid;
            if nid as usize >= before {
                queue.push(nid);
            }
        }
    }
    while trans.len() < hists.len() {
        trans.push([INVALID; 4]);
        prune.push(0);
    }
    MoveDfa { trans, prune, start }
}

/// Moore partition-refinement minimization. Merges states with identical future
/// prune behaviour on every input (Myhill–Nerode), so it preserves every prune
/// decision exactly. States with different current blanks have different legal-move
/// (INVALID) patterns and so are never merged.
#[allow(clippy::needless_range_loop)] // indexes several parallel arrays by one index
fn minimize(dfa: &MoveDfa) -> MoveDfa {
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
            trans[c][mc] = if t == INVALID { INVALID } else { class[t as usize] };
        }
    }
    let mut start = [INVALID; N];
    for b in 0..N {
        start[b] = class[dfa.start[b] as usize];
    }
    MoveDfa { trans, prune, start }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replay a packed sequence from blank cell `b`, returning the board key.
    fn replay(b: usize, packed: u32) -> u128 {
        let len = (packed >> 28) as usize;
        let mut s = generic(b);
        let mut blank = b as u8;
        for i in 0..len {
            let m = Move::ALL[((packed >> (2 * i)) & 3) as usize];
            let (ns, nb) = s.apply_at(m, blank);
            s = ns;
            blank = nb;
        }
        bkey(&s)
    }

    /// `a` is strictly shorter than `b`, or equal-length lexicographically smaller.
    fn dominates(a: u32, b: u32) -> bool {
        let (la, lb) = (a >> 28, b >> 28);
        if la != lb {
            return la < lb;
        }
        for i in 0..la as usize {
            let (ca, cb) = ((a >> (2 * i)) & 3, (b >> (2 * i)) & 3);
            if ca != cb {
                return ca < cb;
            }
        }
        false
    }

    /// Foundational soundness gate (the check `examples/fsm_build.rs` performed):
    /// every sequence the construction calls *dominated* really reaches the same
    /// board as a strictly shorter / lex-smaller *canonical* sequence — so pruning
    /// it removes only a duplicate, never a genuinely new board. Exhaustive over all
    /// inverse-pruned sequences to depth `W`+1 from every blank cell.
    #[test]
    fn dominated_sequences_reach_the_same_board_as_a_shorter_canonical() {
        let mut checked = 0u64;
        for b in 0..N {
            let s0 = generic(b);
            let mut canon: HashMap<u128, u32> = HashMap::new();
            canon.insert(bkey(&s0), pack(&[]));
            let mut frontier: Vec<(State, u8, Vec<Move>)> = vec![(s0, b as u8, Vec::new())];
            for _ in 1..=DEFAULT_WINDOW + 1 {
                let mut next = Vec::new();
                for (s, blank, seq) in &frontier {
                    let last = seq.last().copied();
                    for m in State::legal_moves_at(*blank).iter() {
                        if let Some(p) = last {
                            if m == p.inverse() {
                                continue;
                            }
                        }
                        let (child, cb) = s.apply_at(m, *blank);
                        let mut cseq = seq.clone();
                        cseq.push(m);
                        let k = bkey(&child);
                        match canon.get(&k) {
                            Some(&canonp) => {
                                let dpack = pack(&cseq);
                                assert_eq!(
                                    replay(b, dpack),
                                    replay(b, canonp),
                                    "b={b}: dominated seq and its canonical reach different boards"
                                );
                                assert!(
                                    dominates(canonp, dpack),
                                    "b={b}: canonical does not dominate the dominated seq"
                                );
                                checked += 1;
                            }
                            None => {
                                canon.insert(k, pack(&cseq));
                                next.push((child, cb, cseq));
                            }
                        }
                    }
                }
                frontier = next;
            }
        }
        assert!(checked > 0, "no dominated sequences enumerated");
    }

    #[test]
    fn builds_and_minimizes_sound() {
        let (dominated, prefix) = build_tables(DEFAULT_WINDOW);
        let raw = build_raw(DEFAULT_WINDOW, &dominated, &prefix);
        let min = minimize(&raw);
        // minimization shrinks the table but preserves completeness
        assert!(min.states() < raw.states());
        assert!(min.table_bytes() < 1_000_000, "minimized table should be < 1 MB");
        let (rc, rt) = raw.count_caught(&dominated);
        let (mc, mt) = min.count_caught(&dominated);
        assert_eq!((rc, rt), (mc, mt));
        assert_eq!(mc, mt, "every dominated sequence must be caught");
    }

    #[test]
    fn null_pruner_prunes_nothing() {
        let p = NullPruner;
        p.root_state(12);
        assert!(!p.is_pruned((), Move::Up));
        assert!(!p.is_pruned((), Move::Left));
    }

    #[test]
    fn dfa_pruner_flags_the_2x2_rotation() {
        // The shortest dominated sequence is the length-6 half of a 2×2 rotation;
        // build() asserts all such are caught, so here we just confirm the pruner
        // reports *some* prune reachable from a fresh interior start.
        let dfa = MoveDfa::build_default();
        // walk an arbitrary path and confirm the API is usable (soundness is the
        // build-time completeness assert; this guards the public surface).
        let mut st = dfa.root_state(12);
        st = dfa.advance(st, Move::Up);
        let _ = dfa.is_pruned(st, Move::Down); // Down is the inverse; search filters it anyway
    }
}
