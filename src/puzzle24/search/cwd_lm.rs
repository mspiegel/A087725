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
        for (&k, v) in &self.map {
            w.write_all(&k.to_le_bytes())?;
            w.write_all(v)?;
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
                                next.push((pkey, line as u8, crossed));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::cwd::goal_key;

    /// Build the Last-Move Escape-Constrained WD table and save it.
    ///
    ///   cargo test --release build_cwd_lm_table -- --ignored --nocapture
    #[test]
    #[ignore = "builds the ~66M-key refined table (minutes, ~3 GB peak); writes data/cwd_lm.bin"]
    fn build_cwd_lm_table() {
        let gk = goal_key();
        let t0 = std::time::Instant::now();
        let lm = build_cwd_lm(gk);
        eprintln!("built {} keys in {:.0?}", lm.len(), t0.elapsed());
        // Sanity: from the goal, the tracked tile must leave line 3 and
        // return — exactly 2 abstract moves.
        assert_eq!(lm.get(gk, 3), Some(2), "D(goal, line 3) must be 2");
        lm.save(std::path::Path::new("data/cwd_lm.bin")).expect("save");
        eprintln!("saved data/cwd_lm.bin");
    }
}
