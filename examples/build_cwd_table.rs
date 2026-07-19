//! build_cwd_table — build and serialize the single-line escape-constrained WD
//! surcharge overlay for ALL five goal lines, over ALL WD contingencies.
//!
//! For each line `g` this runs the (validated in `build_cwd_single`) backward
//! product-graph BFS to get `D(σ, g, r) = cWD_single(σ, g, r)`, then stores the
//! surcharge `Δ(σ, g, d) = D(σ,g,d) − WD(σ)` for demands `d = 1..=4`. Δ is even and
//! ≤ 8, so `Δ/2 ∈ 0..=4` packs into a nibble; the four demands pack into one `u16`
//! per `(σ, line)`. The per-line WD-layer invariant `D(σ,0) == WD(σ)` is checked
//! for every contingency, plus a small random cross-check against the reference A*.
//!
//! Output `data/cwd_single.bin`:  "CWDS" magic, version, count, then per σ:
//! `u64 key` + `[u16; 5]` (one packed surcharge curve per goal line).
//!
//! Run from repo root (needs `data/wd24.bin`).

use puzzle8::puzzle24::search::walking_distance::{
    build_full_table, load_dist_table, WdBuild, WdTable, FULL_WD_ENTRIES, WD_KIND_FULL,
};
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

const W: usize = 5;
const K: usize = 4;
const UNSEEN: u8 = 255;
type Matrix = [[u8; W]; W];

fn pack(m: &Matrix, blank: u8) -> u64 {
    let mut k: u64 = blank as u64;
    for r in 0..W {
        for c in 0..(W - 1) {
            k = (k << 3) | (m[r][c] as u64);
        }
    }
    k
}

fn unpack(key: u64) -> (Matrix, u8) {
    let blank = ((key >> 60) & 0x7) as u8;
    let mut m = [[0u8; W]; W];
    let mut k = key;
    for r in (0..W).rev() {
        for c in (0..(W - 1)).rev() {
            m[r][c] = (k & 0x7) as u8;
            k >>= 3;
        }
    }
    for (r, row) in m.iter_mut().enumerate() {
        let margin: u8 = if r as u8 == blank { 4 } else { 5 };
        let partial: u8 = row[..W - 1].iter().sum();
        row[W - 1] = margin - partial;
    }
    (m, blank)
}

fn goal_key() -> u64 {
    let mut m = [[0u8; W]; W];
    for d in 0..W {
        m[d][d] = 5;
    }
    m[W - 1][W - 1] = 4;
    pack(&m, (W - 1) as u8)
}

fn r_row_key() -> u64 {
    let mut m = [[0u8; W]; W];
    for r in 0..W {
        m[r][W - 1 - r] = 5;
    }
    m[0][W - 1] = 4;
    pack(&m, 0)
}

/// `D(σ, r)` for the fixed line `g`, all σ and `r = 0..=K` (backward product BFS).
fn build_line(g: usize) -> HashMap<u64, [u8; K + 1], WdBuild> {
    let goal = goal_key();
    let mut dist: HashMap<u64, [u8; K + 1], WdBuild> =
        HashMap::with_capacity_and_hasher(66_000_000, WdBuild::default());
    dist.insert(goal, {
        let mut a = [UNSEEN; K + 1];
        a[0] = 0;
        a
    });
    let mut frontier: Vec<(u64, u8)> = vec![(goal, 0)];
    let mut depth: u8 = 0;
    while !frontier.is_empty() {
        let nd = depth + 1;
        let mut next: Vec<(u64, u8)> = Vec::new();
        for &(sk, r) in &frontier {
            let (m, br) = unpack(sk);
            for from in [br.wrapping_sub(1), br + 1] {
                let from = from as usize;
                if from >= W {
                    continue;
                }
                for j in 0..W {
                    if m[from][j] == 0 {
                        continue;
                    }
                    let mut m2 = m;
                    m2[from][j] -= 1;
                    m2[br as usize][j] += 1;
                    let nk = pack(&m2, from as u8);
                    let esc = (br as usize == g) && (j == g);
                    let e = dist.entry(nk).or_insert([UNSEEN; K + 1]);
                    if !esc {
                        if e[r as usize] == UNSEEN {
                            e[r as usize] = nd;
                            next.push((nk, r));
                        }
                    } else {
                        let rp = (r as usize + 1).min(K);
                        if e[rp] == UNSEEN {
                            e[rp] = nd;
                            next.push((nk, rp as u8));
                        }
                        if r == 0 && e[0] == UNSEEN {
                            e[0] = nd;
                            next.push((nk, 0));
                        }
                    }
                }
            }
        }
        depth = nd;
        frontier = next;
    }
    dist
}

fn cwd_axis_single(
    table: &WdTable,
    m: &Matrix,
    blank: u8,
    goal: u64,
    g: usize,
    dem: u8,
) -> Option<u8> {
    if dem == 0 {
        return Some(*table.get(&pack(m, blank)).expect("start reachable"));
    }
    let start_key = pack(m, blank);
    let h0 = *table.get(&start_key).expect("start reachable") as usize;
    let mut best: HashMap<u128, u8> = HashMap::with_capacity(1 << 14);
    let mut buckets: Vec<Vec<(u64, u32)>> = vec![Vec::new(); 210];
    const CLOSED: u8 = 0x80;
    let statekey = |wd: u64, c: u32| -> u128 { ((wd as u128) << 16) | c as u128 };
    best.insert(statekey(start_key, 0), 0);
    buckets[h0].push((start_key, 0));
    let mut pops: u64 = 0;
    for f in h0..buckets.len() {
        let mut i = 0;
        while i < buckets[f].len() {
            let (key, c) = buckets[f][i];
            i += 1;
            let sk = statekey(key, c);
            let gcur = match best.get(&sk) {
                Some(&v) if v & CLOSED == 0 => v,
                _ => continue,
            };
            best.insert(sk, gcur | CLOSED);
            pops += 1;
            if pops > 40_000_000 {
                return None;
            }
            if key == goal && c as usize == dem as usize {
                return Some(gcur);
            }
            let (mm, br) = unpack(key);
            let g2 = gcur + 1;
            for from in [br.wrapping_sub(1), br + 1] {
                let from = from as usize;
                if from >= W {
                    continue;
                }
                for t in 0..W {
                    if mm[from][t] == 0 {
                        continue;
                    }
                    let mut m2 = mm;
                    m2[from][t] -= 1;
                    m2[br as usize][t] += 1;
                    let child_key = pack(&m2, from as u8);
                    let esc = (from == g && t == g) as u32;
                    let c2 = (c + esc).min(dem as u32);
                    let h = *table.get(&child_key).expect("child reachable") as usize;
                    let csk = statekey(child_key, c2);
                    let slot = best.entry(csk).or_insert(UNSEEN);
                    if *slot == UNSEEN || (*slot & CLOSED == 0 && g2 < *slot) {
                        *slot = g2;
                        buckets[g2 as usize + h].push((child_key, c2));
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let t0 = Instant::now();
    let path = Path::new("data/wd24.bin");
    let wd = if path.exists() {
        load_dist_table(path, WD_KIND_FULL, Some(FULL_WD_ENTRIES)).expect("wd24.bin")
    } else {
        build_full_table()
    };
    eprintln!(
        "WD table: {} entries in {:.1}s",
        wd.len(),
        t0.elapsed().as_secs_f64()
    );
    let goal = goal_key();

    // overlay[σ] = [packed surcharge curve per line g]; nibble (d-1) of line g =
    // Δ(σ,g,d)/2  (d = 1..=4).
    let mut overlay: HashMap<u64, [u16; W], WdBuild> =
        HashMap::with_capacity_and_hasher(wd.len(), WdBuild::default());
    for &k in wd.keys() {
        overlay.insert(k, [0u16; W]);
    }

    let mut rng: u64 = 0x51ED_C0DE;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let keys: Vec<u64> = wd.keys().copied().collect();

    for g in 0..W {
        let tb = Instant::now();
        let dist = build_line(g);
        assert_eq!(
            dist.len(),
            wd.len(),
            "line {g}: reachable set size mismatch"
        );
        // invariant 1: r=0 layer == WD, per contingency; and pack the surcharge.
        for (&k, d) in dist.iter() {
            let wdv = wd[&k];
            assert_eq!(d[0], wdv, "line {g}: D(σ,0) != WD(σ)");
            let mut packed = 0u16;
            for dm in 1..=K {
                let surch = d[dm] - d[0]; // Δ, even, ≤ 8
                debug_assert!(surch % 2 == 0 && surch <= 8, "unexpected Δ={surch}");
                packed |= ((surch / 2) as u16 & 0xF) << (4 * (dm - 1));
            }
            overlay.get_mut(&k).unwrap()[g] = packed;
        }
        if g == 2 {
            let d = dist[&r_row_key()];
            assert_eq!(d, [70, 70, 70, 70, 72], "R row curve wrong: {d:?}");
            eprintln!("  g=2 R-row check OK: {d:?}");
        }
        // cheap random cross-check vs reference A*
        let mut ok = 0u32;
        for _ in 0..150 {
            let sk = keys[(next() as usize) % keys.len()];
            let (m, br) = unpack(sk);
            let dm = ((next() % K as u64) + 1) as u8;
            let surch = 2 * (((overlay[&sk][g] >> (4 * (dm - 1))) & 0xF) as u8);
            let tv = wd[&sk] + surch;
            if let Some(reference) = cwd_axis_single(&wd, &m, br, goal, g, dm) {
                assert_eq!(
                    tv, reference,
                    "line {g} σ={sk:#x} d={dm}: {tv} vs A* {reference}"
                );
                ok += 1;
            }
        }
        eprintln!(
            "line g={g}: built {:.1}s, WD-layer invariant OK, {ok}/150 A* cross-checks matched",
            tb.elapsed().as_secs_f64()
        );
    }

    // ---- serialize ----
    let out = Path::new("data/cwd_single.bin");
    let f = std::fs::File::create(out).expect("create cwd_single.bin");
    let mut w = BufWriter::new(f);
    w.write_all(b"CWDS").unwrap();
    w.write_all(&1u32.to_le_bytes()).unwrap(); // version
    w.write_all(&(overlay.len() as u64).to_le_bytes()).unwrap();
    let mut buf = Vec::with_capacity(18);
    for (&k, curves) in overlay.iter() {
        buf.clear();
        buf.extend_from_slice(&k.to_le_bytes());
        for c in curves {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        w.write_all(&buf).unwrap();
    }
    w.flush().unwrap();
    let bytes = 4 + 4 + 8 + overlay.len() * 18;
    eprintln!(
        "wrote {} ({} entries, {:.0} MB) in total {:.1}s",
        out.display(),
        overlay.len(),
        bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
}
