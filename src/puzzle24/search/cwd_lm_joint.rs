//! Joint-form LM2 **reference oracle** — the demand-carrying tracked
//! constrained A\* that validated the joint heuristic and now serves only as
//! the test oracle for the production table tier (`cwd_lm1l`).
//!
//! History (2026-08-06): this A\* first ran in production behind a memoized
//! cache and measured the joint heuristic's worth — 1.38× fewer nodes than
//! `--lm2` at threshold 144, 545/780 survivor prunes beyond both table-LM2
//! and k8 — and the `lm2j1l` probe then showed a single-demanded-line
//! restriction retains 778/780 of the prunes, making the closed `cwd_lm1l`
//! table the production form. The runtime memo/front-cache/env-knob
//! machinery was deleted with that migration; git history has it. What
//! remains is the exact solver: builder validation gates and composition
//! cross-checks call it; the engine never does.

use std::collections::HashMap;

use super::cwd::{pack, unpack, MergedBacking};
use crate::puzzle24::state::W;

/// One open-list entry: `(bag key, escape-counter index, tracked line A,
/// crossed A, tracked line B, crossed B)`.
type JointNode = (u64, u32, u8, bool, u8, bool);

/// Reusable A\* workspace.
struct JointScratch {
    best: HashMap<u128, u8>,
    buckets: Vec<Vec<JointNode>>,
}

impl JointScratch {
    fn new() -> Self {
        JointScratch {
            best: HashMap::with_capacity(1 << 14),
            buckets: (0..210).map(|_| Vec::new()).collect(),
        }
    }
    fn clear(&mut self) {
        self.best.clear();
        for b in &mut self.buckets {
            b.clear();
        }
    }
}

const UNSEEN: u8 = 0xFF;
const CLOSED: u8 = 0x80;

/// Pop bound per solve. The production tier is a table; this bounds only
/// test/validation runs. Exhaustion returns `None`; validation callers must
/// skip-and-COUNT those samples (never silently) — the heavy tail lives in
/// extreme-demand configs on deep keys.
const POP_BUDGET: u64 = 40_000_000;

/// The demand-carrying pair-tracking constrained A\*. `ta` = tracked type-3
/// tile's current line (obligation: cross 3→4 once, end in line 3), `tb` =
/// tracked type-2 tile (cross 2→3 once, end in line 2); `None` = untracked.
/// Escape demands from `dem` are enforced via saturating per-line counters.
/// The sweep stops at `h0 + cap_margin` and returns the capped value — pass
/// a large margin for uncapped oracle queries.
///
/// Internal heuristic: `wd + max single-line surcharge over the REMAINING
/// demands` — admissible and consistent (each per-line constrained distance
/// is a true shortest-path distance in a projection of the product graph),
/// so the no-reopen bucket A\* is exact below the cap.
#[allow(clippy::too_many_arguments)]
fn joint_axis(
    backing: MergedBacking,
    scratch: &mut JointScratch,
    start_key: u64,
    goal: u64,
    dem: &[u8; W],
    ta: Option<u8>,
    tb: Option<u8>,
    cap_margin: usize,
) -> Option<u8> {
    let (ta_on, tb_on) = (ta.is_some(), tb.is_some());
    let mut lines = [0usize; W];
    let mut nlines = 0usize;
    for g in 0..W {
        if dem[g] > 0 {
            lines[nlines] = g;
            nlines += 1;
        }
    }
    let mut radix = [1u32; W];
    let mut full: u32 = 1;
    for i in 0..nlines {
        radix[i] = full;
        full *= dem[lines[i]] as u32 + 1;
    }
    let full_index = full - 1;
    let counter_of = |g: usize| -> Option<usize> { lines[..nlines].iter().position(|&x| x == g) };

    // Remaining-demand vector per escape-counter index.
    let rem_of: Vec<[u8; W]> = (0..full)
        .map(|ci| {
            let mut rem = [0u8; W];
            for i in 0..nlines {
                let g = lines[i];
                let cur = (ci / radix[i]) % (dem[g] as u32 + 1);
                rem[g] = dem[g] - cur as u8;
            }
            rem
        })
        .collect();
    let h_of = |cell: &super::cwd::CwdCell, ci: u32| -> usize {
        cell.wd as usize
            + super::cwd::surcharge_from_curves(&cell.curves, &rem_of[ci as usize]) as usize
    };

    let h0 = h_of(&backing.cell(start_key)?, 0);
    let cap = (h0 + cap_margin).min(209);
    scratch.clear();
    let statekey = |wd: u64, ci: u32, la: u8, ea: bool, lb: u8, eb: bool| -> u128 {
        ((wd as u128) << 24)
            | ((ci as u128) << 8)
            | ((la as u128) << 5)
            | ((ea as u128) << 4)
            | ((lb as u128) << 1)
            | eb as u128
    };
    let (la0, ea0) = (ta.unwrap_or(7), !ta_on);
    let (lb0, eb0) = (tb.unwrap_or(7), !tb_on);
    scratch
        .best
        .insert(statekey(start_key, 0, la0, ea0, lb0, eb0), 0);
    scratch.buckets[h0].push((start_key, 0, la0, ea0, lb0, eb0));
    let mut pops: u64 = 0;

    for f in h0..cap.min(scratch.buckets.len()) {
        let mut i = 0;
        while i < scratch.buckets[f].len() {
            let (key, ci, la, ea, lb, eb) = scratch.buckets[f][i];
            i += 1;
            let sk = statekey(key, ci, la, ea, lb, eb);
            let g = match scratch.best.get(&sk) {
                Some(&v) if v & CLOSED == 0 => v,
                _ => continue,
            };
            scratch.best.insert(sk, g | CLOSED);
            pops += 1;
            if pops > POP_BUDGET {
                return None;
            }
            if key == goal
                && ci == full_index
                && (!ta_on || (la == 3 && ea))
                && (!tb_on || (lb == 2 && eb))
            {
                return Some(g);
            }
            let (mm, br) = unpack(key);
            let g2 = g + 1;
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
                    let mut ci2 = ci;
                    if from == t {
                        if let Some(idx) = counter_of(t) {
                            let cur = (ci / radix[idx]) % (dem[t] as u32 + 1);
                            if cur < dem[t] as u32 {
                                ci2 += radix[idx];
                            }
                        }
                    }
                    let h = h_of(&backing.cell(child_key)?, ci2);
                    if g2 as usize + h >= cap {
                        continue;
                    }
                    let push =
                        |la2: u8, ea2: bool, lb2: u8, eb2: bool, scratch: &mut JointScratch| {
                            let csk = statekey(child_key, ci2, la2, ea2, lb2, eb2);
                            let slot = scratch.best.entry(csk).or_insert(UNSEEN);
                            if *slot == UNSEEN || (*slot & CLOSED == 0 && g2 < *slot) {
                                *slot = g2;
                                scratch.buckets[g2 as usize + h]
                                    .push((child_key, ci2, la2, ea2, lb2, eb2));
                            }
                        };
                    if ta_on && t == 3 && from == la as usize {
                        let ea2 = ea || (from == 3 && br as usize == 4);
                        push(br, ea2, lb, eb, scratch);
                        if mm[from][3] >= 2 {
                            push(la, ea, lb, eb, scratch);
                        }
                    } else if tb_on && t == 2 && from == lb as usize {
                        let eb2 = eb || (from == 2 && br as usize == 3);
                        push(la, ea, br, eb2, scratch);
                        if mm[from][2] >= 2 {
                            push(la, ea, lb, eb, scratch);
                        }
                    } else {
                        push(la, ea, lb, eb, scratch);
                    }
                }
            }
        }
    }
    // Swept every bucket below the cap without reaching the saturated goal:
    // the true value is ≥ cap, so the capped value IS cap.
    Some(cap.min(u8::MAX as usize) as u8)
}

/// Oracle: the single-demanded-line joint value for `(key, g, d)` with
/// tracked variant `(ta, tb)`, capped at `base + 8`. The `cwd_lm1l` table
/// stores `min((D − base)/2, 3)`, and the A\*'s internal start value equals
/// the base exactly, so a margin of 8 resolves every 2-bit field
/// identically to the uncapped value while bounding the sweep (a margin of
/// 200 explodes on deep keys with d = 4). `None` = pop budget exhausted;
/// callers skip and count. Validation only.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn reference_1l(
    backing: MergedBacking,
    goal: u64,
    key: u64,
    g: usize,
    d: u8,
    ta: Option<u8>,
    tb: Option<u8>,
) -> Option<u8> {
    let mut dem = [0u8; W];
    dem[g] = d;
    let mut scratch = JointScratch::new();
    joint_axis(backing, &mut scratch, key, goal, &dem, ta, tb, 6)
}

/// Oracle: the four-branch joint LM2 value for one child under the retired
/// production composition — full demand vectors, per-variant table floors,
/// the `+2` decision cap, and the tb-only-aware skips (the +201K-anomaly
/// semantics). Computed fresh per call (no memo); the 675-board probe
/// cross-check and the survivor cross-tabs run against it.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_uncached(
    backing: MergedBacking,
    goal: u64,
    rkey: u64,
    ckey: u64,
    dem_r: &[u8; W],
    dem_c: &[u8; W],
    rterm: u8,
    cterm: u8,
    floors: &[u8; 6],
    lp: &[u8; 6],
    h_cwd: u8,
) -> u8 {
    let mut scratch = JointScratch::new();
    let live3 = |l: u8| (l <= 3).then_some(l);
    let live2 = |l: u8| (l <= 2).then_some(l);
    let (a20, a24) = (live3(lp[0]), live3(lp[1]));
    let (b15, b23) = (live2(lp[2]), live2(lp[5]));
    let (a19r, a19c) = (live3(lp[3]), live3(lp[4]));
    let rdead = dem_r.iter().all(|&d| d == 0);
    let cdead = dem_c.iter().all(|&d| d == 0);
    let axis_v = |key: u64,
                  dem: &[u8; W],
                  term: u8,
                  dead: bool,
                  ta: Option<u8>,
                  tb: Option<u8>,
                  floor: u8,
                  scratch: &mut JointScratch|
     -> u8 {
        // A demand-free axis equals the table only for ta-present variants
        // (tb-only variants have no table), and a floor at the cap cannot
        // be improved.
        if (ta.is_none() && tb.is_none())
            || (dead && ta.is_some())
            || floor >= term.saturating_add(2)
        {
            return floor;
        }
        match joint_axis(backing, scratch, key, goal, dem, ta, tb, 2) {
            Some(v) => v.max(floor),
            None => floor,
        }
    };
    let ba =
        axis_v(rkey, dem_r, rterm, rdead, a20, b15, floors[2], &mut scratch).saturating_add(cterm);
    let bb0 = axis_v(
        rkey,
        dem_r,
        rterm,
        rdead,
        a20,
        None,
        floors[0],
        &mut scratch,
    );
    let bc0 = axis_v(
        rkey,
        dem_r,
        rterm,
        rdead,
        a19r,
        None,
        floors[1],
        &mut scratch,
    );
    let bb = bb0.saturating_add(axis_v(
        ckey,
        dem_c,
        cterm,
        cdead,
        a19c,
        None,
        floors[4],
        &mut scratch,
    ));
    let bc = bc0.saturating_add(axis_v(
        ckey,
        dem_c,
        cterm,
        cdead,
        a24,
        None,
        floors[3],
        &mut scratch,
    ));
    let bd = rterm.saturating_add(axis_v(
        ckey,
        dem_c,
        cterm,
        cdead,
        a24,
        b23,
        floors[5],
        &mut scratch,
    ));
    h_cwd.max(ba.min(bb).min(bc).min(bd))
}

#[cfg(all(test, feature = "cwd-table-tests"))]
mod tests {
    use super::*;
    use crate::puzzle24::search::cwd::{
        goal_key, project, shared_merged_cwd, surcharge_from_curves,
    };
    use crate::puzzle24::state::State;

    /// The oracle must reproduce the lm2jprobe DETAIL values — the
    /// independent worktree implementation's joint results on real survivor
    /// boards (data/lm2jprobe_survivors148.txt, 2026-08-06). This is the
    /// anchor of the whole validation chain: the probe validated the
    /// heuristic, this test pins the oracle to the probe, and the builder
    /// gates pin the table to the oracle.
    #[test]
    fn joint_matches_probe_details() {
        let Some(cwd) = shared_merged_cwd() else {
            return;
        };
        let backing = cwd.backing().expect("merged backing");
        let Ok(lm_mm) =
            super::super::cwd_lm::CwdLmMm::load(std::path::Path::new("data/cwd_lm_mm.bin"))
        else {
            eprintln!("cwd_lm_mm.bin absent — skipping joint cross-check");
            return;
        };
        let text = match std::fs::read_to_string("data/lm2jprobe_survivors148.txt") {
            Ok(t) => t,
            Err(_) => {
                eprintln!("probe dump absent — skipping joint cross-check");
                return;
            }
        };
        let goal = goal_key();
        let mut checked = 0usize;
        for line in text.lines().filter(|l| l.starts_with("DETAIL JOINT>TABLE")) {
            let field = |k: &str| -> u32 {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(k))
                    .and_then(|v| v.parse().ok())
                    .unwrap()
            };
            let (h0p_exp, joint_exp) = (field("h0p="), field("joint="));
            let seg = &line[line.find("board=[").unwrap() + 7..];
            let seg = &seg[..seg.find(']').unwrap()];
            let vals: Vec<u8> = seg.split(',').map(|w| w.trim().parse().unwrap()).collect();
            let mut cells = [0u8; 25];
            cells.copy_from_slice(&vals);
            let s = State(cells);

            let (mr, br, dr, mc, bc, dc) = project(&s);
            let (rkey, ckey) = (pack(&mr, br), pack(&mc, bc));
            let rc = backing.cell(rkey).expect("row reachable");
            let cc = backing.cell(ckey).expect("col reachable");
            let rterm = rc.wd + surcharge_from_curves(&rc.curves, &dr);
            let cterm = cc.wd + surcharge_from_curves(&cc.curves, &dc);
            let h_cwd = rterm + cterm;
            assert_eq!(h_cwd as u32, h0p_exp, "h0p mismatch on {line}");

            let pos = |t: u8| s.0.iter().position(|&x| x == t).unwrap();
            let lp = [
                (pos(20) / W) as u8,
                (pos(24) % W) as u8,
                (pos(15) / W) as u8,
                (pos(19) / W) as u8,
                (pos(19) % W) as u8,
                (pos(23) % W) as u8,
            ];
            // Real table-value floors (the skip contract requires them).
            let floors = {
                let or_ = |v: u8, fb: u8| if v != 0xFF { v } else { fb };
                let sv = |single: &[u8], l: u8| if l < 4 { single[l as usize] } else { 0xFF };
                let pv = |pair: &[u8], la: u8, lb: u8| {
                    if la < 4 && lb < 3 {
                        pair[(la * 3 + lb) as usize]
                    } else {
                        0xFF
                    }
                };
                let (rs, rp) = lm_mm.probe(rkey).expect("row key in lm_mm");
                let (cs, cp) = lm_mm.probe(ckey).expect("col key in lm_mm");
                let r20f = or_(sv(rs, lp[0]), rterm);
                let c24f = or_(sv(cs, lp[1]), cterm);
                [
                    r20f,
                    or_(sv(rs, lp[3]), rterm),
                    or_(pv(rp, lp[0], lp[2]), r20f),
                    c24f,
                    or_(sv(cs, lp[4]), cterm),
                    or_(pv(cp, lp[1], lp[5]), c24f),
                ]
            };
            let hj = eval_uncached(
                backing, goal, rkey, ckey, &dr, &dc, rterm, cterm, &floors, &lp, h_cwd,
            );
            assert_eq!(hj as u32, joint_exp, "joint mismatch on {line}");
            checked += 1;
        }
        assert!(checked >= 600, "expected ~675 DETAIL boards, got {checked}");
        eprintln!("joint cross-check: {checked} DETAIL boards reproduced");
    }
}
