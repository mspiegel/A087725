//! catalog24 — Phase-2 (2E) bracket-catalog joiner + flywheel.
//!
//! Merges the hunt's three evidence streams — candidate pools (lineage), bounded
//! lower bounds (`ladder24 --out-tsv`), and learned upper bounds (`gen_corridors
//! --mode ubfile --out-tsv`), plus optional value scores (`--mode score`) — into
//! an **append-only** evidence log `data/catalog24.tsv`, then reduces it (at read
//! time) to the best `[proven LB, replay-verified UB]` bracket per board.
//!
//! The log is one fact per line; nothing is ever rewritten, so re-ingesting is
//! idempotent and the raw run TSVs stay the ground truth:
//!
//!   added  canon  board  kind  value  method  nodes  replay_ok  src  note
//!
//! `kind ∈ {lb, ub, exact, score, lineage}`; `canon` (the reflection-canonical
//! board, `symmetry::canonical`) is the JOIN KEY — dist(s) == dist(reflect(s)),
//! and ladder24 labels are line-index-relative, so we never join on label.
//! Reduction: best_lb = max over {lb, exact}; best_ub = min over {ub, exact}
//! with replay_ok == 1 (unverified UBs are kept as history but excluded from the
//! bracket); an `exact` row pins both. A board with best_ub < best_lb trips a
//! loud warning (a solver bug — the whole point of the bracket).
//!
//!   cargo run --release --example catalog24 -- --catalog data/catalog24.tsv \
//!       --ingest --pool data/pool_g1.txt --lb-tsv runs/lb_g1.tsv \
//!       --ub-tsv runs/ub_g1.tsv --score-tsv runs/score_g1.tsv \
//!       --rank 30 --reseed-out data/reseed_g2.txt --reseed-top 15 \
//!       --escalate-out data/escalate_g1.txt --escalate-top 6 --gap-min 12

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use puzzle8::puzzle24::state::{State, N_CELLS};
use puzzle8::puzzle24::symmetry::canonical;

// ---------------------------------------------------------------- arg parsing

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn args_multi<'a>(argv: &'a [String], flag: &str) -> Vec<&'a String> {
    let mut out = Vec::new();
    for (i, a) in argv.iter().enumerate() {
        if a == flag {
            if let Some(v) = argv.get(i + 1) {
                out.push(v);
            }
        }
    }
    out
}

fn arg_opt(argv: &[String], flag: &str) -> Option<String> {
    argv.iter().position(|a| a == flag).and_then(|i| argv.get(i + 1)).cloned()
}

// --------------------------------------------------------------- board helpers

/// Parse a 25-token board (whitespace, `_`/`.` blank). Returns `None` unless it
/// is a permutation of `0..=24`.
fn parse_board(s: &str) -> Option<State> {
    let toks: Vec<&str> = s.split_whitespace().collect();
    if toks.len() != N_CELLS {
        return None;
    }
    let mut arr = [0u8; N_CELLS];
    let mut seen = [false; N_CELLS];
    for (i, tok) in toks.iter().enumerate() {
        let v = if *tok == "_" || *tok == "." {
            0u8
        } else {
            match tok.parse::<u8>() {
                Ok(v) if v <= 24 => v,
                _ => return None,
            }
        };
        if seen[v as usize] {
            return None;
        }
        seen[v as usize] = true;
        arr[i] = v;
    }
    Some(State(arr))
}

fn board_str(s: &State) -> String {
    s.0.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
}

/// Reflection-canonical 25-token string — the catalog join key.
fn canon_str(board: &State) -> String {
    board_str(&canonical(board).0)
}

// ----------------------------------------------------------------- evidence

#[derive(Clone)]
struct Evidence {
    added: String,
    canon: String,
    board: String,
    kind: String, // lb | ub | exact | score | lineage
    value: f64,
    method: String,
    nodes: String,
    replay_ok: String, // "1" | "0" | "-"
    src: String,
    note: String,
}

/// Value formatted for the log / dedup key: integer for lb/ub/exact, else 1 dp.
fn val_str(kind: &str, value: f64) -> String {
    match kind {
        "lb" | "ub" | "exact" => format!("{}", value as i64),
        "lineage" => "-".to_string(),
        _ => format!("{value:.1}"),
    }
}

impl Evidence {
    fn key(&self) -> (String, String, String, String) {
        (self.canon.clone(), self.kind.clone(), val_str(&self.kind, self.value), self.method.clone())
    }
    fn to_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.added,
            self.canon,
            self.board,
            self.kind,
            val_str(&self.kind, self.value),
            self.method,
            self.nodes,
            self.replay_ok,
            self.src,
            self.note,
        )
    }
}

const HEADER: &str = "added\tcanon\tboard\tkind\tvalue\tmethod\tnodes\treplay_ok\tsrc\tnote";

fn load_catalog(path: &Path) -> Vec<Evidence> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("added\t") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 10 {
            continue;
        }
        let value = f[4].parse::<f64>().unwrap_or(0.0);
        out.push(Evidence {
            added: f[0].into(),
            canon: f[1].into(),
            board: f[2].into(),
            kind: f[3].into(),
            value,
            method: f[5].into(),
            nodes: f[6].into(),
            replay_ok: f[7].into(),
            src: f[8].into(),
            note: f[9].into(),
        });
    }
    out
}

// -------------------------------------------------------------- input parsers

/// Split a TSV line into fields, or `None` for comment / header / blank lines.
fn tsv_fields(line: &str, header_first: &str) -> Option<Vec<String>> {
    if line.starts_with('#') || line.trim().is_empty() {
        return None;
    }
    let f: Vec<String> = line.split('\t').map(|s| s.to_string()).collect();
    if f.first().map(|s| s.as_str()) == Some(header_first) {
        return None; // column header
    }
    Some(f)
}

fn ingest_lb(path: &Path, added: &str, out: &mut Vec<Evidence>) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("warning: --lb-tsv {}: {}", path.display(), e);
            return;
        }
    };
    let src = path.display().to_string();
    // label board mode heur pick h0 lb solved_depth nodes iters secs outcome
    for line in text.lines() {
        let Some(f) = tsv_fields(line, "label") else { continue };
        if f.len() < 12 {
            continue;
        }
        let Some(board) = parse_board(&f[1]) else { continue };
        let canon = canon_str(&board);
        let method = format!("ladder24:{}:{}", f[3], f[4]);
        let lb: f64 = f[6].parse().unwrap_or(0.0);
        let solved = f[7] != "-";
        if solved {
            let d: f64 = f[7].parse().unwrap_or(lb);
            out.push(Evidence {
                added: added.into(), canon, board: f[1].clone(), kind: "exact".into(),
                value: d, method, nodes: f[8].clone(), replay_ok: "1".into(),
                src: src.clone(), note: f[11].clone(),
            });
        } else {
            out.push(Evidence {
                added: added.into(), canon, board: f[1].clone(), kind: "lb".into(),
                value: lb, method, nodes: f[8].clone(), replay_ok: "-".into(),
                src: src.clone(), note: f[11].clone(),
            });
        }
    }
}

fn ingest_ub(path: &Path, added: &str, out: &mut Vec<Evidence>) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("warning: --ub-tsv {}: {}", path.display(), e);
            return;
        }
    };
    let src = path.display().to_string();
    // idx board wd ub gap nodes replay_ok solver
    for line in text.lines() {
        let Some(f) = tsv_fields(line, "idx") else { continue };
        if f.len() < 8 || f[3] == "-" {
            continue; // BudgetExceeded rows carry no UB
        }
        let Some(board) = parse_board(&f[1]) else { continue };
        let canon = canon_str(&board);
        let ub: f64 = f[3].parse().unwrap_or(0.0);
        out.push(Evidence {
            added: added.into(), canon, board: f[1].clone(), kind: "ub".into(),
            value: ub, method: f[7].clone(), nodes: f[5].clone(),
            replay_ok: f[6].clone(), src: src.clone(), note: format!("wd={}", f[2]),
        });
    }
}

fn ingest_score(path: &Path, added: &str, out: &mut Vec<Evidence>) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("warning: --score-tsv {}: {}", path.display(), e);
            return;
        }
    };
    let src = path.display().to_string();
    // idx board wd lc v_fwd v_pair
    for line in text.lines() {
        let Some(f) = tsv_fields(line, "idx") else { continue };
        if f.len() < 6 {
            continue;
        }
        let Some(board) = parse_board(&f[1]) else { continue };
        let canon = canon_str(&board);
        for (col, meth) in [(4usize, "vfwd"), (5usize, "vpair")] {
            if f[col] == "-" {
                continue;
            }
            if let Ok(v) = f[col].parse::<f64>() {
                out.push(Evidence {
                    added: added.into(), canon: canon.clone(), board: f[1].clone(),
                    kind: "score".into(), value: v, method: meth.into(), nodes: "-".into(),
                    replay_ok: "-".into(), src: src.clone(), note: String::new(),
                });
            }
        }
    }
}

fn ingest_pool(path: &Path, added: &str, out: &mut Vec<Evidence>) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("warning: --pool {}: {}", path.display(), e);
            return;
        }
    };
    let src = path.display().to_string();
    let mut last_comment = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix('#') {
            last_comment = rest.trim().to_string();
            continue;
        }
        if let Some(board) = parse_board(t) {
            let canon = canon_str(&board);
            out.push(Evidence {
                added: added.into(), canon, board: board_str(&board), kind: "lineage".into(),
                value: 0.0, method: "candidates24".into(), nodes: "-".into(),
                replay_ok: "-".into(), src: src.clone(),
                note: if last_comment.is_empty() { "-".into() } else { last_comment.clone() },
            });
        }
    }
}

// ------------------------------------------------------------------ reduction

struct Bracket {
    canon: String,
    board: String,
    best_lb: Option<i64>,
    best_ub: Option<i64>,
    v_fwd: Option<f64>,
    v_pair: Option<f64>,
    lineage: String,
    lb_all_timeout: bool, // every lb/exact evidence was a budget timeout
    n_evidence: usize,
}

fn reduce(all: &[Evidence]) -> Vec<Bracket> {
    let mut by_canon: HashMap<String, Vec<&Evidence>> = HashMap::new();
    for e in all {
        by_canon.entry(e.canon.clone()).or_default().push(e);
    }
    let mut out = Vec::new();
    for (canon, evs) in by_canon {
        let mut best_lb: Option<i64> = None;
        let mut best_ub: Option<i64> = None;
        let mut v_fwd: Option<f64> = None;
        let mut v_pair: Option<f64> = None;
        let mut lineage = String::from("-");
        let mut board = canon.clone();
        let mut saw_lb = false;
        let mut lb_all_timeout = true;
        for e in &evs {
            match e.kind.as_str() {
                "lb" | "exact" => {
                    let v = e.value as i64;
                    best_lb = Some(best_lb.map_or(v, |b| b.max(v)));
                    saw_lb = true;
                    if !e.note.contains("timeout") {
                        lb_all_timeout = false;
                    }
                    if e.kind == "exact" && e.replay_ok == "1" {
                        best_ub = Some(best_ub.map_or(v, |b| b.min(v)));
                    }
                    // prefer a concrete solved/deep board orientation for display
                    board = e.board.clone();
                }
                "ub" => {
                    if e.replay_ok == "1" {
                        let v = e.value as i64;
                        best_ub = Some(best_ub.map_or(v, |b| b.min(v)));
                    }
                }
                "score" => match e.method.as_str() {
                    "vfwd" => v_fwd = Some(v_fwd.map_or(e.value, |x: f64| x.max(e.value))),
                    "vpair" => v_pair = Some(v_pair.map_or(e.value, |x: f64| x.max(e.value))),
                    _ => {}
                },
                "lineage" => {
                    if lineage == "-" && e.note != "-" {
                        lineage = e.note.clone();
                    }
                }
                _ => {}
            }
        }
        if !saw_lb {
            lb_all_timeout = false;
        }
        out.push(Bracket {
            canon,
            board,
            best_lb,
            best_ub,
            v_fwd,
            v_pair,
            lineage,
            lb_all_timeout,
            n_evidence: evs.len(),
        });
    }
    // Rank: proven LB desc, tie-break UB desc, then v_fwd desc.
    out.sort_by(|a, b| {
        b.best_lb
            .cmp(&a.best_lb)
            .then(b.best_ub.cmp(&a.best_ub))
            .then(b.v_fwd.partial_cmp(&a.v_fwd).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

// ---------------------------------------------------------------------- main

fn now_tag() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| format!("t{}", d.as_secs()))
        .unwrap_or_else(|_| "t0".into())
}

fn opt_i(x: Option<i64>) -> String {
    x.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
}
fn opt_f(x: Option<f64>) -> String {
    x.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "catalog24 --catalog FILE [--ingest] [--pool F]... [--lb-tsv F]... \
             [--ub-tsv F]... [--score-tsv F]... [--date TAG] [--rank N] \
             [--reseed-out F --reseed-top K] [--escalate-out F --escalate-top M --gap-min G]"
        );
        return ExitCode::SUCCESS;
    }
    let catalog = PathBuf::from(arg(&argv, "--catalog", "data/catalog24.tsv".to_string()));
    let do_ingest = argv.iter().any(|a| a == "--ingest");
    let added = arg_opt(&argv, "--date").unwrap_or_else(now_tag);

    // Existing evidence + dedup key set.
    let mut existing = load_catalog(&catalog);
    let mut seen: HashSet<(String, String, String, String)> =
        existing.iter().map(|e| e.key()).collect();

    // Collect new evidence from inputs.
    let mut incoming: Vec<Evidence> = Vec::new();
    for p in args_multi(&argv, "--pool") {
        ingest_pool(Path::new(p), &added, &mut incoming);
    }
    for p in args_multi(&argv, "--lb-tsv") {
        ingest_lb(Path::new(p), &added, &mut incoming);
    }
    for p in args_multi(&argv, "--ub-tsv") {
        ingest_ub(Path::new(p), &added, &mut incoming);
    }
    for p in args_multi(&argv, "--score-tsv") {
        ingest_score(Path::new(p), &added, &mut incoming);
    }

    // Idempotent filter.
    let mut fresh: Vec<Evidence> = Vec::new();
    for e in incoming {
        if seen.insert(e.key()) {
            fresh.push(e);
        }
    }

    if do_ingest && !fresh.is_empty() {
        let is_new = std::fs::metadata(&catalog).map(|m| m.len() == 0).unwrap_or(true);
        let mut f = match std::fs::OpenOptions::new().create(true).append(true).open(&catalog) {
            Ok(f) => std::io::BufWriter::new(f),
            Err(e) => {
                eprintln!("error opening {}: {}", catalog.display(), e);
                return ExitCode::FAILURE;
            }
        };
        if is_new {
            writeln!(f, "# catalog24 append-only evidence log").ok();
            writeln!(f, "{HEADER}").ok();
        }
        for e in &fresh {
            writeln!(f, "{}", e.to_line()).ok();
        }
        f.flush().ok();
        eprintln!("ingested {} new evidence rows -> {}", fresh.len(), catalog.display());
    } else if !do_ingest && !fresh.is_empty() {
        eprintln!("(dry run: {} new evidence rows would be ingested; pass --ingest)", fresh.len());
    }

    // Combined view (existing + fresh, whether or not persisted).
    let n_fresh = fresh.len();
    existing.extend(fresh);
    let brackets = reduce(&existing);

    // Integrity: ub < lb is a solver bug.
    let mut bug = 0;
    for b in &brackets {
        if let (Some(lb), Some(ub)) = (b.best_lb, b.best_ub) {
            if ub < lb {
                eprintln!(
                    "WARNING: bracket inversion ub {} < lb {} on {} — solver bug!",
                    ub, lb, b.canon
                );
                bug += 1;
            }
        }
    }
    if bug > 0 {
        eprintln!("WARNING: {bug} bracket inversions (best_ub < best_lb)");
    }

    // Ranked view.
    let rank_n: usize = arg(&argv, "--rank", 50);
    println!(
        "catalog: {} boards, {} evidence rows ({} fresh this run)",
        brackets.len(),
        existing.len(),
        n_fresh
    );
    println!("{:>4}  {:>4} {:>4} {:>4}  {:>6} {:>6}  {:>4}  lineage", "rank", "lb", "ub", "gap", "v_fwd", "v_pair", "n");
    for (i, b) in brackets.iter().take(rank_n).enumerate() {
        let gap = match (b.best_lb, b.best_ub) {
            (Some(lb), Some(ub)) => (ub - lb).to_string(),
            _ => "-".to_string(),
        };
        let short = b.lineage.chars().take(48).collect::<String>();
        println!(
            "{:>4}  {:>4} {:>4} {:>4}  {:>6} {:>6}  {:>4}  {}",
            i,
            opt_i(b.best_lb),
            opt_i(b.best_ub),
            gap,
            opt_f(b.v_fwd),
            opt_f(b.v_pair),
            b.n_evidence,
            short
        );
    }

    // --reseed-out: top-K deepest boards for the next candidates24 generation.
    if let Some(path) = arg_opt(&argv, "--reseed-out") {
        let k: usize = arg(&argv, "--reseed-top", 15);
        let mut text = String::new();
        let _ = writeln!(text, "# catalog24 reseed: top {k} by proven LB (added {added})");
        for (i, b) in brackets.iter().take(k).enumerate() {
            let _ = writeln!(
                text,
                "# reseed#{} rank={} lb={} ub={} lineage={}",
                i, i, opt_i(b.best_lb), opt_i(b.best_ub), b.lineage
            );
            let _ = writeln!(text, "{}", b.board);
        }
        if let Err(e) = std::fs::write(&path, &text) {
            eprintln!("error writing {path}: {e}");
        } else {
            eprintln!("wrote {} reseed boards -> {}", brackets.len().min(k), path);
        }
    }

    // --escalate-out: wide-gap, high-UB boards that deserve a bigger LB budget.
    if let Some(path) = arg_opt(&argv, "--escalate-out") {
        let m: usize = arg(&argv, "--escalate-top", 6);
        let gap_min: i64 = arg(&argv, "--gap-min", 12);
        let mut cands: Vec<&Bracket> = brackets
            .iter()
            .filter(|b| match (b.best_lb, b.best_ub) {
                (Some(lb), Some(ub)) => ub - lb >= gap_min,
                _ => false,
            })
            .collect();
        // Prefer boards still budget-limited (all LB evidence timed out), then
        // higher UB (deeper-looking).
        cands.sort_by(|a, b| {
            b.lb_all_timeout
                .cmp(&a.lb_all_timeout)
                .then(b.best_ub.cmp(&a.best_ub))
        });
        let mut text = String::new();
        let _ = writeln!(
            text,
            "# catalog24 escalate: top {m} by (budget-limited, UB) with gap >= {gap_min} (added {added})"
        );
        for (i, b) in cands.iter().take(m).enumerate() {
            let _ = writeln!(
                text,
                "# escalate#{} lb={} ub={} gap={} timeout={} lineage={}",
                i,
                opt_i(b.best_lb),
                opt_i(b.best_ub),
                b.best_ub.unwrap() - b.best_lb.unwrap(),
                b.lb_all_timeout as u8,
                b.lineage
            );
            let _ = writeln!(text, "{}", b.board);
        }
        if let Err(e) = std::fs::write(&path, &text) {
            eprintln!("error writing {path}: {e}");
        } else {
            eprintln!("wrote {} escalation boards -> {}", cands.len().min(m), path);
        }
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(canon: &str, board: &str, kind: &str, value: f64, method: &str, replay: &str) -> Evidence {
        Evidence {
            added: "t1".into(),
            canon: canon.into(),
            board: board.into(),
            kind: kind.into(),
            value,
            method: method.into(),
            nodes: "-".into(),
            replay_ok: replay.into(),
            src: "test".into(),
            note: "-".into(),
        }
    }

    #[test]
    fn reduction_max_lb_min_verified_ub() {
        let c = "C";
        let all = vec![
            ev(c, "B", "lb", 130.0, "m1", "-"),
            ev(c, "B", "lb", 134.0, "m2", "-"),      // max lb
            ev(c, "B", "ub", 150.0, "u1", "1"),      // verified
            ev(c, "B", "ub", 148.0, "u2", "0"),      // unverified -> ignored for bracket
            ev(c, "B", "ub", 152.0, "u3", "1"),      // verified but larger
        ];
        let r = reduce(&all);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].best_lb, Some(134));
        assert_eq!(r[0].best_ub, Some(150)); // min of verified {150,152}, not 148
    }

    #[test]
    fn exact_pins_both_sides() {
        let all = vec![ev("C", "B", "exact", 88.0, "ladder24:select-k6:cheap", "1")];
        let r = reduce(&all);
        assert_eq!(r[0].best_lb, Some(88));
        assert_eq!(r[0].best_ub, Some(88));
    }

    #[test]
    fn reflection_join_merges_board_and_its_reflection() {
        // R and reflect(R) canonicalize to the same key.
        let mut r = [0u8; N_CELLS];
        for (i, slot) in r.iter_mut().enumerate().skip(1) {
            *slot = (25 - i) as u8;
        }
        let rb = State(r);
        let refl = puzzle8::puzzle24::symmetry::reflect(&rb);
        let ck_r = canon_str(&rb);
        let ck_refl = canon_str(&refl);
        assert_eq!(ck_r, ck_refl, "R and reflect(R) must share a canon key");
        let all = vec![
            ev(&ck_r, &board_str(&rb), "lb", 140.0, "m", "-"),
            ev(&ck_refl, &board_str(&refl), "ub", 156.0, "u", "1"),
        ];
        let red = reduce(&all);
        assert_eq!(red.len(), 1, "must merge to a single bracket");
        assert_eq!(red[0].best_lb, Some(140));
        assert_eq!(red[0].best_ub, Some(156));
    }

    #[test]
    fn dedup_key_is_idempotent() {
        let e = ev("C", "B", "lb", 134.0, "m", "-");
        let mut set: HashSet<_> = HashSet::new();
        assert!(set.insert(e.key()));
        let e2 = ev("C", "B", "lb", 134.0, "m", "-");
        assert!(!set.insert(e2.key()), "identical evidence must dedup");
        let e3 = ev("C", "B", "lb", 136.0, "m", "-");
        assert!(set.insert(e3.key()), "different value is new evidence");
    }

    #[test]
    fn parse_board_validates_permutation() {
        assert!(parse_board("0 1 2 3").is_none());
        // duplicate tile 1
        assert!(parse_board("1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24").is_none());
        assert!(parse_board("_ 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1").is_some());
    }

    #[test]
    fn val_str_formats_by_kind() {
        assert_eq!(val_str("lb", 134.0), "134");
        assert_eq!(val_str("score", 149.4), "149.4");
        assert_eq!(val_str("lineage", 0.0), "-");
    }
}
