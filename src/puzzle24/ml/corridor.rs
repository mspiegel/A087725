//! Geodesic-corridor data for value training and evaluation.
//!
//! The corridor diagnosis (data/corridor_r_compare.txt): V trained on
//! WD-maximal endpoint boards with loose walk labels misprices the states real
//! deep solutions pass through. This module provides (a) the corridor-dataset
//! file loader (produced by `gen_corridors`), and (b) the canonical **R-156
//! corridor** — the replicated published solution to `R` — used as a held-out
//! OOD probe during training (never train on it).

use crate::puzzle24::ml::eval::r_board;
use crate::puzzle24::state::{Move, State, GOAL, N_CELLS};

/// Optimal 78-STM solution of `W = ρ(GOAL)` (90° clockwise-rotated goal), found
/// by `solve24 --heuristic select --parallel` (12,189 nodes, 2.2 s, verified).
const W_SOLUTION: &str = "U U U U R R R D D D D L L L U U U U R R R R D D D D L L L L \
U U U U R R R R D D D L L L U U R R D D L L U U U R R R D D D D L L L L U U U U R R R R D D D D";

/// `W = ρ(GOAL)`: 90° clockwise rotation of the goal (blank at cell 20).
pub fn w_board() -> State {
    let mut c = [0u8; N_CELLS];
    for row in 0..5u8 {
        for col in 0..5u8 {
            let cell = (row * 5 + col) as usize;
            c[cell] = if (row, col) == (4, 0) { 0 } else { 21 + row - 5 * col };
        }
    }
    State(c)
}

fn parse_moves(s: &str) -> Vec<Move> {
    s.split_whitespace()
        .map(|t| match t {
            "U" => Move::Up,
            "D" => Move::Down,
            "L" => Move::Left,
            "R" => Move::Right,
            other => panic!("bad move token {:?}", other),
        })
        .collect()
}

/// Rotate a move direction 90° clockwise.
fn cw(m: Move) -> Move {
    match m {
        Move::Up => Move::Right,
        Move::Right => Move::Down,
        Move::Down => Move::Left,
        Move::Left => Move::Up,
    }
}

/// The replicated 156-move solution to `R` (conjugated half + W half),
/// replay-verified on construction. See corridor_r / PUZZLE24.md appendix.
pub fn r156_moves() -> Vec<Move> {
    let s78 = parse_moves(W_SOLUTION);
    let w = w_board();
    debug_assert_eq!(s78.len(), 78);
    // R -> W: the W-solution conjugated by the rotation (try both directions;
    // exactly one replays R -> W).
    let conj: Vec<Move> = s78.iter().map(|&m| cw(m)).collect();
    let first_half = if replay(&r_board(), &conj) == w {
        conj
    } else {
        let ccw: Vec<Move> = s78.iter().map(|&m| cw(cw(cw(m)))).collect();
        assert_eq!(replay(&r_board(), &ccw), w, "conjugation failed both ways");
        ccw
    };
    let mut moves = first_half;
    moves.extend_from_slice(&s78);
    assert_eq!(replay(&r_board(), &moves), GOAL, "r156 replay failed");
    moves
}

fn replay(start: &State, moves: &[Move]) -> State {
    let mut s = *start;
    for &m in moves {
        s = s.apply(m);
    }
    s
}

/// The R-156 corridor as `(state, remaining)` pairs, `remaining = 156 − k`
/// (exact if `optimal(R) = 156`; proven ∈ [152, 156]). **Held-out OOD probe —
/// never feed these to training.**
pub fn r156_states() -> Vec<(State, u32)> {
    let moves = r156_moves();
    let len = moves.len() as u32;
    let mut out = Vec::with_capacity(moves.len() + 1);
    let mut s = r_board();
    out.push((s, len));
    for (k, &m) in moves.iter().enumerate() {
        s = s.apply(m);
        out.push((s, len - (k as u32 + 1)));
    }
    out
}

/// Load a `gen_corridors` dataset file: lines `rem exact_flag t0 … t24`.
/// Returns `(state, label)` pairs (the exact flag is not currently used in
/// training — both kinds are regression targets).
pub fn load_file(path: &std::path::Path) -> Result<Vec<(State, f32)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() != 2 + N_CELLS {
            return Err(format!("{}:{}: expected {} tokens, got {}", path.display(), ln + 1, 2 + N_CELLS, toks.len()));
        }
        let rem: f32 = toks[0].parse().map_err(|e| format!("{}:{}: rem: {}", path.display(), ln + 1, e))?;
        let mut cells = [0u8; N_CELLS];
        for (i, t) in toks[2..].iter().enumerate() {
            cells[i] = t.parse().map_err(|e| format!("{}:{}: tile: {}", path.display(), ln + 1, e))?;
        }
        let s = State(cells);
        if !s.is_solvable() {
            return Err(format!("{}:{}: unsolvable state", path.display(), ln + 1));
        }
        out.push((s, rem));
    }
    Ok(out)
}

/// Mean `V − rem` over the R-156 corridor in three depth bands — the standing
/// OOD calibration probe logged each eval round. Returns
/// `(band_120_156, band_80_119, band_40_79)`.
pub fn r156_verr<F>(value_of: &F) -> (f32, f32, f32)
where
    F: Fn(&[State]) -> Vec<f32>,
{
    let pairs = r156_states();
    let states: Vec<State> = pairs.iter().map(|&(s, _)| s).collect();
    let vs = value_of(&states);
    let mut sums = [0f32; 3];
    let mut ns = [0usize; 3];
    for ((_, rem), v) in pairs.iter().zip(vs.iter()) {
        let band = match rem {
            120..=156 => 0,
            80..=119 => 1,
            40..=79 => 2,
            _ => continue,
        };
        sums[band] += v - *rem as f32;
        ns[band] += 1;
    }
    (
        sums[0] / ns[0].max(1) as f32,
        sums[1] / ns[1].max(1) as f32,
        sums[2] / ns[2].max(1) as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r156_replays_and_is_monotone() {
        let states = r156_states();
        assert_eq!(states.len(), 157);
        assert_eq!(states[0].1, 156);
        assert_eq!(states[156].1, 0);
        assert_eq!(states[156].0, GOAL);
        assert_eq!(states[78].0, w_board(), "midpoint must be W");
        for (s, _) in &states {
            assert!(s.is_solvable());
        }
    }

    #[test]
    fn corridor_file_round_trip() {
        let path = std::env::temp_dir().join(format!("corr_{}.txt", std::process::id()));
        let states = r156_states();
        let mut text = String::from("# comment\n");
        for (s, rem) in states.iter().take(5) {
            let tiles: Vec<String> = s.0.iter().map(|t| t.to_string()).collect();
            text.push_str(&format!("{} 1 {}\n", rem, tiles.join(" ")));
        }
        std::fs::write(&path, text).unwrap();
        let loaded = load_file(&path).unwrap();
        assert_eq!(loaded.len(), 5);
        assert_eq!(loaded[0].0, states[0].0);
        assert_eq!(loaded[0].1, 156.0);
        let _ = std::fs::remove_file(&path);
    }
}
