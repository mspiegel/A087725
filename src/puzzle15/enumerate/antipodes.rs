//! Parse the 17 depth-80 antipodes from `data/pdb15_antipodes.txt` — the seeds
//! for the top-down frontier.

use std::path::Path;

use crate::puzzle15::rank::rank;
use crate::puzzle15::state::{State, N_CELLS};

/// Parse one 16-token row-major board line (`_` or `0` = blank, `1..=15`
/// tiles) into a [`State`].
fn parse_board(line: &str) -> Result<State, String> {
    let mut cells = [0u8; N_CELLS];
    let mut seen = [false; N_CELLS];
    let mut count = 0usize;
    for tok in line.split_whitespace() {
        if count >= N_CELLS {
            return Err(format!("more than {N_CELLS} tokens in {line:?}"));
        }
        let v = if tok == "_" || tok == "." {
            0
        } else {
            tok.parse::<u8>().map_err(|e| format!("token {tok:?}: {e}"))?
        };
        if v as usize >= N_CELLS {
            return Err(format!("value {v} out of range in {line:?}"));
        }
        if seen[v as usize] {
            return Err(format!("value {v} repeats in {line:?}"));
        }
        seen[v as usize] = true;
        cells[count] = v;
        count += 1;
    }
    if count != N_CELLS {
        return Err(format!("expected {N_CELLS} tokens, got {count} in {line:?}"));
    }
    Ok(State(cells))
}

/// Load the antipodes as ranks. Each must be solvable; otherwise an error.
pub fn load_ranks(path: &Path) -> Result<Vec<u64>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut ranks = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let s = parse_board(line)?;
        if !s.is_solvable() {
            return Err(format!("antipode not solvable: {:?}", s.0));
        }
        ranks.push(rank(&s));
    }
    Ok(ranks)
}
