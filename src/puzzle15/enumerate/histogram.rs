//! Parse the complete A087725 depth distribution (`N(d)`, the termination
//! oracle) from `data/pdb15_depth_histogram.txt`.

use std::path::Path;

use crate::puzzle15::state::DIAMETER;

/// `N(d)` for `d` in `0..=80`, the exact number of solvable 15-puzzle states at
/// optimal depth `d`.
pub type Histogram = [u64; DIAMETER as usize + 1];

/// Read the histogram file. Each non-comment, non-blank line is `<depth>
/// <count>`. Returns an error string on a malformed line or a missing depth.
pub fn load(path: &Path) -> Result<Histogram, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut hist: Histogram = [0; DIAMETER as usize + 1];
    let mut seen = [false; DIAMETER as usize + 1];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let d: usize = it
            .next()
            .ok_or("empty data line")?
            .parse()
            .map_err(|e| format!("bad depth in {line:?}: {e}"))?;
        let n: u64 = it
            .next()
            .ok_or_else(|| format!("missing count in {line:?}"))?
            .parse()
            .map_err(|e| format!("bad count in {line:?}: {e}"))?;
        if d > DIAMETER as usize {
            return Err(format!("depth {d} exceeds diameter {DIAMETER}"));
        }
        hist[d] = n;
        seen[d] = true;
    }
    if let Some(d) = seen.iter().position(|&s| !s) {
        return Err(format!("histogram missing depth {d}"));
    }
    Ok(hist)
}
