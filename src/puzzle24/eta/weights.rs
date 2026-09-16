//! Branching factor `b` and blank-cell weights `w(v)` for η.
//!
//! η weights each state by how often it occurs "at equilibrium". On a sliding
//! puzzle every quantity involved depends only on the blank's cell, so a
//! weighting is a vector over cells, normalised to mean 1 (Clausecker's
//! `Σ_v w(v) = |V|`, with each cell holding `|V|/25` states).
//!
//! - [`Weighting::Uniform`] — `w = 1`. Clausecker's η_perfect (2.5063×10⁻²⁴) is
//!   reproduced only with this weighting.
//! - [`Weighting::Tree`] — the node distribution deep in the brute-force search
//!   tree with inverse-move pruning (Korf, Reid & Edelkamp 2001). Its growth rate
//!   is the asymptotic branching factor `b`.
//! - [`Weighting::Degree`] — the stationary distribution of a simple random walk
//!   on the state graph, proportional to the blank's degree.

use crate::puzzle24::state::{N_CELLS, W};

/// Which equilibrium weighting to apply to the blank cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    Uniform,
    Tree,
    Degree,
}

impl Weighting {
    pub const ALL: [Weighting; 3] = [Weighting::Uniform, Weighting::Tree, Weighting::Degree];

    pub fn name(self) -> &'static str {
        match self {
            Weighting::Uniform => "uniform",
            Weighting::Tree => "tree",
            Weighting::Degree => "degree",
        }
    }
}

/// Power iterations for the tree equilibrium. The chain has at most 4·25+25
/// states; convergence to 1e-15 takes a few hundred steps.
const POWER_ITERATIONS: usize = 20_000;

/// Neighbour cells of `c` on a `width × width` grid, indexed by move code
/// (Up, Down, Left, Right); `None` where the move leaves the board.
fn grid_neighbours(width: usize, c: usize) -> [Option<usize>; 4] {
    let (r, k) = (c / width, c % width);
    [
        (r > 0).then(|| c - width),
        (r + 1 < width).then(|| c + width),
        (k > 0).then(|| c - 1),
        (k + 1 < width).then(|| c + 1),
    ]
}

/// Asymptotic branching factor with inverse-move pruning and the per-cell
/// share of deep tree nodes, normalised to mean 1, for a `width × width` board.
///
/// The chain's state is `(cell, incoming move)`, plus a root state per cell.
/// Blank moves alternate between two colour classes, so the child-count matrix
/// `A` has eigenvalues `±λ` and plain power iteration oscillates; iterating
/// `A + I` converges to the Perron vector with eigenvalue `λ + 1`.
pub fn tree_equilibrium(width: usize) -> (f64, Vec<f64>) {
    const ROOT: usize = 4;
    let n = width * width;
    let idx = |cell: usize, inc: usize| cell * 5 + inc;
    let mut v = vec![1.0f64; n * 5];
    let mut lam = 0.0;
    for _ in 0..POWER_ITERATIONS {
        let mut next = v.clone();
        for c in 0..n {
            for inc in 0..5 {
                let x = v[idx(c, inc)];
                if x == 0.0 {
                    continue;
                }
                for (m, nb) in grid_neighbours(width, c).into_iter().enumerate() {
                    let Some(c2) = nb else { continue };
                    if inc != ROOT && m == (inc ^ 1) {
                        continue;
                    }
                    next[idx(c2, m)] += x;
                }
            }
        }
        let total: f64 = next.iter().sum();
        lam = total / v.iter().sum::<f64>() - 1.0;
        for x in &mut next {
            *x /= total;
        }
        v = next;
    }
    let mut share: Vec<f64> = (0..n).map(|c| (0..5).map(|i| v[idx(c, i)]).sum()).collect();
    let sum: f64 = share.iter().sum();
    for s in &mut share {
        *s *= n as f64 / sum;
    }
    (lam, share)
}

/// Blank-cell weights for the 24-puzzle, mean 1 over the 25 cells.
pub fn blank_weights(weighting: Weighting) -> [f64; N_CELLS] {
    let mut w = [1.0f64; N_CELLS];
    match weighting {
        Weighting::Uniform => {}
        Weighting::Tree => {
            let (_, share) = tree_equilibrium(W);
            w.copy_from_slice(&share);
        }
        Weighting::Degree => {
            let deg: Vec<f64> = (0..N_CELLS)
                .map(|c| grid_neighbours(W, c).iter().flatten().count() as f64)
                .collect();
            let sum: f64 = deg.iter().sum();
            for (wc, d) in w.iter_mut().zip(&deg) {
                *wc = d * N_CELLS as f64 / sum;
            }
        }
    }
    w
}

/// The 24-puzzle's asymptotic branching factor with inverse-move pruning.
pub fn branching_factor() -> f64 {
    tree_equilibrium(W).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branching_factors_match_known_values() {
        // 8-puzzle: √3 (Edelkamp & Korf 1998). 15-puzzle: 2.1304.
        // 24-puzzle: 2.3676, the value Clausecker's η_perfect implies.
        let (b3, _) = tree_equilibrium(3);
        assert!((b3 - 3f64.sqrt()).abs() < 1e-12, "b3 = {b3}");
        let (b4, _) = tree_equilibrium(4);
        assert!((b4 - 2.130395434767).abs() < 1e-9, "b4 = {b4}");
        let b5 = branching_factor();
        assert!((b5 - 2.367604543724).abs() < 1e-9, "b5 = {b5}");
    }

    #[test]
    fn weights_have_mean_one_and_board_symmetry() {
        for wt in Weighting::ALL {
            let w = blank_weights(wt);
            let mean: f64 = w.iter().sum::<f64>() / N_CELLS as f64;
            assert!((mean - 1.0).abs() < 1e-12, "{} mean {mean}", wt.name());
            for c in 0..N_CELLS {
                let (r, k) = (c / W, c % W);
                let mirror = r * W + (W - 1 - k);
                let transpose = k * W + r;
                assert!((w[c] - w[mirror]).abs() < 1e-12);
                assert!((w[c] - w[transpose]).abs() < 1e-12);
            }
        }
        let tree = blank_weights(Weighting::Tree);
        assert!((tree[0] - 0.624881270497095).abs() < 1e-9);
        assert!((tree[12] - 1.47271341229645).abs() < 1e-9);
        assert_eq!(blank_weights(Weighting::Degree)[0], 0.625);
    }
}
