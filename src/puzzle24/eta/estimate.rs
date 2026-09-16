//! Estimators for stratified η sampling.
//!
//! Every stratum is estimated from independent *attempts*. An attempt that does
//! not yield a sample (a rejected or dead-ended walk) contributes `X = 0`; an
//! accepted sample `v` drawn with probability `P(v)` contributes `X = f(v)/P(v)`.
//! Then `E[X] = Σ_{v ∈ stratum} f(v)`, the Horvitz–Thompson estimator: with
//! `f = 1` it estimates the stratum size, with `f = w(v)·b^−h(v)` it estimates
//! `|V|·ηₖ`. This is Clausecker & Schintke's SoCS Eq. 18 once its yield and
//! stratum-size factors cancel.
//!
//! Confidence intervals come from the sample variance of the per-attempt `X`,
//! which includes the dispersion of the `1/P(v)` weights. The effective sample
//! size and the largest single term's share flag strata where a few heavy
//! weights dominate and the normal interval is not trustworthy.

/// Two-sided 95% normal quantile.
pub const Z95: f64 = 1.959_963_984_540_054;

/// Running moments of per-attempt values `X`, including zeros.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Accum {
    /// Attempts, including those contributing zero.
    pub n: u64,
    pub sum: f64,
    pub sum_sq: f64,
    /// Largest single `X`.
    pub max: f64,
}

impl Accum {
    #[inline]
    pub fn add(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sum_sq += x * x;
        if x > self.max {
            self.max = x;
        }
    }

    /// Record `count` attempts that contributed zero.
    #[inline]
    pub fn add_zeros(&mut self, count: u64) {
        self.n += count;
    }

    pub fn merge(&mut self, other: &Accum) {
        self.n += other.n;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.max = self.max.max(other.max);
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }

    /// Unbiased sample variance of `X`.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        ((self.sum_sq - self.sum * self.sum / n) / (n - 1.0)).max(0.0)
    }

    /// Standard error of [`mean`](Self::mean).
    pub fn std_error(&self) -> f64 {
        (self.variance() / self.n as f64).sqrt()
    }

    /// Kish effective sample size of the nonzero terms, `(ΣX)² / ΣX²`.
    pub fn effective_n(&self) -> f64 {
        if self.sum_sq == 0.0 {
            0.0
        } else {
            self.sum * self.sum / self.sum_sq
        }
    }

    /// Share of `ΣX` carried by the single largest term.
    pub fn max_share(&self) -> f64 {
        if self.sum == 0.0 {
            0.0
        } else {
            self.max / self.sum
        }
    }
}

/// Paired moments of `(A, B)` per attempt, for the ratio estimator
/// `T_B · ΣA / ΣB` when the true total `T_B` (e.g. an exact stratum size) is
/// known.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PairAccum {
    pub a: Accum,
    pub b: Accum,
    pub sum_ab: f64,
}

impl PairAccum {
    #[inline]
    pub fn add(&mut self, a: f64, b: f64) {
        self.a.add(a);
        self.b.add(b);
        self.sum_ab += a * b;
    }

    #[inline]
    pub fn add_zeros(&mut self, count: u64) {
        self.a.add_zeros(count);
        self.b.add_zeros(count);
    }

    pub fn merge(&mut self, other: &PairAccum) {
        self.a.merge(&other.a);
        self.b.merge(&other.b);
        self.sum_ab += other.sum_ab;
    }

    /// Ratio estimate `T_B · ΣA / ΣB` and its delta-method standard error.
    pub fn ratio(&self, total_b: f64) -> (f64, f64) {
        let n = self.a.n as f64;
        if self.b.sum == 0.0 || n < 2.0 {
            return (f64::NAN, f64::NAN);
        }
        let r = self.a.sum / self.b.sum;
        // Residuals e_i = A_i − r·B_i have mean 0 by construction.
        let ss = self.a.sum_sq - 2.0 * r * self.sum_ab + r * r * self.b.sum_sq;
        let var_e = (ss / (n - 1.0)).max(0.0);
        let mean_b = self.b.sum / n;
        let se_r = (var_e / n).sqrt() / mean_b;
        (total_b * r, total_b * se_r)
    }
}

/// Batch-means estimate from per-chunk totals `(attempts, sum of X)`.
///
/// Chunks are dealt round-robin into `batches` groups; each group gives an
/// independent per-attempt mean, and the standard error is the spread of the
/// group means over `√batches`. Unlike [`Accum::std_error`], this does not
/// lean on the sample variance of a heavy-tailed `X`, which understates the
/// error when a few terms dominate. Returns `(mean, standard error)`; the
/// error is NaN with fewer than two non-empty batches.
pub fn batch_means(per_chunk: &[(u64, f64)], batches: usize) -> (f64, f64) {
    let batches = batches.max(1);
    let mut att = vec![0u64; batches];
    let mut sum = vec![0.0f64; batches];
    for (i, &(a, s)) in per_chunk.iter().enumerate() {
        att[i % batches] += a;
        sum[i % batches] += s;
    }
    let total_att: u64 = att.iter().sum();
    let mean = if total_att == 0 {
        0.0
    } else {
        sum.iter().sum::<f64>() / total_att as f64
    };
    let means: Vec<f64> = att
        .iter()
        .zip(&sum)
        .filter(|(&a, _)| a > 0)
        .map(|(&a, &s)| s / a as f64)
        .collect();
    if means.len() < 2 {
        return (mean, f64::NAN);
    }
    let m = means.iter().sum::<f64>() / means.len() as f64;
    let var = means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (means.len() - 1) as f64;
    (mean, (var / means.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::eta::rng::Rng;

    #[test]
    fn batch_means_matches_direct_mean_and_scales_error() {
        // 400 chunks of 100 attempts; X = 1 with probability 0.1, else 0.
        let mut rng = Rng::stream(8, 0, 0);
        let per_chunk: Vec<(u64, f64)> = (0..400)
            .map(|_| {
                let hits = (0..100).filter(|_| rng.below(10) == 0).count();
                (100, hits as f64)
            })
            .collect();
        let (mean, se) = batch_means(&per_chunk, 20);
        let direct: f64 = per_chunk.iter().map(|c| c.1).sum::<f64>() / 40_000.0;
        assert!((mean - direct).abs() < 1e-15);
        // Binomial standard error of the mean: sqrt(0.09 / 40000) = 0.0015.
        assert!(se > 0.0008 && se < 0.0025, "se {se}");
        assert!((mean - 0.1).abs() < 5.0 * 0.0015);
        let (_, se1) = batch_means(&per_chunk, 1);
        assert!(se1.is_nan());
    }

    #[test]
    fn accum_moments() {
        let mut a = Accum::default();
        for x in [1.0, 2.0, 3.0, 4.0] {
            a.add(x);
        }
        a.add_zeros(4);
        assert_eq!(a.n, 8);
        assert!((a.mean() - 1.25).abs() < 1e-15);
        // Values 1,2,3,4,0,0,0,0: Σx² = 30, variance = (30 − 8·1.25²)/7.
        assert!((a.variance() - (30.0 - 12.5) / 7.0).abs() < 1e-12);
        assert!((a.effective_n() - 100.0 / 30.0).abs() < 1e-12);
        assert!((a.max_share() - 0.4).abs() < 1e-15);
        let mut b = Accum::default();
        b.add(10.0);
        a.merge(&b);
        assert_eq!(a.n, 9);
        assert_eq!(a.max, 10.0);
    }

    /// Horvitz–Thompson on a toy population: draw item i with probability p_i
    /// (or nothing), and check the size and total estimates cover the truth.
    #[test]
    fn horvitz_thompson_recovers_population_totals() {
        let p = [0.05, 0.1, 0.2, 0.3]; // Σ = 0.65: 35% of attempts yield nothing
        let f = [7.0, 1.0, 3.0, 0.5];
        let mut size = Accum::default();
        let mut total = Accum::default();
        let mut pair = PairAccum::default();
        let mut rng = Rng::stream(42, 0, 0);
        for _ in 0..400_000 {
            let u = rng.next_u64() as f64 / 2f64.powi(64);
            let mut acc = 0.0;
            let mut hit = None;
            for (i, &pi) in p.iter().enumerate() {
                acc += pi;
                if u < acc {
                    hit = Some(i);
                    break;
                }
            }
            match hit {
                Some(i) => {
                    size.add(1.0 / p[i]);
                    total.add(f[i] / p[i]);
                    pair.add(f[i] / p[i], 1.0 / p[i]);
                }
                None => {
                    size.add_zeros(1);
                    total.add_zeros(1);
                    pair.add_zeros(1);
                }
            }
        }
        let truth: f64 = f.iter().sum();
        assert!((size.mean() - 4.0).abs() < 4.0 * size.std_error() + 1e-12);
        assert!((total.mean() - truth).abs() < 4.0 * total.std_error());
        let (r, se) = pair.ratio(4.0);
        assert!((r - truth).abs() < 4.0 * se, "ratio {r} ± {se}");
    }
}
