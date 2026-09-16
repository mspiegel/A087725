//! Deterministic, splittable pseudo-random numbers for the samplers.
//!
//! xoshiro256\*\* (Blackman & Vigna), seeded through SplitMix64. Every sampling
//! stream is keyed by `(seed, stratum, chunk)`, so a run is reproducible and a
//! chunk can be regenerated on its own (the tail stratum stores only its seed).

/// SplitMix64 step: advances `x` and returns the next output.
#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256\*\* generator.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Generator for one sampling stream. Distinct `(seed, stratum, chunk)`
    /// triples give independent-looking streams.
    pub fn stream(seed: u64, stratum: u64, chunk: u64) -> Rng {
        let mut x = seed;
        let a = splitmix64(&mut x);
        let mut x = a ^ stratum.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let b = splitmix64(&mut x);
        let mut x = b ^ chunk.wrapping_mul(0xA076_1D64_78BD_642F);
        let mut s = [0u64; 4];
        for v in &mut s {
            *v = splitmix64(&mut x);
        }
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n` with no modulo bias (Lemire's multiply-shift
    /// with rejection). Panics if `n == 0`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0)");
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = (self.next_u64() as u128) * (n as u128);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_deterministic_and_distinct() {
        let a: Vec<u64> = (0..4)
            .map({
                let mut r = Rng::stream(7, 3, 11);
                move |_| r.next_u64()
            })
            .collect();
        let b: Vec<u64> = (0..4)
            .map({
                let mut r = Rng::stream(7, 3, 11);
                move |_| r.next_u64()
            })
            .collect();
        assert_eq!(a, b);
        let mut c = Rng::stream(7, 3, 12);
        let mut d = Rng::stream(7, 4, 11);
        assert_ne!(a[0], c.next_u64());
        assert_ne!(a[0], d.next_u64());
    }

    #[test]
    fn below_is_uniform_for_small_n() {
        // Chi-square over 3 buckets, 300k draws: the 99.9% critical value for
        // 2 degrees of freedom is 13.8.
        let mut r = Rng::stream(1, 2, 3);
        let n = 300_000u64;
        let mut counts = [0u64; 3];
        for _ in 0..n {
            counts[r.below(3) as usize] += 1;
        }
        let e = n as f64 / 3.0;
        let chi2: f64 = counts.iter().map(|&c| (c as f64 - e).powi(2) / e).sum();
        assert!(chi2 < 13.8, "chi2 = {chi2}, counts = {counts:?}");
        for _ in 0..1000 {
            assert_eq!(r.below(1), 0);
        }
    }
}
