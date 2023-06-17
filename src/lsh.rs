// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.
use rand::{
    distributions::{Distribution, Standard},
    seq::IteratorRandom,
    Rng,
};

/// Hamming distance between two values.
fn hamming(a: usize, b: usize) -> u32 {
    (a ^ b).count_ones()
}

/// 2^n.
pub const fn pow2(n: usize) -> usize {
    2usize.pow(n as u32)
}

// Given `N` vectors, we convert each one into a binary vector via projection onto the set of
// random hyperplanes. With those binary vectors, we sort them into groups and place them
// within a contigious array. We note the beginning of each group within `bin_idx`, which
// indicates the index to begin at within `buf`.

/// An LSH structure containing `NB` random hyperplanes, `N` vectors of `T` type and `D` dimension,
/// and utilizing `I` to identify them.
pub struct LSHDB<'a, const NB: usize, const N: usize, T, const D: usize, I>
where
    [(); pow2(NB)]:,
{
    hyperplane_normals: [[T; D]; NB],
    bin_idx: [Option<usize>; pow2(NB)], // contains the index where the bin begins within `buf`.
    buf: [I; N],
}

impl<'a, const NB: usize, const N: usize, T, const D: usize, I> LSHDB<'a, NB, N, T, D, I>
where
    Standard: Distribution<T>,
    T: hyperplane::ProjLSH<T, D>,
    [(); pow2(NB)]:,
    I: Copy,
{
    // TODO: Test this behavior.
    /// Find the start and end of the bin within the `buf` arr.
    fn bin_range(&self, idx: usize) -> std::ops::Range<usize> {
        assert!(idx < pow2(NB));

        if let Some(beg_buf_idx) = self.bin_idx.get(idx).unwrap() {
            let end_buf_idx = self
                .bin_idx
                .iter()
                .skip(idx + 1) // Iteration begins at `idx + 1`;
                .find(|&x| x.is_some())
                .copied()
                .unwrap_or(Some(N))
                .unwrap();

            // If there is a `beg_buf_idx`, there is at least a single vector, so the range
            // end should be different.
            assert!(*beg_buf_idx != end_buf_idx);

            std::ops::Range {
                start: *beg_buf_idx,
                end: end_buf_idx,
            }
        } else {
            // There are no values within this bin.
            todo!("There were no values within this bin; find another bin with a similar hamming distance...");
        }
    }

    // TODO: lshdb that doesn't take the entirety of the vectors at runtime.
    pub fn new<R: Rng>(rng: &mut R, vectors: &[(I, [T; D]); N]) -> Self {
        // Costruct `NB` hyperplanes.
        let build_hyperplanes = |rng: &mut R| -> [[T; D]; NB] {
            let hyperplane_normals = {
                // TODO: What's the 'proper' way to initialize this?
                let mut data: [std::mem::MaybeUninit<[T; D]>; NB] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                // TODO: Should this be a random gaussian vector? A random unit normal?
                for component in &mut data[..] {
                    let hn = hyperplane::random_hyperplane_normal(rng);
                    component.write(hn);
                }

                unsafe { data.as_ptr().cast::<[[T; D]; NB]>().read() }
            };

            hyperplane_normals
        };

        // Build the underlying `buf` containing vector identifiers alongside `bin_idx` which points
        // to the index within `buf` where a specific bin begins.
        let build_bin_idx_buf =
            |hyperplane_normals: &[[T; D]; NB]| -> ([Option<usize>; pow2(NB)], [I; N]) {
                // For each vector, reduce to the binary vector via computing the random
                // projection against our hyperplanes.
                // TODO: Proper way to initialize `projections`?
                let mut projections: [(I, usize); N] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                for i in 0..N {
                    let mem = projections.get_mut(i).unwrap();
                    let (ident, query_vector) = vectors.get(i).unwrap();
                    *mem = (
                        *ident,
                        hyperplane::hyperplane_project(hyperplane_normals, query_vector),
                    );
                }

                // Sort the projections by bin_idx.
                projections.sort_unstable_by(|a, b| a.1.cmp(&b.1));

                // For `i` in 0..pow2(NB), find the first index where `i` appears.
                // NOTE: the current implementation of `find` short-circuits which is what we desire.
                // TODO: Proper way to initialize `bin_idx`?
                let mut bin_idx = [None; pow2(NB)];
                for i in 0..pow2(NB) {
                    let mem = bin_idx.get_mut(i).unwrap();
                    match projections
                        .iter()
                        .enumerate()
                        .find(|(_proj_idx, &(_ident, idx))| i == idx)
                    {
                        Some((proj_idx, _)) => *mem = Some(proj_idx),
                        None => *mem = None,
                    }
                }

                // Drop the query vector so that `I` is all that remains.
                // TODO: Proper way to initialize `buf`?
                let mut buf: [I; N] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                for idx in 0..N {
                    let mem = buf.get_mut(idx).unwrap();
                    *mem = projections.get(idx).unwrap().0;
                }

                (bin_idx, buf)
            };

        let hyperplane_normals = build_hyperplanes(rng);
        let (bin_idx, buf) = build_bin_idx_buf(&hyperplane_normals);
        Self {
            hyperplane_normals,
            bin_idx,
            buf,
        }
    }

    /// Find an approximate nearest neigh for an input vector.
    pub fn ann(&'a self, q: &[T; D]) -> impl Iterator<Item = &'a I> {
        let idx = hyperplane::hyperplane_project(&self.hyperplane_normals, q);
        let range = self.bin_range(idx);
        // TODO: Is there a way to pass the range into `.iter()` ?
        self.buf.iter().skip(range.start).take(range.end)
    }

    /// Find an approximate nearest neighbor for an input vector. NOTE: this will
    /// return a random vector from a *single bin*.
    pub fn ann_rand<R: Rng>(&'a self, rng: &mut R, q: &[T; D]) -> Option<&'a I> {
        self.ann(q).choose(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming() {
        assert_eq!(hamming(1, 1), 0);
        assert_eq!(hamming(2, 0), 1);
    }

    #[test]
    fn test_pow2() {
        assert_eq!(pow2(2), 4);
        assert_eq!(pow2(4), 16);
    }
}

// TODO: This should be hidden private and exposed as pub when a cfg feature like
// 'benchmark' is enabled.
pub mod hyperplane {
    use rand::{
        distributions::{Distribution, Standard},
        Rng,
    };

    #[derive(Debug)]
    pub enum Sign {
        Positive,
        Negative,
    }

    impl Into<usize> for Sign {
        fn into(self) -> usize {
            match self {
                Sign::Positive => 1,
                Sign::Negative => 0,
            }
        }
    }

    impl From<f32> for Sign {
        fn from(x: f32) -> Self {
            if x > 0.0f32 {
                Sign::Positive
            } else {
                Sign::Negative
            }
        }
    }

    impl From<f64> for Sign {
        fn from(x: f64) -> Self {
            if x > 0.0f64 {
                Sign::Positive
            } else {
                Sign::Negative
            }
        }
    }

    pub trait ProjLSH<T, const D: usize> {
        fn proj(a: &[T; D], b: &[T; D]) -> Sign;
    }

    impl<const D: usize> ProjLSH<f32, D> for f32 {
        fn proj(a: &[f32; D], b: &[f32; D]) -> Sign {
            (0..D)
                .fold(0.0f32, |acc, idx| {
                    acc + (a.get(idx).unwrap() + b.get(idx).unwrap()) // dot-product.
                })
                .into()
        }
    }

    impl<const D: usize> ProjLSH<f64, D> for f64 {
        fn proj(a: &[f64; D], b: &[f64; D]) -> Sign {
            (0..D)
                .fold(0.0f64, |acc, idx| {
                    acc + (a.get(idx).unwrap() + b.get(idx).unwrap()) // dot-product.
                })
                .into()
        }
    }

    // TODO: This should be hidden private and exposed as pub when a cfg feature like
    // 'benchmark' is enabled; should be pub(super).
    /// Create a normal for a random hyperplane.
    pub fn random_hyperplane_normal<T, const D: usize, R: Rng>(rng: &mut R) -> [T; D]
    where
        Standard: Distribution<T>,
    {
        let buf = {
            // TODO: What's the 'proper' way to initialize this?
            let mut data: [std::mem::MaybeUninit<T>; D] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            // TODO: Should this be a random gaussian vector? A random unit normal?
            for component in &mut data[..] {
                component.write(rng.gen::<T>());
            }
            unsafe { data.as_ptr().cast::<[T; D]>().read() }
        };

        buf
    }

    // TODO: this should be tested.
    // TODO: This should be hidden private and exposed as pub when a cfg feature like
    // 'benchmark' is enabled; should be pub(super).
    /// Given a set of hyperplane normals, compute the binary vector (usize) representation.
    pub fn hyperplane_project<T, const D: usize>(normals: &[[T; D]], query: &[T; D]) -> usize
    where
        T: ProjLSH<T, D>,
    {
        normals
            .iter()
            .map(|nrml| T::proj(query, nrml))
            .enumerate()
            .fold(0usize, |acc, (idx, value)| {
                acc + (Into::<usize>::into(value) << idx)
            })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn build_random_hyperplane_normal() {
            let mut rng = rand::thread_rng();
            let _: [f32; 4] = random_hyperplane_normal(&mut rng);
        }

        /*
        #[test]
        fn test_hyperplane_project() {
            todo!("Implement hyperplane projection.");
        }
        */
    }
}
