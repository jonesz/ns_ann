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
pub struct LSHDB<const NB: usize, const N: usize, T, const D: usize, I>
where
    [(); pow2(NB)]:,
{
    hyperplane_normals: [[T; D]; NB],
    bin_idx: [Option<usize>; pow2(NB)], // contains the index where the bin begins within `buf`.
    buf: [I; N],
}

impl<const NB: usize, const N: usize, T, const D: usize, I> LSHDB<NB, N, T, D, I>
where
    Standard: Distribution<T>,
    T: hyperplane::ProjLSH<T, D>,
    [(); pow2(NB)]:,
    I: Copy,
{
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

    /// Find an approximate nearest neighbor for an input vector. NOTE: this will
    /// return a random vector from a *single bin*.
    pub fn ann<R: Rng>(&self, rng: &mut R, q: &[T; D]) -> Option<I> {
        let idx = hyperplane::hyperplane_project(&self.hyperplane_normals, q);
        if let Some(buf_idx) = self.bin_idx.get(idx).unwrap() {
            // Find the beginning an end of the bin within the `buf` arr, then select
            // a random identifier.
            let beg: usize = *buf_idx;
            let end: usize = {
                let mut ctr: usize = *buf_idx + 1;
                loop {
                    if let Some(x) = self.bin_idx.get(ctr) {
                        match x {
                            Some(value) => break *value,
                            None => {
                                ctr = ctr + 1;
                                continue;
                            }
                        }
                    } else {
                        break N; // the bin was the last index: the `end` is `N`.
                    }
                }
            };

            self.buf
                .iter()
                .skip(beg) // range begins at `beg`
                .take(end - beg) // range continues for `end - beg` values.
                .choose(rng)
                .copied() // `Option<&I>` to `Option<I>`.
        } else {
            None // TODO: If we don't find any vectors within this bin, attempt to find one in a similar bin?
        }
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

    #[test]
    fn test_LSHDB_new() {
        let mut rng = rand::thread_rng();
        // let _: LSHDB<4, f32, 8> = LSHDB::new(&mut rng);
        todo!("Rather than utilize thread_rng, use a reproducible seed.");
    }
}

mod hyperplane {
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

    /// Create a normal for a random hyperplane.
    pub(super) fn random_hyperplane_normal<R, T, const D: usize>(rng: &mut R) -> [T; D]
    where
        R: Rng,
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

    /// Given a set of hyperplane normals, compute the binary vector (usize) representation.
    pub(super) fn hyperplane_project<T, const D: usize>(normals: &[[T; D]], query: &[T; D]) -> usize
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
            todo!("Rather than utilize thread_rng, utilize a reproducible seed.");
        }

        #[test]
        fn test_hyperplane_project() {
            todo!("Implement hyperplane projection.");
        }
    }
}
