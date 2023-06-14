use rand::{
    distributions::{Distribution, Standard},
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

        // Build the underlying `buf` containing vector indices alongside `bin_idx` which points
        // to the index withing `buf` where that bin begins.
        let build_bin_idx_buf =
            |hyperplane_normals: &[[T; D]; NB]| -> ([Option<usize>; pow2(NB)], [I; N]) {
                todo!("Implement");
            };

        let hyperplane_normals = build_hyperplanes(rng);
        let (bin_idx, buf) = build_bin_idx_buf(&hyperplane_normals);
        Self {
            hyperplane_normals,
            bin_idx,
            buf,
        }
    }

    pub fn ann<R: Rng>(&self, rng: &mut R, q: &[T; D]) -> I {
        let idx = hyperplane::hyperplane_project(&self.hyperplane_normals, q);

        let (beg, end) = {
            // TODO: Is there something nicer than dereferencing
            // the reference here?
            let beg = self.bin_idx.get(idx).unwrap();
            let end = self.bin_idx.get(idx + 1).unwrap(); // TODO: This can overflow: should unwrap_or to `N`.
            (beg, end)
        };

        todo!("Sample a value I from `self.buf` between [beg, end)");
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

    impl<const D: usize> ProjLSH<f32, D> for [f32; D] {
        fn proj(a: &[f32; D], b: &[f32; D]) -> Sign {
            (0..D)
                .fold(0.0f32, |acc, idx| {
                    acc + (a.get(idx).unwrap() + b.get(idx).unwrap()) // dot-product.
                })
                .into()
        }
    }

    impl<const D: usize> ProjLSH<f64, D> for [f64; D] {
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
