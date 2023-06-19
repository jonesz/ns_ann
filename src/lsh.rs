// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.
use rand::{
    distributions::{Distribution, Standard},
    rngs::SmallRng,
    seq::IteratorRandom,
    Rng, SeedableRng,
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

pub enum HyperplaneTiming<const NB: usize, T, const D: usize> {
    OnInitialization([[T; D]; NB]),
    OnDemand(u64),
}

impl<const NB: usize, T, const D: usize> HyperplaneTiming<NB, T, D>
where
    Standard: Distribution<T>,
{
    pub fn build_on_init<R: Rng>(rng: &mut R) -> Self {
        // TODO: Does the compiler complain if this is the case? It should be moved to
        // type-system const-generic magic.
        assert!(NB > 0 && D > 0);

        HyperplaneTiming::OnInitialization({
            // TODO: What's the 'proper' way to initialize this?
            let mut data: [std::mem::MaybeUninit<[T; D]>; NB] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };

            // TODO: Should this be a random gaussian vector? A random unit normal?
            for component in &mut data[..] {
                let hn = hyperplane::random_hyperplane_normal(rng);
                component.write(hn);
            }

            unsafe { data.as_ptr().cast::<[[T; D]; NB]>().read() }
        })
    }
}

/// An LSH structure containing `NB` random hyperplanes, `N` vectors of `T` type and `D` dimension,
/// and utilizing `I` to identify them.
pub struct LSHDB<'a, const NB: usize, const N: usize, T, const D: usize, I, ST>
where
    [(); pow2(NB)]:,
{
    hyperplane_normals: HyperplaneTiming<NB, T, D>,
    st: ST,
    buf: [I; N],
    phantom: std::marker::PhantomData<&'a I>,
}

impl<'a, const NB: usize, const N: usize, T, const D: usize, I, ST> LSHDB<'a, NB, N, T, D, I, ST>
where
    Standard: Distribution<T>,
    T: hyperplane::ProjLSH<T, D>,
    ST: search::Search<usize, std::ops::Range<usize>>,
    [(); pow2(NB)]:,
{
    /// Compute the binary vector representation of `q`.
    fn to_hyperplane_proj(method: &HyperplaneTiming<NB, T, D>, q: &[T; D]) -> usize {
        // TODO: Does the compiler complain if this is the case? It should be moved to
        // type-system const-generic magic.
        assert!(NB > 0 && D > 0);
        let mut sign_arr: [hyperplane::Sign; NB] = [hyperplane::Sign::default(); NB]; // TODO: This could be MaybeUninit initialized.
        match method {
            HyperplaneTiming::OnDemand(seed) => {
                let mut rng = SmallRng::seed_from_u64(*seed);
                for mem in sign_arr.iter_mut() {
                    let hn = hyperplane::random_hyperplane_normal(&mut rng);
                    *mem = T::proj(q, &hn);
                }
            }
            HyperplaneTiming::OnInitialization(hyperplanes) => {
                for (mem, hn) in sign_arr.iter_mut().zip(hyperplanes) {
                    *mem = T::proj(q, &hn);
                }
            }
        };

        hyperplane::Sign::to_usize(sign_arr)
    }

    pub fn new(ht: HyperplaneTiming<NB, T, D>, st: ST, buf: [I; N]) -> Self {
        Self {
            hyperplane_normals: ht,
            st,
            buf,
            phantom: std::marker::PhantomData,
        }
    }

    /// Find an approximate nearest neigh for an input vector.
    pub fn ann(&'a self, qv: &[T; D]) -> impl Iterator<Item = &'a I> {
        let idx = LSHDB::<NB, N, T, D, I, ST>::to_hyperplane_proj(&self.hyperplane_normals, qv);
        let range = self.st.search(&idx).unwrap();
        // TODO: Is there a way to pass the range into `.iter()` ?
        self.buf.iter().skip(range.start).take(range.end)
    }

    /// Find an approximate nearest neighbor for an input vector. NOTE: this will
    /// return a random vector from a *single bin*.
    pub fn ann_rand<R: Rng>(&'a self, rng: &mut R, qv: &[T; D]) -> Option<&'a I> {
        self.ann(qv).choose(rng)
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

    #[derive(Copy, Clone, Debug, Default)]
    pub enum Sign {
        #[default]
        Positive,
        Negative,
    }

    impl Sign {
        // Convert an arr of `Sign` into a single usize.
        pub fn to_usize<const L: usize>(sign_arr: [Sign; L]) -> usize {
            // TODO: There's const-generic type-system magic to force this check
            // at compile time.
            assert!(L <= usize::BITS.try_into().unwrap());
            sign_arr
                .into_iter()
                .enumerate()
                .fold(0usize, |acc, (idx, value)| {
                    acc + (Into::<usize>::into(value) << idx)
                })
        }
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_sign_to_usize() {
            assert_eq!(
                Sign::to_usize([
                    Sign::Positive,
                    Sign::Negative,
                    Sign::Negative,
                    Sign::Positive,
                    Sign::Positive
                ]),
                0b11001
            );
        }

        #[test]
        fn build_random_hyperplane_normal() {
            let mut rng = rand::thread_rng();
            let _: [f32; 4] = random_hyperplane_normal(&mut rng);
        }
    }
}

mod search {

    pub trait Search<K, V> {
        fn search(&self, k: &K) -> Option<&V>;
    }

    pub(super) struct BST<K, V, const N: usize> {
        _buf: [(K, V); N],
    }

    impl<K, V, const N: usize> BST<K, V, N> {}

    impl<K, V, const N: usize> Search<K, V> for BST<K, V, N> {
        fn search(&self, _k: &K) -> Option<&V> {
            todo!()
        }
    }

    #[cfg(test)]
    mod tests {}
}
