// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.
use super::distribution::RandomUnitVector;
use rand::{rngs::SmallRng, Rng, SeedableRng};

// We utilize `seed_from_u64` within `SeedableRng`.
pub type Seed = u64;

/// Given the output of a projection: f(q, h), determine how to build an
/// an identifier.
pub enum IdentifierMethod {
    /// Consider the output of f(q, h) as the next index to go to within
    /// a balanced binary tree.
    Tree,
    /// Consider the output of f(q, h) as a single bit; concatenate all N
    /// bits into a usize.
    BinaryVec,
}

/// A Method on how to construct or retrieve a hyperplane `h` to compute
/// f(q, h).
pub enum HyperplaneMethod<const N: usize, T, const D: usize> {
    Precomputed([[T; D]; N]),
    /// Generate each hyperplane from an RNG initialized from a single seed.
    OnDemand(Seed),
}

impl<const N: usize, T, const D: usize> HyperplaneMethod<N, T, D>
where
    T: RandomUnitVector<D, Output = [T; D]> + Default + Copy,
{
    /// Produce a new `HyperplaneMethod` composed of precomputed hyperplanes.
    pub fn precompute_random_unit_vector<R: Rng>(rng: &mut R) -> Self {
        let mut hyperplanes = [[T::default(); D]; N];
        for mem in hyperplanes.iter_mut() {
            *mem = T::sample(rng);
        }

        HyperplaneMethod::Precomputed(hyperplanes)
    }
}

// NOTE: This iterator requires a *CLONE*; even though `Precomputed` could be accomplished
// without a clone, the lifetime 'a can't match the lifetime 'b of the `OnDemandIterator` (which)
// is stored on the function call stack rather than say, 'static.
// TODO: Can we coalesce the longer lifetime 'a to the shorter lifetime 'b?. Or does this enum
// need to be split and we utilize a trait with a set of structs?
pub enum HyperplaneMethodIterator<'a, const N: usize, T, const D: usize> {
    PrecomputedIterator(&'a [[T; D]; N], usize),
    OnDemandIterator(rand::rngs::SmallRng, usize),
}

impl<'a, const N: usize, T, const D: usize> Iterator for HyperplaneMethodIterator<'a, N, T, D>
where
    T: RandomUnitVector<D, Output = [T; D]> + Default + Copy,
{
    type Item = [T; D];
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            HyperplaneMethodIterator::PrecomputedIterator(hyperplane_slice, ctr) => {
                hyperplane_slice
                    .get({
                        *ctr += 1;
                        *ctr
                    })
                    .cloned()
            }
            HyperplaneMethodIterator::OnDemandIterator(rng, ctr) => {
                if *ctr < N {
                    *ctr += 1;
                    Some(T::sample(rng))
                } else {
                    None
                }
            }
        }
    }
}

// NOTE: This iterator produces values via *CLONE*.
impl<'a, const N: usize, T, const D: usize> IntoIterator for &'a HyperplaneMethod<N, T, D>
where
    T: RandomUnitVector<D, Output = [T; D]> + Default + Copy,
{
    type Item = [T; D];
    type IntoIter = HyperplaneMethodIterator<'a, N, T, D>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            HyperplaneMethod::Precomputed(x) => {
                HyperplaneMethodIterator::PrecomputedIterator(x, 0usize)
            }
            HyperplaneMethod::OnDemand(seed) => {
                use rand::{rngs::SmallRng, SeedableRng};
                HyperplaneMethodIterator::OnDemandIterator(SmallRng::seed_from_u64(*seed), 0usize)
            }
        }
    }
}

pub struct RandomProjection<const N: usize, T, const D: usize>(
    IdentifierMethod,
    HyperplaneMethod<N, T, D>,
);

impl<const N: usize, T, const D: usize> RandomProjection<N, T, D>
where
    T: hyperplane::Projection<T, D>,
{
    /// Return the bin from which to select an ANN.
    pub fn bin(&self, qv: &[T; D]) -> usize {
        todo!();
    }
}

mod hyperplane {
    #[derive(Copy, Clone, Debug, Default)]
    pub enum Sign {
        #[default]
        Positive,
        Negative,
    }

    impl Sign {
        // Convert an arr of `Sign` into a single usize.
        pub fn to_usize<const N: usize>(sign_arr: [Sign; N]) -> usize {
            // TODO: There's const-generic type-system magic to force this check
            // at compile time.
            assert!(N <= usize::BITS.try_into().unwrap());
            sign_arr
                .into_iter()
                .enumerate()
                .fold(0usize, |acc, (idx, value)| {
                    acc + (Into::<usize>::into(value) << idx)
                })
        }
    }

    impl From<Sign> for usize {
        fn from(val: Sign) -> usize {
            match val {
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

    pub trait Projection<T, const D: usize> {
        fn project(a: &[T; D], b: &[T; D]) -> Sign;
    }

    impl<const D: usize> Projection<f32, D> for f32 {
        fn project(a: &[f32; D], b: &[f32; D]) -> Sign {
            (0..D)
                .fold(0.0f32, |acc, idx| {
                    acc + (a.get(idx).unwrap() + b.get(idx).unwrap()) // dot-product.
                })
                .into()
        }
    }

    impl<const D: usize> Projection<f64, D> for f64 {
        fn project(a: &[f64; D], b: &[f64; D]) -> Sign {
            (0..D)
                .fold(0.0f64, |acc, idx| {
                    acc + (a.get(idx).unwrap() + b.get(idx).unwrap()) // dot-product.
                })
                .into()
        }
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
    }
}
