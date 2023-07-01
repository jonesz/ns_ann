// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.
use super::distribution::RandomUnitVector;
use rand::Rng;

pub type Seed = [u8; 12];

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
    OnDemandSingle(Seed),
    /// Generate each specific hyperplane from an RNG intitialized from a specific seed.
    OnDemandMultiple([Seed; N]),
}

impl<const N: usize, T, const D: usize> HyperplaneMethod<N, T, D>
where
    T: RandomUnitVector<D, Output = [T; D]> + Default + Copy,
{
    pub fn precompute_random_unit_vector<R: Rng>(rng: &mut R) -> Self {
        let mut hyperplanes = [[T::default(); D]; N];
        for mem in hyperplanes.iter_mut() {
            *mem = T::sample(rng);
        }

        HyperplaneMethod::Precomputed(hyperplanes)
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
