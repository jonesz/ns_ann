// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.

pub trait LSH<'a, const NB: usize, T, const D: usize>
where
    T: hyperplane::CosineApproximate<'a, T, D>,
{
    fn bin(query: &'a [T; D], hp: &'a [[T; D]; NB]) -> usize;
}

// Allow the type system to bind `N` to be lte `usize::BITS`; this depends on
// the `ConstructionMethod` which computes different expressions with `N`.
// See: https://github.com/rust-lang/rust/issues/68436#issuecomment-709786363
pub struct ConstAssert<const ASSERT: ()>;
pub const fn fits_in_usize(cm: ConstructionMethod, n: usize) {
    match cm {
        ConstructionMethod::Tree => assert!(
            n.ilog2() <= usize::BITS,
            "Within a tree construction, the depth of the tree (N) must be lte to usize::BITS."
        ),
        ConstructionMethod::Concatenate => assert!(
            n <= usize::BITS as usize,
            "Within a concatenative construction, N must be lte to usize::BITS."
        ),
    }
}

/// Given the output of the cosine approximation: sign(f(q, h)), determine how to construct the
/// lower-dimensional identifier.
#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum ConstructionMethod {
    /// Consider the output of sign(f(q, h)) as the next index to go to within
    /// a balanced binary tree.
    Tree,
    /// Consider the output of sign(f(q, h)) as a single bit; concatenate all `N`
    /// bits into a `usize`.
    Concatenate,
}

pub struct RandomProjection<const N: usize, T, const D: usize, const CM: ConstructionMethod>
where
    ConstAssert<{ fits_in_usize(CM, N) }>:,
{
    t_marker: core::marker::PhantomData<T>,
}

impl<'a, const NB: usize, T, const D: usize, const CM: ConstructionMethod> LSH<'a, NB, T, D>
    for RandomProjection<NB, T, D, CM>
where
    ConstAssert<{ fits_in_usize(CM, NB) }>:,
    [T; NB.ilog2() as usize]: Sized,
    T: hyperplane::CosineApproximate<'a, T, D>,
{
    fn bin(query: &'a [T; D], hp: &'a [[T; D]; NB]) -> usize {
        match CM {
            ConstructionMethod::Tree => RandomProjection::tree(query, hp),
            ConstructionMethod::Concatenate => RandomProjection::concatenate(query, hp),
        }
    }
}

impl<'a, const N: usize, T, const D: usize, const CM: ConstructionMethod>
    RandomProjection<N, T, D, CM>
where
    ConstAssert<{ fits_in_usize(CM, N) }>:,
    // Within a tree construction, we require a `Sign` arr of `log2(N)`; this bound
    // allows for the stack construction of that arr.
    [T; N.ilog2() as usize]: Sized,
    T: hyperplane::CosineApproximate<'a, T, D>,
{
    fn tree(query: &'a [T; D], hp: &'a [[T; D]; N]) -> usize {
        let mut arr = [hyperplane::Sign::default(); N.ilog2() as usize];
        let mut idx = 0;
        for mem in arr.iter_mut() {
            let hp_i = hp.get(idx).unwrap();
            *mem = T::sign_ip(query, hp_i);
            // Choose the left/right node for a perfect BT.
            idx = (idx * 2) + Into::<usize>::into(*mem) + 1;
        }

        hyperplane::Sign::to_usize(&arr)
    }

    fn concatenate(query: &'a [T; D], hp: &'a [[T; D]; N]) -> usize {
        let mut arr = [hyperplane::Sign::default(); N];
        for (mem, hp_i) in arr.iter_mut().zip(hp.iter()) {
            *mem = T::sign_ip(query, hp_i);
        }

        hyperplane::Sign::to_usize(&arr)
    }
}

pub mod hyperplane {
    use super::{fits_in_usize, ConstAssert, ConstructionMethod};
    #[derive(Copy, Clone, Debug, Default)]
    pub enum Sign {
        Positive,
        #[default]
        // Make `Negative` the default: it converts to zero, so the resulting upper bits of
        // the usize won't matter in regards to an index.
        Negative,
    }

    impl Sign {
        // Convert an arr of `Sign` into a single usize.
        pub fn to_usize<const CM: ConstructionMethod, const N: usize>(sign_arr: &[Sign]) -> usize
        where
            ConstAssert<{ fits_in_usize(CM, N) }>:,
        {
            sign_arr
                .iter()
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

    impl From<&Sign> for usize {
        fn from(val: &Sign) -> usize {
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

    pub trait CosineApproximate<'c, T, const D: usize> {
        /// Return the sign of the inner product of two vectors.
        fn sign_ip(a: &'c [T; D], b: &'c [T; D]) -> Sign;
    }

    impl<'c, T, const D: usize> CosineApproximate<'c, T, D> for T
    where
        T: Default + core::ops::Add<Output = T> + Into<Sign>,
        &'c T: core::ops::Mul<&'c T, Output = T> + 'c,
    {
        fn sign_ip(a: &'c [T; D], b: &'c [T; D]) -> Sign {
            a.iter()
                .zip(b.iter())
                .fold(T::default(), |acc, (x, y)| acc + (x * y))
                .into()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_sign_to_usize() {
            const CM: ConstructionMethod = ConstructionMethod::Tree;
            const SZ: usize = 5;
            const ARR: [Sign; SZ] = [
                Sign::Positive,
                Sign::Negative,
                Sign::Negative,
                Sign::Positive,
                Sign::Positive,
            ];

            assert_eq!(Sign::to_usize::<CM, SZ>(&ARR), 0b11001);
        }
    }
}
