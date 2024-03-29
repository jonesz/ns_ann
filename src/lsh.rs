// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.

/// Anything that maps [T; D] to a length `log2(usize)` Hamming space.
pub trait LSH<'a, T, const D: usize> {
    fn bin(&self, q: &'a [T; D]) -> usize;
}

// Allow the type system to bind `N` to be lte `usize::BITS`; this is dependent on
// the `ConstructionMethod` being utilized.
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

/// Specify how to construct the bin identifier given the output of sign(f(q, h)).
#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum ConstructionMethod {
    /// Consider the output of sign(f(q, h)) as the next index to go to within a perfect binary tree.
    Tree,
    /// Consider the output of sign(f(q, h)) as a single bit; concatenate all bits into a `usize`.
    Concatenate,
}

pub struct RandomProjection<'a, T, const D: usize, const NP: usize, const CM: ConstructionMethod> {
    hp: &'a [[T; D]; NP],
}

impl<'a, T, const D: usize, const NP: usize, const CM: ConstructionMethod> LSH<'a, T, D>
    for RandomProjection<'a, T, D, NP, CM>
where
    T: hyperplane::ArcCos<'a, T, D>,
    ConstAssert<{ fits_in_usize(CM, NP) }>:,
    // Within a tree construction, we require a `Sign` arr of `log2(N)`; this bound
    // allows for the stack construction of that arr.
    [(); NP.ilog2() as usize]: Sized,
{
    fn bin(&self, q: &'a [T; D]) -> usize {
        match CM {
            ConstructionMethod::Tree => RandomProjection::tree(q, self.hp),
            ConstructionMethod::Concatenate => RandomProjection::concatenate(q, self.hp),
        }
    }
}

impl<'a, T, const D: usize, const NP: usize, const CM: ConstructionMethod>
    RandomProjection<'a, T, D, NP, CM>
where
    T: hyperplane::ArcCos<'a, T, D>,
    ConstAssert<{ fits_in_usize(CM, NP) }>:,
    // Within a tree construction, we require a `Sign` arr of `log2(N)`; this bound
    // allows for the stack construction of that arr.
    [(); NP.ilog2() as usize]: Sized,
{
    fn tree(query: &'a [T; D], hp: &'a [[T; D]; NP]) -> usize {
        let mut arr = [hyperplane::Sign::default(); NP.ilog2() as usize];
        let mut idx = 0;
        for mem in arr.iter_mut() {
            let hp_i = hp.get(idx).unwrap();
            *mem = T::sign(query, hp_i);
            // Choose the left/right node for a perfect BT.
            idx = (idx * 2) + Into::<usize>::into(*mem) + 1;
        }

        hyperplane::Sign::to_usize(&arr)
    }

    fn concatenate(query: &'a [T; D], hp: &'a [[T; D]; NP]) -> usize {
        let mut arr = [hyperplane::Sign::default(); NP];
        for (mem, hp_i) in arr.iter_mut().zip(hp.iter()) {
            *mem = T::sign(query, hp_i);
        }

        hyperplane::Sign::to_usize(&arr)
    }

    pub fn new(hp: &'a [[T; D]; NP]) -> Self {
        Self { hp }
    }
}

mod hyperplane {
    use super::{fits_in_usize, ConstAssert, ConstructionMethod};

    #[derive(Copy, Clone, Debug, Default)]
    pub enum Sign {
        Positive,
        #[default]
        Negative,
    }

    impl Sign {
        // Convert a slice of `Sign` into a single `usize`.
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

    /// Charikar's SimHash.
    pub trait ArcCos<'c, T, const D: usize> {
        /// Return the sign of the inner product of two vectors.
        fn sign(a: &'c [T; D], b: &'c [T; D]) -> Sign;
    }

    impl<'c, T, const D: usize> ArcCos<'c, T, D> for T
    where
        T: Default + core::ops::Add<Output = T> + Into<Sign>,
        &'c T: core::ops::Mul<&'c T, Output = T> + 'c,
    {
        fn sign(a: &'c [T; D], b: &'c [T; D]) -> Sign {
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
