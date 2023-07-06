// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.

// Allow the type system to bind `N` to be lte to `usize::BITS`; this depends on
// the `ConstructionMethod` which computes different expressions with `N`.
// See: https://github.com/rust-lang/rust/issues/68436#issuecomment-709786363
pub struct ConstAssert<const ASSERT: ()>;
pub const fn fits_in_usize<const CM: ConstructionMethod, const N: usize>() {
    match CM {
        ConstructionMethod::Tree => assert!(
            N.ilog2() < usize::BITS,
            "Within a tree construction, the depth of the tree (N) must be less usize::BITS."
        ),
        ConstructionMethod::Concatenate => assert!(
            N < usize::BITS as usize,
            "Within a concatenative construction, N must be less than usize::BITS."
        ),
    }
}

/// Given the output of a projection: f(q, h), determine how to construct the
/// lower-dimensional identifier.
#[derive(PartialEq, Eq, core::marker::ConstParamTy)]
pub enum ConstructionMethod {
    /// Consider the output of f(q, h) as the next index to go to within
    /// a balanced binary tree.
    Tree,
    /// Consider the output of f(q, h) as a single bit; concatenate all N
    /// bits into a usize.
    Concatenate,
}

pub struct RandomProjection<'a, const N: usize, T, const D: usize, const CM: ConstructionMethod>
where
    ConstAssert<{ fits_in_usize::<CM, N>() }>:,
{
    hp: &'a [[T; D]; N],
}

impl<'a, const N: usize, T, const D: usize, const CM: ConstructionMethod>
    RandomProjection<'a, N, T, D, CM>
where
    ConstAssert<{ fits_in_usize::<CM, N>() }>:,
    // Within a tree construction, we require a `Sign` arr of `log2(N)`; this bound
    // allows for the stack construction of that arr.
    [T; N.ilog2() as usize]: Sized,
    T: hyperplane::Projection<T, D>,
{
    fn tree(&self, query: &[T; D]) -> usize {
        // TODO: `MaybeUnint` this?
        let mut arr = [hyperplane::Sign::default(); N.ilog2() as usize];

        let mut idx = 0;
        for mem in arr.iter_mut() {
            let hp = self.hp.get(idx).unwrap();
            *mem = T::project(query, hp);
            // Choose the left/right node for a perfect BT.
            idx = (idx * 2) + Into::<usize>::into(*mem) + 1;
        }

        hyperplane::Sign::to_usize(&arr)
    }

    fn concatenate(&self, query: &[T; D]) -> usize {
        // TODO: MaybeUninit this?
        let mut arr = [hyperplane::Sign::default(); N];
        for (mem, hp) in arr.iter_mut().zip(self.hp.iter()) {
            *mem = T::project(query, hp);
        }

        hyperplane::Sign::to_usize(&arr)
    }

    /// Return the bin from which to select an ANN.
    pub fn bin(&self, query: &[T; D]) -> usize {
        match CM {
            ConstructionMethod::Tree => self.tree(query),
            ConstructionMethod::Concatenate => self.concatenate(query),
        }
    }

    /// Create a new RandomProjection.
    pub fn new(hp: &'a [[T; D]; N]) -> Self {
        Self { hp }
    }
}

mod hyperplane {
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
            ConstAssert<{ fits_in_usize::<CM, N>() }>:,
        {
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
