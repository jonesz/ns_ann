// src/lsh.rs; Copyright 2023, Ethan Jones. See LICENSE for licensing information.

/// Given the output of a projection: f(q, h), determine how to construct the
/// lower-dimensional identifier.
pub enum ConstructionMethod {
    /// Consider the output of f(q, h) as the next index to go to within
    /// a balanced binary tree.
    Tree,
    /// Consider the output of f(q, h) as a single bit; concatenate all N
    /// bits into a usize.
    Concatenate,
}

pub struct RandomProjection<'a, const N: usize, T, const D: usize>(
    ConstructionMethod,
    &'a [[T; D]; N],
);

impl<'a, const N: usize, T, const D: usize> RandomProjection<'a, N, T, D>
where
    T: hyperplane::Projection<T, D>,
    [(); N.ilog2() as usize]:,
{
    fn tree(&self, query: &[T; D]) -> usize {
        // TODO: `MaybeUnint` this?
        let mut arr = [hyperplane::Sign::default(); N.ilog2() as usize];

        // TODO: Is there an iterator construction that mimics the behavior of search through
        // a BST? I can imagine how the BTree construction implements search, is that the way
        // to do it here?
        let mut iter = self.1.iter().enumerate();
        let mut arr_idx: usize = 0;

        while let Some((idx, hp)) = iter.next() {
            let mem = arr.get_mut(arr_idx).unwrap();
            *mem = T::project(query, &hp);
            arr_idx += 1;

            let next_idx = (idx * 2) + Into::<usize>::into(*mem) + 1;
            let _ = iter.advance_by(next_idx - idx);
        }

        hyperplane::Sign::to_usize(arr)
    }

    fn concatenate(&self, query: &[T; D]) -> usize {
        // TODO: MaybeUninit this?
        let mut arr = [hyperplane::Sign::default(); N];
        for (mem, hp) in arr.iter_mut().zip(self.1.iter()) {
            *mem = T::project(query, hp);
        }

        hyperplane::Sign::to_usize(arr)
    }

    /// Return the bin from which to select an ANN.
    pub fn bin(&self, query: &[T; D]) -> usize {
        match self.0 {
            ConstructionMethod::Tree => self.tree(query),
            ConstructionMethod::Concatenate => self.concatenate(query),
        }
    }

    /// Create a new RandomProjection.
    pub fn new(cm: ConstructionMethod, arr: &'a [[T; D]; N]) -> Self {
        Self(cm, arr)
    }
}

mod hyperplane {
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
