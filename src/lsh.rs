use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

/// Hamming distance between two values.
fn hamming(a: usize, b: usize) -> u32 {
    (a ^ b).count_ones()
}

pub struct LSHDB<const NB: usize, T, const D: usize> {
    hyperplane_normals: [T; NB],
}

impl<const NB: usize, T, const D: usize> LSHDB<NB, T, D>
where
    Standard: Distribution<T>,
{
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let hyperplane_normals = {
            // TODO: What's the 'proper' way to initialize this?
            let mut data: [std::mem::MaybeUninit<[T; D]>; NB] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            // TODO: Should this be a random gaussian vector? A random unit normal?
            for component in &mut data[..] {
                let hn = hyperplane::random_hyperplane_normal(rng);
                component.write(hn);
            }

            unsafe { data.as_ptr().cast::<[T; NB]>().read() }
        };

        Self { hyperplane_normals }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_LSHDB_new() {
        let mut rng = rand::thread_rng();
        let _: LSHDB<4, f32, 8> = LSHDB::new(&mut rng);
        todo!("Rather than utilize thread_rng, use a reproducible seed.");
    }
}

mod hyperplane {
    use rand::{
        distributions::{Distribution, Standard},
        Rng,
    };

    #[derive(Debug)]
    pub(super) enum Sign {
        Positive,
        Negative,
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

    pub(super) trait ProjLSH<T, const D: usize> {
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn build_random_hyperplane_normal() {
            let mut rng = rand::thread_rng();
            let _: [f32; 4] = random_hyperplane_normal(&mut rng);
            todo!("Rather than utilize thread_rng, utilize a reproducible seed.");
        }
    }
}
