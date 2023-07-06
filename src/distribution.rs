//! Generate random unit vectors to serve as hyperplane normals.
use num_traits::real::Real;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn build_random_unit_hyperplanes<const N: usize, T, const D: usize, R>(
    rng: &mut R,
) -> [[T; D]; N]
where
    T: RandomUnitVector<D, Output = [T; D]> + Default + Copy,
    R: Rng,
{
    let mut arr = [[T::default(); D]; N];
    for mem in arr.iter_mut() {
        *mem = T::sample(rng);
    }

    arr
}

pub trait RandomUnitVector<const D: usize> {
    type Output;
    fn sample<R: Rng>(rng: &mut R) -> Self::Output;
}

impl<const D: usize, T> RandomUnitVector<D> for T
where
    StandardNormal: Distribution<T>,
    T: Real + Default + core::ops::DivAssign + core::iter::Sum,
{
    type Output = [Self; D];
    fn sample<R: Rng>(rng: &mut R) -> Self::Output {
        // ∑(xi^2) |> sqrt
        let mag = |x: &[Self]| -> Self { x.iter().map(|&xi| xi * xi).sum::<Self>().sqrt() };

        let mut out = [T::default(); D];
        for mem in out.iter_mut() {
            *mem = rng.sample(StandardNormal);
        }

        let out_mag = mag(&out);
        assert!(out_mag > T::default());

        for mem in out.iter_mut() {
            *mem /= out_mag;
        }

        out
    }
}
