//! Generate random unit vectors to serve as hyperplane normals.
use rand::Rng;
use rand_distr::StandardNormal;

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

// TODO: This would capture [f32, f64], but requires some unsafe array-initialization due to the
// const-generic.
//impl<const D: usize, T> RandomUnitVector<D> for T
//where
//    StandardNormal: Distribution<T>,
//{
//    type Output = [Self; D];
//    fn sample<R: Rng>(rng: &mut R) -> Self::Output {
//        todo!();
//    }
//}

impl<const D: usize> RandomUnitVector<D> for f32 {
    type Output = [Self; D];
    fn sample<R: Rng>(rng: &mut R) -> Self::Output {
        // ∑(xi^2) |> sqrt
        let mag = |x: &[Self]| -> Self { x.iter().map(|&xi| xi * xi).sum::<Self>().sqrt() };

        let mut out = [0.0_f32; D];
        for mem in out.iter_mut() {
            *mem = rng.sample(StandardNormal);
        }

        let out_mag = mag(&out);
        assert!(out_mag > 0.0_f32);

        for mem in out.iter_mut() {
            *mem /= out_mag;
        }

        out
    }
}
