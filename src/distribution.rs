//! Generate random unit vectors to serve as hyperplane normals.
use rand::Rng;

pub trait RandomUnitVector<const D: usize> {
    type Output;
    fn sample<R: Rng>(rng: &mut R) -> Self::Output;
}

impl<const D: usize> RandomUnitVector<D> for f32 {
    type Output = [f32; D];
    fn sample<R: Rng>(rng: &mut R) -> Self::Output {
        todo!();
    }
}
