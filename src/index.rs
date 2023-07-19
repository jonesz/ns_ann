use super::lsh::{hyperplane::CosineApproximate, ConstructionMethod, LSH};
use core::ops::Range;
use rand::Rng;

trait Index<I> {
    /// Given a bin, return the set of identifiers that belong to that bin.
    /// TODO: Utilize an iterator.
    fn get(&self, bin: usize) -> Option<&[I]>;
}

struct ArrIndex<const N: usize, I, const NB: usize, const CM: ConstructionMethod> {
    ranges: [Option<(usize, usize)>; NB],
    arr: [I; N],

    marker: core::marker::PhantomData<ConstructionMethod>,
}

impl<const N: usize, I, const NB: usize, const CM: ConstructionMethod> Index<I>
    for ArrIndex<N, I, NB, CM>
{
    fn get(&self, bin: usize) -> Option<&[I]> {
        if let Some(range) = self.ranges.get(bin).cloned().flatten() {
            Some(&self.arr[range.0..range.1])
        } else {
            None
        }
    }
}

// 'c, N: number of vectors indexed, NB: number of hyperplanes. T: scalar type.
// D: dimension of each vector, CM: ConstructionMethod.
impl<const N: usize, const NB: usize, const CM: ConstructionMethod> ArrIndex<N, usize, NB, CM> {
    fn build_concatenate<'c, R: Rng, T, const D: usize, L: LSH<'c, NB, T, D>>(
        rng: &mut R,
        x: &'c [[T; D]; N],
        h: &'c [[T; D]; NB],
    ) -> Self
    where
        T: Default + Copy,
        T: CosineApproximate<'c, T, D>,
    {
        let mut x_proj = [(0usize, 0usize); N];
        for (idx, (mem, query)) in x_proj.iter_mut().zip(x).enumerate() {
            *mem = (idx, L::bin(query, h));
        }

        // TODO: `sort_unstable_by_key` throws some lifetime issues.
        x_proj.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

        let mut ranges: [Option<(usize, usize)>; NB] = [None; NB];
        let mut arr = [0usize; N];

        for (idx, (a, (id, bin))) in arr.iter_mut().zip(x_proj).enumerate() {
            *a = id;

            // Update the range.
            let r = ranges.get_mut(bin).unwrap();
            match r {
                Some(range) => *r = Some((range.0, idx)),
                None => *r = Some((idx, N)),
            }
        }

        ArrIndex {
            ranges,
            arr,
            marker: core::marker::PhantomData,
        }
    }
}
