use core::ops::Range;
use super::lsh::ConstructionMethod;

trait Index<T> {
    fn get(&self, bin: usize) -> Option<&[T]>;
}

struct ArrIndex<T, const N: usize, const D: usize> {
    ranges: [Option<Range<usize>>; N],
    arr: [usize; N],
}

impl<T, const N: usize> Index<T> for ArrIndex<T, N> {
    fn get(&self, bin: usize) -> Option<&[T]> {
        if let Some(range) = self.ranges.get(bin).cloned().flatten() {
            Some(&self.arr[range])
        } else {
            None
        }
    }
}

impl<T, const N: usize, const D: usize> ArrIndex<T, const N: usize> {
    fn build(arr: [T; N], p: ConstructionMethod) -> (Self, [[T; D]; N]) {
        todo!();
    }
}
