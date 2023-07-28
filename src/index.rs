use super::lsh::LSH;

type TupleRange = (usize, usize);

trait Index<I> {
    /// Given a bin, return the set of identifiers that belong to that bin.
    /// TODO: Utilize an iterator.
    fn get(&self, bin: usize) -> Option<&[I]>;
}

/// An index with the range search mechanism facilitated with an array.
struct ArrIndex<const N: usize, I, const NB: usize> {
    ranges: [Option<TupleRange>; NB],
    arr: [I; N],
}

impl<const N: usize, I, const NB: usize> Index<I> for ArrIndex<N, I, NB> {
    fn get(&self, bin: usize) -> Option<&[I]> {
        if let Some(range) = self.ranges.get(bin).cloned().flatten() {
            Some(&self.arr[range.0..range.1])
        } else {
            None
        }
    }
}

impl<const N: usize, const NB: usize> ArrIndex<N, usize, NB> {
    fn build_concatenate<'c, T, const D: usize, L: LSH<'c, T, D>>(
        x: &'c [[T; D]; N],
        l: &L,
    ) -> Self {
        let mut ranges: [Option<TupleRange>; NB] = [None; NB];
        let mut arr = [0usize; N];

        // Build `(idx, proj)` then sort by `proj`. Compute the range for each `proj` value.
        // Drop `proj` within `tmp_idx_proj` to build `arr`.
        let mut tmp_idx_proj = [(0usize, 0usize); N];
        for (idx, (proj_mem, query)) in tmp_idx_proj.iter_mut().zip(x).enumerate() {
            *proj_mem = (idx, l.bin(query));
        }

        // TODO: `sort_unstable_by_key` throws some lifetime issues.
        tmp_idx_proj
            .sort_unstable_by(|(_, a_proj), (_, b_proj)| a_proj.partial_cmp(b_proj).unwrap());

        for (idx, (arr_mem, (id, proj))) in arr.iter_mut().zip(tmp_idx_proj).enumerate() {
            *arr_mem = id; // This value of `arr` becomes the current idx.

            // Update the range.
            let potential_range = ranges.get_mut(proj).unwrap();
            match potential_range {
                // TODO: This update writes a new value when `idx + 1` is technically all that's needed.
                Some(existing_range) => *potential_range = Some((existing_range.0, idx)), // Update the range.
                None => *potential_range = Some((idx, N)), // If this range is unset, the range exists from `idx` to `N`.
            }
        }

        ArrIndex { ranges, arr }
    }
}
