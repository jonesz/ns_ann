const SKIP_LIST_PROB: f32 = 0.15f32;

pub trait Metric<I, O: Ord> {
    fn d(a: &I, b: &I) -> O;
}

mod hnsw_ptr {
    use super::Metric;
    use rand::Rng;

    type Vertex = usize;
    /// A directed edge to another vertex.
    type Edge = Vertex;

    /// The neighborhood of a vertex containing M edges.
    #[derive(Debug)]
    struct Neighborhood<const M: usize> {
        neighbors: [Edge; M],
    }

    impl<const M: usize> IntoIterator for Neighborhood<M> {
        type Item = Edge;
        type IntoIter = std::array::IntoIter<Self::Item, M>;

        fn into_iter(self) -> Self::IntoIter {
            self.neighbors.into_iter()
        }
    }

    #[derive(Debug)]
    struct Layer<const N: usize, const M: usize, I, O> {
        vertices: [I; N],
        neighbors: [Neighborhood<M>; N],

        _metric: std::marker::PhantomData<O>,
    }

    impl<const N: usize, const M: usize, I, O: Ord> Layer<N, M, I, O> {
        fn entry<R: Rng>(rng: &mut R) -> usize {
            return rng.gen::<usize>() % N;
        }

        pub fn search<R: Rng>(&self, rng: &mut R, m: impl Metric<I, O>) -> usize {
            let idx = Layer::<N, M, I, O>::entry(rng);
            // TODO: If for some reason this evaluates to `None`, would we rather panic (notifying the programmer
            // that `entry` bugged or should we fallback on a pre-defined entry.
            let search_neighborhood = self.neighbors.get(idx).unwrap();

            // Find the minimum distance.
            // let idx = search_neighborhood.into_iter().min_by()
            todo!()
        }
    }

    struct HNSWDB<const L: usize, const M: usize> {}

    #[cfg(test)]
    mod tests {
        use super::*;
    }
}
