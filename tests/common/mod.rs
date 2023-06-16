use rand::Rng;

/// Vector dimension.
pub const V_DIM: usize = 16;
/// Number of vectors within the produced set.
pub const N: usize = 1024;

/// Construct a set of vectos to work with.
pub fn build_vectors() -> Vec<(usize, [f32; V_DIM])> {
    let mut out = Vec::with_capacity(N);
    let mut rng = rand::thread_rng();

    for i in 0..N {
        let mut v = [0.0f32; V_DIM];
        rng.fill(&mut v);
        out.push((i, v));
    }

    out
}
