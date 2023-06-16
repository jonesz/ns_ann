mod common;

use eann_db::lsh::LSHDB;

// Number of hyperplanes.
const NB: usize = 4;

#[test]
fn test_lshdb() {
    let v_set: [(usize, [f32; common::V_DIM]); N] = common::build_vectors().try_into().unwrap();
    let mut rng = rand::thread_rng();

    const V_DIM: usize = common::V_DIM;
    const N: usize = common::N;
    let mut db = LSHDB::<NB, N, f32, V_DIM, usize>::new(&mut rng, &v_set);
}
