mod common;

use eann_db::lsh::LSHDB;

// Number of hyperplanes.
const NB: usize = 4;
// Dimension of the vector.
const V_DIM: usize = common::V_DIM;
// Number of vectors.
const N: usize = common::N;

#[test]
fn test_lshdb() {
    let v_set: [(usize, [f32; V_DIM]); N] = common::build_vectors().try_into().unwrap();
    let mut rng = rand::thread_rng();

    let db = LSHDB::<NB, N, f32, V_DIM, usize>::new(&mut rng, &v_set, None);

    let (q_ident, q_vector) = v_set.get(0).unwrap();
    assert_eq!(db.ann(q_vector).find(|&x| x == q_ident).is_some(), true);
}
