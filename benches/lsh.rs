use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eann_db::lsh::LSHDB;
use rand::{seq::SliceRandom, Fill, Rng};

fn bench_lsh_ann_f32_on_init(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    const V_DIM: usize = 128;
    const N: usize = 4096;
    const NB: usize = 5;

    // TODO: This should be in some common code shared between benches, tests, etc.
    let build_vectors = || {
        let mut out = Vec::with_capacity(N);
        let mut rng = rand::thread_rng();

        for i in 0..N {
            let mut v = [0.0f32; V_DIM];
            rng.fill(&mut v);
            out.push((i, v));
        }

        out
    };

    let v_set: [(usize, [f32; V_DIM]); N] = build_vectors().try_into().unwrap();
    let db = LSHDB::<NB, N, f32, V_DIM, usize>::new(&mut rng, &v_set, None);

    c.bench_function("bench_lsh_ann_f32_large_on_init", |b| {
        let (_, q_vector) = v_set.choose(&mut rng).unwrap();
        b.iter(|| {
            let _ = db.ann(black_box(q_vector));
        })
    });
}

criterion_group!(benches, bench_lsh_ann_f32_on_init);
criterion_main!(benches);
