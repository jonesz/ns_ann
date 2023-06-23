use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ns_ann::lsh::LSHDB;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn bench_lsh_ann_f32_on_init(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);

    const V_DIM: usize = 128;
    const N: usize = 4096;
    const NB: usize = 5;

    // TODO: This should be in some common code shared between benches, tests, etc.
    let mut build_vectors = || {
        let mut out = Vec::with_capacity(N);

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
        let (_, q_vector) = v_set.get(0).unwrap();
        b.iter(|| {
            let _ = black_box(db.ann(black_box(q_vector)).count());
        })
    });
}

criterion_group!(benches, bench_lsh_ann_f32_on_init);
criterion_main!(benches);
