use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ns_ann::lsh;
use rand::{rngs::StdRng, SeedableRng};

fn bench_randomproj_16_f32_1024(c: &mut Criterion) {
    const D: usize = 1024;
    const N: usize = 16;

    let mut rng = StdRng::seed_from_u64(0u64);
    let hm = lsh::HyperplaneMethod::<N, f32, D>::precompute_random_unit_vector(&mut rng);
    let im = lsh::IdentifierMethod::Tree;
    let rp = lsh::RandomProjection::<N, f32, D>::new(im, hm);

    c.bench_function("bench_randomproj_f32_tree_precomputed", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            black_box(rp.bin(&qv));
        })
    });

    let hm = lsh::HyperplaneMethod::<N, f32, D>::precompute_random_unit_vector(&mut rng);
    let im = lsh::IdentifierMethod::BinaryVec;
    let rp = lsh::RandomProjection::<N, f32, D>::new(im, hm);

    c.bench_function("bench_randomproj_f32_binvec_precomputed", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            black_box(rp.bin(&qv));
        })
    });

    let hm = lsh::HyperplaneMethod::<N, f32, D>::OnDemand(0u64);
    let im = lsh::IdentifierMethod::Tree;
    let rp = lsh::RandomProjection::<N, f32, D>::new(im, hm);

    c.bench_function("bench_randomproj_f32_tree_ondemand", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            black_box(rp.bin(&qv));
        })
    });

    let hm = lsh::HyperplaneMethod::<N, f32, D>::OnDemand(0u64);
    let im = lsh::IdentifierMethod::BinaryVec;
    let rp = lsh::RandomProjection::<N, f32, D>::new(im, hm);

    c.bench_function("bench_randomproj_f32_binvec_ondemand", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            black_box(rp.bin(&qv));
        })
    });
}

criterion_group!(benches, bench_randomproj_16_f32_1024);
criterion_main!(benches);
