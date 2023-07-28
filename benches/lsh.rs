use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ns_ann::{distribution, lsh, lsh::LSH};
use rand::{rngs::SmallRng, SeedableRng};

fn bench_randomproj_16_f32_1024(c: &mut Criterion) {
    const D: usize = 1024;
    const N: usize = 16;

    let mut rng = SmallRng::seed_from_u64(0u64);
    let arr: [[f32; D]; N] = distribution::build_random_unit_hyperplanes(&mut rng);

    const CM_TREE: lsh::ConstructionMethod = lsh::ConstructionMethod::Tree;
    const CM_CONCAT: lsh::ConstructionMethod = lsh::ConstructionMethod::Concatenate;

    let rp = lsh::RandomProjection::<'_, f32, D, N, CM_TREE>::new(&arr);
    c.bench_function("bench_randomproj_f32_tree", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            rp.bin(black_box(&qv));
        })
    });

    let rp = lsh::RandomProjection::<'_, f32, D, N, CM_CONCAT>::new(&arr);
    c.bench_function("bench_randomproj_f32_concatenate", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            rp.bin(black_box(&qv));
        })
    });
}

criterion_group!(benches, bench_randomproj_16_f32_1024);
criterion_main!(benches);
