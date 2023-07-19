use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ns_ann::{distribution, lsh, lsh::LSH};
use rand::{rngs::StdRng, SeedableRng};

fn bench_randomproj_16_f32_1024(c: &mut Criterion) {
    const D: usize = 1024;
    const N: usize = 16;

    let mut rng = StdRng::seed_from_u64(0u64);
    let arr: [[f32; D]; N] = distribution::build_random_unit_hyperplanes(&mut rng);

    const CM_TREE: lsh::ConstructionMethod = lsh::ConstructionMethod::Tree;
    const CM_CONCAT: lsh::ConstructionMethod = lsh::ConstructionMethod::Concatenate;

    c.bench_function("bench_randomproj_f32_tree", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            lsh::RandomProjection::<N, f32, D, CM_TREE>::bin(black_box(&qv), black_box(&arr));
        })
    });

    c.bench_function("bench_randomproj_f32_concatenate", |b| {
        let qv = [0.0f32; D];
        b.iter(|| {
            lsh::RandomProjection::<N, f32, D, CM_CONCAT>::bin(black_box(&qv), black_box(&arr));
        })
    });
}

criterion_group!(benches, bench_randomproj_16_f32_1024);
criterion_main!(benches);
