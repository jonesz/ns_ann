use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ns_ann::lsh::hyperplane::{self, Sign};
use rand::{Fill, Rng};

const V_DIM_LARGE: usize = 1024;

// Benchmark for the creation of hyperplane normals.
fn bench_random_hyperplane_normal_f32_large(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    c.bench_function("random_hyperplane_normal_f32_1024", |b| {
        b.iter(|| {
            let _: [f32; V_DIM_LARGE] = hyperplane::random_hyperplane_normal(&mut rng);
        })
    });
}

criterion_group!(benches, bench_random_hyperplane_normal_f32_large,);
criterion_main!(benches);
