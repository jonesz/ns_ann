use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eann_db::lsh::hyperplane::{self, Sign};
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

// Benchmark for computing `proj` (which ends up being the dot product) for a query
// and a set of hyperplane normals.
fn bench_hyperplane_project_f32_large(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    const NB: usize = 5;
    let mut hyperplane_normals: Vec<[f32; V_DIM_LARGE]> = Vec::with_capacity(NB);
    for _ in 0..NB {
        hyperplane_normals.push(hyperplane::random_hyperplane_normal(&mut rng));
    }

    let mut query: [f32; V_DIM_LARGE] = [0.0f32; V_DIM_LARGE];
    rng.fill(&mut query);

    c.bench_function("hyperplane_project_f32_1024_32", |b| {
        b.iter(|| {
            hyperplane::hyperplane_project(black_box(&hyperplane_normals), black_box(&query));
        })
    });
}

criterion_group!(
    benches,
    bench_random_hyperplane_normal_f32_large,
    bench_hyperplane_project_f32_large
);
criterion_main!(benches);
