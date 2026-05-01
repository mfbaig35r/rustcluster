//! Drill-down: what fraction of `ip_batch` is alloc vs faer matmul vs
//! wrapping? This tells us whether output-buffer reuse (Phase 2a in the
//! faer-tuning plan) has a meaningful ceiling.
//!
//! Run with:
//!
//!     cargo run --release --example profile_ip_batch --no-default-features

use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TILE: usize = 384;
const D: usize = 1536;
const REPS: usize = 1431; // matches the tile-pair count for n=20k tile=384

fn make(n: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n * d)
        .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
        .collect();
    Array2::from_shape_vec((n, d), data).unwrap()
}

fn instrumented_ip_batch(
    queries: ArrayView2<f32>,
    data: ArrayView2<f32>,
    par: Par,
) -> (Array2<f32>, u128, u128, u128) {
    let (nq, dq) = queries.dim();
    let (n, _) = data.dim();
    let d = dq;

    // Phase A: allocate output buffer
    let a0 = Instant::now();
    let mut out_data: Vec<f32> = vec![0.0; nq * n];
    let alloc_ns = a0.elapsed().as_nanos();

    // Phase B: wrap inputs/outputs into faer MatRef/MatMut
    let w0 = Instant::now();
    let q_slice = queries.as_slice().expect("contiguous queries");
    let x_slice = data.as_slice().expect("contiguous data");
    let q_ref = MatRef::from_row_major_slice(q_slice, nq, d);
    let x_ref = MatRef::from_row_major_slice(x_slice, n, d);
    let wrap_ns = w0.elapsed().as_nanos();

    // Phase C: faer matmul (this is "the kernel" — pack + microkernel)
    let m0 = Instant::now();
    {
        let out_mut = MatMut::from_row_major_slice_mut(&mut out_data, nq, n);
        matmul(
            out_mut,
            Accum::Replace,
            q_ref,
            x_ref.transpose(),
            1.0_f32,
            par,
        );
    }
    let matmul_ns = m0.elapsed().as_nanos();

    let out = Array2::from_shape_vec((nq, n), out_data).expect("size matches by construction");
    (out, alloc_ns, wrap_ns, matmul_ns)
}

fn main() {
    println!(
        "Repeating ip_batch({}x{} × {}x{}^T) {} times, single-threaded.",
        TILE, D, TILE, D, REPS
    );
    println!("(Tile shape and rep count match a full similarity_graph at n=20k, d=1536.)\n");

    let xi = make(TILE, D, 1);
    let xj = make(TILE, D, 2);

    // Warmup
    let _ = instrumented_ip_batch(xi.view(), xj.view(), Par::Seq);

    let mut alloc_total: u128 = 0;
    let mut wrap_total: u128 = 0;
    let mut matmul_total: u128 = 0;

    let t0 = Instant::now();
    for _ in 0..REPS {
        let (_, a, w, m) = instrumented_ip_batch(xi.view(), xj.view(), Par::Seq);
        alloc_total += a;
        wrap_total += w;
        matmul_total += m;
    }
    let total_wall = t0.elapsed().as_nanos();

    let inner_total = alloc_total + wrap_total + matmul_total;
    println!("wall total:        {:>8.1} ms", total_wall as f64 / 1e6);
    println!(
        "  alloc:           {:>8.1} ms  ({:>4.1}% of wall, {:>4.1}% of inner)",
        alloc_total as f64 / 1e6,
        100.0 * alloc_total as f64 / total_wall as f64,
        100.0 * alloc_total as f64 / inner_total as f64
    );
    println!(
        "  wrap (MatRef):   {:>8.1} ms  ({:>4.1}% of wall, {:>4.1}% of inner)",
        wrap_total as f64 / 1e6,
        100.0 * wrap_total as f64 / total_wall as f64,
        100.0 * wrap_total as f64 / inner_total as f64
    );
    println!(
        "  matmul (faer):   {:>8.1} ms  ({:>4.1}% of wall, {:>4.1}% of inner)",
        matmul_total as f64 / 1e6,
        100.0 * matmul_total as f64 / total_wall as f64,
        100.0 * matmul_total as f64 / inner_total as f64
    );
    let other = total_wall as i128 - inner_total as i128;
    println!(
        "  other (Vec→Array2 reshape, etc.): {:>8.1} ms",
        other as f64 / 1e6
    );

    println!();
    println!("Decision: if matmul fraction is high (≥90%), output-buffer reuse");
    println!("(Phase 2a) has a small ceiling and probably isn't worth doing.");
}
