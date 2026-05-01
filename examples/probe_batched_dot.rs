//! Milestone 4 acceptance bench: rustcluster_simd::batched_dot_tile vs
//! faer::matmul at the tile sizes that matter (96, 256, 384) at d=1536.
//!
//! Acceptance criterion: within 15-30% of faer at d=1536. If we miss
//! that band, register blocking is the follow-up.
//!
//! Run with:
//!
//!     cargo run --release --example probe_batched_dot --no-default-features

use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rustcluster_simd::{batched_dot_tile, batched_dot_tile_scalar};

const D: usize = 1536;
const TILES: &[usize] = &[96, 256, 384];
const RUNS: usize = 6; // first is warmup; report best of remaining

fn rand_block(rows: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

fn faer_dot(q: &[f32], q_h: usize, x: &[f32], x_w: usize, d: usize, out: &mut [f32]) {
    let q_ref = MatRef::from_row_major_slice(q, q_h, d);
    let x_ref = MatRef::from_row_major_slice(x, x_w, d);
    let out_mut = MatMut::from_row_major_slice_mut(out, q_h, x_w);
    matmul(out_mut, Accum::Replace, q_ref, x_ref.transpose(), 1.0, Par::Seq);
}

fn time_runs<F: FnMut()>(label: &str, runs: usize, mut f: F) -> f64 {
    use std::hint::black_box;
    let mut wall_ms: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        black_box(f());
        wall_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let measured = &wall_ms[1..];
    let best = measured.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = measured.iter().sum::<f64>() / measured.len() as f64;
    println!("    {:24} best {:>7.2} ms  mean {:>7.2} ms", label, best, mean);
    best
}

fn parity_check(simd: &[f32], scalar: &[f32], q: &[f32], x: &[f32], q_h: usize, x_w: usize, d: usize) {
    for i in 0..q_h {
        let q_row = &q[i * d..(i + 1) * d];
        for j in 0..x_w {
            let x_row = &x[j * d..(j + 1) * d];
            let abs_sum: f32 = q_row
                .iter()
                .zip(x_row.iter())
                .map(|(a, b)| (a * b).abs())
                .sum();
            let tol = 2.0 * (d as f32) * f32::EPSILON * abs_sum + 1e-3;
            let diff = (simd[i * x_w + j] - scalar[i * x_w + j]).abs();
            assert!(
                diff <= tol,
                "parity fail at ({i},{j}): simd={} scalar={} diff={} tol={}",
                simd[i * x_w + j],
                scalar[i * x_w + j],
                diff,
                tol
            );
        }
    }
}

fn main() {
    println!("Milestone 4 bench: batched_dot_tile vs faer at d={D}");
    println!("--------------------------------------------------------");
    println!("(single-threaded; runs={RUNS}, first is warmup)\n");

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();

    for &t in TILES {
        let q = rand_block(t, D, 1);
        let x = rand_block(t, D, 2);
        let mut out_simd = vec![0.0f32; t * t];
        let mut out_faer = vec![0.0f32; t * t];

        // Parity check (cheap at small tiles, do at every size).
        let mut out_scalar = vec![0.0f32; t.min(8) * t.min(8)];
        let parity_t = t.min(8);
        let q_small = &q[..parity_t * D];
        let x_small = &x[..parity_t * D];
        batched_dot_tile_scalar(q_small, parity_t, x_small, parity_t, D, &mut out_scalar);
        let mut out_simd_small = vec![0.0f32; parity_t * parity_t];
        batched_dot_tile(q_small, parity_t, x_small, parity_t, D, &mut out_simd_small);
        parity_check(&out_simd_small, &out_scalar, q_small, x_small, parity_t, parity_t, D);

        println!("=== tile = {t}x{t}, d = {D} ===");
        let ms_simd = time_runs("rustcluster_simd", RUNS, || {
            batched_dot_tile(&q, t, &x, t, D, &mut out_simd)
        });
        let ms_faer = time_runs("faer matmul", RUNS, || {
            faer_dot(&q, t, &x, t, D, &mut out_faer)
        });
        let ratio = ms_simd / ms_faer;
        println!(
            "    ratio simd/faer: {:.3}  ({})",
            ratio,
            if ratio <= 1.15 {
                "within 15% — meets acceptance"
            } else if ratio <= 1.30 {
                "15-30% slower — meets acceptance"
            } else if ratio <= 2.0 {
                "30-100% slower — register blocking needed"
            } else {
                ">2x slower — register blocking definitely needed"
            }
        );
        println!();

        summary.push((t, ms_simd, ms_faer));
    }

    println!("Summary:");
    println!("  {:>5}  {:>10}  {:>10}  {:>8}", "tile", "simd ms", "faer ms", "ratio");
    for (t, s, f) in &summary {
        println!("  {:>5}  {:>10.2}  {:>10.2}  {:>8.3}", t, s, f, s / f);
    }

    println!("\nDecision (from docs/v1.2-execution-plan.md milestone 4):");
    println!("  if all ratios ≤ 1.30 → ship; proceed to milestone 5");
    println!("  if any ratio > 1.30 → add 4x4 register-blocked microkernel");
}
