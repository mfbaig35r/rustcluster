//! Pure-Rust SIMD distance kernels for vector-search workloads.
//!
//! Ships f32 dot-product, squared-L2, batched-dot, and a fused
//! threshold-emit kernel. All kernels are pure Rust — no BLAS, no C
//! dependencies. They use the `pulp` crate for runtime SIMD dispatch
//! (AVX2, AVX-512, NEON).
//!
//! **Status (v1.2):** library is correct and well-tested but does NOT
//! yet match faer's GEMM at production shapes. `rustcluster` itself
//! still uses faer on the hot path. The kernels here are a foundation
//! for future work — closing the remaining ~15-20% perf gap to faer
//! requires deeper kernel tuning (software prefetching, BLIS-style
//! packed kernels, bigger register tiles). See
//! `docs/v1.2-prototype-result.md` and `examples/probe_batched_dot.rs`
//! for the perf comparison.

mod batched_dot;
mod dot;
mod fused;
mod l2;
mod microkernel;

pub use batched_dot::{batched_dot_tile, batched_dot_tile_scalar};
pub use dot::{dot_f32, dot_f32_scalar};
pub use fused::{fused_threshold_emit_tile_ip, fused_threshold_emit_tile_l2};
pub use l2::{l2sq_f32, l2sq_f32_scalar};
