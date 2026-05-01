//! Pure-Rust SIMD distance kernels for vector-search workloads.
//!
//! Ships f32 dot-product, squared-L2, batched-dot, and a fused
//! threshold-emit kernel. Used internally by `rustcluster` and intended
//! to become a publishable standalone artifact in v1.3 once the API
//! stabilizes.
//!
//! All kernels are pure Rust — no BLAS, no C dependencies. They use
//! the `pulp` crate for runtime SIMD dispatch (AVX2, AVX-512, NEON).

mod dot;
mod l2;

pub use dot::{dot_f32, dot_f32_scalar};
pub use l2::{l2sq_f32, l2sq_f32_scalar};
