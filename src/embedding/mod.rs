//! Purpose-built embedding clustering pipeline.
//!
//! Pipeline: L2 normalize → (optional PCA) → Spherical K-means → Evaluation
//! Optimized for dense embedding vectors (OpenAI, Cohere, etc.).
//!
//! This module is experimental. API may change.

pub mod normalize;
pub mod spherical_kmeans;

pub use spherical_kmeans::{run_spherical_kmeans, SphericalKMeansState};
