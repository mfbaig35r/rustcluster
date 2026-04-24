//! Snapshot serialization: safetensors for numeric arrays, JSON for metadata.
//!
//! Format: directory containing `centroids.safetensors` and `metadata.json`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

use crate::embedding::reduction::PcaProjection;
use crate::error::ClusterError;
use crate::snapshot::{ClusterSnapshot, Preprocessing, SnapshotAlgorithm};

const SNAPSHOT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SnapshotMetadata {
    version: u32,
    algorithm: String,
    metric: String,
    spherical: bool,
    k: usize,
    d: usize,
    input_dim: usize,
    preprocessing: String, // "none", "l2_normalize", "embedding_pipeline"
    fit_mean_distances: Vec<f64>,
    fit_cluster_sizes: Vec<usize>,
    fit_n_samples: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pca_input_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pca_output_dim: Option<usize>,
}

/// Save a snapshot to a directory.
///
/// Creates `{dir}/centroids.safetensors` and `{dir}/metadata.json`.
pub fn save_snapshot(snapshot: &ClusterSnapshot, dir: &str) -> Result<(), ClusterError> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)
        .map_err(|e| ClusterError::SnapshotIo(format!("create dir: {e}")))?;

    // Build safetensors data
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();

    let centroids_bytes = f64_slice_to_bytes(&snapshot.centroids);
    let centroids_view = TensorView::new(
        Dtype::F64,
        vec![snapshot.k, snapshot.d],
        centroids_bytes,
    )
    .map_err(|e| ClusterError::SnapshotFormat(format!("centroids tensor: {e}")))?;
    tensors.insert("centroids".to_string(), centroids_view);

    // PCA tensors (optional)
    let (pca_input_dim, pca_output_dim) = match &snapshot.preprocessing {
        Preprocessing::EmbeddingPipeline { pca, .. } => {
            let comp_bytes = f64_slice_to_bytes(&pca.components);
            let comp_view = TensorView::new(
                Dtype::F64,
                vec![pca.input_dim, pca.output_dim],
                comp_bytes,
            )
            .map_err(|e| ClusterError::SnapshotFormat(format!("pca_components tensor: {e}")))?;
            tensors.insert("pca_components".to_string(), comp_view);

            let mean_bytes = f64_slice_to_bytes(&pca.mean);
            let mean_view =
                TensorView::new(Dtype::F64, vec![pca.input_dim], mean_bytes)
                    .map_err(|e| ClusterError::SnapshotFormat(format!("pca_mean tensor: {e}")))?;
            tensors.insert("pca_mean".to_string(), mean_view);

            (Some(pca.input_dim), Some(pca.output_dim))
        }
        _ => (None, None),
    };

    // Write safetensors
    let tensors_path = dir_path.join("centroids.safetensors");
    safetensors::serialize_to_file(&tensors, &None, &tensors_path)
        .map_err(|e| ClusterError::SnapshotIo(format!("write safetensors: {e}")))?;

    // Build and write metadata
    let preprocessing_str = match &snapshot.preprocessing {
        Preprocessing::None => "none",
        Preprocessing::L2Normalize => "l2_normalize",
        Preprocessing::EmbeddingPipeline { .. } => "embedding_pipeline",
    };

    let metadata = SnapshotMetadata {
        version: snapshot.version,
        algorithm: snapshot.algorithm.as_str().to_string(),
        metric: metric_to_str(snapshot.metric).to_string(),
        spherical: snapshot.spherical,
        k: snapshot.k,
        d: snapshot.d,
        input_dim: snapshot.input_dim,
        preprocessing: preprocessing_str.to_string(),
        fit_mean_distances: snapshot.fit_mean_distances.clone(),
        fit_cluster_sizes: snapshot.fit_cluster_sizes.clone(),
        fit_n_samples: snapshot.fit_n_samples,
        pca_input_dim,
        pca_output_dim,
    };

    let json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| ClusterError::SnapshotFormat(format!("serialize metadata: {e}")))?;
    let meta_path = dir_path.join("metadata.json");
    fs::write(&meta_path, json)
        .map_err(|e| ClusterError::SnapshotIo(format!("write metadata: {e}")))?;

    Ok(())
}

/// Load a snapshot from a directory.
pub fn load_snapshot(dir: &str) -> Result<ClusterSnapshot, ClusterError> {
    let dir_path = Path::new(dir);

    // Read metadata
    let meta_path = dir_path.join("metadata.json");
    let meta_json = fs::read_to_string(&meta_path)
        .map_err(|e| ClusterError::SnapshotIo(format!("read metadata: {e}")))?;
    let metadata: SnapshotMetadata = serde_json::from_str(&meta_json)
        .map_err(|e| ClusterError::SnapshotFormat(format!("parse metadata: {e}")))?;

    // Version check
    if metadata.version != SNAPSHOT_VERSION {
        return Err(ClusterError::SnapshotFormat(format!(
            "unsupported version {}, expected {}",
            metadata.version, SNAPSHOT_VERSION
        )));
    }

    // Read safetensors
    let tensors_path = dir_path.join("centroids.safetensors");
    let tensors_bytes = fs::read(&tensors_path)
        .map_err(|e| ClusterError::SnapshotIo(format!("read safetensors: {e}")))?;
    let tensors = SafeTensors::deserialize(&tensors_bytes)
        .map_err(|e| ClusterError::SnapshotFormat(format!("parse safetensors: {e}")))?;

    // Extract centroids
    let centroids_tensor = tensors
        .tensor("centroids")
        .map_err(|e| ClusterError::SnapshotFormat(format!("missing centroids: {e}")))?;
    let centroids = bytes_to_f64_vec(centroids_tensor.data());

    // Reconstruct preprocessing
    let preprocessing = match metadata.preprocessing.as_str() {
        "none" => Preprocessing::None,
        "l2_normalize" => Preprocessing::L2Normalize,
        "embedding_pipeline" => {
            let pca_input_dim = metadata.pca_input_dim.ok_or_else(|| {
                ClusterError::SnapshotFormat("missing pca_input_dim".to_string())
            })?;
            let pca_output_dim = metadata.pca_output_dim.ok_or_else(|| {
                ClusterError::SnapshotFormat("missing pca_output_dim".to_string())
            })?;

            let comp_tensor = tensors
                .tensor("pca_components")
                .map_err(|e| ClusterError::SnapshotFormat(format!("missing pca_components: {e}")))?;
            let components = bytes_to_f64_vec(comp_tensor.data());

            let mean_tensor = tensors
                .tensor("pca_mean")
                .map_err(|e| ClusterError::SnapshotFormat(format!("missing pca_mean: {e}")))?;
            let mean = bytes_to_f64_vec(mean_tensor.data());

            Preprocessing::EmbeddingPipeline {
                input_dim: pca_input_dim,
                pca: PcaProjection {
                    components,
                    mean,
                    input_dim: pca_input_dim,
                    output_dim: pca_output_dim,
                },
            }
        }
        other => {
            return Err(ClusterError::SnapshotFormat(format!(
                "unknown preprocessing: {other}"
            )))
        }
    };

    let algorithm = SnapshotAlgorithm::from_str(&metadata.algorithm)?;
    let metric = metadata
        .metric
        .parse()
        .map_err(|_| ClusterError::SnapshotFormat(format!("bad metric: {}", metadata.metric)))?;

    Ok(ClusterSnapshot {
        algorithm,
        metric,
        spherical: metadata.spherical,
        centroids: Arc::new(centroids),
        k: metadata.k,
        d: metadata.d,
        input_dim: metadata.input_dim,
        preprocessing,
        fit_mean_distances: metadata.fit_mean_distances,
        fit_cluster_sizes: metadata.fit_cluster_sizes,
        fit_n_samples: metadata.fit_n_samples,
        version: metadata.version,
    })
}

fn metric_to_str(m: crate::distance::Metric) -> &'static str {
    match m {
        crate::distance::Metric::Euclidean => "euclidean",
        crate::distance::Metric::Cosine => "cosine",
        crate::distance::Metric::Manhattan => "manhattan",
    }
}

/// Reinterpret &[f64] as &[u8] for safetensors.
fn f64_slice_to_bytes(data: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) }
}

/// Convert bytes back to Vec<f64>.
fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
    assert_eq!(bytes.len() % 8, 0);
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let arr: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        out.push(f64::from_le_bytes(arr));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::snapshot::ClusterSnapshot;

    fn make_test_snapshot() -> ClusterSnapshot {
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::KMeans,
            metric: Metric::Euclidean,
            spherical: false,
            centroids: Arc::new(vec![0.0, 0.0, 10.0, 10.0]),
            k: 2,
            d: 2,
            input_dim: 2,
            preprocessing: Preprocessing::None,
            fit_mean_distances: vec![1.5, 2.3],
            fit_cluster_sizes: vec![50, 50],
            fit_n_samples: 100,
            version: 1,
        }
    }

    fn make_embedding_snapshot() -> ClusterSnapshot {
        let pca = PcaProjection {
            components: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // 4x2 identity-ish
            mean: vec![0.1, 0.2, 0.3, 0.4],
            input_dim: 4,
            output_dim: 2,
        };
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::EmbeddingCluster,
            metric: Metric::Cosine,
            spherical: true,
            centroids: Arc::new(vec![1.0, 0.0, 0.0, 1.0]),
            k: 2,
            d: 2,
            input_dim: 4,
            preprocessing: Preprocessing::EmbeddingPipeline {
                input_dim: 4,
                pca,
            },
            fit_mean_distances: vec![0.9, 0.85],
            fit_cluster_sizes: vec![30, 30],
            fit_n_samples: 60,
            version: 1,
        }
    }

    #[test]
    fn test_save_load_roundtrip_kmeans() {
        let snap = make_test_snapshot();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap");
        let path_str = path.to_str().unwrap();

        save_snapshot(&snap, path_str).unwrap();

        // Verify files exist
        assert!(path.join("centroids.safetensors").exists());
        assert!(path.join("metadata.json").exists());

        let loaded = load_snapshot(path_str).unwrap();
        assert_eq!(loaded.k, 2);
        assert_eq!(loaded.d, 2);
        assert_eq!(loaded.input_dim, 2);
        assert_eq!(*loaded.centroids, vec![0.0, 0.0, 10.0, 10.0]);
        assert_eq!(loaded.fit_mean_distances, vec![1.5, 2.3]);
        assert_eq!(loaded.fit_cluster_sizes, vec![50, 50]);
        assert_eq!(loaded.algorithm, SnapshotAlgorithm::KMeans);
        assert!(!loaded.spherical);
    }

    #[test]
    fn test_save_load_roundtrip_embedding() {
        let snap = make_embedding_snapshot();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("emb_snap");
        let path_str = path.to_str().unwrap();

        save_snapshot(&snap, path_str).unwrap();
        let loaded = load_snapshot(path_str).unwrap();

        assert_eq!(loaded.k, 2);
        assert_eq!(loaded.d, 2);
        assert_eq!(loaded.input_dim, 4);
        assert!(loaded.spherical);
        assert_eq!(loaded.algorithm, SnapshotAlgorithm::EmbeddingCluster);

        // Verify PCA was restored
        match &loaded.preprocessing {
            Preprocessing::EmbeddingPipeline { input_dim, pca } => {
                assert_eq!(*input_dim, 4);
                assert_eq!(pca.input_dim, 4);
                assert_eq!(pca.output_dim, 2);
                assert_eq!(pca.mean, vec![0.1, 0.2, 0.3, 0.4]);
            }
            _ => panic!("Expected EmbeddingPipeline preprocessing"),
        }
    }

    #[test]
    fn test_load_invalid_dir() {
        let result = load_snapshot("/nonexistent/path/abc123");
        assert!(matches!(result, Err(ClusterError::SnapshotIo(_))));
    }

    #[test]
    fn test_load_bad_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad");
        fs::create_dir_all(&path).unwrap();

        // Write metadata with bad version
        let meta = r#"{"version":99,"algorithm":"kmeans","metric":"euclidean","spherical":false,"k":1,"d":2,"input_dim":2,"preprocessing":"none","fit_mean_distances":[],"fit_cluster_sizes":[],"fit_n_samples":0}"#;
        fs::write(path.join("metadata.json"), meta).unwrap();

        // Write a minimal safetensors file
        let tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        safetensors::serialize_to_file(&tensors, &None, &path.join("centroids.safetensors"))
            .unwrap();

        let result = load_snapshot(path.to_str().unwrap());
        assert!(matches!(result, Err(ClusterError::SnapshotFormat(_))));
    }

    #[test]
    fn test_roundtrip_preserves_assignment() {
        let snap = make_test_snapshot();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap");
        let path_str = path.to_str().unwrap();

        save_snapshot(&snap, path_str).unwrap();
        let loaded = load_snapshot(path_str).unwrap();

        let data = vec![1.0, 1.0, 9.0, 9.0];
        let r1 = snap.assign_batch(&data, 2).unwrap();
        let r2 = loaded.assign_batch(&data, 2).unwrap();
        assert_eq!(r1.labels, r2.labels);
    }
}
