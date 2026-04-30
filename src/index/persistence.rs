//! Save/load for flat indexes.
//!
//! Format: a directory containing
//! - `metadata.json` — type, dim, ntotal, metric, format_version, has_ids
//! - `vectors.safetensors` — the (n, d) f32 matrix as a single tensor
//! - `ids.safetensors` (optional) — the (n,) u64 external id vector
//!
//! Reuses the same convention as `ClusterSnapshot`: safetensors for numeric
//! payload, JSON for metadata. Architecture-portable (safetensors mandates
//! little-endian).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ndarray::Array2;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

use crate::error::ClusterError;

use super::flat::{IndexFlatIP, IndexFlatL2};
use super::ids::IdMap;

/// Bumped on any breaking change to the on-disk layout. `load` accepts any
/// version `<= INDEX_FORMAT_VERSION` so older saves keep working.
pub const INDEX_FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct IndexMetadata {
    format_version: u32,
    /// `"IndexFlatL2"` or `"IndexFlatIP"`.
    index_type: String,
    /// `"l2"` or `"ip"`.
    metric: String,
    dim: usize,
    ntotal: usize,
    has_ids: bool,
}

/// Save an `IndexFlatL2` to a directory.
pub fn save_flat_l2(index: &IndexFlatL2, dir: &str) -> Result<(), ClusterError> {
    save_flat(
        dir,
        "IndexFlatL2",
        "l2",
        index.dim_internal(),
        index.vectors_internal(),
        index.ids_internal(),
    )
}

/// Save an `IndexFlatIP` to a directory.
pub fn save_flat_ip(index: &IndexFlatIP, dir: &str) -> Result<(), ClusterError> {
    save_flat(
        dir,
        "IndexFlatIP",
        "ip",
        index.dim_internal(),
        index.vectors_internal(),
        index.ids_internal(),
    )
}

/// Load an `IndexFlatL2` from a directory.
pub fn load_flat_l2(dir: &str) -> Result<IndexFlatL2, ClusterError> {
    let (meta, vectors, ids) = load_flat(dir, "IndexFlatL2", "l2")?;
    Ok(IndexFlatL2::from_parts(meta.dim, vectors, ids))
}

/// Load an `IndexFlatIP` from a directory.
pub fn load_flat_ip(dir: &str) -> Result<IndexFlatIP, ClusterError> {
    let (meta, vectors, ids) = load_flat(dir, "IndexFlatIP", "ip")?;
    Ok(IndexFlatIP::from_parts(meta.dim, vectors, ids))
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

fn save_flat(
    dir: &str,
    index_type: &str,
    metric: &str,
    dim: usize,
    vectors: &Array2<f32>,
    ids: &IdMap,
) -> Result<(), ClusterError> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)
        .map_err(|e| ClusterError::SnapshotIo(format!("create dir: {e}")))?;

    // Vectors tensor.
    let n = vectors.nrows();
    let vec_bytes = f32_slice_to_bytes(
        vectors
            .as_slice()
            .expect("vectors must be C-contiguous (constructed via append/from_parts)"),
    );
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    let vec_view = TensorView::new(Dtype::F32, vec![n, dim], vec_bytes)
        .map_err(|e| ClusterError::SnapshotFormat(format!("vectors tensor: {e}")))?;
    tensors.insert("vectors".to_string(), vec_view);

    let vectors_path = dir_path.join("vectors.safetensors");
    safetensors::serialize_to_file(&tensors, &None, &vectors_path)
        .map_err(|e| ClusterError::SnapshotIo(format!("write vectors: {e}")))?;

    // Optional ids tensor in its own file (so we don't need to know whether
    // ids are present at the safetensors-deserialize stage).
    let has_ids = matches!(ids, IdMap::Explicit { .. });
    if let IdMap::Explicit { ids: id_vec, .. } = ids {
        let mut id_tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        let id_bytes = u64_slice_to_bytes(id_vec);
        let id_view = TensorView::new(Dtype::U64, vec![id_vec.len()], id_bytes)
            .map_err(|e| ClusterError::SnapshotFormat(format!("ids tensor: {e}")))?;
        id_tensors.insert("ids".to_string(), id_view);
        let ids_path = dir_path.join("ids.safetensors");
        safetensors::serialize_to_file(&id_tensors, &None, &ids_path)
            .map_err(|e| ClusterError::SnapshotIo(format!("write ids: {e}")))?;
    }

    let metadata = IndexMetadata {
        format_version: INDEX_FORMAT_VERSION,
        index_type: index_type.to_string(),
        metric: metric.to_string(),
        dim,
        ntotal: n,
        has_ids,
    };
    let json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| ClusterError::SnapshotFormat(format!("serialize metadata: {e}")))?;
    fs::write(dir_path.join("metadata.json"), json)
        .map_err(|e| ClusterError::SnapshotIo(format!("write metadata: {e}")))?;

    Ok(())
}

fn load_flat(
    dir: &str,
    expected_type: &str,
    expected_metric: &str,
) -> Result<(IndexMetadata, Array2<f32>, IdMap), ClusterError> {
    let dir_path = Path::new(dir);
    let meta_json = fs::read_to_string(dir_path.join("metadata.json"))
        .map_err(|e| ClusterError::SnapshotIo(format!("read metadata: {e}")))?;
    let metadata: IndexMetadata = serde_json::from_str(&meta_json)
        .map_err(|e| ClusterError::SnapshotFormat(format!("parse metadata: {e}")))?;

    if metadata.format_version == 0 || metadata.format_version > INDEX_FORMAT_VERSION {
        return Err(ClusterError::SnapshotFormat(format!(
            "unsupported index format_version {}; this build supports up to {}",
            metadata.format_version, INDEX_FORMAT_VERSION
        )));
    }
    if metadata.index_type != expected_type {
        return Err(ClusterError::SnapshotFormat(format!(
            "index type mismatch: file is {}, loader expects {}",
            metadata.index_type, expected_type
        )));
    }
    if metadata.metric != expected_metric {
        return Err(ClusterError::SnapshotFormat(format!(
            "metric mismatch: file is {}, loader expects {}",
            metadata.metric, expected_metric
        )));
    }

    let vec_bytes = fs::read(dir_path.join("vectors.safetensors"))
        .map_err(|e| ClusterError::SnapshotIo(format!("read vectors: {e}")))?;
    let tensors = SafeTensors::deserialize(&vec_bytes)
        .map_err(|e| ClusterError::SnapshotFormat(format!("parse vectors: {e}")))?;
    let vec_tensor = tensors
        .tensor("vectors")
        .map_err(|e| ClusterError::SnapshotFormat(format!("missing vectors tensor: {e}")))?;
    if vec_tensor.dtype() != Dtype::F32 {
        return Err(ClusterError::SnapshotFormat(format!(
            "vectors dtype must be f32, got {:?}",
            vec_tensor.dtype()
        )));
    }
    let shape = vec_tensor.shape();
    if shape.len() != 2 || shape[0] != metadata.ntotal || shape[1] != metadata.dim {
        return Err(ClusterError::SnapshotFormat(format!(
            "vectors shape {:?} does not match metadata ({}, {})",
            shape, metadata.ntotal, metadata.dim
        )));
    }
    let vec_data = bytes_to_f32_vec(vec_tensor.data());
    let vectors = Array2::from_shape_vec((metadata.ntotal, metadata.dim), vec_data)
        .map_err(|e| ClusterError::SnapshotFormat(format!("reshape vectors: {e}")))?;

    let ids = if metadata.has_ids {
        let id_bytes = fs::read(dir_path.join("ids.safetensors"))
            .map_err(|e| ClusterError::SnapshotIo(format!("read ids: {e}")))?;
        let id_tensors = SafeTensors::deserialize(&id_bytes)
            .map_err(|e| ClusterError::SnapshotFormat(format!("parse ids: {e}")))?;
        let id_tensor = id_tensors
            .tensor("ids")
            .map_err(|e| ClusterError::SnapshotFormat(format!("missing ids tensor: {e}")))?;
        if id_tensor.dtype() != Dtype::U64 {
            return Err(ClusterError::SnapshotFormat(format!(
                "ids dtype must be u64, got {:?}",
                id_tensor.dtype()
            )));
        }
        let id_shape = id_tensor.shape();
        if id_shape.len() != 1 || id_shape[0] != metadata.ntotal {
            return Err(ClusterError::SnapshotFormat(format!(
                "ids shape {:?} does not match ntotal {}",
                id_shape, metadata.ntotal
            )));
        }
        let id_vec = bytes_to_u64_vec(id_tensor.data());
        let mut map = IdMap::new();
        map.extend_explicit(&id_vec)?;
        map
    } else {
        let mut map = IdMap::new();
        if metadata.ntotal > 0 {
            map.extend_sequential(metadata.ntotal)?;
        }
        map
    };

    Ok((metadata, vectors, ids))
}

fn f32_slice_to_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn u64_slice_to_bytes(data: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let arr: [u8; 4] = bytes[i * 4..(i + 1) * 4].try_into().unwrap();
        out.push(f32::from_le_bytes(arr));
    }
    out
}

fn bytes_to_u64_vec(bytes: &[u8]) -> Vec<u64> {
    assert_eq!(bytes.len() % 8, 0);
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let arr: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        out.push(u64::from_le_bytes(arr));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use tempfile::tempdir;

    use crate::index::VectorIndex;

    #[test]
    fn ip_roundtrip_sequential_ids() {
        let data = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [0.7071, 0.7071]]);
        let mut idx = IndexFlatIP::new(2);
        idx.add(data.view()).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        save_flat_ip(&idx, path).unwrap();

        let loaded = load_flat_ip(path).unwrap();
        assert_eq!(loaded.dim(), 2);
        assert_eq!(loaded.ntotal(), 3);
        assert_eq!(loaded.metric(), super::super::Metric::InnerProduct);

        let q = arr2(&[[1.0f32, 0.0]]);
        let r = loaded
            .search(q.view(), 3, super::super::SearchOpts::default())
            .unwrap();
        assert_eq!(r.labels[(0, 0)], 0);
    }

    #[test]
    fn l2_roundtrip_with_external_ids() {
        let data = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
        let mut idx = IndexFlatL2::new(2);
        idx.add_with_ids(data.view(), &[100, 200]).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        save_flat_l2(&idx, path).unwrap();

        let loaded = load_flat_l2(path).unwrap();
        let q = arr2(&[[1.0f32, 0.0]]);
        let r = loaded
            .search(q.view(), 1, super::super::SearchOpts::default())
            .unwrap();
        assert_eq!(r.labels[(0, 0)], 100);
    }

    #[test]
    fn type_mismatch_errors() {
        let mut idx = IndexFlatIP::new(2);
        idx.add(arr2(&[[1.0f32, 0.0]]).view()).unwrap();
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        save_flat_ip(&idx, path).unwrap();

        // Loading as L2 should fail.
        let err = load_flat_l2(path).unwrap_err();
        assert!(err.to_string().contains("type mismatch"));
    }

    #[test]
    fn version_mismatch_errors() {
        let mut idx = IndexFlatL2::new(2);
        idx.add(arr2(&[[1.0f32, 0.0]]).view()).unwrap();
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        save_flat_l2(&idx, path).unwrap();

        // Tamper with metadata to claim a future version.
        let meta_path = dir.path().join("metadata.json");
        let txt = fs::read_to_string(&meta_path).unwrap();
        let bumped = txt.replace("\"format_version\": 1", "\"format_version\": 999");
        fs::write(&meta_path, bumped).unwrap();

        let err = load_flat_l2(path).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }
}
