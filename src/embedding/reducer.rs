//! Standalone embedding dimensionality reducer.
//!
//! Wraps randomized PCA (or Matryoshka truncation) as a first-class artifact
//! with fit/transform/save/load. Users pay the PCA cost once, then iterate on
//! clustering for free.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use super::normalize;
use super::reduction::{compute_pca, project_data, PcaProjection};

/// Magic bytes for the binary format: "RCPC" (RustCluster PCA).
const MAGIC: [u8; 4] = *b"RCPC";
/// Current serialization format version.
const FORMAT_VERSION: u32 = 1;

/// Fitted state of an EmbeddingReducer.
pub struct EmbeddingReducerState {
    pub method: String,
    pub input_dim: usize,
    pub target_dim: usize,
    /// Column means for centering (PCA only).
    pub mean: Vec<f64>,
    /// Projection matrix, shape (input_dim, target_dim), flat row-major (PCA only).
    pub components: Vec<f64>,
}

/// Fit a PCA reducer on the given data.
///
/// `data`: flat row-major f64, shape (n, d).
/// Returns a fitted `EmbeddingReducerState`.
pub fn fit_pca(
    data: &[f64],
    n: usize,
    d: usize,
    target_dim: usize,
    seed: u64,
) -> EmbeddingReducerState {
    let mut rng = StdRng::seed_from_u64(seed);
    let proj = compute_pca(data, n, d, target_dim, 10, &mut rng);
    EmbeddingReducerState {
        method: "pca".to_string(),
        input_dim: proj.input_dim,
        target_dim: proj.output_dim,
        mean: proj.mean,
        components: proj.components,
    }
}

/// Create a Matryoshka (prefix truncation) reducer state. No fitting needed.
pub fn fit_matryoshka(input_dim: usize, target_dim: usize) -> EmbeddingReducerState {
    EmbeddingReducerState {
        method: "matryoshka".to_string(),
        input_dim,
        target_dim,
        mean: Vec::new(),
        components: Vec::new(),
    }
}

/// Transform data using a fitted reducer state. Returns L2-normalized output.
///
/// `data`: flat row-major f64, shape (n, d).
/// Returns flat row-major f64, shape (n, target_dim).
pub fn transform(
    data: &[f64],
    n: usize,
    d: usize,
    state: &EmbeddingReducerState,
) -> Result<Vec<f64>, String> {
    if d != state.input_dim {
        return Err(format!(
            "Input has {} features, expected {}",
            d, state.input_dim
        ));
    }

    let target = state.target_dim;
    let mut result = match state.method.as_str() {
        "pca" => {
            let proj = PcaProjection {
                components: state.components.clone(),
                mean: state.mean.clone(),
                input_dim: state.input_dim,
                output_dim: state.target_dim,
            };
            project_data::<f64>(data, n, &proj)
        }
        "matryoshka" => {
            let mut truncated = Vec::with_capacity(n * target);
            for i in 0..n {
                truncated.extend_from_slice(&data[i * d..i * d + target]);
            }
            truncated
        }
        other => return Err(format!("Unknown method: {}", other)),
    };

    // L2 re-normalize rows
    normalize::l2_normalize_rows_inplace(&mut result, n, target);
    Ok(result)
}

/// Save reducer state to a binary file.
///
/// Format:
/// ```text
/// [magic: 4 bytes "RCPC"]
/// [version: u32 LE]
/// [input_dim: u64 LE]
/// [target_dim: u64 LE]
/// [method_len: u32 LE]
/// [method: UTF-8 bytes]
/// [mean: input_dim × f64 LE]        (PCA only, empty for matryoshka)
/// [components: input_dim × target_dim × f64 LE]  (PCA only)
/// ```
pub fn save_state(state: &EmbeddingReducerState, path: &str) -> Result<(), String> {
    let file =
        File::create(Path::new(path)).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut w = BufWriter::new(file);

    // Header
    write_bytes(&mut w, &MAGIC)?;
    write_u32(&mut w, FORMAT_VERSION)?;
    write_u64(&mut w, state.input_dim as u64)?;
    write_u64(&mut w, state.target_dim as u64)?;

    // Method string
    let method_bytes = state.method.as_bytes();
    write_u32(&mut w, method_bytes.len() as u32)?;
    write_bytes(&mut w, method_bytes)?;

    // Mean and components (PCA only)
    if state.method == "pca" {
        for &v in &state.mean {
            write_f64(&mut w, v)?;
        }
        for &v in &state.components {
            write_f64(&mut w, v)?;
        }
    }

    Ok(())
}

/// Load reducer state from a binary file.
pub fn load_state(path: &str) -> Result<EmbeddingReducerState, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut r = BufReader::new(file);

    // Magic
    let mut magic = [0u8; 4];
    read_exact(&mut r, &mut magic)?;
    if magic != MAGIC {
        return Err("Invalid file format (bad magic bytes)".to_string());
    }

    // Version
    let version = read_u32(&mut r)?;
    if version != FORMAT_VERSION {
        return Err(format!(
            "Unsupported format version {} (expected {})",
            version, FORMAT_VERSION
        ));
    }

    let input_dim = read_u64(&mut r)? as usize;
    let target_dim = read_u64(&mut r)? as usize;

    let method_len = read_u32(&mut r)? as usize;
    let mut method_bytes = vec![0u8; method_len];
    read_exact(&mut r, &mut method_bytes)?;
    let method =
        String::from_utf8(method_bytes).map_err(|e| format!("Invalid method string: {}", e))?;

    let (mean, components) = if method == "pca" {
        let mut mean = vec![0.0f64; input_dim];
        for v in mean.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        let comp_len = input_dim * target_dim;
        let mut components = vec![0.0f64; comp_len];
        for v in components.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        (mean, components)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(EmbeddingReducerState {
        method,
        input_dim,
        target_dim,
        mean,
        components,
    })
}

// ---- Binary I/O helpers ----

fn write_bytes(w: &mut BufWriter<File>, data: &[u8]) -> Result<(), String> {
    w.write_all(data).map_err(|e| format!("Write error: {}", e))
}

fn write_u32(w: &mut BufWriter<File>, v: u32) -> Result<(), String> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))
}

fn write_u64(w: &mut BufWriter<File>, v: u64) -> Result<(), String> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))
}

fn write_f64(w: &mut BufWriter<File>, v: f64) -> Result<(), String> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))
}

fn read_exact(r: &mut BufReader<File>, buf: &mut [u8]) -> Result<(), String> {
    r.read_exact(buf).map_err(|e| format!("Read error: {}", e))
}

fn read_u32(r: &mut BufReader<File>) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut BufReader<File>) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    read_exact(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut BufReader<File>) -> Result<f64, String> {
    let mut buf = [0u8; 8];
    read_exact(r, &mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_fit_pca_dimensions() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;
        let d = 20;
        let target = 5;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let state = fit_pca(&data, n, d, target, 42);
        assert_eq!(state.input_dim, d);
        assert_eq!(state.target_dim, target);
        assert_eq!(state.mean.len(), d);
        assert_eq!(state.components.len(), d * target);
    }

    #[test]
    fn test_transform_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 50;
        let d = 20;
        let target = 5;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let state = fit_pca(&data, n, d, target, 42);
        let result = transform(&data, n, d, &state).unwrap();
        assert_eq!(result.len(), n * target);
    }

    #[test]
    fn test_transform_unit_norm() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 50;
        let d = 20;
        let target = 5;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let state = fit_pca(&data, n, d, target, 42);
        let result = transform(&data, n, d, &state).unwrap();
        for i in 0..n {
            let norm: f64 = (0..target)
                .map(|j| result[i * target + j] * result[i * target + j])
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Row {} norm = {}", i, norm);
        }
    }

    #[test]
    fn test_matryoshka_truncation() {
        let n = 10;
        let d = 20;
        let target = 5;
        let data: Vec<f64> = (0..n * d).map(|i| i as f64).collect();

        let state = fit_matryoshka(d, target);
        let result = transform(&data, n, d, &state).unwrap();
        assert_eq!(result.len(), n * target);
        // Matryoshka takes first target columns, then L2-normalizes
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 50;
        let d = 20;
        let target = 5;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let state = fit_pca(&data, n, d, target, 42);
        let path = "/tmp/rustcluster_test_reducer.bin";
        save_state(&state, path).unwrap();
        let loaded = load_state(path).unwrap();

        assert_eq!(loaded.method, "pca");
        assert_eq!(loaded.input_dim, d);
        assert_eq!(loaded.target_dim, target);
        assert_eq!(loaded.mean.len(), state.mean.len());
        for (a, b) in loaded.mean.iter().zip(state.mean.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        for (a, b) in loaded.components.iter().zip(state.components.iter()) {
            assert!((a - b).abs() < 1e-15);
        }

        // Transform should produce identical results
        let r1 = transform(&data, n, d, &state).unwrap();
        let r2 = transform(&data, n, d, &loaded).unwrap();
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_matryoshka() {
        let state = fit_matryoshka(1536, 128);
        let path = "/tmp/rustcluster_test_matryoshka.bin";
        save_state(&state, path).unwrap();
        let loaded = load_state(path).unwrap();

        assert_eq!(loaded.method, "matryoshka");
        assert_eq!(loaded.input_dim, 1536);
        assert_eq!(loaded.target_dim, 128);
        assert!(loaded.mean.is_empty());
        assert!(loaded.components.is_empty());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let state = fit_matryoshka(20, 5);
        let data = vec![0.0f64; 10 * 30]; // 30 features, state expects 20
        let result = transform(&data, 10, 30, &state);
        assert!(result.is_err());
    }
}
