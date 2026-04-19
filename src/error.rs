#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    // ---- Shared ----
    #[error("Input array must be 2-dimensional")]
    InvalidShape,

    #[error("Input array must be C-contiguous float32 or float64")]
    NotContiguous,

    #[error("Input array is empty (0 samples or 0 features)")]
    EmptyInput,

    #[error("Input contains NaN or infinite values")]
    NonFinite,

    #[error("Model has not been fitted yet — call fit() first")]
    NotFitted,

    #[error("Feature dimension mismatch: model expects {expected} features, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    // ---- K-means ----
    #[error("n_clusters ({k}) must be > 0 and <= n_samples ({n})")]
    InvalidClusters { k: usize, n: usize },

    #[error("max_iter must be > 0, got {0}")]
    InvalidMaxIter(usize),

    #[error("n_init must be > 0, got {0}")]
    InvalidNInit(usize),

    #[error("tol must be >= 0, got {0}")]
    InvalidTol(f64),

    #[error("Unknown algorithm '{0}' — use \"auto\", \"lloyd\", or \"hamerly\"")]
    InvalidAlgorithm(String),

    // ---- DBSCAN ----
    #[error("eps must be > 0, got {0}")]
    InvalidEps(f64),

    #[error("min_samples must be >= 1, got {0}")]
    InvalidMinSamples(usize),

    #[error("Unknown metric '{0}' — only \"euclidean\" is supported")]
    InvalidMetric(String),
}

#[cfg(feature = "python")]
impl From<ClusterError> for pyo3::PyErr {
    fn from(err: ClusterError) -> pyo3::PyErr {
        match &err {
            ClusterError::NotFitted => {
                pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
            }
            _ => pyo3::exceptions::PyValueError::new_err(err.to_string()),
        }
    }
}
