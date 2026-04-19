use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

#[derive(Debug, thiserror::Error)]
pub enum KMeansError {
    #[error("Input array must be 2-dimensional")]
    InvalidShape,

    #[error("Input array must be C-contiguous float64")]
    NotContiguous,

    #[error("Input array is empty (0 samples or 0 features)")]
    EmptyInput,

    #[error("n_clusters ({k}) must be > 0 and <= n_samples ({n})")]
    InvalidClusters { k: usize, n: usize },

    #[error("max_iter must be > 0, got {0}")]
    InvalidMaxIter(usize),

    #[error("n_init must be > 0, got {0}")]
    InvalidNInit(usize),

    #[error("tol must be >= 0, got {0}")]
    InvalidTol(f64),

    #[error("Input contains NaN or infinite values")]
    NonFinite,

    #[error("Model has not been fitted yet — call fit() first")]
    NotFitted,

    #[error("Feature dimension mismatch: model expects {expected} features, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Unknown algorithm '{0}' — use \"auto\", \"lloyd\", or \"hamerly\"")]
    InvalidAlgorithm(String),
}

impl From<KMeansError> for PyErr {
    fn from(err: KMeansError) -> PyErr {
        match &err {
            KMeansError::NotFitted => PyRuntimeError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}
