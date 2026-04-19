pub mod dbscan;
pub mod distance;
mod error;
mod hamerly;
pub mod kmeans;
pub mod utils;

/// Re-exports for Criterion benchmarks.
#[doc(hidden)]
pub mod _bench_api {
    pub use crate::distance::{Distance, Scalar, SquaredEuclidean};
    pub use crate::kmeans::{kmeans_plus_plus_init, run_kmeans_n_init, Algorithm};
    pub use crate::utils::{assign_nearest, assign_nearest_two, squared_euclidean};
}

#[cfg(feature = "python")]
mod python_bindings {
    use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use rayon::prelude::*;

    use crate::error::ClusterError;
    use crate::kmeans::{run_kmeans_n_init, run_kmeans_n_init_f32, Algorithm, KMeansState};
    use crate::utils::{validate_predict_data, validate_predict_data_generic};

    /// Fitted state holding either f64 or f32 results.
    enum FittedState {
        F64(KMeansState<f64>),
        F32(KMeansState<f32>),
    }

    #[pyclass]
    struct KMeans {
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        random_state: u64,
        n_init: usize,
        algorithm: Algorithm,
        fitted: Option<FittedState>,
    }

    #[pymethods]
    impl KMeans {
        #[new]
        #[pyo3(signature = (n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10, algorithm="auto"))]
        fn new(
            n_clusters: usize,
            max_iter: usize,
            tol: f64,
            random_state: u64,
            n_init: usize,
            algorithm: &str,
        ) -> PyResult<Self> {
            if n_clusters == 0 {
                return Err(ClusterError::InvalidClusters { k: 0, n: 0 }.into());
            }
            if max_iter == 0 {
                return Err(ClusterError::InvalidMaxIter(0).into());
            }
            if n_init == 0 {
                return Err(ClusterError::InvalidNInit(0).into());
            }
            if tol < 0.0 {
                return Err(ClusterError::InvalidTol(tol).into());
            }

            let algo = Algorithm::from_str(algorithm)?;

            Ok(KMeans {
                n_clusters,
                max_iter,
                tol,
                random_state,
                n_init,
                algorithm: algo,
                fitted: None,
            })
        }

        /// Fit the K-means model (f64 input).
        fn fit_f64(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
            if !x.is_c_contiguous() {
                return Err(ClusterError::NotContiguous.into());
            }
            let view = x.as_array();
            let k = self.n_clusters;
            let max_iter = self.max_iter;
            let tol = self.tol;
            let seed = self.random_state;
            let n_init = self.n_init;
            let algo = self.algorithm;

            let state = py.allow_threads(move || {
                run_kmeans_n_init(&view, k, max_iter, tol, seed, n_init, algo)
            })?;
            self.fitted = Some(FittedState::F64(state));
            Ok(())
        }

        /// Fit the K-means model (f32 input).
        fn fit_f32(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
            if !x.is_c_contiguous() {
                return Err(ClusterError::NotContiguous.into());
            }
            let view = x.as_array();
            let k = self.n_clusters;
            let max_iter = self.max_iter;
            let tol = self.tol;
            let seed = self.random_state;
            let n_init = self.n_init;
            let algo = self.algorithm;

            let state = py.allow_threads(move || {
                run_kmeans_n_init_f32(&view, k, max_iter, tol, seed, n_init, algo)
            })?;
            self.fitted = Some(FittedState::F32(state));
            Ok(())
        }

        /// Fit — dispatches to f64 or f32 based on input dtype.
        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            // Try f64 first (most common)
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                return self.fit_f64(py, arr);
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                return self.fit_f32(py, arr);
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected a C-contiguous float32 or float64 NumPy array",
            ))
        }

        /// Predict cluster labels for new data.
        fn predict<'py>(
            &self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f64>>()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                            "Model was fit with float64 — predict input must also be float64"
                        ))?;
                    if !arr.is_c_contiguous() {
                        return Err(ClusterError::NotContiguous.into());
                    }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data(&view, expected_d)?;

                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;

                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let data_slice = view.as_slice().expect("data is C-contiguous");
                        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");
                        (0..n)
                            .into_par_iter()
                            .map(|i| {
                                let point = &data_slice[i * d..(i + 1) * d];
                                let (idx, _) = crate::utils::assign_nearest(point, centroids_slice, k, d);
                                idx as i64
                            })
                            .collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
                FittedState::F32(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f32>>()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                            "Model was fit with float32 — predict input must also be float32"
                        ))?;
                    if !arr.is_c_contiguous() {
                        return Err(ClusterError::NotContiguous.into());
                    }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data_generic(&view, expected_d)?;

                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;

                    let labels = py.allow_threads(move || {
                        use crate::distance::SquaredEuclidean;
                        let (n, d) = view.dim();
                        let data_slice = view.as_slice().expect("data is C-contiguous");
                        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");
                        (0..n)
                            .into_par_iter()
                            .map(|i| {
                                let point = &data_slice[i * d..(i + 1) * d];
                                let (idx, _) = crate::utils::assign_nearest_with::<f32, SquaredEuclidean>(point, centroids_slice, k, d);
                                idx as i64
                            })
                            .collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
            }
        }

        /// Fit the model and return cluster labels.
        fn fit_predict<'py>(
            &mut self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                FittedState::F64(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                FittedState::F32(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                FittedState::F32(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(s) => {
                    Ok(PyArray2::from_owned_array(py, s.centroids.clone()).into_any().unbind())
                }
                FittedState::F32(s) => {
                    Ok(PyArray2::from_owned_array(py, s.centroids.clone()).into_any().unbind())
                }
            }
        }

        #[getter]
        fn inertia_(&self) -> PyResult<f64> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(s) => Ok(s.inertia),
                FittedState::F32(s) => Ok(s.inertia),
            }
        }

        #[getter]
        fn n_iter_(&self) -> PyResult<usize> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(s) => Ok(s.n_iter),
                FittedState::F32(s) => Ok(s.n_iter),
            }
        }

        fn __repr__(&self) -> String {
            let algo_str = match self.algorithm {
                Algorithm::Auto => "auto",
                Algorithm::Lloyd => "lloyd",
                Algorithm::Hamerly => "hamerly",
            };
            format!(
                "KMeans(n_clusters={}, max_iter={}, tol={}, random_state={}, n_init={}, algorithm=\"{}\")",
                self.n_clusters, self.max_iter, self.tol, self.random_state, self.n_init, algo_str
            )
        }
    }

    // ---- DBSCAN pyclass ----

    use crate::dbscan::{run_dbscan, run_dbscan_f32, DbscanState};

    enum DbscanFitted {
        F64(DbscanState<f64>),
        F32(DbscanState<f32>),
    }

    #[pyclass]
    struct Dbscan {
        eps: f64,
        min_samples: usize,
        metric: String,
        fitted: Option<DbscanFitted>,
    }

    #[pymethods]
    impl Dbscan {
        #[new]
        #[pyo3(signature = (eps=0.5, min_samples=5, metric="euclidean"))]
        fn new(eps: f64, min_samples: usize, metric: &str) -> PyResult<Self> {
            if eps <= 0.0 || !eps.is_finite() {
                return Err(ClusterError::InvalidEps(eps).into());
            }
            if min_samples == 0 {
                return Err(ClusterError::InvalidMinSamples(0).into());
            }
            let metric_lower = metric.to_lowercase();
            if metric_lower != "euclidean" {
                return Err(ClusterError::InvalidMetric(metric.to_string()).into());
            }

            Ok(Dbscan {
                eps,
                min_samples,
                metric: metric_lower,
                fitted: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let eps = self.eps;
                let min_samples = self.min_samples;
                let state = py.allow_threads(move || run_dbscan(&view, eps, min_samples))?;
                self.fitted = Some(DbscanFitted::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let eps = self.eps;
                let min_samples = self.min_samples;
                let state = py.allow_threads(move || run_dbscan_f32(&view, eps, min_samples))?;
                self.fitted = Some(DbscanFitted::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected a C-contiguous float32 or float64 NumPy array",
            ))
        }

        fn fit_predict<'py>(
            &mut self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                DbscanFitted::F64(s) => s.labels.clone(),
                DbscanFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                DbscanFitted::F64(s) => s.labels.clone(),
                DbscanFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn core_sample_indices_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let indices = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                DbscanFitted::F64(s) => s.core_sample_indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                DbscanFitted::F32(s) => s.core_sample_indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
            };
            Ok(PyArray1::from_vec(py, indices))
        }

        #[getter]
        fn components_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                DbscanFitted::F64(s) => {
                    Ok(PyArray2::from_owned_array(py, s.components.clone()).into_any().unbind())
                }
                DbscanFitted::F32(s) => {
                    Ok(PyArray2::from_owned_array(py, s.components.clone()).into_any().unbind())
                }
            }
        }

        fn __repr__(&self) -> String {
            format!(
                "DBSCAN(eps={}, min_samples={}, metric=\"{}\")",
                self.eps, self.min_samples, self.metric
            )
        }
    }

    #[pymodule]
    pub fn _rustcluster(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<KMeans>()?;
        m.add_class::<Dbscan>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::_rustcluster;
