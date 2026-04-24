pub mod agglomerative;
pub mod dbscan;
pub mod distance;
pub mod embedding;
pub mod snapshot;
pub mod snapshot_io;
// dead_code: KdTree::build is superseded by build_v2, kept for reference
mod error;
mod hamerly;
pub mod hdbscan;
#[allow(dead_code)]
pub mod kdtree;
pub mod kmeans;
pub mod metrics;
pub mod minibatch_kmeans;
pub mod utils;

/// Re-exports for Criterion benchmarks.
#[doc(hidden)]
pub mod _bench_api {
    pub use crate::distance::{
        CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean,
    };
    pub use crate::kmeans::{kmeans_plus_plus_init, run_kmeans_n_init, Algorithm};
    pub use crate::utils::{assign_nearest, assign_nearest_two, squared_euclidean};
}

#[cfg(feature = "python")]
mod python_bindings {
    use std::str::FromStr;
    use std::sync::Arc;

    use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use rayon::prelude::*;

    use crate::dbscan::{run_dbscan_with_metric, run_dbscan_with_metric_f32, DbscanState};
    use crate::distance::Metric;
    use crate::error::ClusterError;
    use crate::kmeans::{
        run_kmeans_with_metric, run_kmeans_with_metric_f32, Algorithm, KMeansState,
    };
    use crate::utils::{assign_nearest_with, validate_predict_data, validate_predict_data_generic};

    /// Extract a 2D array as f64 or f32, run a closure inside `allow_threads`,
    /// and store the result in the given fitted-state enum variant.
    macro_rules! dispatch_fit {
        ($self:expr, $py:expr, $x:expr, $Enum:ident, $run_f64:expr, $run_f32:expr) => {{
            if let Ok(arr) = $x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = $py.allow_threads(move || $run_f64(view))?;
                $self.fitted = Some($Enum::F64(state));
                return Ok(());
            }
            if let Ok(arr) = $x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = $py.allow_threads(move || $run_f32(view))?;
                $self.fitted = Some($Enum::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected a C-contiguous float32 or float64 NumPy array",
            ))
        }};
    }

    // ---- KMeans ----

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
        metric: Metric,
        fitted: Option<FittedState>,
    }

    #[pymethods]
    impl KMeans {
        #[new]
        #[pyo3(signature = (n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10, algorithm="auto", metric="euclidean"))]
        fn new(
            n_clusters: usize,
            max_iter: usize,
            tol: f64,
            random_state: u64,
            n_init: usize,
            algorithm: &str,
            metric: &str,
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
            let met = Metric::from_str(metric)?;

            Ok(KMeans {
                n_clusters,
                max_iter,
                tol,
                random_state,
                n_init,
                algorithm: algo,
                metric: met,
                fitted: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let k = self.n_clusters;
            let max_iter = self.max_iter;
            let tol = self.tol;
            let seed = self.random_state;
            let n_init = self.n_init;
            let algo = self.algorithm;
            let metric = self.metric;

            dispatch_fit!(
                self,
                py,
                x,
                FittedState,
                |view| run_kmeans_with_metric(&view, k, max_iter, tol, seed, n_init, algo, metric),
                |view| run_kmeans_with_metric_f32(
                    &view, k, max_iter, tol, seed, n_init, algo, metric
                )
            )
        }

        fn predict<'py>(
            &self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let metric = self.metric;
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f64>>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Model was fit with float64 — predict input must also be float64",
                        )
                    })?;
                    if !arr.is_c_contiguous() {
                        return Err(ClusterError::NotContiguous.into());
                    }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data(&view, expected_d)?;

                    let centroids = Arc::clone(&state.centroids_flat);
                    let k = self.n_clusters;

                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let data_slice = view.as_slice().expect("C-contiguous");
                        let centroids_slice = &centroids[..];
                        (0..n)
                            .into_par_iter()
                            .map(|i| {
                                let point = &data_slice[i * d..(i + 1) * d];
                                let (idx, _) = match metric {
                                    Metric::Euclidean => assign_nearest_with::<
                                        f64,
                                        crate::distance::SquaredEuclidean,
                                    >(
                                        point, centroids_slice, k, d
                                    ),
                                    Metric::Cosine => assign_nearest_with::<
                                        f64,
                                        crate::distance::CosineDistance,
                                    >(
                                        point, centroids_slice, k, d
                                    ),
                                    Metric::Manhattan => assign_nearest_with::<
                                        f64,
                                        crate::distance::ManhattanDistance,
                                    >(
                                        point, centroids_slice, k, d
                                    ),
                                };
                                idx as i64
                            })
                            .collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
                FittedState::F32(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f32>>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Model was fit with float32 — predict input must also be float32",
                        )
                    })?;
                    if !arr.is_c_contiguous() {
                        return Err(ClusterError::NotContiguous.into());
                    }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data_generic(&view, expected_d)?;

                    let centroids = Arc::clone(&state.centroids_flat);
                    let k = self.n_clusters;

                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let data_slice = view.as_slice().expect("C-contiguous");
                        let centroids_slice = &centroids[..];
                        (0..n)
                            .into_par_iter()
                            .map(|i| {
                                let point = &data_slice[i * d..(i + 1) * d];
                                let (idx, _) = match metric {
                                    Metric::Euclidean => assign_nearest_with::<
                                        f32,
                                        crate::distance::SquaredEuclidean,
                                    >(
                                        point, centroids_slice, k, d
                                    ),
                                    Metric::Cosine => assign_nearest_with::<
                                        f32,
                                        crate::distance::CosineDistance,
                                    >(
                                        point, centroids_slice, k, d
                                    ),
                                    Metric::Manhattan => assign_nearest_with::<
                                        f32,
                                        crate::distance::ManhattanDistance,
                                    >(
                                        point, centroids_slice, k, d
                                    ),
                                };
                                idx as i64
                            })
                            .collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
            }
        }

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
                FittedState::F64(s) => Ok(PyArray2::from_owned_array(py, s.centroids.clone())
                    .into_any()
                    .unbind()),
                FittedState::F32(s) => Ok(PyArray2::from_owned_array(py, s.centroids.clone())
                    .into_any()
                    .unbind()),
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
            let met_str = match self.metric {
                Metric::Euclidean => "euclidean",
                Metric::Cosine => "cosine",
                Metric::Manhattan => "manhattan",
            };
            format!(
                "KMeans(n_clusters={}, max_iter={}, tol={}, random_state={}, n_init={}, algorithm=\"{}\", metric=\"{}\")",
                self.n_clusters, self.max_iter, self.tol, self.random_state, self.n_init, algo_str, met_str
            )
        }

        fn snapshot(&self) -> PyResult<PyClusterSnapshot> {
            let metric = self.metric;
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                FittedState::F64(state) => Ok(PyClusterSnapshot {
                    inner: snapshot::ClusterSnapshot::from_kmeans(state, metric),
                }),
                FittedState::F32(state) => Ok(PyClusterSnapshot {
                    inner: snapshot::ClusterSnapshot::from_kmeans_f32(state, metric),
                }),
            }
        }

        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("n_clusters", self.n_clusters)?;
            dict.set_item("max_iter", self.max_iter)?;
            dict.set_item("tol", self.tol)?;
            dict.set_item("random_state", self.random_state)?;
            dict.set_item("n_init", self.n_init)?;
            dict.set_item(
                "algorithm",
                match self.algorithm {
                    Algorithm::Auto => "auto",
                    Algorithm::Lloyd => "lloyd",
                    Algorithm::Hamerly => "hamerly",
                },
            )?;
            dict.set_item("metric", metric_str(self.metric))?;
            match &self.fitted {
                None => {
                    dict.set_item("fitted", false)?;
                }
                Some(FittedState::F64(s)) => {
                    dict.set_item("fitted", true)?;
                    dict.set_item("dtype", "float64")?;
                    dict.set_item(
                        "centroids",
                        PyArray2::from_owned_array(py, s.centroids.clone()),
                    )?;
                    dict.set_item(
                        "labels",
                        s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                    )?;
                    dict.set_item("inertia", s.inertia)?;
                    dict.set_item("n_iter", s.n_iter)?;
                }
                Some(FittedState::F32(s)) => {
                    dict.set_item("fitted", true)?;
                    dict.set_item("dtype", "float32")?;
                    dict.set_item(
                        "centroids",
                        PyArray2::from_owned_array(py, s.centroids.clone()),
                    )?;
                    dict.set_item(
                        "labels",
                        s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                    )?;
                    dict.set_item("inertia", s.inertia)?;
                    dict.set_item("n_iter", s.n_iter)?;
                }
            }
            Ok(dict.into())
        }

        fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
            self.n_clusters = state.get_item("n_clusters")?.unwrap().extract()?;
            self.max_iter = state.get_item("max_iter")?.unwrap().extract()?;
            self.tol = state.get_item("tol")?.unwrap().extract()?;
            self.random_state = state.get_item("random_state")?.unwrap().extract()?;
            self.n_init = state.get_item("n_init")?.unwrap().extract()?;
            let algo_s: String = state.get_item("algorithm")?.unwrap().extract()?;
            self.algorithm = Algorithm::from_str(&algo_s)?;
            let met_s: String = state.get_item("metric")?.unwrap().extract()?;
            self.metric = Metric::from_str(&met_s)?;
            let is_fitted: bool = state.get_item("fitted")?.unwrap().extract()?;
            if is_fitted {
                let dtype: String = state.get_item("dtype")?.unwrap().extract()?;
                let labels_i64: Vec<i64> = state.get_item("labels")?.unwrap().extract()?;
                let labels: Vec<usize> = labels_i64.iter().map(|&l| l as usize).collect();
                let inertia: f64 = state.get_item("inertia")?.unwrap().extract()?;
                let n_iter: usize = state.get_item("n_iter")?.unwrap().extract()?;
                if dtype == "float32" {
                    let arr = state
                        .get_item("centroids")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f32>>()?;
                    let centroids = arr.as_array().to_owned();
                    let centroids_flat =
                        Arc::new(centroids.as_slice().expect("C-contiguous").to_vec());
                    self.fitted = Some(FittedState::F32(KMeansState {
                        centroids,
                        centroids_flat,
                        labels,
                        inertia,
                        n_iter,
                    }));
                } else {
                    let arr = state
                        .get_item("centroids")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f64>>()?;
                    let centroids = arr.as_array().to_owned();
                    let centroids_flat =
                        Arc::new(centroids.as_slice().expect("C-contiguous").to_vec());
                    self.fitted = Some(FittedState::F64(KMeansState {
                        centroids,
                        centroids_flat,
                        labels,
                        inertia,
                        n_iter,
                    }));
                }
            }
            Ok(())
        }

        fn __getnewargs__(&self) -> (usize,) {
            (self.n_clusters,)
        }
    }

    fn metric_str(m: Metric) -> &'static str {
        match m {
            Metric::Euclidean => "euclidean",
            Metric::Cosine => "cosine",
            Metric::Manhattan => "manhattan",
        }
    }

    // ---- DBSCAN ----

    enum DbscanFitted {
        F64(DbscanState<f64>),
        F32(DbscanState<f32>),
    }

    #[pyclass]
    struct Dbscan {
        eps: f64,
        min_samples: usize,
        metric: Metric,
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
            let met = Metric::from_str(metric)?;

            Ok(Dbscan {
                eps,
                min_samples,
                metric: met,
                fitted: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let eps = self.eps;
            let min_samples = self.min_samples;
            let metric = self.metric;

            dispatch_fit!(
                self,
                py,
                x,
                DbscanFitted,
                |view| run_dbscan_with_metric(&view, eps, min_samples, metric),
                |view| run_dbscan_with_metric_f32(&view, eps, min_samples, metric)
            )
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
        fn core_sample_indices_<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let indices = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                DbscanFitted::F64(s) => s
                    .core_sample_indices
                    .iter()
                    .map(|&i| i as i64)
                    .collect::<Vec<_>>(),
                DbscanFitted::F32(s) => s
                    .core_sample_indices
                    .iter()
                    .map(|&i| i as i64)
                    .collect::<Vec<_>>(),
            };
            Ok(PyArray1::from_vec(py, indices))
        }

        #[getter]
        fn components_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                DbscanFitted::F64(s) => Ok(PyArray2::from_owned_array(py, s.components.clone())
                    .into_any()
                    .unbind()),
                DbscanFitted::F32(s) => Ok(PyArray2::from_owned_array(py, s.components.clone())
                    .into_any()
                    .unbind()),
            }
        }

        fn __repr__(&self) -> String {
            let met_str = match self.metric {
                Metric::Euclidean => "euclidean",
                Metric::Cosine => "cosine",
                Metric::Manhattan => "manhattan",
            };
            format!(
                "DBSCAN(eps={}, min_samples={}, metric=\"{}\")",
                self.eps, self.min_samples, met_str
            )
        }

        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("eps", self.eps)?;
            dict.set_item("min_samples", self.min_samples)?;
            dict.set_item("metric", metric_str(self.metric))?;
            match &self.fitted {
                None => {
                    dict.set_item("fitted", false)?;
                }
                Some(DbscanFitted::F64(s)) => {
                    dict.set_item("fitted", true)?;
                    dict.set_item("dtype", "float64")?;
                    dict.set_item("labels", s.labels.clone())?;
                    dict.set_item(
                        "core_sample_indices",
                        s.core_sample_indices
                            .iter()
                            .map(|&i| i as i64)
                            .collect::<Vec<_>>(),
                    )?;
                    dict.set_item(
                        "components",
                        PyArray2::from_owned_array(py, s.components.clone()),
                    )?;
                    dict.set_item("n_clusters", s.n_clusters)?;
                }
                Some(DbscanFitted::F32(s)) => {
                    dict.set_item("fitted", true)?;
                    dict.set_item("dtype", "float32")?;
                    dict.set_item("labels", s.labels.clone())?;
                    dict.set_item(
                        "core_sample_indices",
                        s.core_sample_indices
                            .iter()
                            .map(|&i| i as i64)
                            .collect::<Vec<_>>(),
                    )?;
                    dict.set_item(
                        "components",
                        PyArray2::from_owned_array(py, s.components.clone()),
                    )?;
                    dict.set_item("n_clusters", s.n_clusters)?;
                }
            }
            Ok(dict.into())
        }

        fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
            self.eps = state.get_item("eps")?.unwrap().extract()?;
            self.min_samples = state.get_item("min_samples")?.unwrap().extract()?;
            let met_s: String = state.get_item("metric")?.unwrap().extract()?;
            self.metric = Metric::from_str(&met_s)?;
            let is_fitted: bool = state.get_item("fitted")?.unwrap().extract()?;
            if is_fitted {
                let dtype: String = state.get_item("dtype")?.unwrap().extract()?;
                let labels: Vec<i64> = state.get_item("labels")?.unwrap().extract()?;
                let csi: Vec<i64> = state.get_item("core_sample_indices")?.unwrap().extract()?;
                let core_sample_indices: Vec<usize> = csi.iter().map(|&i| i as usize).collect();
                let n_clusters: usize = state.get_item("n_clusters")?.unwrap().extract()?;
                if dtype == "float32" {
                    let arr = state
                        .get_item("components")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f32>>()?;
                    self.fitted = Some(DbscanFitted::F32(DbscanState {
                        labels,
                        core_sample_indices,
                        components: arr.as_array().to_owned(),
                        n_clusters,
                    }));
                } else {
                    let arr = state
                        .get_item("components")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f64>>()?;
                    self.fitted = Some(DbscanFitted::F64(DbscanState {
                        labels,
                        core_sample_indices,
                        components: arr.as_array().to_owned(),
                        n_clusters,
                    }));
                }
            }
            Ok(())
        }

        fn __getnewargs__(&self) -> (f64, usize, &str) {
            (self.eps, self.min_samples, metric_str(self.metric))
        }
    }

    // ---- HDBSCAN ----

    use crate::hdbscan::{
        run_hdbscan_with_metric, run_hdbscan_with_metric_f32, ClusterSelectionMethod, HdbscanState,
    };

    enum HdbscanFitted {
        F64(HdbscanState<f64>),
        F32(HdbscanState<f32>),
    }

    #[pyclass]
    struct Hdbscan {
        min_cluster_size: usize,
        min_samples: usize,
        metric: Metric,
        cluster_selection_method: ClusterSelectionMethod,
        fitted: Option<HdbscanFitted>,
    }

    #[pymethods]
    impl Hdbscan {
        #[new]
        #[pyo3(signature = (min_cluster_size=5, min_samples=None, metric="euclidean", cluster_selection_method="eom"))]
        fn new(
            min_cluster_size: usize,
            min_samples: Option<usize>,
            metric: &str,
            cluster_selection_method: &str,
        ) -> PyResult<Self> {
            if min_cluster_size < 2 {
                return Err(ClusterError::InvalidMinClusterSize(min_cluster_size).into());
            }
            let ms = min_samples.unwrap_or(min_cluster_size);
            if ms == 0 {
                return Err(ClusterError::InvalidMinSamples(0).into());
            }
            let met = Metric::from_str(metric)?;
            let sel = ClusterSelectionMethod::from_str(cluster_selection_method)?;

            Ok(Hdbscan {
                min_cluster_size,
                min_samples: ms,
                metric: met,
                cluster_selection_method: sel,
                fitted: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let mcs = self.min_cluster_size;
            let ms = self.min_samples;
            let metric = self.metric;
            let sel = self.cluster_selection_method;

            dispatch_fit!(
                self,
                py,
                x,
                HdbscanFitted,
                |view| run_hdbscan_with_metric(&view, mcs, ms, metric, sel),
                |view| run_hdbscan_with_metric_f32(&view, mcs, ms, metric, sel)
            )
        }

        fn fit_predict<'py>(
            &mut self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                HdbscanFitted::F64(s) => s.labels.clone(),
                HdbscanFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                HdbscanFitted::F64(s) => s.labels.clone(),
                HdbscanFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let probs = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                HdbscanFitted::F64(s) => s.probabilities.clone(),
                HdbscanFitted::F32(s) => s.probabilities.clone(),
            };
            Ok(PyArray1::from_vec(py, probs))
        }

        #[getter]
        fn cluster_persistence_<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let pers = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                HdbscanFitted::F64(s) => s.cluster_persistence.clone(),
                HdbscanFitted::F32(s) => s.cluster_persistence.clone(),
            };
            Ok(PyArray1::from_vec(py, pers))
        }

        fn __repr__(&self) -> String {
            let met_str = match self.metric {
                Metric::Euclidean => "euclidean",
                Metric::Cosine => "cosine",
                Metric::Manhattan => "manhattan",
            };
            let sel_str = match self.cluster_selection_method {
                ClusterSelectionMethod::Eom => "eom",
                ClusterSelectionMethod::Leaf => "leaf",
            };
            format!(
                "HDBSCAN(min_cluster_size={}, min_samples={}, metric=\"{}\", cluster_selection_method=\"{}\")",
                self.min_cluster_size, self.min_samples, met_str, sel_str
            )
        }

        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("min_cluster_size", self.min_cluster_size)?;
            dict.set_item("min_samples", self.min_samples)?;
            dict.set_item("metric", metric_str(self.metric))?;
            dict.set_item(
                "cluster_selection_method",
                match self.cluster_selection_method {
                    ClusterSelectionMethod::Eom => "eom",
                    ClusterSelectionMethod::Leaf => "leaf",
                },
            )?;
            if let Some(ref f) = self.fitted {
                dict.set_item("fitted", true)?;
                let (labels, probs, pers, nc) = match f {
                    HdbscanFitted::F64(s) => (
                        s.labels.clone(),
                        s.probabilities.clone(),
                        s.cluster_persistence.clone(),
                        s.n_clusters,
                    ),
                    HdbscanFitted::F32(s) => (
                        s.labels.clone(),
                        s.probabilities.clone(),
                        s.cluster_persistence.clone(),
                        s.n_clusters,
                    ),
                };
                dict.set_item("labels", labels)?;
                dict.set_item("probabilities", probs)?;
                dict.set_item("cluster_persistence", pers)?;
                dict.set_item("n_clusters", nc)?;
            } else {
                dict.set_item("fitted", false)?;
            }
            Ok(dict.into())
        }

        fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
            self.min_cluster_size = state.get_item("min_cluster_size")?.unwrap().extract()?;
            self.min_samples = state.get_item("min_samples")?.unwrap().extract()?;
            let met_s: String = state.get_item("metric")?.unwrap().extract()?;
            self.metric = Metric::from_str(&met_s)?;
            let sel_s: String = state
                .get_item("cluster_selection_method")?
                .unwrap()
                .extract()?;
            self.cluster_selection_method = ClusterSelectionMethod::from_str(&sel_s)?;
            let is_fitted: bool = state.get_item("fitted")?.unwrap().extract()?;
            if is_fitted {
                let labels: Vec<i64> = state.get_item("labels")?.unwrap().extract()?;
                let probabilities: Vec<f64> =
                    state.get_item("probabilities")?.unwrap().extract()?;
                let cluster_persistence: Vec<f64> =
                    state.get_item("cluster_persistence")?.unwrap().extract()?;
                let n_clusters: usize = state.get_item("n_clusters")?.unwrap().extract()?;
                // HDBSCAN state is dtype-independent (all f64), use F64 variant
                self.fitted = Some(HdbscanFitted::F64(HdbscanState {
                    labels,
                    probabilities,
                    cluster_persistence,
                    n_clusters,
                    _phantom: std::marker::PhantomData,
                }));
            }
            Ok(())
        }

        fn __getnewargs__(&self) -> (usize,) {
            (self.min_cluster_size,)
        }
    }

    // ---- Agglomerative ----

    use crate::agglomerative::{
        run_agglomerative_with_metric, run_agglomerative_with_metric_f32, AgglomerativeState,
        Linkage,
    };

    enum AgglomerativeFitted {
        F64(AgglomerativeState<f64>),
        F32(AgglomerativeState<f32>),
    }

    #[pyclass]
    struct AgglomerativeClustering {
        n_clusters: usize,
        linkage: Linkage,
        metric: Metric,
        fitted: Option<AgglomerativeFitted>,
    }

    #[pymethods]
    impl AgglomerativeClustering {
        #[new]
        #[pyo3(signature = (n_clusters=2, linkage="ward", metric="euclidean"))]
        fn new(n_clusters: usize, linkage: &str, metric: &str) -> PyResult<Self> {
            if n_clusters == 0 {
                return Err(ClusterError::InvalidClusters { k: 0, n: 0 }.into());
            }
            let link = Linkage::from_str(linkage)?;
            let met = Metric::from_str(metric)?;
            if link == Linkage::Ward && met != Metric::Euclidean {
                return Err(ClusterError::WardRequiresEuclidean.into());
            }
            Ok(AgglomerativeClustering {
                n_clusters,
                linkage: link,
                metric: met,
                fitted: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let nc = self.n_clusters;
            let link = self.linkage;
            let metric = self.metric;

            dispatch_fit!(
                self,
                py,
                x,
                AgglomerativeFitted,
                |view| run_agglomerative_with_metric(&view, nc, link, metric),
                |view| run_agglomerative_with_metric_f32(&view, nc, link, metric)
            )
        }

        fn fit_predict<'py>(
            &mut self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                AgglomerativeFitted::F64(s) => s.labels.clone(),
                AgglomerativeFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => s.labels.clone(),
                AgglomerativeFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn n_clusters_(&self) -> PyResult<usize> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => Ok(s.n_clusters),
                AgglomerativeFitted::F32(s) => Ok(s.n_clusters),
            }
        }

        #[getter]
        fn children_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
            let children = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => &s.children,
                AgglomerativeFitted::F32(s) => &s.children,
            };
            let n = children.len();
            let mut arr = ndarray::Array2::<i64>::zeros((n, 2));
            for (i, &(a, b)) in children.iter().enumerate() {
                arr[[i, 0]] = a as i64;
                arr[[i, 1]] = b as i64;
            }
            Ok(PyArray2::from_owned_array(py, arr))
        }

        #[getter]
        fn distances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let dists = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => s.distances.clone(),
                AgglomerativeFitted::F32(s) => s.distances.clone(),
            };
            Ok(PyArray1::from_vec(py, dists))
        }

        fn __repr__(&self) -> String {
            let link_str = match self.linkage {
                Linkage::Ward => "ward",
                Linkage::Complete => "complete",
                Linkage::Average => "average",
                Linkage::Single => "single",
            };
            let met_str = match self.metric {
                Metric::Euclidean => "euclidean",
                Metric::Cosine => "cosine",
                Metric::Manhattan => "manhattan",
            };
            format!(
                "AgglomerativeClustering(n_clusters={}, linkage=\"{}\", metric=\"{}\")",
                self.n_clusters, link_str, met_str
            )
        }

        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("n_clusters", self.n_clusters)?;
            dict.set_item(
                "linkage",
                match self.linkage {
                    Linkage::Ward => "ward",
                    Linkage::Complete => "complete",
                    Linkage::Average => "average",
                    Linkage::Single => "single",
                },
            )?;
            dict.set_item("metric", metric_str(self.metric))?;
            if let Some(ref f) = self.fitted {
                dict.set_item("fitted", true)?;
                let (labels, children, distances, nc) = match f {
                    AgglomerativeFitted::F64(s) => (
                        s.labels.clone(),
                        s.children.clone(),
                        s.distances.clone(),
                        s.n_clusters,
                    ),
                    AgglomerativeFitted::F32(s) => (
                        s.labels.clone(),
                        s.children.clone(),
                        s.distances.clone(),
                        s.n_clusters,
                    ),
                };
                dict.set_item("labels", labels)?;
                dict.set_item(
                    "children",
                    children
                        .iter()
                        .map(|&(a, b)| vec![a as i64, b as i64])
                        .collect::<Vec<_>>(),
                )?;
                dict.set_item("distances", distances)?;
                dict.set_item("result_n_clusters", nc)?;
            } else {
                dict.set_item("fitted", false)?;
            }
            Ok(dict.into())
        }

        fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
            self.n_clusters = state.get_item("n_clusters")?.unwrap().extract()?;
            let link_s: String = state.get_item("linkage")?.unwrap().extract()?;
            self.linkage = Linkage::from_str(&link_s)?;
            let met_s: String = state.get_item("metric")?.unwrap().extract()?;
            self.metric = Metric::from_str(&met_s)?;
            let is_fitted: bool = state.get_item("fitted")?.unwrap().extract()?;
            if is_fitted {
                let labels: Vec<i64> = state.get_item("labels")?.unwrap().extract()?;
                let children_raw: Vec<Vec<i64>> = state.get_item("children")?.unwrap().extract()?;
                let children: Vec<(usize, usize)> = children_raw
                    .iter()
                    .map(|v| (v[0] as usize, v[1] as usize))
                    .collect();
                let distances: Vec<f64> = state.get_item("distances")?.unwrap().extract()?;
                let nc: usize = state.get_item("result_n_clusters")?.unwrap().extract()?;
                self.fitted = Some(AgglomerativeFitted::F64(AgglomerativeState {
                    labels,
                    children,
                    distances,
                    n_clusters: nc,
                    _phantom: std::marker::PhantomData,
                }));
            }
            Ok(())
        }

        fn __getnewargs__(&self) -> (usize,) {
            (self.n_clusters,)
        }
    }

    // ---- Mini-batch K-means ----

    use crate::minibatch_kmeans::{
        run_minibatch_kmeans_with_metric, run_minibatch_kmeans_with_metric_f32,
        MiniBatchKMeansState,
    };

    enum MiniBatchFittedState {
        F64(MiniBatchKMeansState<f64>),
        F32(MiniBatchKMeansState<f32>),
    }

    #[pyclass]
    struct MiniBatchKMeans {
        n_clusters: usize,
        batch_size: usize,
        max_iter: usize,
        tol: f64,
        random_state: u64,
        max_no_improvement: usize,
        metric: Metric,
        fitted: Option<MiniBatchFittedState>,
    }

    #[pymethods]
    impl MiniBatchKMeans {
        #[new]
        #[pyo3(signature = (n_clusters, batch_size=1024, max_iter=100, tol=0.0, random_state=0, max_no_improvement=10, metric="euclidean"))]
        fn new(
            n_clusters: usize,
            batch_size: usize,
            max_iter: usize,
            tol: f64,
            random_state: u64,
            max_no_improvement: usize,
            metric: &str,
        ) -> PyResult<Self> {
            if n_clusters == 0 {
                return Err(ClusterError::InvalidClusters { k: 0, n: 0 }.into());
            }
            if batch_size == 0 {
                return Err(ClusterError::InvalidBatchSize(0).into());
            }
            if max_iter == 0 {
                return Err(ClusterError::InvalidMaxIter(0).into());
            }
            if max_no_improvement == 0 {
                return Err(ClusterError::InvalidMaxNoImprovement(0).into());
            }
            let met = Metric::from_str(metric)?;

            Ok(MiniBatchKMeans {
                n_clusters,
                batch_size,
                max_iter,
                tol,
                random_state,
                max_no_improvement,
                metric: met,
                fitted: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let k = self.n_clusters;
            let bs = self.batch_size;
            let mi = self.max_iter;
            let tol = self.tol;
            let seed = self.random_state;
            let mni = self.max_no_improvement;
            let metric = self.metric;

            dispatch_fit!(
                self,
                py,
                x,
                MiniBatchFittedState,
                |view| run_minibatch_kmeans_with_metric(&view, k, bs, mi, tol, seed, mni, metric),
                |view| run_minibatch_kmeans_with_metric_f32(
                    &view, k, bs, mi, tol, seed, mni, metric
                )
            )
        }

        fn predict<'py>(
            &self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let metric = self.metric;
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f64>>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("Expected float64 array")
                    })?;
                    if !arr.is_c_contiguous() {
                        return Err(ClusterError::NotContiguous.into());
                    }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data(&view, expected_d)?;
                    let centroids = Arc::clone(&state.centroids_flat);
                    let k = self.n_clusters;
                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let ds = view.as_slice().expect("C-contiguous");
                        let cs = &centroids[..];
                        (0..n)
                                .into_par_iter()
                                .map(|i| {
                                    let point = &ds[i * d..(i + 1) * d];
                                    let (idx, _) = match metric {
                                        Metric::Euclidean => assign_nearest_with::<
                                            f64,
                                            crate::distance::SquaredEuclidean,
                                        >(
                                            point, cs, k, d
                                        ),
                                        Metric::Cosine => assign_nearest_with::<
                                            f64,
                                            crate::distance::CosineDistance,
                                        >(
                                            point, cs, k, d
                                        ),
                                        Metric::Manhattan => assign_nearest_with::<
                                            f64,
                                            crate::distance::ManhattanDistance,
                                        >(
                                            point, cs, k, d
                                        ),
                                    };
                                    idx as i64
                                })
                                .collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
                MiniBatchFittedState::F32(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f32>>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("Expected float32 array")
                    })?;
                    if !arr.is_c_contiguous() {
                        return Err(ClusterError::NotContiguous.into());
                    }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data_generic(&view, expected_d)?;
                    let centroids = Arc::clone(&state.centroids_flat);
                    let k = self.n_clusters;
                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let ds = view.as_slice().expect("C-contiguous");
                        let cs = &centroids[..];
                        (0..n)
                                .into_par_iter()
                                .map(|i| {
                                    let point = &ds[i * d..(i + 1) * d];
                                    let (idx, _) = match metric {
                                        Metric::Euclidean => assign_nearest_with::<
                                            f32,
                                            crate::distance::SquaredEuclidean,
                                        >(
                                            point, cs, k, d
                                        ),
                                        Metric::Cosine => assign_nearest_with::<
                                            f32,
                                            crate::distance::CosineDistance,
                                        >(
                                            point, cs, k, d
                                        ),
                                        Metric::Manhattan => assign_nearest_with::<
                                            f32,
                                            crate::distance::ManhattanDistance,
                                        >(
                                            point, cs, k, d
                                        ),
                                    };
                                    idx as i64
                                })
                                .collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
            }
        }

        fn fit_predict<'py>(
            &mut self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                MiniBatchFittedState::F64(s) => {
                    s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>()
                }
                MiniBatchFittedState::F32(s) => {
                    s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>()
                }
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => {
                    s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>()
                }
                MiniBatchFittedState::F32(s) => {
                    s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>()
                }
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter]
        fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => {
                    Ok(PyArray2::from_owned_array(py, s.centroids.clone())
                        .into_any()
                        .unbind())
                }
                MiniBatchFittedState::F32(s) => {
                    Ok(PyArray2::from_owned_array(py, s.centroids.clone())
                        .into_any()
                        .unbind())
                }
            }
        }

        #[getter]
        fn inertia_(&self) -> PyResult<f64> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => Ok(s.inertia),
                MiniBatchFittedState::F32(s) => Ok(s.inertia),
            }
        }

        #[getter]
        fn n_iter_(&self) -> PyResult<usize> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => Ok(s.n_iter),
                MiniBatchFittedState::F32(s) => Ok(s.n_iter),
            }
        }

        fn __repr__(&self) -> String {
            let met_str = match self.metric {
                Metric::Euclidean => "euclidean",
                Metric::Cosine => "cosine",
                Metric::Manhattan => "manhattan",
            };
            format!(
                "MiniBatchKMeans(n_clusters={}, batch_size={}, max_iter={}, tol={}, random_state={}, max_no_improvement={}, metric=\"{}\")",
                self.n_clusters, self.batch_size, self.max_iter, self.tol, self.random_state, self.max_no_improvement, met_str
            )
        }

        fn snapshot(&self) -> PyResult<PyClusterSnapshot> {
            let metric = self.metric;
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(state) => Ok(PyClusterSnapshot {
                    inner: snapshot::ClusterSnapshot::from_minibatch_kmeans(state, metric),
                }),
                MiniBatchFittedState::F32(state) => Ok(PyClusterSnapshot {
                    inner: snapshot::ClusterSnapshot::from_minibatch_kmeans_f32(state, metric),
                }),
            }
        }

        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("n_clusters", self.n_clusters)?;
            dict.set_item("batch_size", self.batch_size)?;
            dict.set_item("max_iter", self.max_iter)?;
            dict.set_item("tol", self.tol)?;
            dict.set_item("random_state", self.random_state)?;
            dict.set_item("max_no_improvement", self.max_no_improvement)?;
            dict.set_item("metric", metric_str(self.metric))?;
            match &self.fitted {
                None => {
                    dict.set_item("fitted", false)?;
                }
                Some(MiniBatchFittedState::F64(s)) => {
                    dict.set_item("fitted", true)?;
                    dict.set_item("dtype", "float64")?;
                    dict.set_item(
                        "centroids",
                        PyArray2::from_owned_array(py, s.centroids.clone()),
                    )?;
                    dict.set_item(
                        "labels",
                        s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                    )?;
                    dict.set_item("inertia", s.inertia)?;
                    dict.set_item("n_iter", s.n_iter)?;
                }
                Some(MiniBatchFittedState::F32(s)) => {
                    dict.set_item("fitted", true)?;
                    dict.set_item("dtype", "float32")?;
                    dict.set_item(
                        "centroids",
                        PyArray2::from_owned_array(py, s.centroids.clone()),
                    )?;
                    dict.set_item(
                        "labels",
                        s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                    )?;
                    dict.set_item("inertia", s.inertia)?;
                    dict.set_item("n_iter", s.n_iter)?;
                }
            }
            Ok(dict.into())
        }

        fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
            self.n_clusters = state.get_item("n_clusters")?.unwrap().extract()?;
            self.batch_size = state.get_item("batch_size")?.unwrap().extract()?;
            self.max_iter = state.get_item("max_iter")?.unwrap().extract()?;
            self.tol = state.get_item("tol")?.unwrap().extract()?;
            self.random_state = state.get_item("random_state")?.unwrap().extract()?;
            self.max_no_improvement = state.get_item("max_no_improvement")?.unwrap().extract()?;
            let met_s: String = state.get_item("metric")?.unwrap().extract()?;
            self.metric = Metric::from_str(&met_s)?;
            let is_fitted: bool = state.get_item("fitted")?.unwrap().extract()?;
            if is_fitted {
                let dtype: String = state.get_item("dtype")?.unwrap().extract()?;
                let labels_i64: Vec<i64> = state.get_item("labels")?.unwrap().extract()?;
                let labels: Vec<usize> = labels_i64.iter().map(|&l| l as usize).collect();
                let inertia: f64 = state.get_item("inertia")?.unwrap().extract()?;
                let n_iter: usize = state.get_item("n_iter")?.unwrap().extract()?;
                if dtype == "float32" {
                    let arr = state
                        .get_item("centroids")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f32>>()?;
                    let centroids = arr.as_array().to_owned();
                    let centroids_flat =
                        Arc::new(centroids.as_slice().expect("C-contiguous").to_vec());
                    self.fitted = Some(MiniBatchFittedState::F32(MiniBatchKMeansState {
                        centroids,
                        centroids_flat,
                        labels,
                        inertia,
                        n_iter,
                    }));
                } else {
                    let arr = state
                        .get_item("centroids")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f64>>()?;
                    let centroids = arr.as_array().to_owned();
                    let centroids_flat =
                        Arc::new(centroids.as_slice().expect("C-contiguous").to_vec());
                    self.fitted = Some(MiniBatchFittedState::F64(MiniBatchKMeansState {
                        centroids,
                        centroids_flat,
                        labels,
                        inertia,
                        n_iter,
                    }));
                }
            }
            Ok(())
        }

        fn __getnewargs__(&self) -> (usize,) {
            (self.n_clusters,)
        }
    }

    // ---- EmbeddingCluster (experimental) ----

    use crate::embedding::{evaluation, normalize, reducer, reduction, spherical_kmeans, vmf};

    /// Core fitted state from spherical K-means + evaluation.
    struct EmbeddingClusterFitted {
        labels: Vec<usize>,
        centroids: Vec<f64>, // flat, unit-norm, in reduced space
        fitted_d: usize,
        objective: f64,
        n_iter: usize,
        representatives: Vec<usize>,
        intra_similarity: Vec<f64>,
        resultant_lengths: Vec<f64>,
        pca_projection: Option<reduction::PcaProjection>,
        // Ephemeral cache (not serialized)
        reduced_data: Option<Vec<f64>>,
        reduced_n: usize,
        reduced_d: usize,
    }

    /// Optional vMF refinement state.
    struct VmfState {
        probabilities: Vec<f64>,
        concentrations: Vec<f64>,
        bic: f64,
    }

    #[pyclass]
    struct EmbeddingCluster {
        n_clusters: usize,
        reduction_dim: Option<usize>,
        reduction: String, // "pca" or "matryoshka"
        max_iter: usize,
        tol: f64,
        random_state: u64,
        n_init: usize,
        fitted: Option<EmbeddingClusterFitted>,
        vmf: Option<VmfState>,
    }

    #[pymethods]
    impl EmbeddingCluster {
        #[new]
        #[pyo3(signature = (n_clusters=50, reduction_dim=Some(128), max_iter=100, tol=1e-6, random_state=0, n_init=5, reduction="pca"))]
        fn new(
            n_clusters: usize,
            reduction_dim: Option<usize>,
            max_iter: usize,
            tol: f64,
            random_state: u64,
            n_init: usize,
            reduction: &str,
        ) -> PyResult<Self> {
            if n_clusters == 0 {
                return Err(ClusterError::InvalidClusters { k: 0, n: 0 }.into());
            }
            let reduction_str = reduction.to_lowercase();
            if reduction_str != "pca" && reduction_str != "matryoshka" && reduction_str != "none" {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "reduction must be 'pca', 'matryoshka', or 'none', got '{}'",
                    reduction
                )));
            }
            Ok(EmbeddingCluster {
                n_clusters,
                reduction_dim,
                reduction: reduction_str,
                max_iter,
                tol,
                random_state,
                n_init,
                fitted: None,
                vmf: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            // Extract as f32 first (primary for embeddings), then f64
            let (data_f64, n, d) = if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                let view = arr.as_array();
                let (n, d) = view.dim();
                let data: Vec<f64> = view
                    .as_slice()
                    .expect("contiguous")
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                (data, n, d)
            } else if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                let view = arr.as_array();
                let (n, d) = view.dim();
                let data: Vec<f64> = view.as_slice().expect("contiguous").to_vec();
                (data, n, d)
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Expected float32 or float64 array",
                ));
            };

            let k = self.n_clusters;
            let max_iter = self.max_iter;
            let tol = self.tol;
            let seed = self.random_state;
            let n_init = self.n_init;
            let reduction_dim = self.reduction_dim;
            let reduction_method = self.reduction.clone();

            // Stage 1: L2 normalize (GIL released)
            let mut data = py.allow_threads(move || {
                let mut data = data_f64;
                normalize::l2_normalize_rows_inplace(&mut data, n, d);
                data
            });
            py.check_signals()?;

            // Stage 2: Dimensionality reduction (GIL released)
            let (work_data, work_d, pca_proj) =
                py.allow_threads(|| match (reduction_dim, reduction_method.as_str()) {
                    (None, _) | (Some(_), "none") => (data, d, None),
                    (Some(target), _) if target >= d => (data, d, None),
                    (Some(target), "matryoshka") => {
                        let mut truncated = Vec::with_capacity(n * target);
                        for i in 0..n {
                            truncated.extend_from_slice(&data[i * d..i * d + target]);
                        }
                        normalize::l2_normalize_rows_inplace(&mut truncated, n, target);
                        (truncated, target, None)
                    }
                    (Some(target), _) => {
                        let mut rng =
                            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
                        let proj = reduction::compute_pca(&data, n, d, target, 10, &mut rng);
                        let projected = reduction::project_data::<f64>(&data, n, &proj);
                        let mut proj_data = projected;
                        normalize::l2_normalize_rows_inplace(&mut proj_data, n, proj.output_dim);
                        (proj_data, proj.output_dim, Some(proj))
                    }
                });
            py.check_signals()?;

            // Cache reduced data if reduction was performed
            let cached_reduced = if work_d != d {
                Some((work_data.clone(), n, work_d))
            } else {
                None
            };

            // Stage 3: Spherical K-means (GIL released)
            let skmeans = py.allow_threads(|| {
                spherical_kmeans::run_spherical_kmeans(
                    &work_data, n, work_d, k, max_iter, tol, seed, n_init,
                )
            })?;
            py.check_signals()?;

            // Stage 4: Evaluation (GIL released)
            let centroids_flat = skmeans.centroids.as_slice().unwrap().to_vec();
            let result = py.allow_threads(|| {
                let reps = evaluation::find_representatives(
                    &work_data,
                    &skmeans.labels,
                    &centroids_flat,
                    n,
                    work_d,
                    k,
                );
                let intra_sim = evaluation::intra_cluster_similarity(
                    &work_data,
                    &skmeans.labels,
                    &centroids_flat,
                    n,
                    work_d,
                    k,
                );
                let res_lens =
                    evaluation::resultant_lengths(&work_data, &skmeans.labels, n, work_d, k);
                (
                    skmeans.labels,
                    centroids_flat,
                    work_d,
                    skmeans.objective,
                    skmeans.n_iter,
                    reps,
                    intra_sim,
                    res_lens,
                    pca_proj,
                )
            });

            let (labels, centroids, work_d, objective, n_iter, reps, intra_sim, res_lens, pca_proj) =
                result;
            let (reduced_data, reduced_n, reduced_d) = match cached_reduced {
                Some((rd, rn, rdim)) => (Some(rd), rn, rdim),
                None => (None, 0, 0),
            };
            self.fitted = Some(EmbeddingClusterFitted {
                labels,
                centroids,
                fitted_d: work_d,
                objective,
                n_iter,
                representatives: reps,
                intra_similarity: intra_sim,
                resultant_lengths: res_lens,
                pca_projection: pca_proj,
                reduced_data,
                reduced_n,
                reduced_d,
            });
            self.vmf = None;
            Ok(())
        }

        fn refine_vmf(
            &mut self,
            py: Python<'_>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<()> {
            let f = self.fitted.as_mut().ok_or(ClusterError::NotFitted)?;

            // Need the working data again — extract and process
            let (data_f64, n, d) = extract_f64_2d(x)?;

            let k = self.n_clusters;
            let work_d = f.fitted_d;
            let labels_clone = f.labels.clone();
            let centroids_clone = f.centroids.clone();
            let pca_proj = f.pca_projection.take();

            let vmf_result = py.allow_threads(move || {
                let mut data = data_f64;
                normalize::l2_normalize_rows_inplace(&mut data, n, d);

                let work_data = if let Some(ref proj) = pca_proj {
                    let projected = reduction::project_data::<f64>(&data, n, proj);
                    let mut proj_data = projected;
                    normalize::l2_normalize_rows_inplace(&mut proj_data, n, work_d);
                    proj_data
                } else {
                    data
                };

                let vmf_state = vmf::fit_vmf(
                    &work_data,
                    n,
                    work_d,
                    k,
                    &centroids_clone,
                    &labels_clone,
                    50,
                    1e-6,
                );
                (vmf_state, pca_proj)
            });

            let (vmf_state, pca_proj) = vmf_result;
            self.fitted.as_mut().unwrap().pca_projection = pca_proj;
            self.vmf = Some(VmfState {
                probabilities: vmf_state.responsibilities,
                concentrations: vmf_state.concentrations,
                bic: vmf_state.bic,
            });
            Ok(())
        }

        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let f = self.fitted.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(
                py,
                f.labels.iter().map(|&l| l as i64).collect(),
            ))
        }

        #[getter]
        fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let f = self.fitted.as_ref().ok_or(ClusterError::NotFitted)?;
            let arr =
                ndarray::Array2::from_shape_vec((self.n_clusters, f.fitted_d), f.centroids.clone())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }

        #[getter]
        fn objective_(&self) -> PyResult<f64> {
            Ok(self
                .fitted
                .as_ref()
                .ok_or(ClusterError::NotFitted)?
                .objective)
        }

        #[getter]
        fn n_iter_(&self) -> PyResult<usize> {
            Ok(self.fitted.as_ref().ok_or(ClusterError::NotFitted)?.n_iter)
        }

        #[getter]
        fn representatives_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let f = self.fitted.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(
                py,
                f.representatives.iter().map(|&r| r as i64).collect(),
            ))
        }

        #[getter]
        fn intra_similarity_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let f = self.fitted.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(py, f.intra_similarity.clone()))
        }

        #[getter]
        fn resultant_lengths_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let f = self.fitted.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(py, f.resultant_lengths.clone()))
        }

        #[getter]
        fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let v = self.vmf.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call refine_vmf() first")
            })?;
            let n = v.probabilities.len() / self.n_clusters;
            let arr =
                ndarray::Array2::from_shape_vec((n, self.n_clusters), v.probabilities.clone())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }

        #[getter]
        fn concentrations_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let v = self.vmf.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call refine_vmf() first")
            })?;
            Ok(PyArray1::from_vec(py, v.concentrations.clone()))
        }

        #[getter]
        fn bic_(&self) -> PyResult<f64> {
            let v = self.vmf.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call refine_vmf() first")
            })?;
            Ok(v.bic)
        }

        #[getter]
        fn reduced_data_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let f = match self.fitted.as_ref() {
                Some(f) => f,
                None => return Ok(py.None()),
            };
            match f.reduced_data {
                Some(ref data) => {
                    let arr =
                        ndarray::Array2::from_shape_vec((f.reduced_n, f.reduced_d), data.clone())
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, arr).into_any().unbind())
                }
                None => Ok(py.None()),
            }
        }

        fn snapshot(&self) -> PyResult<PyClusterSnapshot> {
            let f = self.fitted.as_ref().ok_or(ClusterError::NotFitted)?;
            let input_dim = match &f.pca_projection {
                Some(proj) => proj.input_dim,
                None => f.fitted_d,
            };
            Ok(PyClusterSnapshot {
                inner: snapshot::ClusterSnapshot::from_embedding_cluster(
                    &f.centroids,
                    self.n_clusters,
                    f.fitted_d,
                    input_dim,
                    f.pca_projection.as_ref(),
                    &f.labels,
                    &f.intra_similarity,
                ),
            })
        }

        fn __repr__(&self) -> String {
            format!(
                "EmbeddingCluster(n_clusters={}, reduction_dim={:?}, max_iter={}, n_init={})",
                self.n_clusters, self.reduction_dim, self.max_iter, self.n_init
            )
        }

        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let dict = pyo3::types::PyDict::new(py);
            // Config
            dict.set_item("n_clusters", self.n_clusters)?;
            dict.set_item("reduction_dim", self.reduction_dim)?;
            dict.set_item("reduction", &self.reduction)?;
            dict.set_item("max_iter", self.max_iter)?;
            dict.set_item("tol", self.tol)?;
            dict.set_item("random_state", self.random_state)?;
            dict.set_item("n_init", self.n_init)?;

            // Fitted state
            let is_fitted = self.fitted.is_some();
            dict.set_item("fitted", is_fitted)?;
            if let Some(ref f) = self.fitted {
                dict.set_item(
                    "labels",
                    f.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                )?;
                dict.set_item("centroids", f.centroids.clone())?;
                dict.set_item("fitted_d", f.fitted_d)?;
                dict.set_item("objective", f.objective)?;
                dict.set_item("n_iter", f.n_iter)?;
                dict.set_item(
                    "representatives",
                    f.representatives
                        .iter()
                        .map(|&r| r as i64)
                        .collect::<Vec<_>>(),
                )?;
                dict.set_item("intra_similarity", f.intra_similarity.clone())?;
                dict.set_item("resultant_lengths", f.resultant_lengths.clone())?;

                // PCA projection (optional)
                let has_pca = f.pca_projection.is_some();
                dict.set_item("has_pca", has_pca)?;
                if let Some(ref proj) = f.pca_projection {
                    dict.set_item("pca_components", proj.components.clone())?;
                    dict.set_item("pca_mean", proj.mean.clone())?;
                    dict.set_item("pca_input_dim", proj.input_dim)?;
                    dict.set_item("pca_output_dim", proj.output_dim)?;
                }

                // vMF state (optional)
                let vmf_fitted = self.vmf.is_some();
                dict.set_item("vmf_fitted", vmf_fitted)?;
                if let Some(ref v) = self.vmf {
                    dict.set_item("vmf_probabilities", v.probabilities.clone())?;
                    dict.set_item("vmf_concentrations", v.concentrations.clone())?;
                    dict.set_item("vmf_bic", v.bic)?;
                }
            }
            Ok(dict.into())
        }

        fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
            // Config
            self.n_clusters = state.get_item("n_clusters")?.unwrap().extract()?;
            self.reduction_dim = state.get_item("reduction_dim")?.unwrap().extract()?;
            self.reduction = state.get_item("reduction")?.unwrap().extract()?;
            self.max_iter = state.get_item("max_iter")?.unwrap().extract()?;
            self.tol = state.get_item("tol")?.unwrap().extract()?;
            self.random_state = state.get_item("random_state")?.unwrap().extract()?;
            self.n_init = state.get_item("n_init")?.unwrap().extract()?;

            let is_fitted: bool = state.get_item("fitted")?.unwrap().extract()?;
            if is_fitted {
                let labels_i64: Vec<i64> = state.get_item("labels")?.unwrap().extract()?;
                let labels = labels_i64.iter().map(|&l| l as usize).collect();
                let centroids = state.get_item("centroids")?.unwrap().extract()?;
                let fitted_d = state.get_item("fitted_d")?.unwrap().extract()?;
                let objective = state.get_item("objective")?.unwrap().extract()?;
                let n_iter = state.get_item("n_iter")?.unwrap().extract()?;
                let reps_i64: Vec<i64> = state.get_item("representatives")?.unwrap().extract()?;
                let representatives = reps_i64.iter().map(|&r| r as usize).collect();
                let intra_similarity = state.get_item("intra_similarity")?.unwrap().extract()?;
                let resultant_lengths = state.get_item("resultant_lengths")?.unwrap().extract()?;

                let has_pca: bool = state.get_item("has_pca")?.unwrap().extract()?;
                let pca_projection = if has_pca {
                    Some(reduction::PcaProjection {
                        components: state.get_item("pca_components")?.unwrap().extract()?,
                        mean: state.get_item("pca_mean")?.unwrap().extract()?,
                        input_dim: state.get_item("pca_input_dim")?.unwrap().extract()?,
                        output_dim: state.get_item("pca_output_dim")?.unwrap().extract()?,
                    })
                } else {
                    None
                };

                self.fitted = Some(EmbeddingClusterFitted {
                    labels,
                    centroids,
                    fitted_d,
                    objective,
                    n_iter,
                    representatives,
                    intra_similarity,
                    resultant_lengths,
                    pca_projection,
                    reduced_data: None,
                    reduced_n: 0,
                    reduced_d: 0,
                });

                let vmf_fitted: bool = state.get_item("vmf_fitted")?.unwrap().extract()?;
                self.vmf = if vmf_fitted {
                    Some(VmfState {
                        probabilities: state.get_item("vmf_probabilities")?.unwrap().extract()?,
                        concentrations: state.get_item("vmf_concentrations")?.unwrap().extract()?,
                        bic: state.get_item("vmf_bic")?.unwrap().extract()?,
                    })
                } else {
                    None
                };
            } else {
                self.fitted = None;
                self.vmf = None;
            }

            Ok(())
        }

        fn __getnewargs__(&self) -> (usize,) {
            (self.n_clusters,)
        }
    }

    // ---- EmbeddingReducer ----

    #[pyclass]
    struct EmbeddingReducer {
        target_dim: usize,
        method: String,
        seed: u64,
        state: Option<reducer::EmbeddingReducerState>,
    }

    #[pymethods]
    impl EmbeddingReducer {
        #[new]
        #[pyo3(signature = (target_dim=128, method="pca", random_state=0))]
        fn new(target_dim: usize, method: &str, random_state: u64) -> PyResult<Self> {
            let method_str = method.to_lowercase();
            if method_str != "pca" && method_str != "matryoshka" {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "method must be 'pca' or 'matryoshka', got '{}'",
                    method
                )));
            }
            Ok(EmbeddingReducer {
                target_dim,
                method: method_str,
                seed: random_state,
                state: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let (data_f64, n, d) = extract_f64_2d(x)?;
            let target_dim = self.target_dim;
            let seed = self.seed;
            let method = self.method.clone();

            let state = py.allow_threads(move || match method.as_str() {
                "pca" => reducer::fit_pca(&data_f64, n, d, target_dim, seed),
                "matryoshka" => reducer::fit_matryoshka(d, target_dim),
                _ => unreachable!(),
            });

            self.state = Some(state);
            Ok(())
        }

        fn transform<'py>(
            &self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let state = self.state.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call fit() before transform()")
            })?;
            let (data_f64, n, d) = extract_f64_2d(x)?;
            let state_method = state.method.clone();
            let state_input_dim = state.input_dim;
            let state_target_dim = state.target_dim;
            let state_mean = state.mean.clone();
            let state_components = state.components.clone();

            let result = py
                .allow_threads(move || {
                    let st = reducer::EmbeddingReducerState {
                        method: state_method,
                        input_dim: state_input_dim,
                        target_dim: state_target_dim,
                        mean: state_mean,
                        components: state_components,
                    };
                    reducer::transform(&data_f64, n, d, &st)
                })
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

            let target = self.state.as_ref().unwrap().target_dim;
            let n_out = result.len() / target;
            let arr = ndarray::Array2::from_shape_vec((n_out, target), result)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }

        fn fit_transform<'py>(
            &mut self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            self.fit(py, x)?;
            self.transform(py, x)
        }

        fn save(&self, path: &str) -> PyResult<()> {
            let state = self.state.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call fit() before save()")
            })?;
            reducer::save_state(state, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
        }

        #[staticmethod]
        fn load(path: &str) -> PyResult<Self> {
            let state =
                reducer::load_state(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
            let target_dim = state.target_dim;
            let method = state.method.clone();
            Ok(EmbeddingReducer {
                target_dim,
                method,
                seed: 0,
                state: Some(state),
            })
        }

        #[getter]
        fn target_dim(&self) -> usize {
            self.target_dim
        }

        #[getter]
        fn method(&self) -> &str {
            &self.method
        }

        fn __repr__(&self) -> String {
            let fitted = if self.state.is_some() {
                "fitted"
            } else {
                "unfitted"
            };
            format!(
                "EmbeddingReducer(target_dim={}, method='{}', {})",
                self.target_dim, self.method, fitted
            )
        }
    }

    // ---- ClusterSnapshot ----

    use crate::snapshot;
    use crate::snapshot_io;

    #[pyclass(name = "ClusterSnapshot")]
    struct PyClusterSnapshot {
        inner: snapshot::ClusterSnapshot,
    }

    #[pymethods]
    impl PyClusterSnapshot {
        /// Assign new points to nearest cluster.
        fn assign<'py>(
            &self,
            py: Python<'py>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let (data, n, d) = extract_f64_2d(x)?;
            if d != self.inner.input_dim {
                return Err(ClusterError::DimensionMismatch {
                    expected: self.inner.input_dim,
                    got: d,
                }
                .into());
            }
            let inner = &self.inner;
            let result =
                py.allow_threads(move || inner.assign_batch(&data, n))?;
            Ok(PyArray1::from_vec(py, result.labels))
        }

        /// Assign with confidence scores and optional rejection.
        #[pyo3(signature = (x, distance_threshold=None, confidence_threshold=None))]
        fn assign_with_scores(
            &self,
            py: Python<'_>,
            x: &Bound<'_, pyo3::types::PyAny>,
            distance_threshold: Option<f64>,
            confidence_threshold: Option<f64>,
        ) -> PyResult<PyAssignmentResult> {
            let (data, n, d) = extract_f64_2d(x)?;
            if d != self.inner.input_dim {
                return Err(ClusterError::DimensionMismatch {
                    expected: self.inner.input_dim,
                    got: d,
                }
                .into());
            }
            let inner = &self.inner;
            let spherical = self.inner.spherical;
            let mut result =
                py.allow_threads(move || inner.assign_batch(&data, n))?;
            result.apply_rejection(distance_threshold, confidence_threshold, spherical);
            Ok(PyAssignmentResult { inner: result })
        }

        /// Save snapshot to directory.
        fn save(&self, path: &str) -> PyResult<()> {
            snapshot_io::save_snapshot(&self.inner, path)?;
            Ok(())
        }

        /// Compute drift statistics against new data.
        fn drift_report(
            &self,
            py: Python<'_>,
            x: &Bound<'_, pyo3::types::PyAny>,
        ) -> PyResult<PyDriftReport> {
            let (data, n, d) = extract_f64_2d(x)?;
            if d != self.inner.input_dim {
                return Err(ClusterError::DimensionMismatch {
                    expected: self.inner.input_dim,
                    got: d,
                }
                .into());
            }
            let inner = &self.inner;
            let report =
                py.allow_threads(move || inner.drift_report(&data, n))?;
            Ok(PyDriftReport { inner: report })
        }

        /// Load snapshot from directory.
        #[staticmethod]
        fn load(path: &str) -> PyResult<Self> {
            let inner = snapshot_io::load_snapshot(path)?;
            Ok(PyClusterSnapshot { inner })
        }

        #[getter]
        fn k(&self) -> usize {
            self.inner.k
        }

        #[getter]
        fn d(&self) -> usize {
            self.inner.d
        }

        #[getter]
        fn input_dim(&self) -> usize {
            self.inner.input_dim
        }

        #[getter]
        fn algorithm(&self) -> &str {
            self.inner.algorithm.as_str()
        }

        #[getter]
        fn metric(&self) -> &str {
            match self.inner.metric {
                Metric::Euclidean => "euclidean",
                Metric::Cosine => "cosine",
                Metric::Manhattan => "manhattan",
            }
        }

        #[getter]
        fn spherical(&self) -> bool {
            self.inner.spherical
        }

        fn __repr__(&self) -> String {
            format!(
                "ClusterSnapshot(algorithm=\"{}\", k={}, d={}, input_dim={})",
                self.inner.algorithm.as_str(),
                self.inner.k,
                self.inner.d,
                self.inner.input_dim
            )
        }
    }

    #[pyclass(name = "DriftReport")]
    struct PyDriftReport {
        inner: snapshot::DriftReport,
    }

    #[pymethods]
    impl PyDriftReport {
        #[getter]
        fn new_mean_distances_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_vec(py, self.inner.new_mean_distances.clone())
        }

        #[getter]
        fn new_cluster_sizes_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
            PyArray1::from_vec(
                py,
                self.inner
                    .new_cluster_sizes
                    .iter()
                    .map(|&c| c as i64)
                    .collect(),
            )
        }

        #[getter]
        fn relative_drift_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_vec(py, self.inner.relative_drift.clone())
        }

        #[getter]
        fn global_mean_distance_(&self) -> f64 {
            self.inner.global_mean_distance
        }

        #[getter]
        fn rejection_rate_(&self) -> f64 {
            self.inner.rejection_rate
        }

        #[getter]
        fn n_samples_(&self) -> usize {
            self.inner.n_samples
        }

        fn __repr__(&self) -> String {
            format!(
                "DriftReport(n_samples={}, global_mean_distance={:.4}, rejection_rate={:.2}%)",
                self.inner.n_samples,
                self.inner.global_mean_distance,
                self.inner.rejection_rate * 100.0
            )
        }
    }

    #[pyclass(name = "AssignmentResult")]
    struct PyAssignmentResult {
        inner: snapshot::AssignmentResult,
    }

    #[pymethods]
    impl PyAssignmentResult {
        #[getter]
        fn labels_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
            PyArray1::from_vec(py, self.inner.labels.clone())
        }

        #[getter]
        fn distances_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_vec(py, self.inner.distances.clone())
        }

        #[getter]
        fn second_distances_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_vec(py, self.inner.second_distances.clone())
        }

        #[getter]
        fn confidences_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_vec(py, self.inner.confidences.clone())
        }

        #[getter]
        fn rejected_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
            Ok(PyArray1::from_vec(py, self.inner.rejected.clone()))
        }

        fn __repr__(&self) -> String {
            let n = self.inner.labels.len();
            let rejected = self.inner.rejected.iter().filter(|&&r| r).count();
            format!("AssignmentResult(n={n}, rejected={rejected})")
        }
    }

    /// Helper: extract a 2D array as Vec<f64> + (n, d).
    fn extract_f64_2d(x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            let (n, d) = view.dim();
            let data: Vec<f64> = view
                .as_slice()
                .expect("contiguous")
                .iter()
                .map(|&v| v as f64)
                .collect();
            Ok((data, n, d))
        } else if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            let (n, d) = view.dim();
            let data: Vec<f64> = view.as_slice().expect("contiguous").to_vec();
            Ok((data, n, d))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected float32 or float64 array",
            ))
        }
    }

    // ---- Metrics ----

    #[pyfunction]
    fn silhouette_score(
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        labels: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<f64> {
        let labels_slice = labels
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("labels must be C-contiguous"))?;
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::silhouette_score(&view, labels_slice))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::silhouette_score(&view, labels_slice))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "Expected float32 or float64 array",
        ))
    }

    #[pyfunction]
    fn calinski_harabasz_score(
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        labels: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<f64> {
        let labels_slice = labels
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("labels must be C-contiguous"))?;
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::calinski_harabasz_score(&view, labels_slice))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::calinski_harabasz_score(&view, labels_slice))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "Expected float32 or float64 array",
        ))
    }

    #[pyfunction]
    fn davies_bouldin_score(
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        labels: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<f64> {
        let labels_slice = labels
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("labels must be C-contiguous"))?;
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::davies_bouldin_score(&view, labels_slice))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::davies_bouldin_score(&view, labels_slice))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "Expected float32 or float64 array",
        ))
    }

    #[pymodule]
    pub fn _rustcluster(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<KMeans>()?;
        m.add_class::<MiniBatchKMeans>()?;
        m.add_class::<Dbscan>()?;
        m.add_class::<Hdbscan>()?;
        m.add_class::<AgglomerativeClustering>()?;
        m.add_class::<EmbeddingCluster>()?;
        m.add_class::<EmbeddingReducer>()?;
        m.add_class::<PyClusterSnapshot>()?;
        m.add_class::<PyAssignmentResult>()?;
        m.add_class::<PyDriftReport>()?;
        m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
        m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
        m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::_rustcluster;
