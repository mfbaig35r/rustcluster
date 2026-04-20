pub mod agglomerative;
pub mod dbscan;
pub mod distance;
pub mod embedding;
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
    use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use rayon::prelude::*;

    use crate::dbscan::{run_dbscan_with_metric, run_dbscan_with_metric_f32, DbscanState};
    use crate::distance::Metric;
    use crate::error::ClusterError;
    use crate::kmeans::{
        run_kmeans_with_metric, run_kmeans_with_metric_f32, Algorithm, KMeansState,
    };
    use crate::utils::{assign_nearest_with, validate_predict_data, validate_predict_data_generic};

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

            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_kmeans_with_metric(&view, k, max_iter, tol, seed, n_init, algo, metric)
                })?;
                self.fitted = Some(FittedState::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_kmeans_with_metric_f32(&view, k, max_iter, tol, seed, n_init, algo, metric)
                })?;
                self.fitted = Some(FittedState::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected a C-contiguous float32 or float64 NumPy array",
            ))
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

                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;

                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let data_slice = view.as_slice().expect("C-contiguous");
                        let centroids_slice = centroids.as_slice().expect("C-contiguous");
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

                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;

                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let data_slice = view.as_slice().expect("C-contiguous");
                        let centroids_slice = centroids.as_slice().expect("C-contiguous");
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
                    self.fitted = Some(FittedState::F32(KMeansState {
                        centroids: arr.as_array().to_owned(),
                        labels,
                        inertia,
                        n_iter,
                    }));
                } else {
                    let arr = state
                        .get_item("centroids")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f64>>()?;
                    self.fitted = Some(FittedState::F64(KMeansState {
                        centroids: arr.as_array().to_owned(),
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

            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_dbscan_with_metric(&view, eps, min_samples, metric)
                })?;
                self.fitted = Some(DbscanFitted::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_dbscan_with_metric_f32(&view, eps, min_samples, metric)
                })?;
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

            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state =
                    py.allow_threads(move || run_hdbscan_with_metric(&view, mcs, ms, metric, sel))?;
                self.fitted = Some(HdbscanFitted::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_hdbscan_with_metric_f32(&view, mcs, ms, metric, sel)
                })?;
                self.fitted = Some(HdbscanFitted::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected float32 or float64 array",
            ))
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

            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_agglomerative_with_metric(&view, nc, link, metric)
                })?;
                self.fitted = Some(AgglomerativeFitted::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_agglomerative_with_metric_f32(&view, nc, link, metric)
                })?;
                self.fitted = Some(AgglomerativeFitted::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected float32 or float64 array",
            ))
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

            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_minibatch_kmeans_with_metric(&view, k, bs, mi, tol, seed, mni, metric)
                })?;
                self.fitted = Some(MiniBatchFittedState::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() {
                    return Err(ClusterError::NotContiguous.into());
                }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_minibatch_kmeans_with_metric_f32(&view, k, bs, mi, tol, seed, mni, metric)
                })?;
                self.fitted = Some(MiniBatchFittedState::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected float32 or float64 array",
            ))
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
                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;
                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let ds = view.as_slice().expect("C-contiguous");
                        let cs = centroids.as_slice().expect("C-contiguous");
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
                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;
                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let ds = view.as_slice().expect("C-contiguous");
                        let cs = centroids.as_slice().expect("C-contiguous");
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
                    self.fitted = Some(MiniBatchFittedState::F32(MiniBatchKMeansState {
                        centroids: arr.as_array().to_owned(),
                        labels,
                        inertia,
                        n_iter,
                    }));
                } else {
                    let arr = state
                        .get_item("centroids")?
                        .unwrap()
                        .extract::<PyReadonlyArray2<'_, f64>>()?;
                    self.fitted = Some(MiniBatchFittedState::F64(MiniBatchKMeansState {
                        centroids: arr.as_array().to_owned(),
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

    use crate::embedding::{
        evaluation, normalize, reduction, spherical_kmeans, vmf,
    };

    #[pyclass]
    struct EmbeddingCluster {
        n_clusters: usize,
        reduction_dim: Option<usize>,
        reduction: String,  // "pca" or "matryoshka"
        max_iter: usize,
        tol: f64,
        random_state: u64,
        n_init: usize,
        // Fitted state
        labels: Option<Vec<usize>>,
        centroids: Option<Vec<f64>>,  // flat, unit-norm, in reduced space
        fitted_d: usize,              // dimensionality of fitted centroids
        objective: Option<f64>,
        n_iter: Option<usize>,
        representatives: Option<Vec<usize>>,
        intra_similarity: Option<Vec<f64>>,
        resultant_lengths: Option<Vec<f64>>,
        // vMF state
        vmf_probabilities: Option<Vec<f64>>,
        vmf_concentrations: Option<Vec<f64>>,
        vmf_bic: Option<f64>,
        // PCA projection for later use
        pca_projection: Option<reduction::PcaProjection>,
    }

    #[pymethods]
    impl EmbeddingCluster {
        #[new]
        #[pyo3(signature = (n_clusters=50, reduction_dim=Some(128), max_iter=100, tol=1e-6, random_state=0, n_init=5, reduction="pca"))]
        fn new(n_clusters: usize, reduction_dim: Option<usize>, max_iter: usize, tol: f64, random_state: u64, n_init: usize, reduction: &str) -> PyResult<Self> {
            if n_clusters == 0 { return Err(ClusterError::InvalidClusters { k: 0, n: 0 }.into()); }
            let reduction_str = reduction.to_lowercase();
            if reduction_str != "pca" && reduction_str != "matryoshka" && reduction_str != "none" {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("reduction must be 'pca', 'matryoshka', or 'none', got '{}'", reduction)
                ));
            }
            Ok(EmbeddingCluster {
                n_clusters, reduction_dim, reduction: reduction_str, max_iter, tol, random_state, n_init,
                labels: None, centroids: None, fitted_d: 0, objective: None, n_iter: None,
                representatives: None, intra_similarity: None, resultant_lengths: None,
                vmf_probabilities: None, vmf_concentrations: None, vmf_bic: None,
                pca_projection: None,
            })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            // Extract as f32 first (primary for embeddings), then f64
            let (data_f64, n, d) = if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                let view = arr.as_array();
                let (n, d) = view.dim();
                let data: Vec<f64> = view.as_slice().expect("contiguous").iter().map(|&v| v as f64).collect();
                (data, n, d)
            } else if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                let view = arr.as_array();
                let (n, d) = view.dim();
                let data: Vec<f64> = view.as_slice().expect("contiguous").to_vec();
                (data, n, d)
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err("Expected float32 or float64 array"));
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
            let (work_data, work_d, pca_proj) = py.allow_threads(|| {
                match (reduction_dim, reduction_method.as_str()) {
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
                        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
                        let proj = reduction::compute_pca(&data, n, d, target, 10, &mut rng);
                        let projected = reduction::project_data::<f64>(&data, n, &proj);
                        let mut proj_data = projected;
                        normalize::l2_normalize_rows_inplace(&mut proj_data, n, proj.output_dim);
                        (proj_data, proj.output_dim, Some(proj))
                    }
                }
            });
            py.check_signals()?;

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
                let reps = evaluation::find_representatives(&work_data, &skmeans.labels, &centroids_flat, n, work_d, k);
                let intra_sim = evaluation::intra_cluster_similarity(&work_data, &skmeans.labels, &centroids_flat, n, work_d, k);
                let res_lens = evaluation::resultant_lengths(&work_data, &skmeans.labels, n, work_d, k);
                (skmeans.labels, centroids_flat, work_d, skmeans.objective, skmeans.n_iter, reps, intra_sim, res_lens, pca_proj)
            });

            let (labels, centroids, work_d, objective, n_iter, reps, intra_sim, res_lens, pca_proj) = result;
            self.labels = Some(labels);
            self.centroids = Some(centroids);
            self.fitted_d = work_d;
            self.objective = Some(objective);
            self.n_iter = Some(n_iter);
            self.representatives = Some(reps);
            self.intra_similarity = Some(intra_sim);
            self.resultant_lengths = Some(res_lens);
            self.pca_projection = pca_proj;
            self.vmf_probabilities = None;
            self.vmf_concentrations = None;
            self.vmf_bic = None;
            Ok(())
        }

        fn refine_vmf(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let labels = self.labels.as_ref().ok_or(ClusterError::NotFitted)?;
            let centroids = self.centroids.as_ref().ok_or(ClusterError::NotFitted)?;

            // Need the working data again — extract and process
            let (data_f64, n, d) = if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                let view = arr.as_array();
                let (n, d) = view.dim();
                (view.as_slice().expect("contiguous").iter().map(|&v| v as f64).collect::<Vec<_>>(), n, d)
            } else if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                let view = arr.as_array();
                let (n, d) = view.dim();
                (view.as_slice().expect("contiguous").to_vec(), n, d)
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err("Expected float32 or float64 array"));
            };

            let k = self.n_clusters;
            let work_d = self.fitted_d;
            let labels_clone = labels.clone();
            let centroids_clone = centroids.clone();
            let pca_proj = self.pca_projection.take();

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

                let vmf_state = vmf::fit_vmf(&work_data, n, work_d, k, &centroids_clone, &labels_clone, 50, 1e-6);
                (vmf_state, pca_proj)
            });

            let (vmf_state, pca_proj) = vmf_result;
            self.pca_projection = pca_proj;
            self.vmf_probabilities = Some(vmf_state.responsibilities);
            self.vmf_concentrations = Some(vmf_state.concentrations);
            self.vmf_bic = Some(vmf_state.bic);
            Ok(())
        }

        #[getter] fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = self.labels.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(py, labels.iter().map(|&l| l as i64).collect()))
        }

        #[getter] fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let centroids = self.centroids.as_ref().ok_or(ClusterError::NotFitted)?;
            let arr = ndarray::Array2::from_shape_vec((self.n_clusters, self.fitted_d), centroids.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }

        #[getter] fn objective_(&self) -> PyResult<f64> {
            self.objective.ok_or_else(|| ClusterError::NotFitted.into())
        }

        #[getter] fn n_iter_(&self) -> PyResult<usize> {
            self.n_iter.ok_or_else(|| ClusterError::NotFitted.into())
        }

        #[getter] fn representatives_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let reps = self.representatives.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(py, reps.iter().map(|&r| r as i64).collect()))
        }

        #[getter] fn intra_similarity_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let sims = self.intra_similarity.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(py, sims.clone()))
        }

        #[getter] fn resultant_lengths_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let rl = self.resultant_lengths.as_ref().ok_or(ClusterError::NotFitted)?;
            Ok(PyArray1::from_vec(py, rl.clone()))
        }

        #[getter] fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let probs = self.vmf_probabilities.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call refine_vmf() first")
            })?;
            let n = probs.len() / self.n_clusters;
            let arr = ndarray::Array2::from_shape_vec((n, self.n_clusters), probs.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }

        #[getter] fn concentrations_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let conc = self.vmf_concentrations.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Call refine_vmf() first")
            })?;
            Ok(PyArray1::from_vec(py, conc.clone()))
        }

        #[getter] fn bic_(&self) -> PyResult<f64> {
            self.vmf_bic.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call refine_vmf() first").into())
        }

        fn __repr__(&self) -> String {
            format!(
                "EmbeddingCluster(n_clusters={}, reduction_dim={:?}, max_iter={}, n_init={})",
                self.n_clusters, self.reduction_dim, self.max_iter, self.n_init
            )
        }
    }

    // ---- Metrics ----

    #[pyfunction]
    fn silhouette_score(
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        labels: Vec<i64>,
    ) -> PyResult<f64> {
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::silhouette_score(&view, &labels))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::silhouette_score(&view, &labels))
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
        labels: Vec<i64>,
    ) -> PyResult<f64> {
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::calinski_harabasz_score(&view, &labels))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::calinski_harabasz_score(&view, &labels))
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
        labels: Vec<i64>,
    ) -> PyResult<f64> {
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::davies_bouldin_score(&view, &labels))
                .map_err(|e| pyo3::PyErr::from(e));
        }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = arr.as_array();
            return py
                .allow_threads(move || crate::metrics::davies_bouldin_score(&view, &labels))
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
        m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
        m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
        m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::_rustcluster;
