pub mod agglomerative;
pub mod dbscan;
pub mod distance;
mod error;
mod hamerly;
pub mod hdbscan;
pub mod kmeans;
pub mod metrics;
pub mod minibatch_kmeans;
pub mod utils;

/// Re-exports for Criterion benchmarks.
#[doc(hidden)]
pub mod _bench_api {
    pub use crate::distance::{CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean};
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
                    let arr = x
                        .extract::<PyReadonlyArray2<'_, f64>>()
                        .map_err(|_| {
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
                    let arr = x
                        .extract::<PyReadonlyArray2<'_, f32>>()
                        .map_err(|_| {
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
                DbscanFitted::F64(s) => {
                    s.core_sample_indices.iter().map(|&i| i as i64).collect::<Vec<_>>()
                }
                DbscanFitted::F32(s) => {
                    s.core_sample_indices.iter().map(|&i| i as i64).collect::<Vec<_>>()
                }
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
    }

    // ---- HDBSCAN ----

    use crate::hdbscan::{
        run_hdbscan_with_metric, run_hdbscan_with_metric_f32,
        ClusterSelectionMethod, HdbscanState,
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
                if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                let view = arr.as_array();
                let state = py.allow_threads(move || run_hdbscan_with_metric(&view, mcs, ms, metric, sel))?;
                self.fitted = Some(HdbscanFitted::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                let view = arr.as_array();
                let state = py.allow_threads(move || run_hdbscan_with_metric_f32(&view, mcs, ms, metric, sel))?;
                self.fitted = Some(HdbscanFitted::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err("Expected float32 or float64 array"))
        }

        fn fit_predict<'py>(&mut self, py: Python<'py>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                HdbscanFitted::F64(s) => s.labels.clone(),
                HdbscanFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter] fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                HdbscanFitted::F64(s) => s.labels.clone(),
                HdbscanFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter] fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let probs = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                HdbscanFitted::F64(s) => s.probabilities.clone(),
                HdbscanFitted::F32(s) => s.probabilities.clone(),
            };
            Ok(PyArray1::from_vec(py, probs))
        }

        #[getter] fn cluster_persistence_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let pers = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                HdbscanFitted::F64(s) => s.cluster_persistence.clone(),
                HdbscanFitted::F32(s) => s.cluster_persistence.clone(),
            };
            Ok(PyArray1::from_vec(py, pers))
        }

        fn __repr__(&self) -> String {
            let met_str = match self.metric { Metric::Euclidean => "euclidean", Metric::Cosine => "cosine", Metric::Manhattan => "manhattan" };
            let sel_str = match self.cluster_selection_method { ClusterSelectionMethod::Eom => "eom", ClusterSelectionMethod::Leaf => "leaf" };
            format!(
                "HDBSCAN(min_cluster_size={}, min_samples={}, metric=\"{}\", cluster_selection_method=\"{}\")",
                self.min_cluster_size, self.min_samples, met_str, sel_str
            )
        }
    }

    // ---- Agglomerative ----

    use crate::agglomerative::{
        run_agglomerative_with_metric, run_agglomerative_with_metric_f32,
        AgglomerativeState, Linkage,
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
            Ok(AgglomerativeClustering { n_clusters, linkage: link, metric: met, fitted: None })
        }

        fn fit(&mut self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
            let nc = self.n_clusters;
            let link = self.linkage;
            let metric = self.metric;

            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() {
                if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                let view = arr.as_array();
                let state = py.allow_threads(move || run_agglomerative_with_metric(&view, nc, link, metric))?;
                self.fitted = Some(AgglomerativeFitted::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                let view = arr.as_array();
                let state = py.allow_threads(move || run_agglomerative_with_metric_f32(&view, nc, link, metric))?;
                self.fitted = Some(AgglomerativeFitted::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err("Expected float32 or float64 array"))
        }

        fn fit_predict<'py>(&mut self, py: Python<'py>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                AgglomerativeFitted::F64(s) => s.labels.clone(),
                AgglomerativeFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter] fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => s.labels.clone(),
                AgglomerativeFitted::F32(s) => s.labels.clone(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter] fn n_clusters_(&self) -> PyResult<usize> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => Ok(s.n_clusters),
                AgglomerativeFitted::F32(s) => Ok(s.n_clusters),
            }
        }

        #[getter] fn children_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
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

        #[getter] fn distances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let dists = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                AgglomerativeFitted::F64(s) => s.distances.clone(),
                AgglomerativeFitted::F32(s) => s.distances.clone(),
            };
            Ok(PyArray1::from_vec(py, dists))
        }

        fn __repr__(&self) -> String {
            let link_str = match self.linkage { Linkage::Ward => "ward", Linkage::Complete => "complete", Linkage::Average => "average", Linkage::Single => "single" };
            let met_str = match self.metric { Metric::Euclidean => "euclidean", Metric::Cosine => "cosine", Metric::Manhattan => "manhattan" };
            format!("AgglomerativeClustering(n_clusters={}, linkage=\"{}\", metric=\"{}\")", self.n_clusters, link_str, met_str)
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
                if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_minibatch_kmeans_with_metric(&view, k, bs, mi, tol, seed, mni, metric)
                })?;
                self.fitted = Some(MiniBatchFittedState::F64(state));
                return Ok(());
            }
            if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() {
                if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                let view = arr.as_array();
                let state = py.allow_threads(move || {
                    run_minibatch_kmeans_with_metric_f32(&view, k, bs, mi, tol, seed, mni, metric)
                })?;
                self.fitted = Some(MiniBatchFittedState::F32(state));
                return Ok(());
            }
            Err(pyo3::exceptions::PyValueError::new_err("Expected float32 or float64 array"))
        }

        fn predict<'py>(&self, py: Python<'py>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let metric = self.metric;
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f64>>()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Expected float64 array"))?;
                    if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data(&view, expected_d)?;
                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;
                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let ds = view.as_slice().expect("C-contiguous");
                        let cs = centroids.as_slice().expect("C-contiguous");
                        (0..n).into_par_iter().map(|i| {
                            let point = &ds[i*d..(i+1)*d];
                            let (idx, _) = match metric {
                                Metric::Euclidean => assign_nearest_with::<f64, crate::distance::SquaredEuclidean>(point, cs, k, d),
                                Metric::Cosine => assign_nearest_with::<f64, crate::distance::CosineDistance>(point, cs, k, d),
                                Metric::Manhattan => assign_nearest_with::<f64, crate::distance::ManhattanDistance>(point, cs, k, d),
                            };
                            idx as i64
                        }).collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
                MiniBatchFittedState::F32(state) => {
                    let arr = x.extract::<PyReadonlyArray2<'_, f32>>()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Expected float32 array"))?;
                    if !arr.is_c_contiguous() { return Err(ClusterError::NotContiguous.into()); }
                    let view = arr.as_array();
                    let (_, expected_d) = state.centroids.dim();
                    validate_predict_data_generic(&view, expected_d)?;
                    let centroids = state.centroids.clone();
                    let k = self.n_clusters;
                    let labels = py.allow_threads(move || {
                        let (n, d) = view.dim();
                        let ds = view.as_slice().expect("C-contiguous");
                        let cs = centroids.as_slice().expect("C-contiguous");
                        (0..n).into_par_iter().map(|i| {
                            let point = &ds[i*d..(i+1)*d];
                            let (idx, _) = match metric {
                                Metric::Euclidean => assign_nearest_with::<f32, crate::distance::SquaredEuclidean>(point, cs, k, d),
                                Metric::Cosine => assign_nearest_with::<f32, crate::distance::CosineDistance>(point, cs, k, d),
                                Metric::Manhattan => assign_nearest_with::<f32, crate::distance::ManhattanDistance>(point, cs, k, d),
                            };
                            idx as i64
                        }).collect::<Vec<i64>>()
                    });
                    Ok(PyArray1::from_vec(py, labels))
                }
            }
        }

        fn fit_predict<'py>(&mut self, py: Python<'py>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            self.fit(py, x)?;
            let labels = match self.fitted.as_ref().unwrap() {
                MiniBatchFittedState::F64(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                MiniBatchFittedState::F32(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter] fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let labels = match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                MiniBatchFittedState::F32(s) => s.labels.iter().map(|&l| l as i64).collect::<Vec<_>>(),
            };
            Ok(PyArray1::from_vec(py, labels))
        }

        #[getter] fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => Ok(PyArray2::from_owned_array(py, s.centroids.clone()).into_any().unbind()),
                MiniBatchFittedState::F32(s) => Ok(PyArray2::from_owned_array(py, s.centroids.clone()).into_any().unbind()),
            }
        }

        #[getter] fn inertia_(&self) -> PyResult<f64> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => Ok(s.inertia),
                MiniBatchFittedState::F32(s) => Ok(s.inertia),
            }
        }

        #[getter] fn n_iter_(&self) -> PyResult<usize> {
            match self.fitted.as_ref().ok_or(ClusterError::NotFitted)? {
                MiniBatchFittedState::F64(s) => Ok(s.n_iter),
                MiniBatchFittedState::F32(s) => Ok(s.n_iter),
            }
        }

        fn __repr__(&self) -> String {
            let met_str = match self.metric { Metric::Euclidean => "euclidean", Metric::Cosine => "cosine", Metric::Manhattan => "manhattan" };
            format!(
                "MiniBatchKMeans(n_clusters={}, batch_size={}, max_iter={}, tol={}, random_state={}, max_no_improvement={}, metric=\"{}\")",
                self.n_clusters, self.batch_size, self.max_iter, self.tol, self.random_state, self.max_no_improvement, met_str
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
        m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
        m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
        m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::_rustcluster;
