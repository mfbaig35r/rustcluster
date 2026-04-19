mod error;
mod kmeans;
mod utils;

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::error::KMeansError;
use crate::kmeans::{run_kmeans_n_init, KMeansState};
use crate::utils::validate_predict_data;

#[pyclass]
struct KMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    random_state: u64,
    n_init: usize,
    fitted: Option<KMeansState>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        random_state: u64,
        n_init: usize,
    ) -> PyResult<Self> {
        if n_clusters == 0 {
            return Err(KMeansError::InvalidClusters { k: 0, n: 0 }.into());
        }
        if max_iter == 0 {
            return Err(KMeansError::InvalidMaxIter(0).into());
        }
        if n_init == 0 {
            return Err(KMeansError::InvalidNInit(0).into());
        }
        if tol < 0.0 {
            return Err(KMeansError::InvalidTol(tol).into());
        }

        Ok(KMeans {
            n_clusters,
            max_iter,
            tol,
            random_state,
            n_init,
            fitted: None,
        })
    }

    /// Fit the K-means model to the input data.
    fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        if !x.is_c_contiguous() {
            return Err(KMeansError::NotContiguous.into());
        }

        let view = x.as_array();
        let k = self.n_clusters;
        let max_iter = self.max_iter;
        let tol = self.tol;
        let seed = self.random_state;
        let n_init = self.n_init;

        let state = py.allow_threads(move || {
            run_kmeans_n_init(&view, k, max_iter, tol, seed, n_init)
        })?;

        self.fitted = Some(state);
        Ok(())
    }

    /// Predict cluster labels for new data.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let state = self.fitted.as_ref().ok_or(KMeansError::NotFitted)?;

        if !x.is_c_contiguous() {
            return Err(KMeansError::NotContiguous.into());
        }

        let view = x.as_array();
        let (_, expected_d) = state.centroids.dim();
        validate_predict_data(&view, expected_d)?;

        let centroids = state.centroids.clone();
        let k = self.n_clusters;

        let labels = py.allow_threads(move || {
            let (n, d) = view.dim();
            let data_slice = view.as_slice().expect("data is C-contiguous");
            let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");

            let labels: Vec<i64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let point = &data_slice[i * d..(i + 1) * d];
                    let (idx, _) = crate::utils::assign_nearest(point, centroids_slice, k, d);
                    idx as i64
                })
                .collect();
            labels
        });

        Ok(PyArray1::from_vec(py, labels))
    }

    /// Fit the model and return cluster labels.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.fit(py, x)?;
        let state = self.fitted.as_ref().unwrap();
        let labels: Vec<i64> = state.labels.iter().map(|&l| l as i64).collect();
        Ok(PyArray1::from_vec(py, labels))
    }

    /// Cluster labels for the training data (available after fit).
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let state = self.fitted.as_ref().ok_or(KMeansError::NotFitted)?;
        let labels: Vec<i64> = state.labels.iter().map(|&l| l as i64).collect();
        Ok(PyArray1::from_vec(py, labels))
    }

    /// Centroid coordinates (k x d), available after fit.
    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let state = self.fitted.as_ref().ok_or(KMeansError::NotFitted)?;
        Ok(PyArray2::from_owned_array(py, state.centroids.clone()))
    }

    /// Sum of squared distances to nearest centroid (available after fit).
    #[getter]
    fn inertia_(&self) -> PyResult<f64> {
        let state = self.fitted.as_ref().ok_or(KMeansError::NotFitted)?;
        Ok(state.inertia)
    }

    /// Number of iterations in the best run (available after fit).
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let state = self.fitted.as_ref().ok_or(KMeansError::NotFitted)?;
        Ok(state.n_iter)
    }

    fn __repr__(&self) -> String {
        format!(
            "KMeans(n_clusters={}, max_iter={}, tol={}, random_state={}, n_init={})",
            self.n_clusters, self.max_iter, self.tol, self.random_state, self.n_init
        )
    }
}

#[pymodule]
fn _rustcluster(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KMeans>()?;
    Ok(())
}
