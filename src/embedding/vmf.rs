//! Von Mises-Fisher mixture model for soft embedding clustering.
//!
//! Provides posterior probabilities per point per cluster, per-cluster
//! concentration parameters (κ), and BIC for model selection.
//!
//! Initialized from spherical K-means results. Uses log-space computation
//! throughout to avoid Bessel function overflow at high dimensions.
//!
//! Reference: Banerjee, Dhillon, Ghosh, Sra (2005), "Clustering on the
//! Unit Hypersphere using von Mises-Fisher Distributions."

use rayon::prelude::*;

/// Result of vMF mixture model fitting.
pub struct VmfState {
    /// Mean directions, shape (k, d), unit-norm. Flat row-major.
    pub means: Vec<f64>,
    /// Concentration parameters, one per cluster.
    pub concentrations: Vec<f64>,
    /// Mixing weights, sum to 1.
    pub weights: Vec<f64>,
    /// Posterior responsibilities, shape (n, k). Flat row-major.
    pub responsibilities: Vec<f64>,
    /// Final log-likelihood.
    pub log_likelihood: f64,
    /// Number of EM iterations.
    pub n_iter: usize,
    /// Bayesian Information Criterion.
    pub bic: f64,
}

/// Fit a vMF mixture model initialized from spherical K-means results.
pub fn fit_vmf(
    data: &[f64],          // unit-normalized, n * d, flat row-major
    n: usize,
    d: usize,
    k: usize,
    init_centroids: &[f64], // k * d, unit-norm, from spherical K-means
    init_labels: &[usize],
    max_iter: usize,
    tol: f64,
) -> VmfState {
    // Initialize means from spherical K-means centroids
    let mut means = init_centroids.to_vec();

    // Initialize weights from label proportions
    let mut weights = vec![0.0f64; k];
    for &label in init_labels {
        weights[label] += 1.0;
    }
    for w in weights.iter_mut() {
        *w /= n as f64;
        if *w < 1e-10 {
            *w = 1e-10; // avoid zero weights
        }
    }
    // Renormalize
    let w_sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= w_sum;
    }

    // Initialize concentrations via Banerjee approximation
    let mut concentrations = compute_initial_kappas(&means, data, init_labels, n, d, k);

    // Responsibilities matrix (n x k)
    let mut resp = vec![0.0f64; n * k];

    let mut log_likelihood = f64::NEG_INFINITY;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        // ---- E-step: compute log-responsibilities ----
        let new_ll = e_step(data, &means, &concentrations, &weights, &mut resp, n, d, k);

        n_iter = iter + 1;

        // Check convergence
        if iter > 0 {
            let ll_change = (new_ll - log_likelihood).abs() / log_likelihood.abs().max(1.0);
            if ll_change < tol {
                log_likelihood = new_ll;
                break;
            }
        }
        log_likelihood = new_ll;

        // ---- M-step: update parameters ----
        m_step(data, &resp, &mut means, &mut concentrations, &mut weights, n, d, k);
    }

    // Compute BIC
    // Parameters: k means (d-1 free params each on sphere) + k kappas + (k-1) weights
    let n_params = k * (d - 1) + k + (k - 1);
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    VmfState {
        means,
        concentrations,
        weights,
        responsibilities: resp,
        log_likelihood,
        n_iter,
        bic,
    }
}

/// E-step: compute posterior responsibilities in log-space.
/// Returns the log-likelihood.
fn e_step(
    data: &[f64],
    means: &[f64],
    kappas: &[f64],
    weights: &[f64],
    resp: &mut [f64],
    n: usize,
    d: usize,
    k: usize,
) -> f64 {
    // Compute log-normalizer for each component
    let log_norms: Vec<f64> = kappas.iter().map(|&kappa| log_vmf_normalizer(kappa, d)).collect();

    // Parallel over points
    let ll_sum: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let point = &data[i * d..(i + 1) * d];
            let r_start = i * k;

            // Compute log r_ik for each cluster
            let mut log_r = vec![0.0f64; k];
            for j in 0..k {
                let mean = &means[j * d..(j + 1) * d];
                let dot: f64 = point.iter().zip(mean).map(|(&a, &b)| a * b).sum();
                log_r[j] = weights[j].ln() + log_norms[j] + kappas[j] * dot;
            }

            // Log-sum-exp for normalization and log-likelihood contribution
            let max_log = log_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = max_log + log_r.iter().map(|&lr| (lr - max_log).exp()).sum::<f64>().ln();

            // Normalize responsibilities
            // Note: we can't write to resp here in a par_iter without unsafe.
            // Instead, return the values and write sequentially below.
            // Actually, we can use par_chunks_mut on resp instead.
            (log_r, log_sum)
        })
        .collect::<Vec<_>>()
        .iter()
        .enumerate()
        .map(|(i, (log_r, log_sum))| {
            for j in 0..k {
                resp[i * k + j] = (log_r[j] - log_sum).exp();
            }
            *log_sum
        })
        .sum();

    ll_sum
}

/// M-step: update means, concentrations, and weights.
fn m_step(
    data: &[f64],
    resp: &[f64],
    means: &mut [f64],
    kappas: &mut [f64],
    weights: &mut [f64],
    n: usize,
    d: usize,
    k: usize,
) {
    for j in 0..k {
        // Effective count
        let n_j: f64 = (0..n).map(|i| resp[i * k + j]).sum();
        weights[j] = (n_j / n as f64).max(1e-10);

        // Weighted sum of points
        let mut sum = vec![0.0f64; d];
        for i in 0..n {
            let r = resp[i * k + j];
            if r < 1e-30 { continue; }
            for dim in 0..d {
                sum[dim] += r * data[i * d + dim];
            }
        }

        // Resultant length
        let norm_sq: f64 = sum.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        let r_bar = if n_j > 1e-30 { norm / n_j } else { 0.0 };

        // Update mean direction (re-normalize)
        if norm > 1e-30 {
            let inv = 1.0 / norm;
            for dim in 0..d {
                means[j * d + dim] = sum[dim] * inv;
            }
        }

        // Update concentration via Banerjee approximation + Newton refinement
        kappas[j] = estimate_kappa(r_bar, d);
    }

    // Renormalize weights
    let w_sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= w_sum;
    }
}

// ---- Numerical utilities ----

/// Banerjee's approximation for vMF concentration parameter.
/// κ ≈ (r̄·d - r̄³) / (1 - r̄²)
fn banerjee_kappa(r_bar: f64, d: usize) -> f64 {
    let d_f = d as f64;
    let r2 = r_bar * r_bar;
    let denom = (1.0 - r2).max(1e-30);
    (r_bar * d_f - r_bar * r2) / denom
}

/// Estimate κ with Banerjee approximation + 2 Newton steps (Sra 2012).
fn estimate_kappa(r_bar: f64, d: usize) -> f64 {
    if r_bar <= 1e-10 {
        return 1e-6; // near-uniform distribution
    }
    if r_bar >= 1.0 - 1e-10 {
        return 1e6; // near-point mass
    }

    let mut kappa = banerjee_kappa(r_bar, d).clamp(1e-6, 1e6);

    // Newton refinement: A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ) ≈ 1 - (d-1)/(2κ)
    // For large κ: A_d(κ) ≈ 1 - (d-1)/(2κ) - (d-1)(d-3)/(8κ²)
    // Newton: κ_new = κ - (A_d(κ) - r̄) / A_d'(κ)
    // A_d'(κ) = 1 - A_d(κ)² - (d-1)/κ * A_d(κ)  [from Bessel recurrence]
    for _ in 0..2 {
        let a_d = approx_a_d(kappa, d);
        let a_d_prime = 1.0 - a_d * a_d - ((d as f64 - 1.0) / kappa) * a_d;
        if a_d_prime.abs() > 1e-30 {
            kappa -= (a_d - r_bar) / a_d_prime;
            kappa = kappa.clamp(1e-6, 1e6);
        }
    }

    kappa
}

/// Approximate A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ).
/// Uses asymptotic expansion for large κ and series for small κ.
fn approx_a_d(kappa: f64, d: usize) -> f64 {
    let nu = d as f64 / 2.0 - 1.0;
    if kappa > 10.0 {
        // Large κ: A ≈ 1 - (d-1)/(2κ)
        1.0 - (2.0 * nu + 1.0) / (2.0 * kappa)
    } else if kappa > 0.1 {
        // Medium κ: more terms
        let v = 2.0 * nu + 1.0;
        1.0 - v / (2.0 * kappa) - v * (v - 2.0) / (8.0 * kappa * kappa)
    } else {
        // Small κ: A ≈ κ / (d + κ²/(d+2))
        kappa / (d as f64 + kappa * kappa / (d as f64 + 2.0))
    }
}

/// Log of vMF normalizing constant: log C_d(κ)
/// C_d(κ) = κ^{d/2-1} / ((2π)^{d/2} · I_{d/2-1}(κ))
/// log C_d = (d/2-1)·log(κ) - (d/2)·log(2π) - log I_{d/2-1}(κ)
///
/// Uses asymptotic expansion for log-Bessel to avoid overflow.
fn log_vmf_normalizer(kappa: f64, d: usize) -> f64 {
    let nu = d as f64 / 2.0 - 1.0;
    let half_d = d as f64 / 2.0;

    let log_kappa_term = nu * kappa.max(1e-30).ln();
    let log_2pi_term = half_d * (2.0 * std::f64::consts::PI).ln();
    let log_bessel = log_bessel_i(nu, kappa);

    log_kappa_term - log_2pi_term - log_bessel
}

/// Log of modified Bessel function of the first kind I_ν(x).
/// Uses asymptotic expansion for large x to avoid overflow.
fn log_bessel_i(nu: f64, x: f64) -> f64 {
    if x < 1e-30 {
        // I_ν(0) = 0 for ν > 0, so log = -inf
        return f64::NEG_INFINITY;
    }

    if x > nu.max(10.0) {
        // Asymptotic expansion: log I_ν(x) ≈ x - 0.5·ln(2πx) - (4ν²-1)/(8x)
        x - 0.5 * (2.0 * std::f64::consts::PI * x).ln()
            - (4.0 * nu * nu - 1.0) / (8.0 * x)
    } else {
        // Series expansion: I_ν(x) = (x/2)^ν · Σ_{k=0}^{∞} (x²/4)^k / (k! · Γ(ν+k+1))
        // Computed in log-space
        let log_half_x = (x / 2.0).ln();
        let x2_over_4 = x * x / 4.0;

        let mut log_sum = 0.0f64; // log(1) for k=0 term relative to (x/2)^ν / Γ(ν+1)
        let mut term = 0.0f64; // log of k-th term relative to k=0
        let mut max_log_term = 0.0f64;
        let mut terms = vec![0.0f64]; // relative log-terms

        for k in 1..100 {
            term += x2_over_4.ln() - (k as f64).ln() - (nu + k as f64).ln();
            terms.push(term);
            if term > max_log_term { max_log_term = term; }
            if term < max_log_term - 50.0 { break; } // converged
        }

        // log-sum-exp of terms
        let lse = max_log_term + terms.iter().map(|&t| (t - max_log_term).exp()).sum::<f64>().ln();

        // Full result: ν·ln(x/2) - ln(Γ(ν+1)) + lse
        nu * log_half_x - lgamma(nu + 1.0) + lse
    }
}

/// Log-gamma function via Stirling's approximation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    // Use Rust's built-in ln_gamma if available, otherwise Stirling
    // Stirling: ln Γ(x) ≈ (x - 0.5)·ln(x) - x + 0.5·ln(2π)
    if x > 10.0 {
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
            + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x)
    } else {
        // For small x, use recurrence: Γ(x+1) = x·Γ(x)
        let mut val = x;
        let mut result = 0.0f64;
        while val < 10.0 {
            result -= val.ln();
            val += 1.0;
        }
        result + (val - 0.5) * val.ln() - val + 0.5 * (2.0 * std::f64::consts::PI).ln()
            + 1.0 / (12.0 * val)
    }
}

/// Compute initial κ values from spherical K-means results.
fn compute_initial_kappas(
    centroids: &[f64],
    data: &[f64],
    labels: &[usize],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f64> {
    let mut r_bars = vec![0.0f64; k];
    let mut counts = vec![0usize; k];

    // Compute mean resultant length per cluster
    let mut sums = vec![0.0f64; k * d];
    for i in 0..n {
        let label = labels[i];
        counts[label] += 1;
        for j in 0..d {
            sums[label * d + j] += data[i * d + j];
        }
    }

    for j in 0..k {
        if counts[j] > 0 {
            let norm_sq: f64 = (0..d).map(|dim| sums[j * d + dim].powi(2)).sum();
            r_bars[j] = norm_sq.sqrt() / counts[j] as f64;
        }
    }

    r_bars.iter().map(|&r| estimate_kappa(r, d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::normalize::l2_normalize_rows;
    use crate::embedding::spherical_kmeans::run_spherical_kmeans;

    #[test]
    fn test_banerjee_kappa_basic() {
        // r_bar close to 1 → high kappa
        let kappa = banerjee_kappa(0.99, 10);
        assert!(kappa > 50.0, "kappa = {kappa}");

        // r_bar close to 0 → low kappa
        let kappa = banerjee_kappa(0.1, 10);
        assert!(kappa < 5.0, "kappa = {kappa}");
    }

    #[test]
    fn test_estimate_kappa_clamps() {
        assert!(estimate_kappa(0.0, 100) > 0.0);
        assert!(estimate_kappa(1.0, 100) < 1e7);
    }

    #[test]
    fn test_log_vmf_normalizer_no_overflow() {
        // d=128 should not overflow
        let result = log_vmf_normalizer(100.0, 128);
        assert!(result.is_finite(), "log_norm = {result}");

        // d=256
        let result = log_vmf_normalizer(500.0, 256);
        assert!(result.is_finite(), "log_norm = {result}");
    }

    #[test]
    fn test_log_bessel_no_overflow() {
        let result = log_bessel_i(62.0, 100.0); // d=128 → ν=62
        assert!(result.is_finite(), "log_bessel = {result}");
    }

    #[test]
    fn test_vmf_em_on_synthetic() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let d = 8;
        let mut rng = StdRng::seed_from_u64(42);

        // Generate two clusters on unit sphere
        let mut data = Vec::new();
        for _ in 0..50 {
            let mut point = vec![0.0f64; d];
            point[0] = 5.0;
            for j in 1..d { point[j] = rng.gen_range(-0.5..0.5); }
            data.extend_from_slice(&point);
        }
        for _ in 0..50 {
            let mut point = vec![0.0f64; d];
            point[0] = -5.0;
            for j in 1..d { point[j] = rng.gen_range(-0.5..0.5); }
            data.extend_from_slice(&point);
        }

        let (norm_data, _) = l2_normalize_rows(&data, 100, d);

        // First run spherical K-means
        let skmeans = run_spherical_kmeans(&norm_data, 100, d, 2, 50, 1e-6, 42, 1).unwrap();
        let centroids_flat: Vec<f64> = skmeans.centroids.as_slice().unwrap().to_vec();

        // Now refine with vMF
        let vmf = fit_vmf(&norm_data, 100, d, 2, &centroids_flat, &skmeans.labels, 20, 1e-6);

        assert_eq!(vmf.responsibilities.len(), 100 * 2);
        assert_eq!(vmf.concentrations.len(), 2);
        assert_eq!(vmf.weights.len(), 2);
        assert!(vmf.log_likelihood.is_finite());
        assert!(vmf.bic.is_finite());

        // Responsibilities should sum to ~1 per point
        for i in 0..100 {
            let sum: f64 = (0..2).map(|j| vmf.responsibilities[i * 2 + j]).sum();
            assert!((sum - 1.0).abs() < 1e-6, "Point {i} resp sum = {sum}");
        }

        // High concentration (tight clusters)
        for &kappa in &vmf.concentrations {
            assert!(kappa > 1.0, "kappa = {kappa}");
        }
    }

    #[test]
    fn test_lgamma_basic() {
        // Γ(1) = 1, ln(Γ(1)) = 0
        assert!(lgamma(1.0).abs() < 0.1);
        // Γ(2) = 1, ln(Γ(2)) = 0
        assert!(lgamma(2.0).abs() < 0.1);
        // ln(Γ(10)) = ln(9!) = ln(362880) ≈ 12.8
        assert!((lgamma(10.0) - 12.8).abs() < 0.5);
    }
}
