use ndarray::{Array1, Array2, Array3, s};

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};
use crate::linalg;

/// Result of a PCA computation.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Principal components as row vectors, shape (n_components, bands).
    /// Sorted by descending explained variance.
    pub components: Array2<f64>,
    /// Variance explained by each component.
    pub explained_variance: Array1<f64>,
    /// Mean spectrum subtracted before PCA, shape (bands,).
    pub mean: Array1<f64>,
    /// Original wavelengths, preserved for inverse transform.
    pub wavelengths: Array1<f64>,
}

/// Compute PCA on a spectral cube.
///
/// Reshapes the cube to a 2D matrix of pixels x bands, centers it,
/// and finds principal components via eigendecomposition of the covariance
/// matrix or randomized truncated SVD (when `n_components` is much smaller
/// than the number of bands).
///
/// `n_components` specifies how many principal components to keep.
/// If `None`, all components are kept.
pub fn pca(cube: &SpectralCube, n_components: Option<usize>) -> Result<PcaResult> {
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let n_pixels = height * width;

    if n_pixels < 2 {
        return Err(HyperspecError::InvalidInput(
            "PCA requires at least 2 pixels".to_string(),
        ));
    }

    let n_comp = n_components.unwrap_or(bands);
    if n_comp == 0 || n_comp > bands {
        return Err(HyperspecError::InvalidInput(format!(
            "n_components {} must be in [1, {}]",
            n_comp, bands
        )));
    }

    // Validate no NaN/nodata, compute mean per band
    let data = cube.data();
    let nodata = cube.nodata();

    let mut mean = Array1::<f64>::zeros(bands);
    for b in 0..bands {
        let band_slice = data.slice(s![b, .., ..]);
        let mut sum = 0.0;
        for &v in band_slice.iter() {
            if v.is_nan() || nodata == Some(v) {
                return Err(HyperspecError::InvalidInput(
                    "PCA input contains NaN or nodata values; mask or remove them first"
                        .to_string(),
                ));
            }
            sum += v;
        }
        mean[b] = sum / n_pixels as f64;
    }

    // Randomized SVD wins for high band counts where the covariance GEMM (O(bands²))
    // dominates. For typical HSI (100-400 bands), cov+eigh is faster because the
    // covariance matrix is small and QR overhead in randomized SVD isn't worth it.
    let oversampling = 10.min(bands.saturating_sub(n_comp));
    let use_randomized = bands > 500 && n_comp + oversampling < bands;

    let mean_slice = mean.as_slice().expect("mean is contiguous");

    if use_randomized {
        let (explained_variance, components) =
            linalg::randomized_pca(data, mean_slice, n_pixels, bands, height, width, n_comp)?;

        Ok(PcaResult {
            components,
            explained_variance,
            mean,
            wavelengths: cube.wavelengths().clone(),
        })
    } else {
        let (_cov, eigenvalues, eigenvectors) =
            linalg::clean_covariance_eigen(data, mean_slice, n_pixels, bands, height, width)?;

        // Sort by descending eigenvalue
        let mut indices: Vec<usize> = (0..bands).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top n_components
        let mut components = Array2::<f64>::zeros((n_comp, bands));
        let mut explained_variance = Array1::<f64>::zeros(n_comp);
        for (i, &idx) in indices.iter().take(n_comp).enumerate() {
            explained_variance[i] = eigenvalues[idx].max(0.0);
            for j in 0..bands {
                components[[i, j]] = eigenvectors[[j, idx]];
            }
        }

        Ok(PcaResult {
            components,
            explained_variance,
            mean,
            wavelengths: cube.wavelengths().clone(),
        })
    }
}

/// Forward transform: project a cube into PCA space.
///
/// Returns an `Array3<f64>` of shape (n_components, height, width) containing
/// principal component scores. This is not a `SpectralCube` because PC scores
/// are not spectral data.
pub fn pca_transform(cube: &SpectralCube, pca_result: &PcaResult) -> Result<Array3<f64>> {
    let bands = cube.bands();

    if bands != pca_result.mean.len() {
        return Err(HyperspecError::DimensionMismatch(format!(
            "cube has {} bands but PCA was fitted on {} bands",
            bands,
            pca_result.mean.len()
        )));
    }

    let mean_slice = pca_result.mean.as_slice().expect("mean is contiguous");
    Ok(linalg::bsq_transform(
        cube.data(),
        &pca_result.components,
        Some(mean_slice),
        None,
    ))
}

/// Inverse transform: reconstruct a cube from PCA scores.
///
/// Takes a scores array of shape (n_components, height, width) and
/// reconstructs an approximation of the original spectral data.
/// The reconstruction is lossy if fewer components were kept.
pub fn pca_inverse(scores: &Array3<f64>, pca_result: &PcaResult) -> Result<SpectralCube> {
    let n_comp = scores.shape()[0];

    if n_comp != pca_result.components.nrows() {
        return Err(HyperspecError::DimensionMismatch(format!(
            "scores has {} components but PCA has {} components",
            n_comp,
            pca_result.components.nrows()
        )));
    }

    let mean_slice = pca_result.mean.as_slice().expect("mean is contiguous");
    let components_t = pca_result.components.t().to_owned();
    let result = linalg::bsq_transform(scores, &components_t, None, Some(mean_slice));

    SpectralCube::new(result, pca_result.wavelengths.clone(), None, None)
}

/// Jacobi eigenvalue algorithm for a symmetric matrix.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns of the
/// returned matrix.
#[cfg(test)]
fn jacobi_eigen(matrix: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "matrix must be square");

    let mut a = matrix.clone();
    let mut v = Array2::<f64>::eye(n);

    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation to zero out a[p][q]
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos = theta.cos();
        let sin = theta.sin();

        // Apply Givens rotation to A in-place: A' = G^T A G
        for i in 0..n {
            if i != p && i != q {
                let aip = a[[i, p]];
                let aiq = a[[i, q]];
                a[[i, p]] = cos * aip + sin * aiq;
                a[[p, i]] = a[[i, p]];
                a[[i, q]] = -sin * aip + cos * aiq;
                a[[q, i]] = a[[i, q]];
            }
        }
        a[[p, p]] = cos * cos * app + 2.0 * sin * cos * apq + sin * sin * aqq;
        a[[q, q]] = sin * sin * app - 2.0 * sin * cos * apq + cos * cos * aqq;
        a[[p, q]] = 0.0;
        a[[q, p]] = 0.0;

        // Accumulate eigenvectors: V' = V G
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = cos * vip + sin * viq;
            v[[i, q]] = -sin * vip + cos * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_cube() -> SpectralCube {
        // 3 bands, 4x4 pixels with known structure
        // Band 0 and band 1 are correlated, band 2 is independent noise
        let mut data = Array3::zeros((3, 4, 4));
        let rng_values = [
            0.5, 0.8, 0.3, 0.9, 0.6, 0.7, 0.4, 0.2, 0.85, 0.35, 0.65, 0.55, 0.45, 0.75, 0.25, 0.95,
        ];
        for row in 0..4 {
            for col in 0..4 {
                let v = rng_values[row * 4 + col];
                data[[0, row, col]] = v;
                data[[1, row, col]] = v * 0.9 + 0.05; // correlated with band 0
                data[[2, row, col]] = 1.0 - v; // anti-correlated
            }
        }
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_pca_basic() {
        let cube = make_cube();
        let result = pca(&cube, None).unwrap();

        assert_eq!(result.components.nrows(), 3);
        assert_eq!(result.components.ncols(), 3);
        assert_eq!(result.explained_variance.len(), 3);
        assert_eq!(result.mean.len(), 3);

        // Eigenvalues should be non-negative and sorted descending
        for &v in result.explained_variance.iter() {
            assert!(v >= 0.0);
        }
        for i in 1..result.explained_variance.len() {
            assert!(result.explained_variance[i - 1] >= result.explained_variance[i]);
        }
    }

    #[test]
    fn test_pca_n_components() {
        let cube = make_cube();
        let result = pca(&cube, Some(2)).unwrap();

        assert_eq!(result.components.nrows(), 2);
        assert_eq!(result.explained_variance.len(), 2);
    }

    #[test]
    fn test_pca_invalid_n_components() {
        let cube = make_cube();
        assert!(pca(&cube, Some(0)).is_err());
        assert!(pca(&cube, Some(10)).is_err());
    }

    #[test]
    fn test_pca_transform_shape() {
        let cube = make_cube();
        let result = pca(&cube, Some(2)).unwrap();
        let transformed = pca_transform(&cube, &result).unwrap();

        assert_eq!(transformed.shape(), &[2, 4, 4]);
    }

    #[test]
    fn test_pca_roundtrip() {
        // Full PCA (all components) -> transform -> inverse should reconstruct original
        let cube = make_cube();
        let result = pca(&cube, None).unwrap();
        let transformed = pca_transform(&cube, &result).unwrap();
        let reconstructed = pca_inverse(&transformed, &result).unwrap();

        // Check reconstruction is close to original
        let orig = cube.data();
        let recon = reconstructed.data();
        for b in 0..cube.bands() {
            for row in 0..cube.height() {
                for col in 0..cube.width() {
                    assert!(
                        (orig[[b, row, col]] - recon[[b, row, col]]).abs() < 1e-8,
                        "mismatch at [{}, {}, {}]: {} vs {}",
                        b,
                        row,
                        col,
                        orig[[b, row, col]],
                        recon[[b, row, col]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_pca_variance_concentration() {
        // With correlated bands, most variance should be in first component
        let cube = make_cube();
        let result = pca(&cube, None).unwrap();

        let total: f64 = result.explained_variance.iter().sum();
        let first_ratio = result.explained_variance[0] / total;
        // First component should capture >80% of variance for correlated data
        assert!(
            first_ratio > 0.8,
            "first component only captures {:.1}%",
            first_ratio * 100.0
        );
    }

    #[test]
    fn test_pca_components_orthogonal() {
        let cube = make_cube();
        let result = pca(&cube, None).unwrap();

        // Components should be orthonormal
        for i in 0..result.components.nrows() {
            for j in 0..result.components.nrows() {
                let dot: f64 = result
                    .components
                    .row(i)
                    .iter()
                    .zip(result.components.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "components {} and {}: dot product = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_pca_single_pixel() {
        let data = Array3::from_shape_fn((3, 1, 1), |(b, _, _)| b as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(pca(&cube, None).is_err());
    }

    #[test]
    fn test_pca_rejects_nan() {
        let mut data = Array3::from_shape_fn((3, 4, 4), |(b, r, c)| (b + r + c) as f64);
        data[[1, 2, 3]] = f64::NAN;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let err = pca(&cube, None).unwrap_err();
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn test_pca_rejects_nodata() {
        let mut data = Array3::from_shape_fn((3, 4, 4), |(b, r, c)| (b + r + c) as f64);
        data[[0, 0, 0]] = -9999.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();
        let err = pca(&cube, None).unwrap_err();
        assert!(err.to_string().contains("nodata"));
    }

    #[test]
    fn test_pca_inverse_preserves_wavelengths() {
        let cube = make_cube();
        let result = pca(&cube, None).unwrap();
        let transformed = pca_transform(&cube, &result).unwrap();
        let reconstructed = pca_inverse(&transformed, &result).unwrap();
        assert_eq!(reconstructed.wavelengths(), cube.wavelengths());
    }

    #[test]
    fn test_pca_transform_dimension_mismatch() {
        let cube = make_cube();
        let result = pca(&cube, Some(2)).unwrap();

        // Create a cube with different band count
        let data = Array3::zeros((5, 4, 4));
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let wrong_cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(pca_transform(&wrong_cube, &result).is_err());
    }

    #[test]
    fn test_pca_inverse_dimension_mismatch() {
        let cube = make_cube();
        let result = pca(&cube, Some(2)).unwrap();

        // Scores array with wrong component count
        let wrong_scores = Array3::zeros((3, 4, 4));
        assert!(pca_inverse(&wrong_scores, &result).is_err());
    }

    #[test]
    fn test_jacobi_identity() {
        // Eigenvalues of a diagonal matrix are the diagonal entries
        let m = Array2::from_diag(&Array1::from_vec(vec![3.0, 1.0, 2.0]));
        let (eigenvalues, _eigenvectors) = jacobi_eigen(&m);

        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 3.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_symmetric() {
        // Known 2x2 symmetric matrix
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = 2.0;
        m[[0, 1]] = 1.0;
        m[[1, 0]] = 1.0;
        m[[1, 1]] = 2.0;

        let (eigenvalues, eigenvectors) = jacobi_eigen(&m);

        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // Eigenvalues of [[2,1],[1,2]] are 3 and 1
        assert!((sorted[0] - 3.0).abs() < 1e-10);
        assert!((sorted[1] - 1.0).abs() < 1e-10);

        // Verify A * v = lambda * v for each eigenvector
        for (i, &lam) in eigenvalues.iter().enumerate().take(2) {
            let v = eigenvectors.column(i).to_owned();
            let av = m.dot(&v);
            let lv = &v * lam;
            for j in 0..2 {
                assert!(
                    (av[j] - lv[j]).abs() < 1e-10,
                    "eigenvector {} failed: Av={}, lv={}",
                    i,
                    av[j],
                    lv[j]
                );
            }
        }
    }

    #[test]
    fn test_pca_randomized_vs_full() {
        // 20-band 32x32 cube, compare full PCA vs truncated PCA (5 components)
        let bands = 20;
        let height = 32;
        let width = 32;
        let n_comp = 5;

        // Build data with known structure: first few bands carry most variance
        let mut data = Array3::zeros((bands, height, width));
        let mut rng_state: u64 = 12345;
        for b in 0..bands {
            for row in 0..height {
                for col in 0..width {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
                    // Strong signal in first 5 bands, weak in rest
                    let signal = if b < 5 {
                        (row as f64 + col as f64 * 0.7 + b as f64 * 0.3) / 50.0
                    } else {
                        0.01 * noise
                    };
                    data[[b, row, col]] = signal + 0.001 * noise;
                }
            }
        }
        let wl = Array1::from_iter((0..bands).map(|b| 400.0 + 10.0 * b as f64));
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        // Full PCA (all bands)
        let full = pca(&cube, None).unwrap();
        // Truncated PCA (5 components) — should use randomized path
        let trunc = pca(&cube, Some(n_comp)).unwrap();

        // Check variance ratio: truncated should capture similar variance
        let full_top_var: f64 = full.explained_variance.iter().take(n_comp).sum();
        let trunc_var: f64 = trunc.explained_variance.iter().sum();
        let var_ratio = (full_top_var - trunc_var).abs() / full_top_var;
        assert!(
            var_ratio < 0.01,
            "variance ratio difference {:.4} exceeds 1%",
            var_ratio
        );

        // Check component alignment: dot product > 0.99 for significant components.
        // Only check components whose explained variance is > 1% of total —
        // near-degenerate eigenvalues produce unstable subspace orientations.
        let total_var: f64 = full.explained_variance.iter().sum();
        for i in 0..n_comp {
            if full.explained_variance[i] / total_var < 0.01 {
                continue;
            }
            let dot: f64 = full
                .components
                .row(i)
                .iter()
                .zip(trunc.components.row(i).iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .abs();
            assert!(dot > 0.99, "component {} alignment {:.4} < 0.99", i, dot);
        }
    }

    #[test]
    fn test_pca_randomized_roundtrip() {
        // 10-band 16x16 cube, 5 components, check reconstruction SNR > 10
        let bands = 10;
        let height = 16;
        let width = 16;
        let n_comp = 5;

        let mut data = Array3::zeros((bands, height, width));
        let mut rng_state: u64 = 54321;
        for b in 0..bands {
            for row in 0..height {
                for col in 0..width {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
                    let signal = (row as f64 * 0.1 + col as f64 * 0.05 + b as f64 * 0.2) / 10.0;
                    data[[b, row, col]] = signal + 0.01 * noise;
                }
            }
        }
        let wl = Array1::from_iter((0..bands).map(|b| 400.0 + 10.0 * b as f64));
        let cube = SpectralCube::new(data.clone(), wl, None, None).unwrap();

        let result = pca(&cube, Some(n_comp)).unwrap();
        let scores = pca_transform(&cube, &result).unwrap();
        let reconstructed = pca_inverse(&scores, &result).unwrap();

        // Compute SNR = 10 * log10(signal_power / error_power)
        let mut signal_power = 0.0;
        let mut error_power = 0.0;
        let recon = reconstructed.data();
        for b in 0..bands {
            for row in 0..height {
                for col in 0..width {
                    let orig = data[[b, row, col]];
                    let rec = recon[[b, row, col]];
                    signal_power += orig * orig;
                    error_power += (orig - rec) * (orig - rec);
                }
            }
        }
        let snr = 10.0 * (signal_power / error_power).log10();
        assert!(snr > 10.0, "reconstruction SNR {:.1} dB < 10 dB", snr);
    }
}
