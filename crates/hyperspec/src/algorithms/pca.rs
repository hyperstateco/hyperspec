use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

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
/// Reshapes the cube to a 2D matrix of pixels × bands, centers it,
/// computes the covariance matrix, and finds eigenvectors via Jacobi
/// eigenvalue decomposition.
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

    // Reshape to (n_pixels, bands), rejecting NaN and nodata
    let data = cube.data();
    let nodata = cube.nodata();
    let mut pixels = Array2::<f64>::zeros((n_pixels, bands));
    for row in 0..height {
        for col in 0..width {
            let px = row * width + col;
            for b in 0..bands {
                let v = data[[b, row, col]];
                if v.is_nan() || nodata == Some(v) {
                    return Err(HyperspecError::InvalidInput(
                        "PCA input contains NaN or nodata values; mask or remove them first"
                            .to_string(),
                    ));
                }
                pixels[[px, b]] = v;
            }
        }
    }

    // Compute mean per band
    let mean = pixels.mean_axis(Axis(0)).expect("pixels is non-empty");

    // Center the data
    for mut row in pixels.rows_mut() {
        row -= &mean;
    }

    // Covariance matrix: (bands, bands) = X^T X / (n - 1)
    let n_f = (n_pixels - 1).max(1) as f64;
    let cov = pixels.t().dot(&pixels) / n_f;

    // Eigendecomposition via Jacobi
    let (eigenvalues, eigenvectors) = jacobi_eigen(&cov);

    // Sort by descending eigenvalue (NaN sorts last)
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

/// Forward transform: project a cube into PCA space.
///
/// Returns an `Array3<f64>` of shape (n_components, height, width) containing
/// principal component scores. This is not a `SpectralCube` because PC scores
/// are not spectral data.
pub fn pca_transform(cube: &SpectralCube, pca_result: &PcaResult) -> Result<Array3<f64>> {
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let n_comp = pca_result.components.nrows();

    if bands != pca_result.mean.len() {
        return Err(HyperspecError::DimensionMismatch(format!(
            "cube has {} bands but PCA was fitted on {} bands",
            bands,
            pca_result.mean.len()
        )));
    }

    let data = cube.data();
    let mean = &pca_result.mean;
    let components = &pca_result.components;

    // Parallel per-row
    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; n_comp * width];
            for col in 0..width {
                // Center the pixel
                let mut centered = vec![0.0; bands];
                for b in 0..bands {
                    centered[b] = data[[b, row, col]] - mean[b];
                }
                // Project onto each component
                for c in 0..n_comp {
                    let mut score = 0.0;
                    for b in 0..bands {
                        score += components[[c, b]] * centered[b];
                    }
                    row_data[c * width + col] = score;
                }
            }
            row_data
        })
        .collect();

    let mut result = Array3::<f64>::zeros((n_comp, height, width));
    for (row, row_data) in rows.iter().enumerate() {
        for c in 0..n_comp {
            for col in 0..width {
                result[[c, row, col]] = row_data[c * width + col];
            }
        }
    }

    Ok(result)
}

/// Inverse transform: reconstruct a cube from PCA scores.
///
/// Takes a scores array of shape (n_components, height, width) and
/// reconstructs an approximation of the original spectral data.
/// The reconstruction is lossy if fewer components were kept.
pub fn pca_inverse(scores: &Array3<f64>, pca_result: &PcaResult) -> Result<SpectralCube> {
    let n_comp = scores.shape()[0];
    let height = scores.shape()[1];
    let width = scores.shape()[2];
    let bands = pca_result.mean.len();

    if n_comp != pca_result.components.nrows() {
        return Err(HyperspecError::DimensionMismatch(format!(
            "scores has {} components but PCA has {} components",
            n_comp,
            pca_result.components.nrows()
        )));
    }

    let mean = &pca_result.mean;
    let components = &pca_result.components;

    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; bands * width];
            for col in 0..width {
                for b in 0..bands {
                    let mut val = mean[b];
                    for c in 0..n_comp {
                        val += scores[[c, row, col]] * components[[c, b]];
                    }
                    row_data[b * width + col] = val;
                }
            }
            row_data
        })
        .collect();

    let mut result = Array3::<f64>::zeros((bands, height, width));
    for (row, row_data) in rows.iter().enumerate() {
        for b in 0..bands {
            for col in 0..width {
                result[[b, row, col]] = row_data[b * width + col];
            }
        }
    }

    SpectralCube::new(result, pca_result.wavelengths.clone(), None, None)
}

/// Jacobi eigenvalue algorithm for a symmetric matrix.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns of the
/// returned matrix.
pub(crate) fn jacobi_eigen(matrix: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
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
        // First update the rows/columns that interact with p and q
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
        // Then update the diagonal and zero out the target
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
        // Full PCA (all components) → transform → inverse should reconstruct original
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
}
