use ndarray::{Array1, Array2, s};
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

/// Result of an MNF computation.
#[derive(Debug, Clone)]
pub struct MnfResult {
    /// MNF components as row vectors, shape (n_components, bands).
    pub components: Array2<f64>,
    /// Eigenvalues (signal-to-noise ratio) per component, descending.
    pub eigenvalues: Array1<f64>,
    /// Mean spectrum subtracted before MNF.
    pub mean: Array1<f64>,
    /// Original wavelengths, preserved for inverse transform.
    pub wavelengths: Array1<f64>,
}

/// Internal intermediates from the MNF computation, used by both `mnf()` and
/// `mnf_denoise()` to avoid recomputation.
struct MnfInternals {
    /// Sorted eigenvalues (descending SNR).
    eigenvalues: Vec<f64>,
    /// Sorted indices into the whitened eigendecomposition.
    sorted_indices: Vec<usize>,
    /// Whitened eigenvectors (bands × bands), columns are eigenvectors.
    whitened_eigenvectors: Array2<f64>,
    /// noise_cov^{-1/2}, shape (bands, bands).
    noise_inv_sqrt: Array2<f64>,
    /// noise_cov^{1/2}, shape (bands, bands).
    noise_sqrt: Array2<f64>,
    /// Mean spectrum subtracted before MNF.
    mean: Array1<f64>,
}

/// Eigendecompose a symmetric ndarray matrix using faer.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns.
fn faer_eigen(matrix: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    // Build row-major slice for faer
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = matrix[[i, j]];
        }
    }
    let mat = faer::MatRef::from_row_major_slice(&data, n, n);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| HyperspecError::InvalidInput("eigendecomposition failed".to_string()))?;

    let s = decomp.S();
    let u = decomp.U();
    let eigenvalues: Vec<f64> = (0..n).map(|i| s[i]).collect();
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            eigenvectors[[i, j]] = u[(i, j)];
        }
    }
    Ok((eigenvalues, eigenvectors))
}

/// Shared MNF computation: validates input, estimates covariances, and
/// performs the noise-whitened eigendecomposition.
fn mnf_core(cube: &SpectralCube, n_components: usize) -> Result<MnfInternals> {
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let n_pixels = height * width;

    if n_pixels < 2 {
        return Err(HyperspecError::InvalidInput(
            "MNF requires at least 2 pixels".to_string(),
        ));
    }

    if width < 2 {
        return Err(HyperspecError::InvalidInput(
            "MNF requires width >= 2 for noise estimation".to_string(),
        ));
    }

    if n_components == 0 || n_components > bands {
        return Err(HyperspecError::InvalidInput(format!(
            "n_components {} must be in [1, {}]",
            n_components, bands
        )));
    }

    let data = cube.data();
    let nodata = cube.nodata();

    // Validate and compute mean per band
    let mut mean = Array1::<f64>::zeros(bands);
    for b in 0..bands {
        let band_slice = data.slice(s![b, .., ..]);
        let mut sum = 0.0;
        for &v in band_slice.iter() {
            if v.is_nan() || nodata == Some(v) {
                return Err(HyperspecError::InvalidInput(
                    "MNF input contains NaN or nodata values; mask or remove them first"
                        .to_string(),
                ));
            }
            sum += v;
        }
        mean[b] = sum / n_pixels as f64;
    }

    // Estimate noise covariance using horizontal spatial differences
    let noise_cov = estimate_noise_covariance(cube);

    // Compute data covariance via tiled GEMM (same approach as PCA)
    let n_f = (n_pixels - 1).max(1) as f64;
    let data_cov = if n_pixels <= 32768 {
        // Small: scalar triangle accumulation
        let tri_size = bands * (bands + 1) / 2;
        let mut tri = vec![0.0; tri_size];
        let mut centered = vec![0.0; bands];
        for row in 0..height {
            for col in 0..width {
                for b in 0..bands {
                    centered[b] = data[[b, row, col]] - mean[b];
                }
                let mut idx = 0;
                for i in 0..bands {
                    for j in i..bands {
                        tri[idx] += centered[i] * centered[j];
                        idx += 1;
                    }
                }
            }
        }
        let mut cov = Array2::<f64>::zeros((bands, bands));
        let inv_n = 1.0 / n_f;
        let mut idx = 0;
        for i in 0..bands {
            for j in i..bands {
                let v = tri[idx] * inv_n;
                cov[[i, j]] = v;
                cov[[j, i]] = v;
                idx += 1;
            }
        }
        cov
    } else {
        // Large: tiled GEMM via faer
        let tile_size = 32768;
        let mut cov_faer = faer::Mat::<f64>::zeros(bands, bands);
        let mut tile_buf = vec![0.0f64; tile_size * bands];

        for tile_start in (0..n_pixels).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n_pixels);
            let rows_in_tile = tile_end - tile_start;

            for b in 0..bands {
                let band_slice = data.slice(s![b, .., ..]);
                let band_data = band_slice.as_standard_layout();
                let band_raw = band_data.as_slice().expect("band is contiguous");
                let col_start = b * tile_size;
                let mean_b = mean[b];
                for i in 0..rows_in_tile {
                    tile_buf[col_start + i] = band_raw[tile_start + i] - mean_b;
                }
            }

            let tile = faer::MatRef::from_column_major_slice(
                &tile_buf[..rows_in_tile * bands],
                rows_in_tile,
                bands,
            );

            faer::linalg::matmul::matmul(
                cov_faer.as_mut(),
                faer::Accum::Add,
                tile.transpose(),
                tile,
                1.0,
                faer::Par::rayon(0),
            );
        }

        let inv_n = 1.0 / n_f;
        let mut cov = Array2::<f64>::zeros((bands, bands));
        for i in 0..bands {
            for j in 0..bands {
                cov[[i, j]] = cov_faer[(i, j)] * inv_n;
            }
        }
        cov
    };

    // Eigendecompose noise covariance via faer
    let (noise_eigenvalues, noise_eigenvectors) = faer_eigen(&noise_cov)?;

    // Check that noise has non-trivial eigenvalues
    let n_valid = noise_eigenvalues.iter().filter(|&&v| v > 1e-12).count();
    if n_valid == 0 {
        return Err(HyperspecError::InvalidInput(
            "noise covariance is zero — data has no spatial variation".to_string(),
        ));
    }

    // Build noise_cov^{-1/2} and noise_cov^{1/2} from the eigendecomposition
    let mut inv_scaled = Array2::<f64>::zeros((bands, n_valid));
    let mut fwd_scaled = Array2::<f64>::zeros((bands, n_valid));
    let mut col_idx = 0;
    for k in 0..bands {
        let lam = noise_eigenvalues[k];
        if lam > 1e-12 {
            let inv_scale = 1.0 / lam.sqrt();
            let fwd_scale = lam.sqrt();
            for i in 0..bands {
                inv_scaled[[i, col_idx]] = noise_eigenvectors[[i, k]] * inv_scale;
                fwd_scaled[[i, col_idx]] = noise_eigenvectors[[i, k]] * fwd_scale;
            }
            col_idx += 1;
        }
    }
    let noise_inv_sqrt = inv_scaled.dot(&inv_scaled.t());
    let noise_sqrt = fwd_scaled.dot(&fwd_scaled.t());

    // Whitened covariance: noise_cov^{-1/2} * data_cov * noise_cov^{-1/2}
    let whitened_cov = noise_inv_sqrt.dot(&data_cov).dot(&noise_inv_sqrt);

    // Eigendecompose the whitened covariance via faer
    let (eigenvalues, whitened_eigenvectors) = faer_eigen(&whitened_cov)?;

    // Sort by descending eigenvalue
    let mut sorted_indices: Vec<usize> = (0..bands).collect();
    sorted_indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(MnfInternals {
        eigenvalues,
        sorted_indices,
        whitened_eigenvectors,
        noise_inv_sqrt,
        noise_sqrt,
        mean,
    })
}

/// Compute the Minimum Noise Fraction transform.
///
/// MNF = PCA on noise-whitened data. Steps:
/// 1. Estimate noise covariance using spatial differences
/// 2. Whiten the data by the noise covariance
/// 3. PCA on the whitened data
///
/// The resulting components are ordered by signal-to-noise ratio (descending).
pub fn mnf(cube: &SpectralCube, n_components: Option<usize>) -> Result<MnfResult> {
    let bands = cube.bands();
    let n_comp = n_components.unwrap_or(bands);
    let internals = mnf_core(cube, n_comp)?;

    // MNF components in original space: noise_cov^{-1/2} * whitened_eigenvectors
    let mnf_vectors = internals
        .noise_inv_sqrt
        .dot(&internals.whitened_eigenvectors);

    // Take top n_components and normalize
    let mut components = Array2::<f64>::zeros((n_comp, bands));
    let mut sorted_eigenvalues = Array1::<f64>::zeros(n_comp);
    for (i, &idx) in internals.sorted_indices.iter().take(n_comp).enumerate() {
        sorted_eigenvalues[i] = internals.eigenvalues[idx].max(0.0);
        let mut norm = 0.0;
        for j in 0..bands {
            norm += mnf_vectors[[j, idx]] * mnf_vectors[[j, idx]];
        }
        let norm = norm.sqrt();
        if norm > 1e-12 {
            for j in 0..bands {
                components[[i, j]] = mnf_vectors[[j, idx]] / norm;
            }
        }
    }

    Ok(MnfResult {
        components,
        eigenvalues: sorted_eigenvalues,
        mean: internals.mean,
        wavelengths: cube.wavelengths().clone(),
    })
}

/// Denoise a cube by forward MNF transform, truncating to n_components,
/// then inverse transforming back.
///
/// Uses the mathematically correct inverse: `noise_sqrt @ Vr @ V^T @ noise_inv_sqrt`
/// where Vr has columns beyond n_components zeroed. The full denoise transform is
/// precomputed as a single (bands, bands) matrix, then applied per-pixel with a
/// single matrix-vector multiply.
pub fn mnf_denoise(cube: &SpectralCube, n_components: usize) -> Result<SpectralCube> {
    let bands = cube.bands();
    let internals = mnf_core(cube, n_components)?;

    // Build Vr: copy of V with columns beyond n_components zeroed (in sorted order)
    let v = &internals.whitened_eigenvectors;
    let mut vr = Array2::<f64>::zeros((bands, bands));
    for (i, &idx) in internals
        .sorted_indices
        .iter()
        .take(n_components)
        .enumerate()
    {
        for j in 0..bands {
            vr[[j, idx]] = v[[j, idx]];
        }
        let _ = i; // used only for take() count
    }

    // Denoise matrix: D = noise_sqrt @ Vr @ V^T @ noise_inv_sqrt
    // This is (bands, bands), precomputed once.
    let denoise_matrix = internals
        .noise_sqrt
        .dot(&vr)
        .dot(&v.t())
        .dot(&internals.noise_inv_sqrt);

    let data = cube.data();
    let height = cube.height();
    let width = cube.width();
    let mean = &internals.mean;

    // Apply D per pixel: result = D @ (pixel - mean) + mean
    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; bands * width];
            for col in 0..width {
                let mut centered = vec![0.0; bands];
                for b in 0..bands {
                    centered[b] = data[[b, row, col]] - mean[b];
                }
                for b in 0..bands {
                    let mut val = mean[b];
                    for k in 0..bands {
                        val += denoise_matrix[[b, k]] * centered[k];
                    }
                    row_data[b * width + col] = val;
                }
            }
            row_data
        })
        .collect();

    let band_size = height * width;
    let mut flat = vec![0.0f64; bands * band_size];
    for (row, row_data) in rows.iter().enumerate() {
        for b in 0..bands {
            let src = &row_data[b * width..(b + 1) * width];
            let dst_start = b * band_size + row * width;
            flat[dst_start..dst_start + width].copy_from_slice(src);
        }
    }
    let out = ndarray::Array3::from_shape_vec((bands, height, width), flat)
        .expect("shape matches total element count");

    SpectralCube::new(
        out,
        cube.wavelengths().clone(),
        cube.fwhm().cloned(),
        cube.nodata(),
    )
}

/// Estimate noise covariance using horizontal spatial differences.
///
/// Rayon-parallel per-row. For each pixel (row, col) where col+1 exists,
/// the difference spectrum is a noise estimate. The covariance of these
/// differences / 2 estimates the noise covariance.
fn estimate_noise_covariance(cube: &SpectralCube) -> Array2<f64> {
    let data = cube.data();
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();

    // Each row contributes a partial (bands, bands) covariance matrix
    let row_covs: Vec<Array2<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut cov = Array2::<f64>::zeros((bands, bands));
            for col in 0..(width - 1) {
                for i in 0..bands {
                    let di = data[[i, row, col + 1]] - data[[i, row, col]];
                    for j in i..bands {
                        let dj = data[[j, row, col + 1]] - data[[j, row, col]];
                        let v = di * dj;
                        cov[[i, j]] += v;
                        if i != j {
                            cov[[j, i]] += v;
                        }
                    }
                }
            }
            cov
        })
        .collect();

    // Sum partial covariances
    let mut total = Array2::<f64>::zeros((bands, bands));
    for cov in &row_covs {
        total += cov;
    }

    let n_diffs = height * (width - 1);
    let n_f = (n_diffs - 1).max(1) as f64;
    // Divide by 2 because var(a - b) = var(a) + var(b) = 2 * var(noise)
    total / (2.0 * n_f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn make_cube() -> SpectralCube {
        // 3 bands, 4x4 with signal + noise structure
        let mut data = Array3::zeros((3, 4, 4));
        let signal = [
            0.5, 0.8, 0.3, 0.9, 0.6, 0.7, 0.4, 0.2, 0.85, 0.35, 0.65, 0.55, 0.45, 0.75, 0.25, 0.95,
        ];
        for row in 0..4 {
            for col in 0..4 {
                let s = signal[row * 4 + col];
                data[[0, row, col]] = s + 0.01;
                data[[1, row, col]] = s * 0.9 + 0.05 + 0.005;
                data[[2, row, col]] = 1.0 - s + 0.008;
            }
        }
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_mnf_basic() {
        let cube = make_cube();
        let result = mnf(&cube, None).unwrap();

        assert_eq!(result.components.nrows(), 3);
        assert_eq!(result.components.ncols(), 3);
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.mean.len(), 3);

        // Eigenvalues sorted descending
        for i in 1..result.eigenvalues.len() {
            assert!(result.eigenvalues[i - 1] >= result.eigenvalues[i]);
        }
    }

    #[test]
    fn test_mnf_n_components() {
        let cube = make_cube();
        let result = mnf(&cube, Some(2)).unwrap();
        assert_eq!(result.components.nrows(), 2);
        assert_eq!(result.eigenvalues.len(), 2);
    }

    #[test]
    fn test_mnf_invalid_n_components() {
        let cube = make_cube();
        assert!(mnf(&cube, Some(0)).is_err());
        assert!(mnf(&cube, Some(10)).is_err());
    }

    #[test]
    fn test_mnf_denoise_shape() {
        let cube = make_cube();
        let denoised = mnf_denoise(&cube, 2).unwrap();
        assert_eq!(denoised.bands(), cube.bands());
        assert_eq!(denoised.height(), cube.height());
        assert_eq!(denoised.width(), cube.width());
    }

    #[test]
    fn test_mnf_denoise_preserves_wavelengths() {
        let cube = make_cube();
        let denoised = mnf_denoise(&cube, 2).unwrap();
        assert_eq!(denoised.wavelengths(), cube.wavelengths());
    }

    #[test]
    fn test_mnf_denoise_full_components_roundtrip() {
        // Denoising with all components should reconstruct original near-exactly
        let cube = make_cube();
        let denoised = mnf_denoise(&cube, 3).unwrap();

        let orig = cube.data();
        let den = denoised.data();
        for b in 0..cube.bands() {
            for row in 0..cube.height() {
                for col in 0..cube.width() {
                    assert!(
                        (orig[[b, row, col]] - den[[b, row, col]]).abs() < 1e-8,
                        "mismatch at [{}, {}, {}]: {} vs {}",
                        b,
                        row,
                        col,
                        orig[[b, row, col]],
                        den[[b, row, col]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mnf_denoise_rank_deficient_noise() {
        // Data with near-rank-deficient noise covariance.
        // Full-component denoise can't perfectly reconstruct signal in
        // dimensions discarded during noise whitening — this is correct
        // mathematical behavior, not a bug.
        let mut data = Array3::zeros((3, 8, 8));
        for row in 0..8 {
            for col in 0..8 {
                let s = (row * 8 + col) as f64 / 64.0;
                data[[0, row, col]] = s + 0.001 * (row as f64);
                data[[1, row, col]] = s * 0.5 + 0.01 * (col as f64);
                data[[2, row, col]] = 1.0 - s + 0.0001 * ((row + col) as f64);
            }
        }
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        // Should not error
        let denoised = mnf_denoise(&cube, 3).unwrap();
        assert_eq!(denoised.shape(), cube.shape());
    }

    #[test]
    fn test_mnf_components_unit_norm() {
        let cube = make_cube();
        let result = mnf(&cube, Some(2)).unwrap();

        // Components with non-trivial eigenvalues should be unit norm
        for i in 0..result.components.nrows() {
            if result.eigenvalues[i] > 1e-10 {
                let norm: f64 = result
                    .components
                    .row(i)
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-8,
                    "component {} has norm {}",
                    i,
                    norm
                );
            }
        }
    }

    #[test]
    fn test_mnf_narrow_cube() {
        // Width == 1: can't compute horizontal differences
        let data = Array3::from_shape_fn((3, 4, 1), |(b, r, _)| (b + r) as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(mnf(&cube, None).is_err());
    }

    #[test]
    fn test_mnf_single_pixel() {
        let data = Array3::from_shape_fn((3, 1, 1), |(b, _, _)| b as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(mnf(&cube, None).is_err());
    }

    #[test]
    fn test_mnf_rejects_nan() {
        let mut data = Array3::from_shape_fn((3, 4, 4), |(b, r, c)| {
            (b as f64) * 0.1 + (r as f64) * 0.3 + (c as f64) * 0.2 + 1.0
        });
        data[[1, 2, 3]] = f64::NAN;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let err = mnf(&cube, None).unwrap_err();
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn test_mnf_rejects_nodata() {
        let mut data = Array3::from_shape_fn((3, 4, 4), |(b, r, c)| {
            (b as f64) * 0.1 + (r as f64) * 0.3 + (c as f64) * 0.2 + 1.0
        });
        data[[0, 0, 0]] = -9999.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();
        let err = mnf(&cube, None).unwrap_err();
        assert!(err.to_string().contains("nodata"));
    }

    #[test]
    fn test_mnf_uniform_data() {
        // All pixels identical → noise covariance is zero → should error
        let data = Array3::from_shape_fn((3, 4, 4), |(b, _, _)| b as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(mnf(&cube, None).is_err());
    }

    #[test]
    fn test_noise_covariance_shape() {
        let cube = make_cube();
        let noise_cov = estimate_noise_covariance(&cube);
        assert_eq!(noise_cov.shape(), &[3, 3]);
    }
}
