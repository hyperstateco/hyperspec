use ndarray::Array3;
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

/// Apply a Savitzky-Golay smoothing filter along the spectral axis.
///
/// `window` must be odd and ≥ 3. `polyorder` must be < `window`.
/// The filter preserves the spectral shape while reducing noise.
///
/// The convolution coefficients are computed in band-index space (uniform spacing),
/// not wavelength space. This is standard SG behavior but means the filter width
/// in nanometers varies if the wavelength grid is non-uniform.
///
/// Pixels where any band is NaN or nodata produce all NaN in the output.
/// Edge bands use mirror-reflection boundary extension.
pub fn savitzky_golay(
    cube: &SpectralCube,
    window: usize,
    polyorder: usize,
) -> Result<SpectralCube> {
    if window < 3 || window.is_multiple_of(2) {
        return Err(HyperspecError::InvalidInput(format!(
            "window must be odd and ≥ 3, got {}",
            window
        )));
    }
    if polyorder >= window {
        return Err(HyperspecError::InvalidInput(format!(
            "polyorder ({}) must be less than window ({})",
            polyorder, window
        )));
    }
    if window > cube.bands() {
        return Err(HyperspecError::InvalidInput(format!(
            "window ({}) exceeds band count ({})",
            window,
            cube.bands()
        )));
    }

    // Precompute SG convolution coefficients
    let coeffs = sg_coefficients(window, polyorder);

    let data = cube.data();
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let nodata = cube.nodata();
    let half = window / 2;

    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; bands * width];
            for col in 0..width {
                // Check for invalid pixel
                let mut has_invalid = false;
                for b in 0..bands {
                    let v = data[[b, row, col]];
                    if v.is_nan() || nodata == Some(v) {
                        has_invalid = true;
                        break;
                    }
                }

                if has_invalid {
                    for b in 0..bands {
                        row_data[b * width + col] = f64::NAN;
                    }
                } else {
                    for b in 0..bands {
                        let mut val = 0.0;
                        for (k, &c) in coeffs.iter().enumerate() {
                            // Mirror at boundaries
                            let idx = b as isize + k as isize - half as isize;
                            let idx = reflect(idx, bands);
                            val += c * data[[idx, row, col]];
                        }
                        row_data[b * width + col] = val;
                    }
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

    SpectralCube::new(
        result,
        cube.wavelengths().clone(),
        cube.fwhm().cloned(),
        cube.nodata(),
    )
}

/// Reflect index at boundaries for the SG stencil.
///
/// Only valid for small out-of-range offsets (bounded by the SG half-window,
/// which is always < len due to the `window <= bands` check).
#[inline]
fn reflect(idx: isize, len: usize) -> usize {
    let n = len as isize;
    if idx < 0 {
        (-idx).min(n - 1) as usize
    } else if idx >= n {
        (2 * n - 2 - idx).max(0) as usize
    } else {
        idx as usize
    }
}

/// Compute Savitzky-Golay convolution coefficients for zeroth derivative (smoothing).
///
/// Builds the Vandermonde matrix J for the window on integer indices, computes
/// `(J^T J)^{-1} e_0` via Gaussian elimination, then evaluates the polynomial
/// at each window position to get the convolution weights.
fn sg_coefficients(window: usize, polyorder: usize) -> Vec<f64> {
    let half = window as isize / 2;
    let m = polyorder + 1;

    // Build Vandermonde J (window × m) and compute J^T J
    let points: Vec<f64> = (-half..=half).map(|i| i as f64).collect();
    let mut jtj = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for &x in &points {
                s += x.powi(i as i32) * x.powi(j as i32);
            }
            jtj[i * m + j] = s;
        }
    }

    // Solve (J^T J) c = e_0 for the smoothing coefficients
    // e_0 = [1, 0, 0, ...] since we want the 0th derivative (value, not slope)
    let mut rhs = vec![0.0; m];
    rhs[0] = 1.0;

    // Gaussian elimination with partial pivoting
    let mut aug = vec![0.0; m * (m + 1)];
    for i in 0..m {
        for j in 0..m {
            aug[i * (m + 1) + j] = jtj[i * m + j];
        }
        aug[i * (m + 1) + m] = rhs[i];
    }

    for col in 0..m {
        // Pivot
        let mut max_row = col;
        let mut max_val = aug[col * (m + 1) + col].abs();
        for row in (col + 1)..m {
            let v = aug[row * (m + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=m {
                aug.swap(col * (m + 1) + j, max_row * (m + 1) + j);
            }
        }

        let pivot = aug[col * (m + 1) + col];
        debug_assert!(
            pivot.abs() > 1e-12,
            "near-zero pivot in SG coefficient solve (window={}, polyorder={})",
            window,
            polyorder
        );
        for row in (col + 1)..m {
            let factor = aug[row * (m + 1) + col] / pivot;
            for j in col..=m {
                aug[row * (m + 1) + j] -= factor * aug[col * (m + 1) + j];
            }
        }
    }

    // Back substitution
    let mut poly_coeffs = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = aug[i * (m + 1) + m];
        for j in (i + 1)..m {
            s -= aug[i * (m + 1) + j] * poly_coeffs[j];
        }
        poly_coeffs[i] = s / aug[i * (m + 1) + i];
    }

    // Convolution coefficients: c_k = sum_j poly_coeffs[j] * x_k^j
    points
        .iter()
        .map(|&x| {
            let mut val = 0.0;
            for (j, &c) in poly_coeffs.iter().enumerate() {
                val += c * x.powi(j as i32);
            }
            val
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    #[test]
    fn test_sg_flat_spectrum() {
        // Flat spectrum → smoothing should preserve it exactly
        let data = Array3::from_elem((10, 1, 1), 5.0);
        let wl = Array1::from_vec((0..10).map(|i| 400.0 + i as f64 * 10.0).collect());
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let smooth = savitzky_golay(&cube, 5, 2).unwrap();
        for b in 0..10 {
            assert!(
                (smooth.data()[[b, 0, 0]] - 5.0).abs() < 1e-10,
                "band {}: expected 5.0, got {}",
                b,
                smooth.data()[[b, 0, 0]]
            );
        }
    }

    #[test]
    fn test_sg_linear_spectrum() {
        // Linear spectrum → SG preserves exactly in the interior (away from edges)
        let wl = Array1::from_vec((0..10).map(|i| 400.0 + i as f64 * 10.0).collect());
        let data = Array3::from_shape_fn((10, 1, 1), |(b, _, _)| b as f64 * 3.0 + 1.0);
        let cube = SpectralCube::new(data.clone(), wl, None, None).unwrap();

        let smooth = savitzky_golay(&cube, 5, 2).unwrap();
        // Interior bands (half-window = 2, so bands 2..8) should be exact
        for b in 2..8 {
            assert!(
                (smooth.data()[[b, 0, 0]] - data[[b, 0, 0]]).abs() < 1e-8,
                "band {}: expected {}, got {}",
                b,
                data[[b, 0, 0]],
                smooth.data()[[b, 0, 0]]
            );
        }
    }

    #[test]
    fn test_sg_reduces_noise() {
        // Noisy signal → smoothed signal should have lower variance
        let wl = Array1::from_vec((0..20).map(|i| 400.0 + i as f64 * 10.0).collect());
        let noise = [
            0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.2, -0.05, 0.1, -0.1, 0.12, -0.18, 0.08, -0.12,
            0.06, -0.14, 0.16, -0.08, 0.11, -0.09,
        ];
        let data = Array3::from_shape_fn((20, 1, 1), |(b, _, _)| 1.0 + noise[b]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let smooth = savitzky_golay(&cube, 5, 2).unwrap();

        let orig_var: f64 = (0..20).map(|b| noise[b] * noise[b]).sum::<f64>() / 20.0;
        let smooth_var: f64 = (0..20)
            .map(|b| (smooth.data()[[b, 0, 0]] - 1.0).powi(2))
            .sum::<f64>()
            / 20.0;

        assert!(smooth_var < orig_var, "smoothing should reduce variance");
    }

    #[test]
    fn test_sg_nan_pixel() {
        let mut data = Array3::from_elem((5, 1, 2), 1.0);
        data[[2, 0, 0]] = f64::NAN;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let smooth = savitzky_golay(&cube, 3, 1).unwrap();
        // Invalid pixel → all NaN
        for b in 0..5 {
            assert!(smooth.data()[[b, 0, 0]].is_nan());
        }
        // Valid pixel preserved
        for b in 0..5 {
            assert!(!smooth.data()[[b, 0, 1]].is_nan());
        }
    }

    #[test]
    fn test_sg_invalid_params() {
        let data = Array3::zeros((10, 1, 1));
        let wl = Array1::from_vec((0..10).map(|i| 400.0 + i as f64 * 10.0).collect());
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        assert!(savitzky_golay(&cube, 2, 1).is_err()); // even window
        assert!(savitzky_golay(&cube, 1, 0).is_err()); // window < 3
        assert!(savitzky_golay(&cube, 5, 5).is_err()); // polyorder >= window
        assert!(savitzky_golay(&cube, 11, 2).is_err()); // window > bands
    }

    #[test]
    fn test_sg_preserves_metadata() {
        let data = Array3::from_elem((10, 2, 3), 1.0);
        let wl = Array1::from_vec((0..10).map(|i| 400.0 + i as f64 * 10.0).collect());
        let cube = SpectralCube::new(data, wl.clone(), None, None).unwrap();

        let smooth = savitzky_golay(&cube, 5, 2).unwrap();
        assert_eq!(smooth.shape(), cube.shape());
        assert_eq!(smooth.wavelengths(), &wl);
    }

    #[test]
    fn test_sg_coefficients_sum_to_one() {
        // SG smoothing coefficients should sum to 1 (preserve DC)
        let coeffs = sg_coefficients(5, 2);
        let sum: f64 = coeffs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "coeffs sum = {}", sum);
    }

    #[test]
    fn test_reflect() {
        assert_eq!(reflect(-1, 10), 1);
        assert_eq!(reflect(-2, 10), 2);
        assert_eq!(reflect(0, 10), 0);
        assert_eq!(reflect(9, 10), 9);
        assert_eq!(reflect(10, 10), 8);
        assert_eq!(reflect(11, 10), 7);
    }
}
