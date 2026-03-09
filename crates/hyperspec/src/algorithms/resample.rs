use ndarray::{Array1, Array3};
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

/// Interpolation method for band resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMethod {
    Linear,
    Cubic,
}

/// Resample a spectral cube to a new set of target wavelengths.
///
/// Interpolates each pixel's spectrum from the cube's wavelength grid to the
/// target wavelength grid. Rayon-parallel per-row.
///
/// Target wavelengths must be strictly increasing and within the range of
/// the source wavelengths (no extrapolation).
///
/// Pixels containing NaN or nodata in any band produce NaN for all output bands.
pub fn resample(
    cube: &SpectralCube,
    target_wavelengths: &Array1<f64>,
    method: ResampleMethod,
) -> Result<SpectralCube> {
    let src_wl = cube.wavelengths();
    let n_target = target_wavelengths.len();

    if n_target == 0 {
        return Err(HyperspecError::InvalidInput(
            "target wavelengths must be non-empty".to_string(),
        ));
    }

    // Validate target wavelengths are strictly increasing
    let target_wl = target_wavelengths.as_standard_layout();
    let target_slice = target_wl
        .as_slice()
        .expect("target wavelengths contiguous after as_standard_layout");
    if target_slice.windows(2).any(|w| w[1] <= w[0]) {
        return Err(HyperspecError::InvalidWavelength(
            "target wavelengths must be strictly increasing".to_string(),
        ));
    }

    // Validate target range is within source range
    let src_min = src_wl[0];
    let src_max = src_wl[src_wl.len() - 1];
    let tgt_min = target_slice[0];
    let tgt_max = target_slice[n_target - 1];
    if tgt_min < src_min || tgt_max > src_max {
        return Err(HyperspecError::InvalidWavelength(format!(
            "target range [{}, {}] exceeds source range [{}, {}]",
            tgt_min, tgt_max, src_min, src_max
        )));
    }

    let data = cube.data();
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let nodata = cube.nodata();
    let src_wl_std = src_wl.as_standard_layout();
    let src_slice = src_wl_std
        .as_slice()
        .expect("source wavelengths contiguous after as_standard_layout");

    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; n_target * width];
            for col in 0..width {
                let mut spectrum = vec![0.0; bands];
                let mut has_invalid = false;
                for b in 0..bands {
                    let v = data[[b, row, col]];
                    if v.is_nan() || nodata == Some(v) {
                        has_invalid = true;
                        break;
                    }
                    spectrum[b] = v;
                }

                if has_invalid {
                    for t in 0..n_target {
                        row_data[t * width + col] = f64::NAN;
                    }
                } else {
                    for (t, &twl) in target_slice.iter().enumerate() {
                        let val = match method {
                            ResampleMethod::Linear => interpolate_linear(src_slice, &spectrum, twl),
                            ResampleMethod::Cubic => interpolate_cubic(src_slice, &spectrum, twl),
                        };
                        row_data[t * width + col] = val;
                    }
                }
            }
            row_data
        })
        .collect();

    let band_size = height * width;
    let mut flat = vec![0.0f64; n_target * band_size];
    for (row, row_data) in rows.iter().enumerate() {
        for t in 0..n_target {
            let src = &row_data[t * width..(t + 1) * width];
            let dst_start = t * band_size + row * width;
            flat[dst_start..dst_start + width].copy_from_slice(src);
        }
    }
    let result = Array3::from_shape_vec((n_target, height, width), flat)
        .expect("shape matches total element count");

    SpectralCube::new(result, target_wavelengths.clone(), None, cube.nodata())
}

/// Linear interpolation.
fn interpolate_linear(x: &[f64], y: &[f64], xi: f64) -> f64 {
    // Find the interval containing xi
    let idx = find_interval(x, xi);
    let x0 = x[idx];
    let x1 = x[idx + 1];
    let t = (xi - x0) / (x1 - x0);
    y[idx] + t * (y[idx + 1] - y[idx])
}

/// Cubic interpolation (Catmull-Rom spline).
fn interpolate_cubic(x: &[f64], y: &[f64], xi: f64) -> f64 {
    let n = x.len();
    let idx = find_interval(x, xi);

    // Get four points for cubic: idx-1, idx, idx+1, idx+2
    // Clamp at boundaries
    let i0 = if idx > 0 { idx - 1 } else { 0 };
    let i1 = idx;
    let i2 = idx + 1;
    let i3 = if idx + 2 < n { idx + 2 } else { n - 1 };

    let t = (xi - x[i1]) / (x[i2] - x[i1]);

    // Catmull-Rom with non-uniform spacing
    let y0 = y[i0];
    let y1 = y[i1];
    let y2 = y[i2];
    let y3 = y[i3];

    // For non-uniform x spacing, use simple Catmull-Rom approximation
    let t2 = t * t;
    let t3 = t2 * t;

    // Tangents at knots
    let m1 = if i1 != i0 {
        0.5 * ((y2 - y1) / (x[i2] - x[i1]) + (y1 - y0) / (x[i1] - x[i0])) * (x[i2] - x[i1])
    } else {
        y2 - y1
    };
    let m2 = if i3 != i2 {
        0.5 * ((y3 - y2) / (x[i3] - x[i2]) + (y2 - y1) / (x[i2] - x[i1])) * (x[i2] - x[i1])
    } else {
        y2 - y1
    };

    // Hermite basis
    (2.0 * t3 - 3.0 * t2 + 1.0) * y1
        + (t3 - 2.0 * t2 + t) * m1
        + (-2.0 * t3 + 3.0 * t2) * y2
        + (t3 - t2) * m2
}

/// Find the index of the interval containing xi: x[idx] <= xi <= x[idx+1].
fn find_interval(x: &[f64], xi: f64) -> usize {
    // Binary search
    let n = x.len();
    match x.binary_search_by(|v| v.partial_cmp(&xi).unwrap()) {
        Ok(i) => {
            // Exact match — return the interval starting here,
            // but clamp to second-to-last for the right endpoint
            if i >= n - 1 { n - 2 } else { i }
        }
        Err(i) => {
            // xi falls between x[i-1] and x[i]
            if i == 0 { 0 } else { i - 1 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cube() -> SpectralCube {
        // 5 bands, 2x2, linear spectrum: value = wavelength / 1000
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let mut data = Array3::zeros((5, 2, 2));
        for b in 0..5 {
            let val = wl[b] / 1000.0;
            for row in 0..2 {
                for col in 0..2 {
                    data[[b, row, col]] = val;
                }
            }
        }
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_resample_linear_identity() {
        // Resampling to same wavelengths should give same values
        let cube = make_cube();
        let target = cube.wavelengths().clone();
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();
        let orig = cube.data();
        let res = resampled.data();
        for b in 0..5 {
            for row in 0..2 {
                for col in 0..2 {
                    assert!((orig[[b, row, col]] - res[[b, row, col]]).abs() < 1e-10,);
                }
            }
        }
    }

    #[test]
    fn test_resample_linear_midpoints() {
        // Linear data resampled at midpoints should give midpoint values
        let cube = make_cube();
        let target = Array1::from_vec(vec![450.0, 550.0, 650.0, 750.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();

        assert_eq!(resampled.bands(), 4);
        // Value at 450nm should be 0.45 (midpoint of 0.4 and 0.5)
        assert!((resampled.data()[[0, 0, 0]] - 0.45).abs() < 1e-10);
        assert!((resampled.data()[[1, 0, 0]] - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_resample_cubic_identity() {
        let cube = make_cube();
        let target = cube.wavelengths().clone();
        let resampled = resample(&cube, &target, ResampleMethod::Cubic).unwrap();
        let orig = cube.data();
        let res = resampled.data();
        for b in 0..5 {
            for row in 0..2 {
                for col in 0..2 {
                    assert!((orig[[b, row, col]] - res[[b, row, col]]).abs() < 1e-8,);
                }
            }
        }
    }

    #[test]
    fn test_resample_cubic_on_linear_data() {
        // Cubic interpolation of linear data should give exact results
        let cube = make_cube();
        let target = Array1::from_vec(vec![450.0, 550.0, 650.0, 750.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Cubic).unwrap();

        assert!((resampled.data()[[0, 0, 0]] - 0.45).abs() < 1e-8);
        assert!((resampled.data()[[1, 0, 0]] - 0.55).abs() < 1e-8);
    }

    #[test]
    fn test_resample_preserves_spatial() {
        let cube = make_cube();
        let target = Array1::from_vec(vec![450.0, 650.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();

        assert_eq!(resampled.height(), cube.height());
        assert_eq!(resampled.width(), cube.width());
    }

    #[test]
    fn test_resample_target_out_of_range() {
        let cube = make_cube();
        let target = Array1::from_vec(vec![300.0, 500.0]); // 300 < 400 (source min)
        assert!(resample(&cube, &target, ResampleMethod::Linear).is_err());
    }

    #[test]
    fn test_resample_empty_target() {
        let cube = make_cube();
        let target = Array1::from_vec(vec![]);
        assert!(resample(&cube, &target, ResampleMethod::Linear).is_err());
    }

    #[test]
    fn test_resample_unsorted_target() {
        let cube = make_cube();
        let target = Array1::from_vec(vec![600.0, 500.0]);
        assert!(resample(&cube, &target, ResampleMethod::Linear).is_err());
    }

    #[test]
    fn test_resample_two_band_cube() {
        // Minimum viable interpolation: 2 bands
        let wl = Array1::from_vec(vec![400.0, 800.0]);
        let mut data = Array3::zeros((2, 1, 1));
        data[[0, 0, 0]] = 0.4;
        data[[1, 0, 0]] = 0.8;
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let target = Array1::from_vec(vec![600.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();
        // Linear interp: 0.4 + (600-400)/(800-400) * (0.8-0.4) = 0.6
        assert!((resampled.data()[[0, 0, 0]] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_resample_single_target() {
        let cube = make_cube();
        let target = Array1::from_vec(vec![500.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();
        assert_eq!(resampled.bands(), 1);
        assert!((resampled.data()[[0, 0, 0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_resample_nan_pixel() {
        // 5 bands, 1x2. Pixel (0,0) has NaN, pixel (0,1) is valid.
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let mut data = Array3::zeros((5, 1, 2));
        for b in 0..5 {
            let val = wl[b] / 1000.0;
            data[[b, 0, 0]] = val;
            data[[b, 0, 1]] = val;
        }
        data[[2, 0, 0]] = f64::NAN;
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let target = Array1::from_vec(vec![450.0, 650.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();
        // invalid pixel → all NaN
        for t in 0..2 {
            assert!(resampled.data()[[t, 0, 0]].is_nan());
        }
        // valid pixel → correct interpolation
        assert!((resampled.data()[[0, 0, 1]] - 0.45).abs() < 1e-10);
        assert!((resampled.data()[[1, 0, 1]] - 0.65).abs() < 1e-10);
    }

    #[test]
    fn test_resample_nodata_pixel() {
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let mut data = Array3::zeros((5, 1, 2));
        for b in 0..5 {
            let val = wl[b] / 1000.0;
            data[[b, 0, 0]] = val;
            data[[b, 0, 1]] = val;
        }
        data[[0, 0, 0]] = -9999.0;
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let target = Array1::from_vec(vec![450.0, 650.0]);
        let resampled = resample(&cube, &target, ResampleMethod::Linear).unwrap();
        for t in 0..2 {
            assert!(resampled.data()[[t, 0, 0]].is_nan());
        }
        assert!((resampled.data()[[0, 0, 1]] - 0.45).abs() < 1e-10);
    }
}
