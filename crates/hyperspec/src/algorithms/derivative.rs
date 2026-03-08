use ndarray::Array3;
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

/// Compute the spectral derivative of order 1 or 2 using finite differences.
///
/// For order 1: `d[i] = (x[i+1] - x[i]) / (wl[i+1] - wl[i])`, producing `bands - 1` output bands
/// at midpoint wavelengths `(wl[i] + wl[i+1]) / 2`.
///
/// For order 2: applies the first-derivative operator twice. On uniform wavelength
/// grids this is equivalent to the standard centered second difference
/// `(x[i+2] - 2*x[i+1] + x[i]) / h²`. On non-uniform grids it is an approximation
/// — the second derivative lives on midpoints-of-midpoints, not the original grid.
/// Output has `bands - 2` bands.
///
/// NaN and nodata pixels produce NaN if either adjacent band is invalid.
pub fn derivative(cube: &SpectralCube, order: usize) -> Result<SpectralCube> {
    match order {
        1 => derivative_first(cube),
        2 => {
            let first = derivative_first(cube)?;
            derivative_first(&first)
        }
        _ => Err(HyperspecError::InvalidInput(format!(
            "derivative order must be 1 or 2, got {}",
            order
        ))),
    }
}

fn derivative_first(cube: &SpectralCube) -> Result<SpectralCube> {
    let bands = cube.bands();
    if bands < 2 {
        return Err(HyperspecError::InvalidInput(
            "need at least 2 bands for first derivative".to_string(),
        ));
    }

    let data = cube.data();
    let height = cube.height();
    let width = cube.width();
    let nodata = cube.nodata();
    let wl = cube.wavelengths();
    let out_bands = bands - 1;

    // Precompute wavelength deltas.
    // SpectralCube::new enforces strictly increasing wavelengths, so dwl > 0
    // is guaranteed. The debug_assert catches any future invariant violation.
    let dwl: Vec<f64> = (0..out_bands)
        .map(|i| {
            let d = wl[i + 1] - wl[i];
            debug_assert!(d > 0.0, "wavelengths must be strictly increasing");
            d
        })
        .collect();
    let mid_wl: Vec<f64> = (0..out_bands).map(|i| (wl[i] + wl[i + 1]) * 0.5).collect();

    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; out_bands * width];
            for col in 0..width {
                for b in 0..out_bands {
                    let v0 = data[[b, row, col]];
                    let v1 = data[[b + 1, row, col]];
                    row_data[b * width + col] =
                        if v0.is_nan() || v1.is_nan() || nodata == Some(v0) || nodata == Some(v1) {
                            f64::NAN
                        } else {
                            (v1 - v0) / dwl[b]
                        };
                }
            }
            row_data
        })
        .collect();

    let mut result = Array3::<f64>::zeros((out_bands, height, width));
    for (row, row_data) in rows.iter().enumerate() {
        for b in 0..out_bands {
            for col in 0..width {
                result[[b, row, col]] = row_data[b * width + col];
            }
        }
    }

    SpectralCube::new(
        result,
        ndarray::Array1::from_vec(mid_wl),
        None, // FWHM is not meaningful for derivatives
        None, // nodata sentinel not meaningful for derivative values
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    #[test]
    fn test_derivative_linear() {
        // Linear spectrum: y = 2*wl → dy/dwl = 2 everywhere
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0]);
        let data = Array3::from_shape_fn((4, 1, 1), |(b, _, _)| 2.0 * wl[b]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let d1 = derivative(&cube, 1).unwrap();
        assert_eq!(d1.bands(), 3);
        for b in 0..3 {
            assert!((d1.data()[[b, 0, 0]] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_derivative_midpoint_wavelengths() {
        let wl = Array1::from_vec(vec![400.0, 500.0, 700.0]);
        let data = Array3::zeros((3, 1, 1));
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let d1 = derivative(&cube, 1).unwrap();
        assert!((d1.wavelengths()[0] - 450.0).abs() < 1e-10);
        assert!((d1.wavelengths()[1] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_second_order() {
        // Quadratic: y = wl^2 → d1 = 2*wl, d2 = 2
        let wl = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data = Array3::from_shape_fn((5, 1, 1), |(b, _, _)| wl[b] * wl[b]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let d2 = derivative(&cube, 2).unwrap();
        assert_eq!(d2.bands(), 3);
        for b in 0..3 {
            assert!(
                (d2.data()[[b, 0, 0]] - 2.0).abs() < 1e-10,
                "band {}: expected 2.0, got {}",
                b,
                d2.data()[[b, 0, 0]]
            );
        }
    }

    #[test]
    fn test_derivative_nan_propagation() {
        let mut data = Array3::zeros((3, 1, 2));
        data[[0, 0, 0]] = 1.0;
        data[[1, 0, 0]] = f64::NAN;
        data[[2, 0, 0]] = 3.0;
        data[[0, 0, 1]] = 1.0;
        data[[1, 0, 1]] = 2.0;
        data[[2, 0, 1]] = 3.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let d1 = derivative(&cube, 1).unwrap();
        // pixel (0,0): band 0→1 has NaN, band 1→2 has NaN
        assert!(d1.data()[[0, 0, 0]].is_nan());
        assert!(d1.data()[[1, 0, 0]].is_nan());
        // pixel (0,1): all valid
        assert!(!d1.data()[[0, 0, 1]].is_nan());
        assert!(!d1.data()[[1, 0, 1]].is_nan());
    }

    #[test]
    fn test_derivative_nodata_propagation() {
        let mut data = Array3::zeros((3, 1, 2));
        data[[0, 0, 0]] = 1.0;
        data[[1, 0, 0]] = -9999.0;
        data[[2, 0, 0]] = 3.0;
        data[[0, 0, 1]] = 1.0;
        data[[1, 0, 1]] = 2.0;
        data[[2, 0, 1]] = 3.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let d1 = derivative(&cube, 1).unwrap();
        // pixel (0,0): nodata in band 1 affects both differences
        assert!(d1.data()[[0, 0, 0]].is_nan());
        assert!(d1.data()[[1, 0, 0]].is_nan());
        // pixel (0,1): all valid
        assert!(!d1.data()[[0, 0, 1]].is_nan());
    }

    #[test]
    fn test_derivative_nonuniform_wavelengths() {
        // Non-uniform spacing: 400, 500, 700 (gaps of 100 and 200)
        // Spectrum: [100, 200, 500]
        // d1[0] = (200 - 100) / 100 = 1.0
        // d1[1] = (500 - 200) / 200 = 1.5
        let wl = Array1::from_vec(vec![400.0, 500.0, 700.0]);
        let mut data = Array3::zeros((3, 1, 1));
        data[[0, 0, 0]] = 100.0;
        data[[1, 0, 0]] = 200.0;
        data[[2, 0, 0]] = 500.0;
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let d1 = derivative(&cube, 1).unwrap();
        assert!((d1.data()[[0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((d1.data()[[1, 0, 0]] - 1.5).abs() < 1e-10);
        // Midpoint wavelengths
        assert!((d1.wavelengths()[0] - 450.0).abs() < 1e-10);
        assert!((d1.wavelengths()[1] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_invalid_order() {
        let data = Array3::zeros((3, 1, 1));
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(derivative(&cube, 0).is_err());
        assert!(derivative(&cube, 3).is_err());
    }

    #[test]
    fn test_derivative_too_few_bands() {
        let data = Array3::zeros((1, 1, 1));
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        assert!(derivative(&cube, 1).is_err());
    }
}
