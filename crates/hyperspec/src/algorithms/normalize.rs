use ndarray::Array3;
use rayon::prelude::*;

use crate::algorithms::stats::band_stats;
use crate::cube::SpectralCube;
use crate::error::Result;

/// Per-band min-max normalization to [0, 1].
///
/// Each band is scaled independently: `(x - min) / (max - min)`.
/// Bands with zero or non-finite range (constant or all-invalid) are set to 0.0.
/// Invalid pixels (NaN or nodata) are written as NaN in the output;
/// the original nodata metadata is preserved for downstream I/O.
pub fn normalize_minmax(cube: &SpectralCube) -> Result<SpectralCube> {
    let stats = band_stats(cube);
    let mins = stats.min.as_slice().unwrap();
    let maxs = stats.max.as_slice().unwrap();

    // Precompute per-band scale factors
    let params: Vec<(f64, f64)> = (0..cube.bands())
        .map(|b| {
            let range = maxs[b] - mins[b];
            (mins[b], range)
        })
        .collect();

    apply_per_band(cube, &params, |v, &(mn, range)| {
        if !range.is_finite() || range == 0.0 { 0.0 } else { (v - mn) / range }
    })
}

/// Per-band z-score normalization: `(x - mean) / std`.
///
/// Each band is centered and scaled independently.
/// Bands with zero or non-finite std (constant or all-invalid) are set to 0.0.
/// Invalid pixels (NaN or nodata) are written as NaN in the output;
/// the original nodata metadata is preserved for downstream I/O.
pub fn normalize_zscore(cube: &SpectralCube) -> Result<SpectralCube> {
    let stats = band_stats(cube);
    let means = stats.mean.as_slice().unwrap();
    let stds = stats.std.as_slice().unwrap();

    let params: Vec<(f64, f64)> = (0..cube.bands())
        .map(|b| (means[b], stds[b]))
        .collect();

    apply_per_band(cube, &params, |v, &(mu, sigma)| {
        if !sigma.is_finite() || sigma == 0.0 { 0.0 } else { (v - mu) / sigma }
    })
}

/// Apply a per-band transformation with precomputed parameters.
/// NaN and nodata pixels produce NaN.
fn apply_per_band<P, F>(cube: &SpectralCube, params: &[P], f: F) -> Result<SpectralCube>
where
    P: Sync,
    F: Fn(f64, &P) -> f64 + Sync,
{
    let data = cube.data();
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let nodata = cube.nodata();

    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; bands * width];
            for col in 0..width {
                for b in 0..bands {
                    let v = data[[b, row, col]];
                    row_data[b * width + col] = if v.is_nan() || nodata == Some(v) {
                        f64::NAN
                    } else {
                        f(v, &params[b])
                    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_cube() -> SpectralCube {
        let mut data = ndarray::Array3::zeros((2, 1, 4));
        // band 0: [0, 10, 20, 30]
        data[[0, 0, 0]] = 0.0;
        data[[0, 0, 1]] = 10.0;
        data[[0, 0, 2]] = 20.0;
        data[[0, 0, 3]] = 30.0;
        // band 1: [5, 5, 5, 5] (constant)
        data[[1, 0, 0]] = 5.0;
        data[[1, 0, 1]] = 5.0;
        data[[1, 0, 2]] = 5.0;
        data[[1, 0, 3]] = 5.0;

        let wl = Array1::from_vec(vec![400.0, 500.0]);
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_minmax_basic() {
        let cube = make_cube();
        let normed = normalize_minmax(&cube).unwrap();
        let d = normed.data();
        // band 0: (x - 0) / 30 → [0.0, 1/3, 2/3, 1.0]
        assert!((d[[0, 0, 0]] - 0.0).abs() < 1e-10);
        assert!((d[[0, 0, 1]] - 1.0 / 3.0).abs() < 1e-10);
        assert!((d[[0, 0, 3]] - 1.0).abs() < 1e-10);
        // band 1: constant → all 0.0
        for c in 0..4 {
            assert!((d[[1, 0, c]] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zscore_basic() {
        let cube = make_cube();
        let normed = normalize_zscore(&cube).unwrap();
        let d = normed.data();
        // band 0: mean=15, std=sqrt(125)=11.180...
        // z(0) = (0-15)/std, z(30) = (30-15)/std = -z(0)
        assert!((d[[0, 0, 0]] + d[[0, 0, 3]]).abs() < 1e-10); // symmetric around mean
        // band 1: constant → all 0.0
        for c in 0..4 {
            assert!((d[[1, 0, c]] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zscore_mean_zero() {
        let cube = make_cube();
        let normed = normalize_zscore(&cube).unwrap();
        // After z-score, mean of band 0 should be ~0
        let sum: f64 = (0..4).map(|c| normed.data()[[0, 0, c]]).sum();
        assert!(sum.abs() < 1e-10);
    }

    #[test]
    fn test_minmax_nan() {
        let mut data = ndarray::Array3::zeros((1, 1, 3));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = f64::NAN;
        data[[0, 0, 2]] = 3.0;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let normed = normalize_minmax(&cube).unwrap();
        assert!(!normed.data()[[0, 0, 0]].is_nan());
        assert!(normed.data()[[0, 0, 1]].is_nan());
        assert!(!normed.data()[[0, 0, 2]].is_nan());
    }

    #[test]
    fn test_zscore_nodata() {
        let mut data = ndarray::Array3::zeros((1, 1, 3));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = -9999.0;
        data[[0, 0, 2]] = 3.0;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let normed = normalize_zscore(&cube).unwrap();
        assert!(!normed.data()[[0, 0, 0]].is_nan());
        assert!(normed.data()[[0, 0, 1]].is_nan());
        assert!(!normed.data()[[0, 0, 2]].is_nan());
    }

    #[test]
    fn test_minmax_all_invalid_band() {
        // band 0: all NaN, band 1: valid data
        let mut data = ndarray::Array3::zeros((2, 1, 2));
        data[[0, 0, 0]] = f64::NAN;
        data[[0, 0, 1]] = f64::NAN;
        data[[1, 0, 0]] = 1.0;
        data[[1, 0, 1]] = 3.0;
        let wl = Array1::from_vec(vec![400.0, 500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let normed = normalize_minmax(&cube).unwrap();
        // All-NaN band: stats are NaN, range is non-finite → output is NaN (invalid pixels)
        assert!(normed.data()[[0, 0, 0]].is_nan());
        assert!(normed.data()[[0, 0, 1]].is_nan());
        // Valid band still normalized correctly
        assert!((normed.data()[[1, 0, 0]] - 0.0).abs() < 1e-10);
        assert!((normed.data()[[1, 0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_all_invalid_band() {
        // band 0: all nodata, band 1: valid data
        let mut data = ndarray::Array3::zeros((2, 1, 2));
        data[[0, 0, 0]] = -9999.0;
        data[[0, 0, 1]] = -9999.0;
        data[[1, 0, 0]] = 1.0;
        data[[1, 0, 1]] = 3.0;
        let wl = Array1::from_vec(vec![400.0, 500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let normed = normalize_zscore(&cube).unwrap();
        // All-nodata band: stats are NaN, sigma is non-finite → output is NaN (invalid pixels)
        assert!(normed.data()[[0, 0, 0]].is_nan());
        assert!(normed.data()[[0, 0, 1]].is_nan());
        // Valid band still normalized
        assert!(!normed.data()[[1, 0, 0]].is_nan());
        assert!(!normed.data()[[1, 0, 1]].is_nan());
    }

    #[test]
    fn test_minmax_preserves_metadata() {
        let cube = make_cube();
        let normed = normalize_minmax(&cube).unwrap();
        assert_eq!(normed.shape(), cube.shape());
        assert_eq!(normed.wavelengths(), cube.wavelengths());
        assert_eq!(normed.nodata(), cube.nodata());
    }
}
