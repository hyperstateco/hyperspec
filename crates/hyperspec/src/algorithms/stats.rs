use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::cube::SpectralCube;

/// Per-band descriptive statistics.
///
/// All statistics use population normalization (`/ n`), not sample (`/ (n - 1)`).
#[derive(Debug, Clone)]
pub struct BandStats {
    pub min: Array1<f64>,
    pub max: Array1<f64>,
    pub mean: Array1<f64>,
    /// Population standard deviation (`/ n`), not sample (`/ (n - 1)`).
    pub std: Array1<f64>,
    pub valid_count: Array1<u64>,
}

/// Compute per-band min, max, mean, and std. Skips NaN and nodata pixels per-band.
///
/// Each band is computed independently — a NaN in band 0 does not affect band 1's stats.
/// Bands where all pixels are excluded produce NaN for min/max/mean/std and 0 for valid_count.
pub fn band_stats(cube: &SpectralCube) -> BandStats {
    let data = cube.data();
    let bands = cube.bands();
    let nodata = cube.nodata();

    // Welford's online algorithm for numerically stable mean and variance.
    let stats: Vec<(f64, f64, f64, f64, u64)> = (0..bands)
        .into_par_iter()
        .map(|b| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let mut mean = 0.0;
            let mut m2 = 0.0;
            let mut count = 0u64;

            for &v in data.slice(ndarray::s![b, .., ..]).iter() {
                if v.is_nan() || nodata == Some(v) {
                    continue;
                }
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                count += 1;
                let delta = v - mean;
                mean += delta / count as f64;
                m2 += delta * (v - mean);
            }

            if count == 0 {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0)
            } else {
                let std = (m2 / count as f64).sqrt();
                (min, max, mean, std, count)
            }
        })
        .collect();

    let mut mins = Array1::zeros(bands);
    let mut maxs = Array1::zeros(bands);
    let mut means = Array1::zeros(bands);
    let mut stds = Array1::zeros(bands);
    let mut counts = Array1::zeros(bands);
    for (i, &(mn, mx, me, st, ct)) in stats.iter().enumerate() {
        mins[i] = mn;
        maxs[i] = mx;
        means[i] = me;
        stds[i] = st;
        counts[i] = ct;
    }

    BandStats {
        min: mins,
        max: maxs,
        mean: means,
        std: stds,
        valid_count: counts,
    }
}

/// Whole-pixel valid-mask covariance matrix, shape (bands, bands).
///
/// Uses population normalization (`/ n`), not sample (`/ (n - 1)`).
///
/// A pixel is included only if **all** bands are valid (not NaN, not nodata).
/// This differs from [`band_stats`], which excludes per-band independently.
/// Whole-pixel exclusion is required here because covariance needs paired
/// observations across all bands simultaneously.
pub fn covariance(cube: &SpectralCube) -> Array2<f64> {
    let data = cube.data();
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let nodata = cube.nodata();

    // Pass 1: compute means and count, skipping invalid pixels.
    let mut mean = vec![0.0; bands];
    let mut n = 0u64;
    for row in 0..height {
        for col in 0..width {
            if (0..bands).any(|b| {
                let v = data[[b, row, col]];
                v.is_nan() || nodata == Some(v)
            }) {
                continue;
            }
            for b in 0..bands {
                mean[b] += data[[b, row, col]];
            }
            n += 1;
        }
    }

    if n == 0 {
        return Array2::from_elem((bands, bands), f64::NAN);
    }

    for m in &mut mean {
        *m /= n as f64;
    }

    // Pass 2: accumulate covariance in parallel over rows.
    // Each row produces a partial upper-triangle sum.
    let tri_size = bands * (bands + 1) / 2;
    let row_sums: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut tri = vec![0.0; tri_size];
            for col in 0..width {
                if (0..bands).any(|b| {
                    let v = data[[b, row, col]];
                    v.is_nan() || nodata == Some(v)
                }) {
                    continue;
                }
                let mut idx = 0;
                for i in 0..bands {
                    let di = data[[i, row, col]] - mean[i];
                    for j in i..bands {
                        tri[idx] += di * (data[[j, row, col]] - mean[j]);
                        idx += 1;
                    }
                }
            }
            tri
        })
        .collect();

    // Reduce row sums
    let mut tri_total = vec![0.0; tri_size];
    for row_tri in &row_sums {
        for (i, &v) in row_tri.iter().enumerate() {
            tri_total[i] += v;
        }
    }

    // Unpack upper triangle into symmetric matrix
    let n_f = n as f64;
    let mut cov = Array2::zeros((bands, bands));
    let mut idx = 0;
    for i in 0..bands {
        for j in i..bands {
            let val = tri_total[idx] / n_f;
            cov[[i, j]] = val;
            cov[[j, i]] = val;
            idx += 1;
        }
    }
    cov
}

/// Compute the band-to-band correlation matrix, shape (bands, bands).
///
/// Derived from the covariance matrix: `cor[i,j] = cov[i,j] / (std[i] * std[j])`.
/// Diagonal is 1.0 for bands with non-zero variance.
/// Bands with zero variance produce NaN in their row/column.
pub fn correlation(cube: &SpectralCube) -> Array2<f64> {
    let cov = covariance(cube);
    let bands = cov.nrows();
    let mut cor = Array2::zeros((bands, bands));

    // When covariance is all-NaN (no valid pixels), stds will be NaN and
    // the division below naturally produces NaN — no special case needed.
    let stds: Vec<f64> = (0..bands).map(|i| cov[[i, i]].max(0.0).sqrt()).collect();

    for i in 0..bands {
        for j in 0..bands {
            let denom = stds[i] * stds[j];
            cor[[i, j]] = if denom == 0.0 {
                f64::NAN
            } else {
                cov[[i, j]] / denom
            };
        }
    }
    cor
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    use crate::cube::SpectralCube;

    fn make_cube() -> SpectralCube {
        // 3 bands, 2x2: known values
        let mut data = Array3::zeros((3, 2, 2));
        // band 0: [1, 2, 3, 4]
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = 2.0;
        data[[0, 1, 0]] = 3.0;
        data[[0, 1, 1]] = 4.0;
        // band 1: [10, 20, 30, 40]
        data[[1, 0, 0]] = 10.0;
        data[[1, 0, 1]] = 20.0;
        data[[1, 1, 0]] = 30.0;
        data[[1, 1, 1]] = 40.0;
        // band 2: [5, 5, 5, 5] (constant)
        data[[2, 0, 0]] = 5.0;
        data[[2, 0, 1]] = 5.0;
        data[[2, 1, 0]] = 5.0;
        data[[2, 1, 1]] = 5.0;

        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_band_stats_basic() {
        let cube = make_cube();
        let s = band_stats(&cube);

        assert_eq!(s.valid_count, Array1::from_vec(vec![4, 4, 4]));
        assert!((s.min[0] - 1.0).abs() < 1e-10);
        assert!((s.max[0] - 4.0).abs() < 1e-10);
        assert!((s.mean[0] - 2.5).abs() < 1e-10);
        assert!((s.min[1] - 10.0).abs() < 1e-10);
        assert!((s.max[1] - 40.0).abs() < 1e-10);
        assert!((s.mean[1] - 25.0).abs() < 1e-10);
        // band 2 is constant → std = 0
        assert!((s.mean[2] - 5.0).abs() < 1e-10);
        assert!(s.std[2].abs() < 1e-10);
    }

    #[test]
    fn test_band_stats_std() {
        let cube = make_cube();
        let s = band_stats(&cube);
        // band 0: [1,2,3,4], mean=2.5, var = ((1-2.5)^2 + ... + (4-2.5)^2)/4 = 1.25
        let expected_std = 1.25_f64.sqrt();
        assert!((s.std[0] - expected_std).abs() < 1e-10);
    }

    #[test]
    fn test_band_stats_nodata() {
        let mut data = Array3::zeros((2, 1, 3));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = -9999.0;
        data[[0, 0, 2]] = 3.0;
        data[[1, 0, 0]] = 10.0;
        data[[1, 0, 1]] = 20.0;
        data[[1, 0, 2]] = 30.0;
        let wl = Array1::from_vec(vec![400.0, 500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let s = band_stats(&cube);
        // band 0: only 1.0 and 3.0 valid
        assert_eq!(s.valid_count[0], 2);
        assert!((s.mean[0] - 2.0).abs() < 1e-10);
        // band 1: all valid
        assert_eq!(s.valid_count[1], 3);
    }

    #[test]
    fn test_band_stats_all_invalid() {
        let mut data = Array3::zeros((1, 1, 2));
        data[[0, 0, 0]] = f64::NAN;
        data[[0, 0, 1]] = f64::NAN;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let s = band_stats(&cube);
        assert_eq!(s.valid_count[0], 0);
        assert!(s.mean[0].is_nan());
        assert!(s.std[0].is_nan());
        assert!(s.min[0].is_nan());
        assert!(s.max[0].is_nan());
    }

    #[test]
    fn test_covariance_basic() {
        let cube = make_cube();
        let cov = covariance(&cube);
        assert_eq!(cov.shape(), &[3, 3]);
        // Symmetric
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
        // band 2 is constant → cov with anything = 0
        assert!(cov[[2, 0]].abs() < 1e-10);
        assert!(cov[[2, 1]].abs() < 1e-10);
        assert!(cov[[2, 2]].abs() < 1e-10);
        // cov(band0, band0) = var(band0) = 1.25
        assert!((cov[[0, 0]] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_symmetric() {
        let data = Array3::from_shape_fn((4, 3, 5), |(b, r, c)| {
            ((b * 7 + r * 13 + c * 3) as f64 * 0.37).sin()
        });
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let cov = covariance(&cube);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (cov[[i, j]] - cov[[j, i]]).abs() < 1e-10,
                    "cov not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_correlation_diagonal() {
        let cube = make_cube();
        let cor = correlation(&cube);
        // Bands 0 and 1 have non-zero variance → diagonal = 1.0
        assert!((cor[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((cor[[1, 1]] - 1.0).abs() < 1e-10);
        // Band 2 is constant → NaN
        assert!(cor[[2, 2]].is_nan());
    }

    #[test]
    fn test_correlation_perfect() {
        let cube = make_cube();
        let cor = correlation(&cube);
        // band 0 and band 1 are perfectly correlated (band1 = 10 * band0)
        assert!((cor[[0, 1]] - 1.0).abs() < 1e-10);
    }
}
