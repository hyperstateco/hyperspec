use ndarray::{Array2, Zip, s};

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

/// Compute the normalized difference index between two bands: (A - B) / (A + B).
///
/// Common examples: NDVI (NIR vs Red), NDWI (Green vs NIR), NDSI (Green vs SWIR).
/// Output values are in [-1, 1]. Pixels where A + B = 0 produce 0.0.
/// Pixels where either band is NaN or nodata produce NaN.
pub fn normalized_difference(
    cube: &SpectralCube,
    band_a: usize,
    band_b: usize,
) -> Result<Array2<f64>> {
    validate_band(cube, band_a)?;
    validate_band(cube, band_b)?;

    let data = cube.data();
    let nodata = cube.nodata();
    let slice_a = data.slice(s![band_a, .., ..]);
    let slice_b = data.slice(s![band_b, .., ..]);

    let mut result = Array2::<f64>::uninit(slice_a.raw_dim());

    if let Some(nd) = nodata {
        // Slow path: must check nodata per pixel
        Zip::from(result.view_mut())
            .and(&slice_a)
            .and(&slice_b)
            .for_each(|r, &a, &b| {
                r.write(if a == nd || b == nd || a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    let sum = a + b;
                    if sum == 0.0 { 0.0 } else { (a - b) / sum }
                });
            });
    } else {
        // Fast path: NaN propagates naturally through arithmetic.
        // Only need explicit check for a + b == 0.
        Zip::from(result.view_mut())
            .and(&slice_a)
            .and(&slice_b)
            .for_each(|r, &a, &b| {
                let sum = a + b;
                // NaN + anything = NaN, so sum.is_nan covers NaN inputs.
                // sum == 0.0 is the only non-NaN special case.
                r.write(if sum == 0.0 { 0.0 } else { (a - b) / sum });
            });
    }

    // SAFETY: every element was written by the Zip above
    Ok(unsafe { result.assume_init() })
}

/// Compute the ratio of two bands: A / B.
///
/// Useful for simple mineral indices (e.g., iron oxide ratio).
/// Pixels where B = 0 produce 0.0.
/// Pixels where either band is NaN or nodata produce NaN.
pub fn band_ratio(cube: &SpectralCube, band_a: usize, band_b: usize) -> Result<Array2<f64>> {
    validate_band(cube, band_a)?;
    validate_band(cube, band_b)?;

    let data = cube.data();
    let nodata = cube.nodata();
    let slice_a = data.slice(s![band_a, .., ..]);
    let slice_b = data.slice(s![band_b, .., ..]);

    let mut result = Array2::<f64>::uninit(slice_a.raw_dim());

    if let Some(nd) = nodata {
        // Slow path: must check nodata per pixel
        Zip::from(result.view_mut())
            .and(&slice_a)
            .and(&slice_b)
            .for_each(|r, &a, &b| {
                r.write(if a == nd || b == nd || a.is_nan() || b.is_nan() {
                    f64::NAN
                } else if b == 0.0 {
                    0.0
                } else {
                    a / b
                });
            });
    } else {
        // Fast path: NaN propagates naturally through division (NaN/x = NaN, x/NaN = NaN).
        // Only need explicit check for b == 0.
        Zip::from(result.view_mut())
            .and(&slice_a)
            .and(&slice_b)
            .for_each(|r, &a, &b| {
                r.write(if b == 0.0 { 0.0 } else { a / b });
            });
    }

    // SAFETY: every element was written by the Zip above
    Ok(unsafe { result.assume_init() })
}

/// Convenience function for NDVI: normalized_difference(cube, nir_band, red_band).
///
/// NDVI = (NIR - Red) / (NIR + Red). Values near 1 indicate dense vegetation,
/// near 0 indicate bare soil, and negative values indicate water or cloud.
pub fn ndvi(cube: &SpectralCube, nir_band: usize, red_band: usize) -> Result<Array2<f64>> {
    normalized_difference(cube, nir_band, red_band)
}

fn validate_band(cube: &SpectralCube, band: usize) -> Result<()> {
    if band >= cube.bands() {
        return Err(HyperspecError::DimensionMismatch(format!(
            "band index {} out of range for cube with {} bands",
            band,
            cube.bands()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    fn make_cube() -> SpectralCube {
        // 3 bands, 2x2: band 0 = 0.2, band 1 = 0.8, band 2 = 0.4
        let mut data = Array3::zeros((3, 2, 2));
        for row in 0..2 {
            for col in 0..2 {
                data[[0, row, col]] = 0.2;
                data[[1, row, col]] = 0.8;
                data[[2, row, col]] = 0.4;
            }
        }
        let wl = Array1::from_vec(vec![650.0, 860.0, 1600.0]);
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_normalized_difference() {
        let cube = make_cube();
        // (band1 - band0) / (band1 + band0) = (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        let result = normalized_difference(&cube, 1, 0).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        for &v in result.iter() {
            assert!((v - 0.6).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalized_difference_zero_sum() {
        let mut data = Array3::zeros((2, 1, 1));
        data[[0, 0, 0]] = 1.0;
        data[[1, 0, 0]] = -1.0;
        let wl = Array1::from_vec(vec![400.0, 500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let result = normalized_difference(&cube, 0, 1).unwrap();
        assert_eq!(result[[0, 0]], 0.0);
    }

    #[test]
    fn test_band_ratio() {
        let cube = make_cube();
        // band1 / band0 = 0.8 / 0.2 = 4.0
        let result = band_ratio(&cube, 1, 0).unwrap();
        for &v in result.iter() {
            assert!((v - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_band_ratio_zero_denominator() {
        let data = Array3::zeros((2, 1, 1));
        let wl = Array1::from_vec(vec![400.0, 500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let result = band_ratio(&cube, 0, 1).unwrap();
        assert_eq!(result[[0, 0]], 0.0);
    }

    #[test]
    fn test_band_out_of_range() {
        let cube = make_cube();
        assert!(normalized_difference(&cube, 10, 0).is_err());
        assert!(band_ratio(&cube, 0, 10).is_err());
    }

    #[test]
    fn test_normalized_difference_nan_pixel() {
        let mut data = Array3::zeros((2, 1, 2));
        data[[0, 0, 0]] = f64::NAN;
        data[[1, 0, 0]] = 0.8;
        data[[0, 0, 1]] = 0.2;
        data[[1, 0, 1]] = 0.8;
        let wl = Array1::from_vec(vec![650.0, 860.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = normalized_difference(&cube, 1, 0).unwrap();
        assert!(result[[0, 0]].is_nan());
        assert!((result[[0, 1]] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_band_ratio_nan_pixel() {
        let mut data = Array3::zeros((2, 1, 2));
        data[[0, 0, 0]] = 0.2;
        data[[1, 0, 0]] = f64::NAN;
        data[[0, 0, 1]] = 0.2;
        data[[1, 0, 1]] = 0.8;
        let wl = Array1::from_vec(vec![650.0, 860.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = band_ratio(&cube, 1, 0).unwrap();
        assert!(result[[0, 0]].is_nan());
        assert!((result[[0, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_difference_nodata_pixel() {
        let mut data = Array3::zeros((2, 1, 2));
        data[[0, 0, 0]] = -9999.0;
        data[[1, 0, 0]] = 0.8;
        data[[0, 0, 1]] = 0.2;
        data[[1, 0, 1]] = 0.8;
        let wl = Array1::from_vec(vec![650.0, 860.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let result = normalized_difference(&cube, 1, 0).unwrap();
        assert!(result[[0, 0]].is_nan());
        assert!((result[[0, 1]] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_band_ratio_nodata_pixel() {
        let mut data = Array3::zeros((2, 1, 2));
        data[[0, 0, 0]] = 0.2;
        data[[1, 0, 0]] = -9999.0;
        data[[0, 0, 1]] = 0.2;
        data[[1, 0, 1]] = 0.8;
        let wl = Array1::from_vec(vec![650.0, 860.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let result = band_ratio(&cube, 1, 0).unwrap();
        assert!(result[[0, 0]].is_nan());
        assert!((result[[0, 1]] - 4.0).abs() < 1e-10);
    }
}
