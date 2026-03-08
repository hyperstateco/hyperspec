use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

/// Compute the Spectral Angle Mapper (SAM) between every pixel in the cube and
/// a reference spectrum.
///
/// SAM measures the angle (in radians) between two spectra treated as vectors
/// in band-dimensional space. It is invariant to illumination intensity.
///
/// Returns an `Array2<f64>` of shape (height, width) with angles in [0, π] radians.
/// A value of 0 means identical spectral shape. For reflectance data (non-negative),
/// the effective range is [0, π/2]. Angles beyond π/2 indicate negative values in
/// the data.
///
/// Pixels containing NaN or nodata in any band produce NaN in the output.
pub fn sam(cube: &SpectralCube, reference: &Array1<f64>) -> Result<Array2<f64>> {
    if reference.len() != cube.bands() {
        return Err(HyperspecError::DimensionMismatch(format!(
            "reference length {} does not match cube band count {}",
            reference.len(),
            cube.bands()
        )));
    }

    let reference = reference.as_standard_layout();
    let ref_slice = reference
        .as_slice()
        .expect("reference is contiguous after as_standard_layout");

    if ref_slice.iter().any(|v| v.is_nan()) {
        return Err(HyperspecError::InvalidInput(
            "reference spectrum contains NaN".to_string(),
        ));
    }

    let ref_norm = dot(ref_slice, ref_slice).sqrt();
    if ref_norm == 0.0 {
        return Err(HyperspecError::InvalidInput(
            "reference spectrum is all zeros".to_string(),
        ));
    }

    let data = cube.data();
    let height = cube.height();
    let width = cube.width();
    let bands = cube.bands();
    let nodata = cube.nodata();

    // Parallel per-row computation
    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_result = vec![0.0; width];
            for col in 0..width {
                let mut pixel_dot_ref = 0.0;
                let mut pixel_norm_sq = 0.0;
                let mut has_invalid = false;
                for b in 0..bands {
                    let v = data[[b, row, col]];
                    if v.is_nan() || nodata == Some(v) {
                        has_invalid = true;
                        break;
                    }
                    pixel_dot_ref += v * ref_slice[b];
                    pixel_norm_sq += v * v;
                }
                if has_invalid {
                    row_result[col] = f64::NAN;
                } else {
                    let pixel_norm = pixel_norm_sq.sqrt();
                    if pixel_norm == 0.0 {
                        // Zero pixel → max angle
                        row_result[col] = std::f64::consts::FRAC_PI_2;
                    } else {
                        let cos_angle = (pixel_dot_ref / (pixel_norm * ref_norm)).clamp(-1.0, 1.0);
                        row_result[col] = cos_angle.acos();
                    }
                }
            }
            row_result
        })
        .collect();

    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((height, width), flat).expect("shape is correct"))
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    use crate::cube::SpectralCube;

    #[test]
    fn test_sam_identical() {
        // All pixels are [1, 2, 3], reference is [1, 2, 3] → angle = 0
        let data = Array3::from_shape_fn((3, 2, 2), |(b, _, _)| (b + 1) as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sam(&cube, &reference).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        for &v in result.iter() {
            assert!(v.abs() < 1e-10, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_sam_orthogonal() {
        // Pixel is [1, 0, 0], reference is [0, 1, 0] → angle = π/2
        let mut data = Array3::zeros((3, 1, 1));
        data[[0, 0, 0]] = 1.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let result = sam(&cube, &reference).unwrap();
        assert!((result[[0, 0]] - std::f64::consts::FRAC_PI_2).abs() < 1e-10,);
    }

    #[test]
    fn test_sam_scaled_invariant() {
        // SAM should be invariant to scaling
        let data = Array3::from_shape_fn((3, 1, 1), |(b, _, _)| (b + 1) as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let ref1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let ref2 = Array1::from_vec(vec![10.0, 20.0, 30.0]);

        let r1 = sam(&cube, &ref1).unwrap();
        let r2 = sam(&cube, &ref2).unwrap();
        assert!((r1[[0, 0]] - r2[[0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_sam_known_angle() {
        // [1, 0] vs [1, 1] → 45° = π/4
        let mut data = Array3::zeros((2, 1, 1));
        data[[0, 0, 0]] = 1.0;
        let wl = Array1::from_vec(vec![400.0, 500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![1.0, 1.0]);

        let result = sam(&cube, &reference).unwrap();
        assert!(
            (result[[0, 0]] - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "expected π/4, got {}",
            result[[0, 0]]
        );
    }

    #[test]
    fn test_sam_zero_reference() {
        let data = Array3::from_shape_fn((3, 1, 1), |(b, _, _)| (b + 1) as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::zeros(3);
        assert!(sam(&cube, &reference).is_err());
    }

    #[test]
    fn test_sam_dimension_mismatch() {
        let data = Array3::from_shape_fn((3, 2, 2), |(b, _, _)| b as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![1.0, 2.0]); // wrong length
        assert!(sam(&cube, &reference).is_err());
    }

    #[test]
    fn test_sam_zero_pixel() {
        // Zero pixel should give π/2
        let data = Array3::zeros((3, 1, 1));
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sam(&cube, &reference).unwrap();
        assert!((result[[0, 0]] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_sam_nan_reference() {
        let data = Array3::from_shape_fn((3, 1, 1), |(b, _, _)| (b + 1) as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(sam(&cube, &reference).is_err());
    }

    #[test]
    fn test_sam_nan_pixel() {
        let mut data = Array3::from_shape_fn((3, 1, 2), |(b, _, _)| (b + 1) as f64);
        data[[1, 0, 0]] = f64::NAN; // pixel (0,0) has NaN
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        let reference = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sam(&cube, &reference).unwrap();
        assert!(result[[0, 0]].is_nan());
        // pixel (0,1) is valid
        assert!(result[[0, 1]].abs() < 1e-10);
    }

    #[test]
    fn test_sam_nodata_pixel() {
        let mut data = Array3::from_shape_fn((3, 1, 2), |(b, _, _)| (b + 1) as f64);
        data[[2, 0, 0]] = -9999.0; // pixel (0,0) has nodata in one band
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();
        let reference = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sam(&cube, &reference).unwrap();
        assert!(result[[0, 0]].is_nan());
        // pixel (0,1) is valid
        assert!(result[[0, 1]].abs() < 1e-10);
    }
}
