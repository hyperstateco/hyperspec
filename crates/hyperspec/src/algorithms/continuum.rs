use ndarray::Array3;
use rayon::prelude::*;

use crate::cube::SpectralCube;
use crate::error::Result;

/// Perform continuum removal on every pixel in the cube.
///
/// Computes the convex hull upper envelope of each pixel's spectrum, then
/// divides the spectrum by that envelope. Output values are in [0, 1], where
/// 1 means the spectrum touches the continuum and values < 1 indicate
/// absorption features.
///
/// Negative input values are clamped to 0 before computing the hull, since
/// negative reflectance is physically meaningless (sensor artifact).
///
/// Pixels containing NaN or nodata in any band produce NaN for all bands in
/// the output.
///
/// Returns a new `SpectralCube` with the same shape, wavelengths, and metadata.
pub fn continuum_removal(cube: &SpectralCube) -> Result<SpectralCube> {
    let data = cube.data();
    let bands = cube.bands();
    let height = cube.height();
    let width = cube.width();
    let nodata = cube.nodata();

    let wl = cube.wavelengths().as_standard_layout();
    let wl_slice = wl
        .as_slice()
        .expect("wavelengths contiguous after as_standard_layout");

    // Process per-row in parallel. Each row reuses spectrum/hull/continuum buffers
    // across all pixels, avoiding per-pixel allocations.
    let row_bufs: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0.0; bands * width];
            let mut spectrum = vec![0.0f64; bands];
            let mut hull_indices: Vec<usize> = Vec::with_capacity(bands);
            let mut continuum = vec![0.0f64; bands];

            for col in 0..width {
                let mut has_invalid = false;
                for b in 0..bands {
                    let v = data[[b, row, col]];
                    if v.is_nan() || nodata == Some(v) {
                        has_invalid = true;
                        break;
                    }
                    spectrum[b] = v.max(0.0);
                }

                if has_invalid {
                    for b in 0..bands {
                        row_data[b * width + col] = f64::NAN;
                    }
                } else {
                    upper_convex_hull_into(wl_slice, &spectrum, &mut hull_indices, &mut continuum);
                    for b in 0..bands {
                        row_data[b * width + col] = if continuum[b] == 0.0 {
                            0.0
                        } else {
                            spectrum[b] / continuum[b]
                        };
                    }
                }
            }
            row_data
        })
        .collect();

    let band_size = height * width;
    let mut flat = vec![0.0f64; bands * band_size];
    for (row, row_data) in row_bufs.iter().enumerate() {
        for b in 0..bands {
            let src = &row_data[b * width..(b + 1) * width];
            let dst_start = b * band_size + row * width;
            flat[dst_start..dst_start + width].copy_from_slice(src);
        }
    }

    let result = Array3::from_shape_vec((bands, height, width), flat)
        .expect("shape matches total element count");

    SpectralCube::new(
        result,
        cube.wavelengths().clone(),
        cube.fwhm().cloned(),
        cube.nodata(),
    )
}

/// Compute the upper convex hull envelope into pre-allocated buffers.
///
/// `hull_indices` and `continuum` are cleared and reused to avoid per-pixel allocation.
fn upper_convex_hull_into(
    wavelengths: &[f64],
    spectrum: &[f64],
    hull_indices: &mut Vec<usize>,
    continuum: &mut [f64],
) {
    let n = wavelengths.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        continuum[0] = spectrum[0];
        return;
    }

    // Find upper convex hull vertices (indices).
    hull_indices.clear();
    for i in 0..n {
        while hull_indices.len() >= 2 {
            let a = hull_indices[hull_indices.len() - 2];
            let b = hull_indices[hull_indices.len() - 1];
            let cross = (wavelengths[b] - wavelengths[a]) * (spectrum[i] - spectrum[a])
                - (spectrum[b] - spectrum[a]) * (wavelengths[i] - wavelengths[a]);
            if cross >= 0.0 {
                hull_indices.pop();
            } else {
                break;
            }
        }
        hull_indices.push(i);
    }

    // Linearly interpolate between hull vertices to get continuum at every band
    let mut seg = 0;
    for i in 0..n {
        while seg + 1 < hull_indices.len() - 1 && i > hull_indices[seg + 1] {
            seg += 1;
        }
        let left = hull_indices[seg];
        let right = hull_indices[seg + 1];
        let t = (wavelengths[i] - wavelengths[left]) / (wavelengths[right] - wavelengths[left]);
        continuum[i] = spectrum[left] + t * (spectrum[right] - spectrum[left]);
    }
}

/// Compute the upper convex hull envelope interpolated at each wavelength.
///
/// Uses the standard monotone chain algorithm on (wavelength, reflectance)
/// points to find the upper hull vertices, then linearly interpolates
/// between them to get the continuum value at every band.
#[cfg(test)]
fn upper_convex_hull(wavelengths: &[f64], spectrum: &[f64]) -> Vec<f64> {
    let mut hull_indices = Vec::with_capacity(wavelengths.len());
    let mut continuum = vec![0.0; wavelengths.len()];
    upper_convex_hull_into(wavelengths, spectrum, &mut hull_indices, &mut continuum);
    continuum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_flat_spectrum() {
        // Flat spectrum → continuum = spectrum → removal = all 1.0
        let data = Array3::from_shape_fn((5, 1, 1), |_| 0.5);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        for b in 0..5 {
            let v = result.data()[[b, 0, 0]];
            assert!(
                (v - 1.0).abs() < 1e-10,
                "band {}: expected 1.0, got {}",
                b,
                v
            );
        }
    }

    #[test]
    fn test_absorption_known_value() {
        // Spectrum: [1.0, 0.5, 1.0] with equal wavelength spacing
        // Continuum is a straight line from (400, 1.0) to (600, 1.0) → hull = [1.0, 1.0, 1.0]
        // Removal: [1.0, 0.5/1.0, 1.0] = [1.0, 0.5, 1.0]
        let mut data = Array3::zeros((3, 1, 1));
        data[[0, 0, 0]] = 1.0;
        data[[1, 0, 0]] = 0.5;
        data[[2, 0, 0]] = 1.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        assert!((result.data()[[0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((result.data()[[1, 0, 0]] - 0.5).abs() < 1e-10);
        assert!((result.data()[[2, 0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_monotonically_increasing() {
        // All points on the hull → removal = all 1.0
        let mut data = Array3::zeros((4, 1, 1));
        data[[0, 0, 0]] = 0.2;
        data[[1, 0, 0]] = 0.4;
        data[[2, 0, 0]] = 0.6;
        data[[3, 0, 0]] = 0.8;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        for b in 0..4 {
            assert!(
                (result.data()[[b, 0, 0]] - 1.0).abs() < 1e-10,
                "band {}: expected 1.0, got {}",
                b,
                result.data()[[b, 0, 0]]
            );
        }
    }

    #[test]
    fn test_output_range() {
        let data = Array3::from_shape_fn((5, 2, 3), |(b, r, c)| {
            ((b * 10 + r * 3 + c) as f64 * 0.17).sin().abs() + 0.1
        });
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        for &v in result.data().iter() {
            assert!((0.0..=1.0).contains(&v), "value out of range: {}", v);
        }
    }

    #[test]
    fn test_negative_values_clamped_to_zero() {
        // Negative input is treated as 0 before hull computation.
        // Input: [1.0, -0.5, 1.0] → clamped to [1.0, 0.0, 1.0]
        // Hull: straight line [1.0, 1.0, 1.0]
        // Removal: [1.0/1.0, 0.0/1.0, 1.0/1.0] = [1.0, 0.0, 1.0]
        let mut data = Array3::zeros((3, 1, 1));
        data[[0, 0, 0]] = 1.0;
        data[[1, 0, 0]] = -0.5;
        data[[2, 0, 0]] = 1.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        assert!((result.data()[[0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((result.data()[[1, 0, 0]] - 0.0).abs() < 1e-10);
        assert!((result.data()[[2, 0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_preserves_shape_and_metadata() {
        let data = Array3::from_shape_fn((5, 3, 4), |_| 1.0);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let cube = SpectralCube::new(data, wl.clone(), None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        assert_eq!(result.bands(), 5);
        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 4);
        assert_eq!(result.wavelengths(), &wl);
    }

    #[test]
    fn test_zero_spectrum() {
        let data = Array3::zeros((3, 1, 1));
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        for &v in result.data().iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_nan_pixel() {
        // Pixel (0,0) has NaN in band 1 → all bands should be NaN
        // Pixel (0,1) is valid
        let mut data = Array3::from_shape_fn((3, 1, 2), |_| 0.5);
        data[[1, 0, 0]] = f64::NAN;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let result = continuum_removal(&cube).unwrap();
        for b in 0..3 {
            assert!(result.data()[[b, 0, 0]].is_nan());
        }
        // valid pixel still produces 1.0 (flat spectrum)
        for b in 0..3 {
            assert!((result.data()[[b, 0, 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nodata_pixel() {
        let mut data = Array3::from_shape_fn((3, 1, 2), |_| 0.5);
        data[[2, 0, 0]] = -9999.0;
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let result = continuum_removal(&cube).unwrap();
        for b in 0..3 {
            assert!(result.data()[[b, 0, 0]].is_nan());
        }
        for b in 0..3 {
            assert!((result.data()[[b, 0, 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_convex_hull_known_values() {
        // Straight line from (400, 1.0) to (800, 1.0), dip in middle
        let wl = [400.0, 500.0, 600.0, 700.0, 800.0];
        let spec = [1.0, 0.5, 0.3, 0.6, 1.0];
        let hull = upper_convex_hull(&wl, &spec);

        // First and last are on the hull
        assert!((hull[0] - 1.0).abs() < 1e-10);
        assert!((hull[4] - 1.0).abs() < 1e-10);

        // Hull at band 2 (600nm): linear interp from (400,1.0) to (800,1.0) = 1.0
        assert!((hull[2] - 1.0).abs() < 1e-10);

        // All hull values >= spectrum values
        for i in 0..5 {
            assert!(hull[i] >= spec[i] - 1e-10);
        }
    }
}
