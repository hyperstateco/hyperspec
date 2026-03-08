use ndarray::{Array1, Array2, Array3, ArrayView3, Axis, s};
use rayon::prelude::*;

use crate::error::{HyperspecError, Result};

/// A hyperspectral data cube: 3D array (bands, height, width) with wavelength metadata.
///
/// Wavelengths are guaranteed strictly increasing, positive, and finite.
/// Data layout is band-interleaved: `(bands, height, width)`.
#[derive(Debug, Clone)]
pub struct SpectralCube {
    /// Reflectance/radiance data with shape (bands, height, width).
    data: Array3<f64>,
    /// Center wavelength of each band in nanometers.
    wavelengths: Array1<f64>,
    /// Full-width at half-maximum for each band (optional).
    fwhm: Option<Array1<f64>>,
    /// No-data sentinel value.
    nodata: Option<f64>,
}

impl SpectralCube {
    /// Create a new SpectralCube.
    ///
    /// `data` shape must be (bands, height, width) and `wavelengths` length must equal `bands`.
    /// Wavelengths must be finite, positive, and strictly increasing.
    pub fn new(
        data: Array3<f64>,
        wavelengths: Array1<f64>,
        fwhm: Option<Array1<f64>>,
        nodata: Option<f64>,
    ) -> Result<Self> {
        let bands = data.shape()[0];

        if bands == 0 || data.shape()[1] == 0 || data.shape()[2] == 0 {
            return Err(HyperspecError::EmptyCube(format!(
                "cube dimensions must be non-zero, got shape {:?}",
                data.shape()
            )));
        }

        if wavelengths.len() != bands {
            return Err(HyperspecError::DimensionMismatch(format!(
                "wavelengths length {} does not match band count {}",
                wavelengths.len(),
                bands
            )));
        }

        if wavelengths.iter().any(|&w| !w.is_finite() || w <= 0.0) {
            return Err(HyperspecError::InvalidWavelength(
                "wavelengths must be finite and positive".to_string(),
            ));
        }

        if wavelengths
            .as_slice()
            .expect("wavelengths contiguous")
            .windows(2)
            .any(|w| w[1] <= w[0])
        {
            return Err(HyperspecError::InvalidWavelength(
                "wavelengths must be strictly increasing".to_string(),
            ));
        }

        if let Some(ref f) = fwhm {
            if f.len() != bands {
                return Err(HyperspecError::DimensionMismatch(format!(
                    "fwhm length {} does not match band count {}",
                    f.len(),
                    bands
                )));
            }
            if f.iter().any(|&v| !v.is_finite() || v <= 0.0) {
                return Err(HyperspecError::InvalidInput(
                    "fwhm values must be finite and positive".to_string(),
                ));
            }
        }

        if let Some(v) = nodata
            && !v.is_finite()
        {
            return Err(HyperspecError::InvalidInput(
                "nodata must be finite".to_string(),
            ));
        }

        Ok(Self {
            data,
            wavelengths,
            fwhm,
            nodata,
        })
    }

    // --- Getters ---

    /// Number of spectral bands.
    pub fn bands(&self) -> usize {
        self.data.shape()[0]
    }

    /// Spatial height (rows).
    pub fn height(&self) -> usize {
        self.data.shape()[1]
    }

    /// Spatial width (columns).
    pub fn width(&self) -> usize {
        self.data.shape()[2]
    }

    /// Shape as (bands, height, width).
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.bands(), self.height(), self.width())
    }

    /// Reference to the underlying 3D array.
    pub fn data(&self) -> &Array3<f64> {
        &self.data
    }

    /// Reference to the wavelength array.
    pub fn wavelengths(&self) -> &Array1<f64> {
        &self.wavelengths
    }

    /// Reference to optional FWHM array.
    pub fn fwhm(&self) -> Option<&Array1<f64>> {
        self.fwhm.as_ref()
    }

    /// No-data value.
    pub fn nodata(&self) -> Option<f64> {
        self.nodata
    }

    // --- Pixel / band access ---

    /// Extract the spectrum at a given pixel (row, col) as a 1D array of length `bands`.
    pub fn spectrum(&self, row: usize, col: usize) -> Result<Array1<f64>> {
        if row >= self.height() || col >= self.width() {
            return Err(HyperspecError::DimensionMismatch(format!(
                "pixel ({}, {}) out of bounds for cube of size ({}, {})",
                row,
                col,
                self.height(),
                self.width()
            )));
        }
        Ok(self.data.slice(s![.., row, col]).to_owned())
    }

    /// Get the band data (2D array, height × width) by band index.
    pub fn band(&self, index: usize) -> Result<Array2<f64>> {
        if index >= self.bands() {
            return Err(HyperspecError::DimensionMismatch(format!(
                "band index {} out of range for cube with {} bands",
                index,
                self.bands()
            )));
        }
        Ok(self.data.slice(s![index, .., ..]).to_owned())
    }

    /// Get the wavelength value at a band index.
    pub fn wavelength(&self, index: usize) -> Result<f64> {
        if index >= self.bands() {
            return Err(HyperspecError::DimensionMismatch(format!(
                "band index {} out of range for cube with {} bands",
                index,
                self.bands()
            )));
        }
        Ok(self.wavelengths[index])
    }

    /// Compute the mean spectrum across all pixels.
    ///
    /// Skips NaN values and nodata pixels (exact equality) per-band.
    /// Bands where all pixels are excluded produce NaN.
    pub fn mean_spectrum(&self) -> Array1<f64> {
        let nodata = self.nodata;
        let bands: Vec<f64> = (0..self.bands())
            .into_par_iter()
            .map(|b| {
                let band_slice = self.data.slice(s![b, .., ..]);
                let mut sum = 0.0;
                let mut count = 0u64;
                for &v in band_slice.iter() {
                    if v.is_nan() {
                        continue;
                    }
                    if let Some(nd) = nodata
                        && v == nd
                    {
                        continue;
                    }
                    sum += v;
                    count += 1;
                }
                if count == 0 {
                    f64::NAN
                } else {
                    sum / count as f64
                }
            })
            .collect();
        Array1::from_vec(bands)
    }

    // --- Wavelength lookup ---

    fn validate_query_wavelength(nm: f64) -> Result<()> {
        if !nm.is_finite() {
            return Err(HyperspecError::InvalidWavelength(
                "query wavelength must be finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Find the band index whose wavelength is nearest to the given value in nm.
    /// Uses binary search (wavelengths are guaranteed sorted).
    pub fn nearest_band_index(&self, nm: f64) -> Result<usize> {
        Self::validate_query_wavelength(nm)?;
        let wl = self.wavelengths.as_slice().expect("wavelengths contiguous");
        let n = wl.len();
        match wl.binary_search_by(|v| v.partial_cmp(&nm).unwrap()) {
            Ok(i) => Ok(i),
            Err(i) => {
                if i == 0 {
                    Ok(0)
                } else if i >= n {
                    Ok(n - 1)
                } else if (wl[i - 1] - nm).abs() <= (wl[i] - nm).abs() {
                    Ok(i - 1)
                } else {
                    Ok(i)
                }
            }
        }
    }

    /// Get the band data (2D array, height × width) at the nearest wavelength.
    pub fn band_nearest(&self, nm: f64) -> Result<Array2<f64>> {
        let idx = self.nearest_band_index(nm)?;
        self.band(idx)
    }

    /// Get the band data at an exact wavelength match.
    /// Returns an error if no band has exactly this wavelength.
    pub fn band_at(&self, nm: f64) -> Result<Array2<f64>> {
        let idx = self.wavelength_index(nm)?;
        self.band(idx)
    }

    /// Find the band index with an exact wavelength match.
    /// Returns an error if no band has exactly this wavelength.
    pub fn wavelength_index(&self, nm: f64) -> Result<usize> {
        Self::validate_query_wavelength(nm)?;
        let wl = self.wavelengths.as_slice().expect("wavelengths contiguous");
        match wl.binary_search_by(|v| v.partial_cmp(&nm).unwrap()) {
            Ok(i) => Ok(i),
            Err(_) => Err(HyperspecError::InvalidWavelength(format!(
                "no band at exactly {} nm",
                nm
            ))),
        }
    }

    // --- Subsetting ---

    /// Build a new cube from a pre-validated slice of band indices.
    fn subset_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let views: Vec<ArrayView3<f64>> = indices
            .iter()
            .map(|&i| self.data.slice(s![i..i + 1, .., ..]))
            .collect();
        let new_data =
            ndarray::concatenate(Axis(0), &views.iter().map(|v| v.view()).collect::<Vec<_>>())
                .expect("concatenate should not fail for valid slices");

        let new_wl = Array1::from_vec(indices.iter().map(|&i| self.wavelengths[i]).collect());
        let new_fwhm = self
            .fwhm
            .as_ref()
            .map(|f| Array1::from_vec(indices.iter().map(|&i| f[i]).collect()));

        Self::new(new_data, new_wl, new_fwhm, self.nodata)
    }

    /// Subset the cube to specific band indices. Returns a new SpectralCube.
    ///
    /// Indices must produce strictly increasing wavelengths (ascending spectral order).
    pub fn sel_bands(&self, indices: &[usize]) -> Result<Self> {
        if indices.is_empty() {
            return Err(HyperspecError::InvalidInput(
                "band indices must be non-empty".to_string(),
            ));
        }

        for &i in indices {
            if i >= self.bands() {
                return Err(HyperspecError::DimensionMismatch(format!(
                    "band index {} out of range for cube with {} bands",
                    i,
                    self.bands()
                )));
            }
        }

        // Verify indices produce strictly increasing wavelengths
        let new_wl_vec: Vec<f64> = indices.iter().map(|&i| self.wavelengths[i]).collect();
        if new_wl_vec.windows(2).any(|w| w[1] <= w[0]) {
            return Err(HyperspecError::InvalidWavelength(
                "selected bands must produce strictly increasing wavelengths".to_string(),
            ));
        }

        self.subset_by_indices(indices)
    }

    /// Subset the cube to bands whose wavelengths fall within [min_nm, max_nm].
    /// Returns a new SpectralCube.
    pub fn sel_wavelengths(&self, min_nm: f64, max_nm: f64) -> Result<Self> {
        if min_nm > max_nm {
            return Err(HyperspecError::InvalidWavelength(format!(
                "min_nm ({}) > max_nm ({})",
                min_nm, max_nm
            )));
        }

        let indices: Vec<usize> = self
            .wavelengths
            .iter()
            .enumerate()
            .filter(|&(_, wl)| *wl >= min_nm && *wl <= max_nm)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            return Err(HyperspecError::InvalidWavelength(format!(
                "no bands found in range [{}, {}]",
                min_nm, max_nm
            )));
        }

        self.subset_by_indices(&indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_cube() -> SpectralCube {
        // 3 bands, 2 rows, 4 cols
        let data = Array3::from_shape_fn((3, 2, 4), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        SpectralCube::new(data, wl, None, None).unwrap()
    }

    #[test]
    fn test_dimensions() {
        let cube = make_cube();
        assert_eq!(cube.bands(), 3);
        assert_eq!(cube.height(), 2);
        assert_eq!(cube.width(), 4);
        assert_eq!(cube.shape(), (3, 2, 4));
    }

    #[test]
    fn test_spectrum() {
        let cube = make_cube();
        let spec = cube.spectrum(1, 2).unwrap();
        assert_eq!(spec.len(), 3);
        assert_eq!(spec[0], 12.0);
        assert_eq!(spec[1], 112.0);
        assert_eq!(spec[2], 212.0);
    }

    #[test]
    fn test_spectrum_out_of_bounds() {
        let cube = make_cube();
        assert!(cube.spectrum(5, 0).is_err());
    }

    #[test]
    fn test_band() {
        let cube = make_cube();
        let b = cube.band(1).unwrap();
        assert_eq!(b.shape(), &[2, 4]);
        assert_eq!(b[[0, 0]], 100.0);
    }

    #[test]
    fn test_band_out_of_range() {
        let cube = make_cube();
        assert!(cube.band(5).is_err());
    }

    #[test]
    fn test_wavelength_at_index() {
        let cube = make_cube();
        assert_eq!(cube.wavelength(0).unwrap(), 450.0);
        assert_eq!(cube.wavelength(2).unwrap(), 650.0);
        assert!(cube.wavelength(5).is_err());
    }

    #[test]
    fn test_mean_spectrum() {
        let cube = make_cube();
        let mean = cube.mean_spectrum();
        assert_eq!(mean.len(), 3);
        // Band 0: values b*100 + r*10 + c for b=0, r in 0..2, c in 0..4
        // = 0,1,2,3,10,11,12,13 → sum=52, mean=52/8=6.5
        assert!((mean[0] - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_band_index() {
        let cube = make_cube();
        assert_eq!(cube.nearest_band_index(500.0).unwrap(), 0); // equidistant, picks first
        assert_eq!(cube.nearest_band_index(510.0).unwrap(), 1); // closer to 550
        assert_eq!(cube.nearest_band_index(440.0).unwrap(), 0);
        assert_eq!(cube.nearest_band_index(700.0).unwrap(), 2);
    }

    #[test]
    fn test_nearest_band_index_nan() {
        let cube = make_cube();
        assert!(cube.nearest_band_index(f64::NAN).is_err());
    }

    #[test]
    fn test_wavelength_index_exact() {
        let cube = make_cube();
        assert_eq!(cube.wavelength_index(550.0).unwrap(), 1);
    }

    #[test]
    fn test_wavelength_index_no_match() {
        let cube = make_cube();
        assert!(cube.wavelength_index(551.0).is_err());
    }

    #[test]
    fn test_band_at_exact() {
        let cube = make_cube();
        let b = cube.band_at(550.0).unwrap();
        assert_eq!(b[[0, 0]], 100.0);
    }

    #[test]
    fn test_band_at_no_match() {
        let cube = make_cube();
        assert!(cube.band_at(551.0).is_err());
    }

    #[test]
    fn test_band_nearest() {
        let cube = make_cube();
        // 560 closer to 550 than 650
        let b = cube.band_nearest(560.0).unwrap();
        assert_eq!(b[[0, 0]], 100.0); // band 1
    }

    #[test]
    fn test_sel_wavelengths() {
        let cube = make_cube();
        let sub = cube.sel_wavelengths(500.0, 700.0).unwrap();
        assert_eq!(sub.bands(), 2);
        assert_eq!(sub.wavelengths()[0], 550.0);
        assert_eq!(sub.wavelengths()[1], 650.0);
    }

    #[test]
    fn test_sel_wavelengths_preserves_data() {
        let cube = make_cube();
        let sub = cube.sel_wavelengths(500.0, 700.0).unwrap();
        assert_eq!(sub.height(), cube.height());
        assert_eq!(sub.width(), cube.width());
        let orig_spec = cube.spectrum(0, 0).unwrap();
        let sub_spec = sub.spectrum(0, 0).unwrap();
        assert_eq!(sub_spec[0], orig_spec[1]);
    }

    #[test]
    fn test_sel_wavelengths_empty() {
        let cube = make_cube();
        assert!(cube.sel_wavelengths(800.0, 900.0).is_err());
    }

    #[test]
    fn test_sel_wavelengths_inverted_range() {
        let cube = make_cube();
        assert!(cube.sel_wavelengths(700.0, 400.0).is_err());
    }

    #[test]
    fn test_sel_bands() {
        let cube = make_cube();
        let sub = cube.sel_bands(&[0, 2]).unwrap();
        assert_eq!(sub.bands(), 2);
        assert_eq!(sub.wavelengths()[0], 450.0);
        assert_eq!(sub.wavelengths()[1], 650.0);
        assert_eq!(
            sub.spectrum(0, 0).unwrap()[0],
            cube.spectrum(0, 0).unwrap()[0]
        );
        assert_eq!(
            sub.spectrum(0, 0).unwrap()[1],
            cube.spectrum(0, 0).unwrap()[2]
        );
    }

    #[test]
    fn test_sel_bands_out_of_range() {
        let cube = make_cube();
        assert!(cube.sel_bands(&[0, 10]).is_err());
    }

    #[test]
    fn test_sel_bands_empty() {
        let cube = make_cube();
        assert!(cube.sel_bands(&[]).is_err());
    }

    #[test]
    fn test_sel_bands_non_increasing() {
        let cube = make_cube();
        assert!(cube.sel_bands(&[2, 0]).is_err());
    }

    #[test]
    fn test_fwhm_mismatch() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let fwhm = Array1::from_vec(vec![10.0, 10.0]); // wrong length
        assert!(SpectralCube::new(data, wl, Some(fwhm), None).is_err());
    }

    #[test]
    fn test_fwhm_non_finite() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let fwhm = Array1::from_vec(vec![10.0, f64::NAN, 10.0]);
        assert!(SpectralCube::new(data, wl, Some(fwhm), None).is_err());
    }

    #[test]
    fn test_fwhm_negative() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let fwhm = Array1::from_vec(vec![10.0, -5.0, 10.0]);
        assert!(SpectralCube::new(data, wl, Some(fwhm), None).is_err());
    }

    #[test]
    fn test_wavelength_mismatch() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![450.0, 550.0]); // wrong length
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_empty_cube() {
        let data = Array3::zeros((0, 2, 4));
        let wl = Array1::from_vec(vec![]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_unsorted_wavelengths() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![600.0, 400.0, 500.0]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_duplicate_wavelengths() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![400.0, 500.0, 500.0]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_negative_wavelengths() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![-100.0, 400.0, 500.0]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_zero_wavelength() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![0.0, 400.0, 500.0]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_nan_wavelength() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![400.0, f64::NAN, 500.0]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_inf_wavelength() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![400.0, f64::INFINITY, 500.0]);
        assert!(SpectralCube::new(data, wl, None, None).is_err());
    }

    #[test]
    fn test_nodata_nan() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        assert!(SpectralCube::new(data, wl, None, Some(f64::NAN)).is_err());
    }

    #[test]
    fn test_nodata_inf() {
        let data = Array3::zeros((3, 2, 4));
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        assert!(SpectralCube::new(data, wl, None, Some(f64::INFINITY)).is_err());
    }

    #[test]
    fn test_mean_spectrum_with_nodata() {
        // 1 band, 2x2: values [1.0, -9999.0, 3.0, -9999.0]
        let mut data = Array3::zeros((1, 2, 2));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = -9999.0;
        data[[0, 1, 0]] = 3.0;
        data[[0, 1, 1]] = -9999.0;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let mean = cube.mean_spectrum();
        // Only 1.0 and 3.0 counted → mean = 2.0
        assert!((mean[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_spectrum_skips_nan() {
        let mut data = Array3::zeros((1, 2, 2));
        data[[0, 0, 0]] = 2.0;
        data[[0, 0, 1]] = f64::NAN;
        data[[0, 1, 0]] = 4.0;
        data[[0, 1, 1]] = f64::NAN;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let mean = cube.mean_spectrum();
        assert!((mean[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_spectrum_all_nodata() {
        let mut data = Array3::zeros((1, 1, 2));
        data[[0, 0, 0]] = -9999.0;
        data[[0, 0, 1]] = -9999.0;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let mean = cube.mean_spectrum();
        assert!(mean[0].is_nan());
    }
}
