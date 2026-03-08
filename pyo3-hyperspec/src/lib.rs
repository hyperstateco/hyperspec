use ndarray::{Array1, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;

use hyperspec::{
    EnviWriteDataType, EnviWriteOptions, Interleave, MnfResult, PcaResult, ResampleMethod,
    SpectralCube, ZarrCompression, ZarrReadOptions, ZarrWriteOptions, band_ratio as rs_band_ratio,
    continuum_removal as rs_continuum_removal, mnf as rs_mnf, mnf_denoise as rs_mnf_denoise,
    ndvi as rs_ndvi, normalized_difference as rs_normalized_difference, pca as rs_pca,
    pca_inverse as rs_pca_inverse, pca_transform as rs_pca_transform, read_envi as rs_read_envi,
    read_zarr as rs_read_zarr, read_zarr_window as rs_read_zarr_window,
    read_zarr_with_options as rs_read_zarr_with_options, resample as rs_resample, sam as rs_sam,
    write_envi_with_options as rs_write_envi_with_options,
    write_zarr_with_options as rs_write_zarr_with_options, zarr_cube_shape as rs_zarr_cube_shape,
};

/// Python wrapper around `hyperspec::SpectralCube`.
#[pyclass(name = "SpectralCube", frozen, from_py_object)]
#[derive(Clone)]
struct PySpectralCube {
    inner: SpectralCube,
}

#[pymethods]
impl PySpectralCube {
    /// Create a new SpectralCube from numpy arrays.
    #[new]
    #[pyo3(signature = (data, wavelengths, fwhm=None, nodata=None))]
    fn new(
        data: PyReadonlyArray3<f64>,
        wavelengths: PyReadonlyArray1<f64>,
        fwhm: Option<PyReadonlyArray1<f64>>,
        nodata: Option<f64>,
    ) -> PyResult<Self> {
        let data: Array3<f64> = data.as_array().to_owned();
        let wl: Array1<f64> = wavelengths.as_array().to_owned();
        let fwhm: Option<Array1<f64>> = fwhm.map(|f| f.as_array().to_owned());

        let inner = SpectralCube::new(data, wl, fwhm, nodata).map_err(to_value_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn bands(&self) -> usize {
        self.inner.bands()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        (self.inner.bands(), self.inner.height(), self.inner.width())
    }

    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        PyArray3::from_owned_array(py, self.inner.data().clone())
    }

    fn wavelengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.wavelengths().clone())
    }

    fn fwhm<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .fwhm()
            .map(|f| PyArray1::from_owned_array(py, f.clone()))
    }

    #[getter]
    fn nodata(&self) -> Option<f64> {
        self.inner.nodata()
    }

    fn spectrum<'py>(
        &self,
        py: Python<'py>,
        row: usize,
        col: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let spec = self.inner.spectrum(row, col).map_err(to_value_err)?;
        Ok(PyArray1::from_owned_array(py, spec))
    }

    /// Get band data (2D array) by index.
    fn band<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let b = self.inner.band(index).map_err(to_value_err)?;
        Ok(PyArray2::from_owned_array(py, b))
    }

    /// Get wavelength value at a band index.
    fn wavelength(&self, index: usize) -> PyResult<f64> {
        self.inner.wavelength(index).map_err(to_value_err)
    }

    /// Compute mean spectrum across all pixels.
    fn mean_spectrum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.mean_spectrum())
    }

    /// Find band index nearest to a wavelength (nm).
    fn nearest_band_index(&self, nm: f64) -> PyResult<usize> {
        self.inner.nearest_band_index(nm).map_err(to_value_err)
    }

    /// Find band index with exact wavelength match.
    fn wavelength_index(&self, nm: f64) -> PyResult<usize> {
        self.inner.wavelength_index(nm).map_err(to_value_err)
    }

    /// Get band data (2D array) at exact wavelength match.
    fn band_at<'py>(&self, py: Python<'py>, nm: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let b = self.inner.band_at(nm).map_err(to_value_err)?;
        Ok(PyArray2::from_owned_array(py, b))
    }

    /// Get band data (2D array) at nearest wavelength.
    fn band_nearest<'py>(&self, py: Python<'py>, nm: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let b = self.inner.band_nearest(nm).map_err(to_value_err)?;
        Ok(PyArray2::from_owned_array(py, b))
    }

    /// Subset to specific band indices (must be in increasing spectral order).
    fn sel_bands(&self, indices: Vec<usize>) -> PyResult<Self> {
        let inner = self.inner.sel_bands(&indices).map_err(to_value_err)?;
        Ok(Self { inner })
    }

    /// Subset to bands within [min_nm, max_nm].
    fn sel_wavelengths(&self, min_nm: f64, max_nm: f64) -> PyResult<Self> {
        let inner = self
            .inner
            .sel_wavelengths(min_nm, max_nm)
            .map_err(to_value_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectralCube(bands={}, height={}, width={})",
            self.inner.bands(),
            self.inner.height(),
            self.inner.width()
        )
    }
}

/// Python wrapper around `hyperspec::PcaResult`.
#[pyclass(name = "PcaResult", frozen, from_py_object)]
#[derive(Clone)]
struct PyPcaResult {
    inner: PcaResult,
}

#[pymethods]
impl PyPcaResult {
    /// Principal components as row vectors, shape (n_components, bands).
    fn components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.inner.components.clone())
    }

    /// Variance explained by each component.
    fn explained_variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.explained_variance.clone())
    }

    /// Mean spectrum subtracted before PCA.
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.mean.clone())
    }

    /// Original wavelengths, preserved for inverse transform.
    fn wavelengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.wavelengths.clone())
    }

    /// Number of components.
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.components.nrows()
    }

    fn __repr__(&self) -> String {
        let total: f64 = self.inner.explained_variance.iter().sum();
        let first_pct = if total > 0.0 {
            self.inner.explained_variance[0] / total * 100.0
        } else {
            0.0
        };
        format!(
            "PcaResult(n_components={}, first_component={:.1}%)",
            self.inner.components.nrows(),
            first_pct
        )
    }
}

fn to_value_err(e: hyperspec::HyperspecError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

// --- Algorithm functions ---

#[pyfunction]
fn sam<'py>(
    py: Python<'py>,
    cube: &PySpectralCube,
    reference: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ref_arr: Array1<f64> = reference.as_array().to_owned();
    let result = rs_sam(&cube.inner, &ref_arr).map_err(to_value_err)?;
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
fn continuum_removal(cube: &PySpectralCube) -> PyResult<PySpectralCube> {
    let inner = rs_continuum_removal(&cube.inner).map_err(to_value_err)?;
    Ok(PySpectralCube { inner })
}

#[pyfunction]
fn normalized_difference<'py>(
    py: Python<'py>,
    cube: &PySpectralCube,
    band_a: usize,
    band_b: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = rs_normalized_difference(&cube.inner, band_a, band_b).map_err(to_value_err)?;
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
fn band_ratio<'py>(
    py: Python<'py>,
    cube: &PySpectralCube,
    band_a: usize,
    band_b: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = rs_band_ratio(&cube.inner, band_a, band_b).map_err(to_value_err)?;
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
fn ndvi<'py>(
    py: Python<'py>,
    cube: &PySpectralCube,
    nir_band: usize,
    red_band: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = rs_ndvi(&cube.inner, nir_band, red_band).map_err(to_value_err)?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Compute PCA on a spectral cube.
///
/// Args:
///     cube: A SpectralCube instance.
///     n_components: Number of components to keep. If None, keeps all.
///
/// Returns:
///     A PcaResult with components, explained_variance, and mean.
#[pyfunction]
#[pyo3(signature = (cube, n_components=None))]
fn pca(cube: &PySpectralCube, n_components: Option<usize>) -> PyResult<PyPcaResult> {
    let inner = rs_pca(&cube.inner, n_components).map_err(to_value_err)?;
    Ok(PyPcaResult { inner })
}

/// Project a cube into PCA space.
///
/// Returns:
///     A numpy array of shape (n_components, height, width) with PC scores.
#[pyfunction]
fn pca_transform<'py>(
    py: Python<'py>,
    cube: &PySpectralCube,
    pca_result: &PyPcaResult,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let scores = rs_pca_transform(&cube.inner, &pca_result.inner).map_err(to_value_err)?;
    Ok(PyArray3::from_owned_array(py, scores))
}

/// Reconstruct a cube from PCA scores.
///
/// Args:
///     scores: numpy array of shape (n_components, height, width).
///     pca_result: PcaResult from a previous pca() call.
///
/// Returns:
///     A SpectralCube approximating the original data.
#[pyfunction]
fn pca_inverse(
    scores: PyReadonlyArray3<f64>,
    pca_result: &PyPcaResult,
) -> PyResult<PySpectralCube> {
    let scores_arr = scores.as_array().to_owned();
    let inner = rs_pca_inverse(&scores_arr, &pca_result.inner).map_err(to_value_err)?;
    Ok(PySpectralCube { inner })
}

/// Python wrapper around `hyperspec::MnfResult`.
#[pyclass(name = "MnfResult", frozen, from_py_object)]
#[derive(Clone)]
struct PyMnfResult {
    inner: MnfResult,
}

#[pymethods]
impl PyMnfResult {
    fn components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.inner.components.clone())
    }

    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.eigenvalues.clone())
    }

    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.mean.clone())
    }

    fn wavelengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, self.inner.wavelengths.clone())
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.inner.components.nrows()
    }

    fn __repr__(&self) -> String {
        format!("MnfResult(n_components={})", self.inner.components.nrows())
    }
}

/// Compute the Minimum Noise Fraction transform.
#[pyfunction]
#[pyo3(signature = (cube, n_components=None))]
fn mnf(cube: &PySpectralCube, n_components: Option<usize>) -> PyResult<PyMnfResult> {
    let inner = rs_mnf(&cube.inner, n_components).map_err(to_value_err)?;
    Ok(PyMnfResult { inner })
}

/// Denoise a cube via MNF: forward transform, truncate, inverse.
#[pyfunction]
fn mnf_denoise(cube: &PySpectralCube, n_components: usize) -> PyResult<PySpectralCube> {
    let inner = rs_mnf_denoise(&cube.inner, n_components).map_err(to_value_err)?;
    Ok(PySpectralCube { inner })
}

/// Resample a cube to target wavelengths.
///
/// Args:
///     cube: Source SpectralCube.
///     target_wavelengths: 1D numpy array of target wavelengths (strictly increasing).
///     method: "linear" or "cubic".
#[pyfunction]
fn resample(
    cube: &PySpectralCube,
    target_wavelengths: PyReadonlyArray1<f64>,
    method: &str,
) -> PyResult<PySpectralCube> {
    let target = target_wavelengths.as_array().to_owned();
    let m = match method {
        "linear" => ResampleMethod::Linear,
        "cubic" => ResampleMethod::Cubic,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown method '{}', expected 'linear' or 'cubic'",
                method
            )));
        }
    };
    let inner = rs_resample(&cube.inner, &target, m).map_err(to_value_err)?;
    Ok(PySpectralCube { inner })
}

// --- I/O functions ---

#[pyfunction]
fn read_envi(path: &str) -> PyResult<PySpectralCube> {
    let inner = rs_read_envi(path).map_err(to_value_err)?;
    Ok(PySpectralCube { inner })
}

#[pyfunction]
#[pyo3(signature = (cube, path, data_type=None, interleave=None, force=false))]
fn write_envi(
    cube: &PySpectralCube,
    path: &str,
    data_type: Option<&str>,
    interleave: Option<&str>,
    force: bool,
) -> PyResult<()> {
    let dt = match data_type {
        None | Some("f64") => EnviWriteDataType::F64,
        Some("f32") => EnviWriteDataType::F32,
        Some("u8") => EnviWriteDataType::U8,
        Some("i16") => EnviWriteDataType::I16,
        Some("u16") => EnviWriteDataType::U16,
        Some("i32") => EnviWriteDataType::I32,
        Some("u32") => EnviWriteDataType::U32,
        Some("i64") => EnviWriteDataType::I64,
        Some("u64") => EnviWriteDataType::U64,
        Some(s) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown data_type '{s}', expected one of: u8, i16, u16, i32, u32, f32, f64, i64, u64"
            )));
        }
    };
    let il = match interleave {
        None | Some("bsq") => Interleave::Bsq,
        Some("bil") => Interleave::Bil,
        Some("bip") => Interleave::Bip,
        Some(s) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown interleave '{s}', expected one of: bsq, bil, bip"
            )));
        }
    };
    let opts = EnviWriteOptions {
        data_type: dt,
        interleave: il,
        force,
    };
    rs_write_envi_with_options(&cube.inner, path, &opts).map_err(to_value_err)
}

// --- Zarr I/O functions ---

#[pyfunction]
#[pyo3(signature = (path, data_path=None, wavelength_path=None, fwhm_path=None, nodata=None, scale_factor=None, add_offset=None))]
fn read_zarr(
    path: &str,
    data_path: Option<&str>,
    wavelength_path: Option<&str>,
    fwhm_path: Option<&str>,
    nodata: Option<f64>,
    scale_factor: Option<f64>,
    add_offset: Option<f64>,
) -> PyResult<PySpectralCube> {
    let inner = if let (Some(dp), Some(wp)) = (data_path, wavelength_path) {
        let opts = ZarrReadOptions {
            data_path: dp.to_string(),
            wavelength_path: wp.to_string(),
            fwhm_path: fwhm_path.map(|s| s.to_string()),
            nodata,
            scale_factor,
            add_offset,
        };
        rs_read_zarr_with_options(path, &opts).map_err(to_value_err)?
    } else {
        rs_read_zarr(path).map_err(to_value_err)?
    };
    Ok(PySpectralCube { inner })
}

/// Query the shape of a Zarr cube without loading data.
#[pyfunction]
fn zarr_cube_shape(path: &str) -> PyResult<(usize, usize, usize)> {
    rs_zarr_cube_shape(path).map_err(to_value_err)
}

/// Read a spatial/spectral window from a Zarr store without loading the full cube.
#[pyfunction]
fn read_zarr_window(
    path: &str,
    bands: [usize; 2],
    rows: [usize; 2],
    cols: [usize; 2],
) -> PyResult<PySpectralCube> {
    let inner = rs_read_zarr_window(path, bands[0]..bands[1], rows[0]..rows[1], cols[0]..cols[1])
        .map_err(to_value_err)?;
    Ok(PySpectralCube { inner })
}

#[pyfunction]
#[pyo3(signature = (cube, path, compression="gzip", gzip_level=5, chunk_shape=None, overwrite=false))]
fn write_zarr(
    cube: &PySpectralCube,
    path: &str,
    compression: &str,
    gzip_level: u8,
    chunk_shape: Option<[usize; 3]>,
    overwrite: bool,
) -> PyResult<()> {
    let comp = match compression {
        "none" => ZarrCompression::None,
        "gzip" => ZarrCompression::Gzip(gzip_level),
        s => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown compression '{s}', expected 'none' or 'gzip'"
            )));
        }
    };
    let opts = ZarrWriteOptions {
        compression: comp,
        chunk_shape,
        overwrite,
    };
    rs_write_zarr_with_options(&cube.inner, path, &opts).map_err(to_value_err)
}

#[pymodule]
fn _hyperspec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PySpectralCube>()?;
    m.add_class::<PyPcaResult>()?;
    m.add_class::<PyMnfResult>()?;
    m.add_function(wrap_pyfunction!(sam, m)?)?;
    m.add_function(wrap_pyfunction!(continuum_removal, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_difference, m)?)?;
    m.add_function(wrap_pyfunction!(band_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ndvi, m)?)?;
    m.add_function(wrap_pyfunction!(pca, m)?)?;
    m.add_function(wrap_pyfunction!(pca_transform, m)?)?;
    m.add_function(wrap_pyfunction!(pca_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(mnf, m)?)?;
    m.add_function(wrap_pyfunction!(mnf_denoise, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(read_envi, m)?)?;
    m.add_function(wrap_pyfunction!(write_envi, m)?)?;
    m.add_function(wrap_pyfunction!(read_zarr, m)?)?;
    m.add_function(wrap_pyfunction!(zarr_cube_shape, m)?)?;
    m.add_function(wrap_pyfunction!(read_zarr_window, m)?)?;
    m.add_function(wrap_pyfunction!(write_zarr, m)?)?;
    Ok(())
}
