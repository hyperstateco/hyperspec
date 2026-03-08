import numpy as np
import pytest

import hyperspec
from hyperspec import (
    SpectralCube,
    band_ratio,
    band_stats,
    continuum_removal,
    correlation,
    covariance,
    derivative,
    mnf,
    mnf_denoise,
    ndvi,
    normalize_minmax,
    normalize_zscore,
    normalized_difference,
    pca,
    pca_inverse,
    pca_transform,
    resample,
    sam,
    savitzky_golay,
)


def test_version():
    assert hyperspec.__version__ == "0.7.0"


def make_cube(bands=3, height=2, width=4):
    data = np.arange(bands * height * width, dtype=np.float64).reshape(
        bands, height, width
    )
    wavelengths = np.linspace(400, 700, bands)
    return SpectralCube(data, wavelengths)


class TestSpectralCube:
    def test_creation_and_data_roundtrip(self):
        data = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
        wl = np.array([400.0, 550.0, 700.0])
        cube = SpectralCube(data, wl)

        assert cube.bands == 3
        assert cube.height == 2
        assert cube.width == 4
        assert cube.shape == (3, 2, 4)
        np.testing.assert_array_equal(cube.data(), data)
        np.testing.assert_array_equal(cube.wavelengths(), wl)
        assert cube.nodata is None
        assert cube.fwhm() is None

    def test_fwhm(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        fwhm = np.array([10.0, 12.0, 15.0])
        cube = SpectralCube(data, wl, fwhm=fwhm)
        np.testing.assert_array_equal(cube.fwhm(), fwhm)

    def test_fwhm_length_mismatch(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        fwhm = np.array([10.0, 12.0])  # wrong length
        with pytest.raises(ValueError, match="fwhm length"):
            SpectralCube(data, wl, fwhm=fwhm)

    def test_nodata(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl, nodata=-9999.0)
        assert cube.nodata == -9999.0

    def test_wavelength_mismatch(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([400.0, 500.0])  # wrong length
        with pytest.raises(ValueError, match="wavelengths length"):
            SpectralCube(data, wl)

    def test_unsorted_wavelengths(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([600.0, 400.0, 500.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            SpectralCube(data, wl)

    def test_duplicate_wavelengths(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([400.0, 500.0, 500.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            SpectralCube(data, wl)

    def test_negative_wavelengths(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([-100.0, 400.0, 500.0])
        with pytest.raises(ValueError, match="positive"):
            SpectralCube(data, wl)

    def test_nodata_nan(self):
        data = np.ones((3, 2, 4), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        with pytest.raises(ValueError, match="nodata must be finite"):
            SpectralCube(data, wl, nodata=float("nan"))

    def test_mean_spectrum_with_nodata(self):
        data = np.zeros((1, 2, 2), dtype=np.float64)
        data[0, 0, 0] = 1.0
        data[0, 0, 1] = -9999.0
        data[0, 1, 0] = 3.0
        data[0, 1, 1] = -9999.0
        wl = np.array([500.0])
        cube = SpectralCube(data, wl, nodata=-9999.0)
        mean = cube.mean_spectrum()
        np.testing.assert_allclose(mean[0], 2.0, atol=1e-10)

    def test_mean_spectrum_skips_nan(self):
        data = np.zeros((1, 2, 2), dtype=np.float64)
        data[0, 0, 0] = 2.0
        data[0, 0, 1] = float("nan")
        data[0, 1, 0] = 4.0
        data[0, 1, 1] = float("nan")
        wl = np.array([500.0])
        cube = SpectralCube(data, wl)
        mean = cube.mean_spectrum()
        np.testing.assert_allclose(mean[0], 3.0, atol=1e-10)

    def test_mean_spectrum_all_nodata(self):
        data = np.full((1, 1, 2), -9999.0, dtype=np.float64)
        wl = np.array([500.0])
        cube = SpectralCube(data, wl, nodata=-9999.0)
        mean = cube.mean_spectrum()
        assert np.isnan(mean[0])

    def test_spectrum_values(self):
        cube = make_cube()
        # band 0: [0..7], band 1: [8..15], band 2: [16..23]
        # pixel (0, 0): values at [0, 0] in each band = 0, 8, 16
        np.testing.assert_array_equal(cube.spectrum(0, 0), [0.0, 8.0, 16.0])
        # pixel (1, 3): last pixel in second row = 7, 15, 23
        np.testing.assert_array_equal(cube.spectrum(1, 3), [7.0, 15.0, 23.0])

    def test_spectrum_out_of_bounds(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="out of bounds"):
            cube.spectrum(10, 0)

    def test_band(self):
        cube = make_cube()
        b = cube.band(1)
        assert b.shape == (2, 4)
        assert b[0, 0] == 8.0  # band 1 of make_cube

    def test_wavelength_at_index(self):
        cube = make_cube()
        assert cube.wavelength(0) == 400.0
        assert cube.wavelength(2) == 700.0

    def test_mean_spectrum(self):
        cube = make_cube()
        mean = cube.mean_spectrum()
        assert mean.shape == (3,)
        # band 0: arange 0..7, mean = 3.5
        np.testing.assert_allclose(mean[0], 3.5, atol=1e-10)

    def test_nearest_band_index(self):
        cube = make_cube()
        # wavelengths are [400, 550, 700]
        assert cube.nearest_band_index(400.0) == 0
        assert cube.nearest_band_index(700.0) == 2
        assert cube.nearest_band_index(560.0) == 1

    def test_wavelength_index_exact(self):
        cube = make_cube()
        assert cube.wavelength_index(550.0) == 1

    def test_wavelength_index_no_match(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="no band at exactly"):
            cube.wavelength_index(551.0)

    def test_band_at_exact(self):
        cube = make_cube()
        b = cube.band_at(550.0)
        assert b.shape == (2, 4)
        assert b[0, 0] == 8.0

    def test_band_at_no_match(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="no band at exactly"):
            cube.band_at(551.0)

    def test_band_nearest(self):
        cube = make_cube()
        # 560 closer to 550 than 700
        b = cube.band_nearest(560.0)
        assert b[0, 0] == 8.0

    def test_sel_wavelengths(self):
        cube = make_cube(bands=5)
        # wavelengths: [400, 475, 550, 625, 700]
        sub = cube.sel_wavelengths(450.0, 600.0)
        assert sub.bands == 2
        np.testing.assert_allclose(sub.wavelengths(), [475.0, 550.0])
        # verify spatial data is preserved
        assert sub.height == cube.height
        assert sub.width == cube.width

    def test_sel_wavelengths_empty_range(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="no bands"):
            cube.sel_wavelengths(800.0, 900.0)

    def test_sel_wavelengths_inverted_range(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="min_nm"):
            cube.sel_wavelengths(700.0, 400.0)

    def test_sel_bands(self):
        cube = make_cube()
        sub = cube.sel_bands([0, 2])
        assert sub.bands == 2
        np.testing.assert_array_equal(sub.wavelengths(), [400.0, 700.0])

    def test_sel_bands_out_of_range(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="out of range"):
            cube.sel_bands([0, 10])

    def test_sel_bands_empty(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="non-empty"):
            cube.sel_bands([])

    def test_repr(self):
        cube = make_cube()
        r = repr(cube)
        assert "SpectralCube" in r
        assert "bands=3" in r
        assert "height=2" in r
        assert "width=4" in r


class TestSAM:
    def test_identical(self):
        data = np.ones((3, 2, 2), dtype=np.float64)
        data[0] = 1.0
        data[1] = 2.0
        data[2] = 3.0
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)

        result = sam(cube, np.array([1.0, 2.0, 3.0]))
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_orthogonal(self):
        data = np.zeros((3, 1, 1), dtype=np.float64)
        data[0, 0, 0] = 1.0
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)

        result = sam(cube, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(result[0, 0], np.pi / 2, atol=1e-10)

    def test_known_angle(self):
        # [1, 0] vs [1, 1] in 2-band space → angle = 45° = π/4
        data = np.zeros((2, 1, 1), dtype=np.float64)
        data[0, 0, 0] = 1.0  # [1, 0]
        wl = np.array([400.0, 500.0])
        cube = SpectralCube(data, wl)

        result = sam(cube, np.array([1.0, 1.0]))
        np.testing.assert_allclose(result[0, 0], np.pi / 4, atol=1e-10)

    def test_scale_invariant(self):
        data = np.array([[[1.0]], [[2.0]], [[3.0]]])
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)

        r1 = sam(cube, np.array([1.0, 2.0, 3.0]))
        r2 = sam(cube, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_allclose(r1, r2, atol=1e-10)

    def test_zero_pixel(self):
        data = np.zeros((3, 1, 1), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)

        result = sam(cube, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result[0, 0], np.pi / 2, atol=1e-10)

    def test_zero_reference(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="all zeros"):
            sam(cube, np.zeros(3))

    def test_dimension_mismatch(self):
        cube = make_cube()
        with pytest.raises(ValueError, match="does not match"):
            sam(cube, np.array([1.0, 2.0]))

    def test_larger_cube(self):
        # Smoke test: enough rows to exercise rayon parallelism
        bands, height, width = 50, 100, 100
        data = np.random.default_rng(42).random((bands, height, width))
        wl = np.linspace(400, 2500, bands)
        cube = SpectralCube(data, wl)
        ref = data[:, 0, 0].copy()  # use first pixel as reference

        result = sam(cube, ref)
        assert result.shape == (height, width)
        # first pixel vs itself should be ~0
        assert result[0, 0] < 1e-10
        # all values should be valid angles in [0, π/2]
        assert np.all(result >= 0)
        assert np.all(result <= np.pi / 2 + 1e-10)


class TestContinuumRemoval:
    def test_flat_spectrum(self):
        data = np.full((5, 2, 2), 0.5, dtype=np.float64)
        wl = np.linspace(400, 800, 5)
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        np.testing.assert_allclose(result.data(), 1.0, atol=1e-10)

    def test_absorption_known_value(self):
        # [1.0, 0.5, 1.0] → hull = [1.0, 1.0, 1.0] → removal = [1.0, 0.5, 1.0]
        data = np.array([[[1.0]], [[0.5]], [[1.0]]])
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        np.testing.assert_allclose(result.data()[:, 0, 0], [1.0, 0.5, 1.0], atol=1e-10)

    def test_monotonically_increasing(self):
        # All points on the hull → all 1.0
        data = np.array([[[0.2]], [[0.4]], [[0.6]], [[0.8]]])
        wl = np.array([400.0, 500.0, 600.0, 700.0])
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        np.testing.assert_allclose(result.data(), 1.0, atol=1e-10)

    def test_output_range(self):
        data = np.abs(np.random.default_rng(42).random((10, 3, 3))) + 0.01
        wl = np.linspace(400, 2500, 10)
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        assert np.all(result.data() >= 0.0)
        assert np.all(result.data() <= 1.0)

    def test_negative_values_clamped_to_zero(self):
        # Input: [1.0, -0.5, 1.0] → clamped to [1.0, 0.0, 1.0]
        # Hull: [1.0, 1.0, 1.0] → removal: [1.0, 0.0, 1.0]
        data = np.array([[[1.0]], [[-0.5]], [[1.0]]])
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        np.testing.assert_allclose(result.data()[:, 0, 0], [1.0, 0.0, 1.0], atol=1e-10)

    def test_preserves_metadata(self):
        data = np.ones((5, 2, 3), dtype=np.float64)
        wl = np.linspace(400, 800, 5)
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        assert result.shape == cube.shape
        np.testing.assert_array_equal(result.wavelengths(), cube.wavelengths())

    def test_zero_spectrum(self):
        data = np.zeros((3, 1, 1), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        result = continuum_removal(cube)
        np.testing.assert_array_equal(result.data(), 0.0)


class TestIndices:
    def _make_index_cube(self):
        # band 0 (red) = 0.2, band 1 (nir) = 0.8
        data = np.zeros((2, 2, 2), dtype=np.float64)
        data[0] = 0.2
        data[1] = 0.8
        wl = np.array([650.0, 860.0])
        return SpectralCube(data, wl)

    def test_normalized_difference(self):
        cube = self._make_index_cube()
        # (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        result = normalized_difference(cube, 1, 0)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 0.6, atol=1e-10)

    def test_normalized_difference_zero_sum(self):
        data = np.zeros((2, 1, 1), dtype=np.float64)
        data[0, 0, 0] = 1.0
        data[1, 0, 0] = -1.0
        wl = np.array([400.0, 500.0])
        cube = SpectralCube(data, wl)
        result = normalized_difference(cube, 0, 1)
        assert result[0, 0] == 0.0

    def test_band_ratio(self):
        cube = self._make_index_cube()
        # 0.8 / 0.2 = 4.0
        result = band_ratio(cube, 1, 0)
        np.testing.assert_allclose(result, 4.0, atol=1e-10)

    def test_band_ratio_zero_denominator(self):
        data = np.zeros((2, 1, 1), dtype=np.float64)
        data[0, 0, 0] = 1.0
        wl = np.array([400.0, 500.0])
        cube = SpectralCube(data, wl)
        result = band_ratio(cube, 0, 1)
        assert result[0, 0] == 0.0

    def test_band_out_of_range(self):
        cube = self._make_index_cube()
        with pytest.raises(ValueError, match="out of range"):
            normalized_difference(cube, 10, 0)
        with pytest.raises(ValueError, match="out of range"):
            band_ratio(cube, 0, 10)


class TestPCA:
    def _make_correlated_cube(self):
        # 3 bands, 4x4. Band 0 and 1 correlated, band 2 anti-correlated.
        rng = np.random.default_rng(42)
        base = rng.random((4, 4))
        data = np.zeros((3, 4, 4), dtype=np.float64)
        data[0] = base
        data[1] = base * 0.9 + 0.05
        data[2] = 1.0 - base
        wl = np.array([400.0, 500.0, 600.0])
        return SpectralCube(data, wl)

    def test_pca_basic(self):
        cube = self._make_correlated_cube()
        result = pca(cube)

        assert result.n_components == 3
        assert result.components().shape == (3, 3)
        assert result.explained_variance().shape == (3,)
        assert result.mean().shape == (3,)

        # Sorted descending
        ev = result.explained_variance()
        for i in range(1, len(ev)):
            assert ev[i - 1] >= ev[i]

    def test_pca_n_components(self):
        cube = self._make_correlated_cube()
        result = pca(cube, n_components=2)
        assert result.n_components == 2

    def test_pca_invalid_n_components(self):
        cube = self._make_correlated_cube()
        with pytest.raises(ValueError, match="n_components"):
            pca(cube, n_components=0)
        with pytest.raises(ValueError, match="n_components"):
            pca(cube, n_components=10)

    def test_pca_transform_shape(self):
        cube = self._make_correlated_cube()
        result = pca(cube, n_components=2)
        scores = pca_transform(cube, result)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2, 4, 4)

    def test_pca_roundtrip(self):
        cube = self._make_correlated_cube()
        result = pca(cube)
        scores = pca_transform(cube, result)
        reconstructed = pca_inverse(scores, result)

        np.testing.assert_allclose(
            reconstructed.data(), cube.data(), atol=1e-8
        )

    def test_pca_variance_concentration(self):
        cube = self._make_correlated_cube()
        result = pca(cube)
        ev = result.explained_variance()
        total = ev.sum()
        first_pct = ev[0] / total
        assert first_pct > 0.8, f"first component only captures {first_pct*100:.1f}%"

    def test_pca_components_orthogonal(self):
        cube = self._make_correlated_cube()
        result = pca(cube)
        comp = result.components()
        gram = comp @ comp.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)

    def test_pca_single_pixel(self):
        data = np.ones((3, 1, 1), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        with pytest.raises(ValueError, match="at least 2 pixels"):
            pca(cube)

    def test_pca_inverse_preserves_wavelengths(self):
        cube = self._make_correlated_cube()
        result = pca(cube)
        scores = pca_transform(cube, result)
        reconstructed = pca_inverse(scores, result)
        np.testing.assert_array_equal(
            reconstructed.wavelengths(), cube.wavelengths()
        )

    def test_pca_repr(self):
        cube = self._make_correlated_cube()
        result = pca(cube, n_components=2)
        r = repr(result)
        assert "PcaResult" in r
        assert "n_components=2" in r


class TestMNF:
    def _make_cube(self):
        rng = np.random.default_rng(42)
        base = rng.random((4, 4))
        data = np.zeros((3, 4, 4), dtype=np.float64)
        data[0] = base + 0.01
        data[1] = base * 0.9 + 0.05 + 0.005
        data[2] = 1.0 - base + 0.008
        wl = np.array([400.0, 500.0, 600.0])
        return SpectralCube(data, wl)

    def test_mnf_basic(self):
        cube = self._make_cube()
        result = mnf(cube)
        assert result.n_components == 3
        assert result.components().shape == (3, 3)
        assert result.eigenvalues().shape == (3,)

        # Sorted descending
        ev = result.eigenvalues()
        for i in range(1, len(ev)):
            assert ev[i - 1] >= ev[i]

    def test_mnf_n_components(self):
        cube = self._make_cube()
        result = mnf(cube, n_components=2)
        assert result.n_components == 2

    def test_mnf_denoise_shape(self):
        cube = self._make_cube()
        denoised = mnf_denoise(cube, 2)
        assert denoised.shape == cube.shape

    def test_mnf_denoise_preserves_wavelengths(self):
        cube = self._make_cube()
        denoised = mnf_denoise(cube, 2)
        np.testing.assert_array_equal(denoised.wavelengths(), cube.wavelengths())

    def test_mnf_single_pixel(self):
        data = np.ones((3, 1, 1), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        with pytest.raises(ValueError, match="at least 2 pixels"):
            mnf(cube)

    def test_mnf_narrow_cube(self):
        data = np.ones((3, 4, 1), dtype=np.float64)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        with pytest.raises(ValueError, match="width"):
            mnf(cube)

    def test_mnf_uniform_data(self):
        # All pixels identical → zero noise covariance → should error
        data = np.zeros((3, 4, 4), dtype=np.float64)
        for b in range(3):
            data[b] = float(b)
        wl = np.array([400.0, 500.0, 600.0])
        cube = SpectralCube(data, wl)
        with pytest.raises(ValueError, match="no spatial variation"):
            mnf(cube)

    def test_mnf_repr(self):
        cube = self._make_cube()
        result = mnf(cube, n_components=2)
        r = repr(result)
        assert "MnfResult" in r
        assert "n_components=2" in r


class TestResample:
    def _make_linear_cube(self):
        # 5 bands, value = wavelength / 1000
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        data = np.zeros((5, 2, 2), dtype=np.float64)
        for b in range(5):
            data[b] = wl[b] / 1000.0
        return SpectralCube(data, wl)

    def test_resample_linear_identity(self):
        cube = self._make_linear_cube()
        target = cube.wavelengths()
        resampled = resample(cube, target, "linear")
        np.testing.assert_allclose(resampled.data(), cube.data(), atol=1e-10)

    def test_resample_linear_midpoints(self):
        cube = self._make_linear_cube()
        target = np.array([450.0, 550.0, 650.0, 750.0])
        resampled = resample(cube, target, "linear")
        assert resampled.bands == 4
        np.testing.assert_allclose(resampled.data()[0, 0, 0], 0.45, atol=1e-10)
        np.testing.assert_allclose(resampled.data()[1, 0, 0], 0.55, atol=1e-10)

    def test_resample_cubic(self):
        cube = self._make_linear_cube()
        target = np.array([450.0, 550.0, 650.0, 750.0])
        resampled = resample(cube, target, "cubic")
        # Cubic on linear data should be exact
        np.testing.assert_allclose(resampled.data()[0, 0, 0], 0.45, atol=1e-8)

    def test_resample_preserves_spatial(self):
        cube = self._make_linear_cube()
        target = np.array([450.0, 650.0])
        resampled = resample(cube, target, "linear")
        assert resampled.height == cube.height
        assert resampled.width == cube.width

    def test_resample_out_of_range(self):
        cube = self._make_linear_cube()
        with pytest.raises(ValueError, match="exceeds source range"):
            resample(cube, np.array([300.0, 500.0]), "linear")

    def test_resample_unsorted_target(self):
        cube = self._make_linear_cube()
        with pytest.raises(ValueError, match="strictly increasing"):
            resample(cube, np.array([600.0, 500.0]), "linear")

    def test_resample_invalid_method(self):
        cube = self._make_linear_cube()
        with pytest.raises(ValueError, match="unknown method"):
            resample(cube, np.array([500.0]), "spline")


class TestBandStats:
    def _make_cube(self):
        data = np.zeros((3, 2, 2))
        data[0] = [[1, 2], [3, 4]]
        data[1] = [[10, 20], [30, 40]]
        data[2] = [[5, 5], [5, 5]]
        wl = np.array([450.0, 550.0, 650.0])
        return SpectralCube(data, wl)

    def test_basic(self):
        cube = self._make_cube()
        s = band_stats(cube)
        assert repr(s) == "BandStats(bands=3)"
        np.testing.assert_allclose(s.mean(), [2.5, 25.0, 5.0])
        np.testing.assert_allclose(s.min(), [1.0, 10.0, 5.0])
        np.testing.assert_allclose(s.max(), [4.0, 40.0, 5.0])
        np.testing.assert_allclose(s.valid_count(), [4, 4, 4])
        assert s.std()[2] == pytest.approx(0.0)

    def test_nodata(self):
        data = np.array([[[1.0, -9999.0, 3.0]]])
        cube = SpectralCube(data, np.array([500.0]), nodata=-9999.0)
        s = band_stats(cube)
        assert s.valid_count()[0] == 2
        assert s.mean()[0] == pytest.approx(2.0)

    def test_covariance(self):
        cube = self._make_cube()
        cov = covariance(cube)
        assert cov.shape == (3, 3)
        # Symmetric
        np.testing.assert_allclose(cov, cov.T)
        # Constant band has zero covariance
        np.testing.assert_allclose(cov[2, :], 0.0, atol=1e-10)

    def test_correlation(self):
        cube = self._make_cube()
        cor = correlation(cube)
        # Perfectly correlated bands
        assert cor[0, 1] == pytest.approx(1.0)
        # Constant band → NaN
        assert np.isnan(cor[2, 2])


class TestNormalize:
    def _make_cube(self):
        data = np.zeros((2, 1, 4))
        data[0, 0] = [0, 10, 20, 30]
        data[1, 0] = [5, 5, 5, 5]
        wl = np.array([400.0, 500.0])
        return SpectralCube(data, wl)

    def test_minmax(self):
        cube = self._make_cube()
        normed = normalize_minmax(cube)
        d = normed.data()
        np.testing.assert_allclose(d[0, 0], [0.0, 1 / 3, 2 / 3, 1.0])
        # Constant band → 0
        np.testing.assert_allclose(d[1, 0], [0.0, 0.0, 0.0, 0.0])

    def test_zscore(self):
        cube = self._make_cube()
        normed = normalize_zscore(cube)
        d = normed.data()
        # Mean should be ~0
        assert np.abs(d[0, 0].mean()) < 1e-10
        # Constant band → 0
        np.testing.assert_allclose(d[1, 0], [0.0, 0.0, 0.0, 0.0])

    def test_nan_passthrough(self):
        data = np.array([[[1.0, np.nan, 3.0]]])
        cube = SpectralCube(data, np.array([500.0]))
        normed = normalize_minmax(cube)
        assert not np.isnan(normed.data()[0, 0, 0])
        assert np.isnan(normed.data()[0, 0, 1])
        assert not np.isnan(normed.data()[0, 0, 2])

    def test_preserves_shape(self):
        cube = self._make_cube()
        assert normalize_minmax(cube).shape == cube.shape
        assert normalize_zscore(cube).shape == cube.shape

    def test_minmax_all_invalid_band(self):
        # band 0: all NaN, band 1: valid
        data = np.zeros((2, 1, 2))
        data[0] = np.nan
        data[1, 0] = [1.0, 3.0]
        cube = SpectralCube(data, np.array([400.0, 500.0]))
        normed = normalize_minmax(cube)
        assert np.all(np.isnan(normed.data()[0]))
        np.testing.assert_allclose(normed.data()[1, 0], [0.0, 1.0])

    def test_zscore_all_invalid_band(self):
        # band 0: all nodata, band 1: valid
        data = np.zeros((2, 1, 2))
        data[0] = -9999.0
        data[1, 0] = [1.0, 3.0]
        cube = SpectralCube(data, np.array([400.0, 500.0]), nodata=-9999.0)
        normed = normalize_zscore(cube)
        assert np.all(np.isnan(normed.data()[0]))
        assert not np.any(np.isnan(normed.data()[1]))


class TestDerivative:
    def test_first_order_linear(self):
        wl = np.array([400.0, 500.0, 600.0, 700.0])
        data = np.zeros((4, 1, 1))
        for b in range(4):
            data[b, 0, 0] = 2.0 * wl[b]
        cube = SpectralCube(data, wl)
        d1 = derivative(cube, 1)
        assert d1.bands == 3
        np.testing.assert_allclose(d1.data()[:, 0, 0], [2.0, 2.0, 2.0])

    def test_second_order_quadratic(self):
        wl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = (wl ** 2).reshape(5, 1, 1)
        cube = SpectralCube(data, wl)
        d2 = derivative(cube, 2)
        assert d2.bands == 3
        np.testing.assert_allclose(d2.data()[:, 0, 0], [2.0, 2.0, 2.0])

    def test_midpoint_wavelengths(self):
        wl = np.array([400.0, 500.0, 700.0])
        data = np.zeros((3, 1, 1))
        cube = SpectralCube(data, wl)
        d1 = derivative(cube, 1)
        np.testing.assert_allclose(d1.wavelengths(), [450.0, 600.0])

    def test_invalid_order(self):
        data = np.zeros((3, 1, 1))
        cube = SpectralCube(data, np.array([400.0, 500.0, 600.0]))
        with pytest.raises(ValueError):
            derivative(cube, 0)
        with pytest.raises(ValueError):
            derivative(cube, 3)

    def test_nan_propagation(self):
        data = np.zeros((3, 1, 2))
        data[:, 0, 0] = [1, np.nan, 3]
        data[:, 0, 1] = [1, 2, 3]
        cube = SpectralCube(data, np.array([400.0, 500.0, 600.0]))
        d1 = derivative(cube, 1)
        assert np.isnan(d1.data()[0, 0, 0])
        assert not np.isnan(d1.data()[0, 0, 1])

    def test_nodata_propagation(self):
        data = np.zeros((3, 1, 2))
        data[:, 0, 0] = [1, -9999, 3]
        data[:, 0, 1] = [1, 2, 3]
        cube = SpectralCube(data, np.array([400.0, 500.0, 600.0]), nodata=-9999.0)
        d1 = derivative(cube, 1)
        assert np.isnan(d1.data()[0, 0, 0])
        assert np.isnan(d1.data()[1, 0, 0])
        assert not np.isnan(d1.data()[0, 0, 1])
        # Derivative output drops nodata metadata
        assert d1.nodata is None

    def test_nonuniform_wavelengths(self):
        wl = np.array([400.0, 500.0, 700.0])
        data = np.array([100.0, 200.0, 500.0]).reshape(3, 1, 1)
        cube = SpectralCube(data, wl)
        d1 = derivative(cube, 1)
        # d[0] = (200-100)/100 = 1.0, d[1] = (500-200)/200 = 1.5
        np.testing.assert_allclose(d1.data()[:, 0, 0], [1.0, 1.5])
        np.testing.assert_allclose(d1.wavelengths(), [450.0, 600.0])


class TestSavitzkyGolay:
    def _make_cube(self, n_bands=20):
        wl = np.linspace(400, 800, n_bands)
        data = np.ones((n_bands, 1, 1))
        return SpectralCube(data, wl)

    def test_flat_preserved(self):
        cube = self._make_cube()
        smooth = savitzky_golay(cube, 5, 2)
        np.testing.assert_allclose(smooth.data()[:, 0, 0], 1.0, atol=1e-10)

    def test_reduces_noise(self):
        wl = np.linspace(400, 800, 20)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.1, 20)
        data = (1.0 + noise).reshape(20, 1, 1)
        cube = SpectralCube(data, wl)
        smooth = savitzky_golay(cube, 5, 2)
        orig_var = np.var(data[:, 0, 0] - 1.0)
        smooth_var = np.var(smooth.data()[:, 0, 0] - 1.0)
        assert smooth_var < orig_var

    def test_preserves_shape(self):
        cube = self._make_cube()
        smooth = savitzky_golay(cube, 5, 2)
        assert smooth.shape == cube.shape
        np.testing.assert_array_equal(smooth.wavelengths(), cube.wavelengths())

    def test_invalid_params(self):
        cube = self._make_cube()
        with pytest.raises(ValueError):
            savitzky_golay(cube, 4, 2)  # even window
        with pytest.raises(ValueError):
            savitzky_golay(cube, 5, 5)  # polyorder >= window
        with pytest.raises(ValueError):
            savitzky_golay(cube, 21, 2)  # window > bands

    def test_nan_pixel(self):
        wl = np.linspace(400, 800, 10)
        data = np.ones((10, 1, 2))
        data[3, 0, 0] = np.nan
        cube = SpectralCube(data, wl)
        smooth = savitzky_golay(cube, 5, 2)
        assert all(np.isnan(smooth.data()[b, 0, 0]) for b in range(10))
        assert all(not np.isnan(smooth.data()[b, 0, 1]) for b in range(10))
