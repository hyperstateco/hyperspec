"""Type stubs for hyperspec — fast processing library for hyperspectral imagery."""

import numpy as np
import numpy.typing as npt

__version__: str

class SpectralCube:
    """A 3D spectral data cube with wavelength metadata.

    Data is stored as a (bands, height, width) array of float64 values.

    Args:
        data: 3D array of shape (bands, height, width).
        wavelengths: 1D array of wavelength values in nanometers, length must equal bands.
        fwhm: Optional 1D array of full-width half-maximum values per band.
        nodata: Optional nodata sentinel value. Pixels with this value are excluded from computations.
    """

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        wavelengths: npt.NDArray[np.float64],
        fwhm: npt.NDArray[np.float64] | None = None,
        nodata: float | None = None,
    ) -> None: ...

    @property
    def bands(self) -> int:
        """Number of spectral bands."""
        ...

    @property
    def height(self) -> int:
        """Number of rows (spatial y-dimension)."""
        ...

    @property
    def width(self) -> int:
        """Number of columns (spatial x-dimension)."""
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as (bands, height, width)."""
        ...

    @property
    def nodata(self) -> float | None:
        """Nodata sentinel value, or None if unset."""
        ...

    def data(self) -> npt.NDArray[np.float64]:
        """Return the full data array, shape (bands, height, width)."""
        ...

    def wavelengths(self) -> npt.NDArray[np.float64]:
        """Return wavelength values in nm, shape (bands,)."""
        ...

    def fwhm(self) -> npt.NDArray[np.float64] | None:
        """Return FWHM values per band, or None if unset."""
        ...

    def spectrum(self, row: int, col: int) -> npt.NDArray[np.float64]:
        """Extract the spectrum at a single pixel, shape (bands,).

        Raises:
            ValueError: If row or col is out of bounds.
        """
        ...

    def band(self, index: int) -> npt.NDArray[np.float64]:
        """Extract a single band image by index, shape (height, width).

        Raises:
            ValueError: If index is out of bounds.
        """
        ...

    def wavelength(self, index: int) -> float:
        """Get the wavelength value at a band index.

        Raises:
            ValueError: If index is out of bounds.
        """
        ...

    def mean_spectrum(self) -> npt.NDArray[np.float64]:
        """Compute the mean spectrum across all spatial pixels, shape (bands,)."""
        ...

    def nearest_band_index(self, nm: float) -> int:
        """Find the band index with the wavelength nearest to ``nm``.

        Raises:
            ValueError: If the cube has no bands.
        """
        ...

    def wavelength_index(self, nm: float) -> int:
        """Find the band index with an exact wavelength match.

        Raises:
            ValueError: If no band has wavelength exactly equal to ``nm``.
        """
        ...

    def band_at(self, nm: float) -> npt.NDArray[np.float64]:
        """Get band image at an exact wavelength match, shape (height, width).

        Raises:
            ValueError: If no band has wavelength exactly equal to ``nm``.
        """
        ...

    def band_nearest(self, nm: float) -> npt.NDArray[np.float64]:
        """Get band image at the nearest wavelength, shape (height, width).

        Raises:
            ValueError: If the cube has no bands.
        """
        ...

    def sel_bands(self, indices: list[int]) -> SpectralCube:
        """Subset the cube to specific band indices.

        Indices must be in strictly increasing order.

        Raises:
            ValueError: If indices are out of bounds or not increasing.
        """
        ...

    def sel_wavelengths(self, min_nm: float, max_nm: float) -> SpectralCube:
        """Subset the cube to bands within a wavelength range [min_nm, max_nm].

        Raises:
            ValueError: If no bands fall within the range.
        """
        ...

class PcaResult:
    """Result of a PCA decomposition."""

    @property
    def n_components(self) -> int:
        """Number of principal components."""
        ...

    def components(self) -> npt.NDArray[np.float64]:
        """Principal components as row vectors, shape (n_components, bands)."""
        ...

    def explained_variance(self) -> npt.NDArray[np.float64]:
        """Variance explained by each component, shape (n_components,)."""
        ...

    def mean(self) -> npt.NDArray[np.float64]:
        """Mean spectrum subtracted before PCA, shape (bands,)."""
        ...

    def wavelengths(self) -> npt.NDArray[np.float64]:
        """Original wavelengths, preserved for inverse transform, shape (bands,)."""
        ...

class MnfResult:
    """Result of a Minimum Noise Fraction transform."""

    @property
    def n_components(self) -> int:
        """Number of MNF components."""
        ...

    def components(self) -> npt.NDArray[np.float64]:
        """MNF components as row vectors, shape (n_components, bands)."""
        ...

    def eigenvalues(self) -> npt.NDArray[np.float64]:
        """Eigenvalues for each component, shape (n_components,)."""
        ...

    def mean(self) -> npt.NDArray[np.float64]:
        """Mean spectrum, shape (bands,)."""
        ...

    def wavelengths(self) -> npt.NDArray[np.float64]:
        """Original wavelengths, shape (bands,)."""
        ...

class BandStats:
    """Per-band statistics for a spectral cube."""

    def min(self) -> npt.NDArray[np.float64]:
        """Minimum value per band, shape (bands,)."""
        ...

    def max(self) -> npt.NDArray[np.float64]:
        """Maximum value per band, shape (bands,)."""
        ...

    def mean(self) -> npt.NDArray[np.float64]:
        """Mean value per band, shape (bands,)."""
        ...

    def std(self) -> npt.NDArray[np.float64]:
        """Standard deviation per band, shape (bands,)."""
        ...

    def valid_count(self) -> npt.NDArray[np.uint64]:
        """Number of valid (non-nodata, non-NaN) pixels per band, shape (bands,)."""
        ...

# --- Spectral algorithms ---

def sam(
    cube: SpectralCube,
    reference: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Spectral Angle Mapper — angular distance between each pixel and a reference spectrum.

    Args:
        cube: Input spectral cube.
        reference: Reference spectrum, shape (bands,). Must match cube band count.

    Returns:
        Angle map in radians, shape (height, width).

    Raises:
        ValueError: If reference length does not match cube bands.
    """
    ...

def continuum_removal(cube: SpectralCube) -> SpectralCube:
    """Continuum removal via convex hull.

    Returns a new cube with reflectance values divided by the continuum envelope.

    Raises:
        ValueError: If the cube has fewer than 2 bands.
    """
    ...

def normalized_difference(
    cube: SpectralCube,
    band_a: int,
    band_b: int,
) -> npt.NDArray[np.float64]:
    """Compute (band_a - band_b) / (band_a + band_b) per pixel.

    Args:
        cube: Input spectral cube.
        band_a: Index of the first band.
        band_b: Index of the second band.

    Returns:
        Normalized difference image, shape (height, width).

    Raises:
        ValueError: If band indices are out of bounds.
    """
    ...

def band_ratio(
    cube: SpectralCube,
    band_a: int,
    band_b: int,
) -> npt.NDArray[np.float64]:
    """Compute band_a / band_b per pixel.

    Args:
        cube: Input spectral cube.
        band_a: Index of the numerator band.
        band_b: Index of the denominator band.

    Returns:
        Ratio image, shape (height, width).

    Raises:
        ValueError: If band indices are out of bounds.
    """
    ...

def ndvi(
    cube: SpectralCube,
    nir_band: int,
    red_band: int,
) -> npt.NDArray[np.float64]:
    """Compute NDVI = (NIR - Red) / (NIR + Red).

    Args:
        cube: Input spectral cube.
        nir_band: Index of the near-infrared band.
        red_band: Index of the red band.

    Returns:
        NDVI image, shape (height, width).

    Raises:
        ValueError: If band indices are out of bounds.
    """
    ...

# --- Dimensionality reduction ---

def pca(
    cube: SpectralCube,
    n_components: int | None = None,
) -> PcaResult:
    """Compute PCA on a spectral cube.

    Args:
        cube: Input spectral cube.
        n_components: Number of components to keep. If None, keeps all.

    Returns:
        PCA decomposition result.
    """
    ...

def pca_transform(
    cube: SpectralCube,
    pca_result: PcaResult,
) -> npt.NDArray[np.float64]:
    """Project a cube into PCA space.

    Args:
        cube: Input spectral cube.
        pca_result: Result from a previous ``pca()`` call.

    Returns:
        PC scores, shape (n_components, height, width).
    """
    ...

def pca_inverse(
    scores: npt.NDArray[np.float64],
    pca_result: PcaResult,
) -> SpectralCube:
    """Reconstruct a spectral cube from PCA scores.

    Args:
        scores: PC scores, shape (n_components, height, width).
        pca_result: Result from a previous ``pca()`` call.

    Returns:
        Reconstructed spectral cube.
    """
    ...

def mnf(
    cube: SpectralCube,
    n_components: int | None = None,
) -> MnfResult:
    """Compute the Minimum Noise Fraction transform.

    Args:
        cube: Input spectral cube.
        n_components: Number of components to keep. If None, keeps all.

    Returns:
        MNF decomposition result.
    """
    ...

def mnf_denoise(
    cube: SpectralCube,
    n_components: int,
) -> SpectralCube:
    """Denoise a cube via MNF: forward transform, truncate, inverse.

    Args:
        cube: Input spectral cube.
        n_components: Number of MNF components to retain.

    Returns:
        Denoised spectral cube.
    """
    ...

# --- Preprocessing ---

def resample(
    cube: SpectralCube,
    target_wavelengths: npt.NDArray[np.float64],
    method: str,
) -> SpectralCube:
    """Resample a cube to target wavelengths.

    Args:
        cube: Input spectral cube.
        target_wavelengths: 1D array of target wavelengths in nm (strictly increasing).
        method: Interpolation method — ``"linear"`` or ``"cubic"``.

    Returns:
        Resampled spectral cube.

    Raises:
        ValueError: If method is not ``"linear"`` or ``"cubic"``, or wavelengths are invalid.
    """
    ...

def normalize_minmax(cube: SpectralCube) -> SpectralCube:
    """Normalize each band to [0, 1] using per-band min/max.

    Returns:
        Normalized spectral cube.

    Raises:
        ValueError: If a band has zero range (min == max).
    """
    ...

def normalize_zscore(cube: SpectralCube) -> SpectralCube:
    """Normalize each band to zero mean and unit variance (z-score).

    Returns:
        Normalized spectral cube.

    Raises:
        ValueError: If a band has zero variance.
    """
    ...

def derivative(cube: SpectralCube, order: int) -> SpectralCube:
    """Compute spectral derivatives using finite differences.

    Args:
        cube: Input spectral cube.
        order: Derivative order (1 for first derivative, 2 for second, etc.).

    Returns:
        Derivative cube with ``bands - order`` bands.

    Raises:
        ValueError: If order is 0 or greater than ``bands - 1``.
    """
    ...

def savitzky_golay(
    cube: SpectralCube,
    window: int,
    polyorder: int,
) -> SpectralCube:
    """Apply Savitzky-Golay smoothing to each pixel spectrum.

    Args:
        cube: Input spectral cube.
        window: Window length (must be odd and > polyorder).
        polyorder: Polynomial order for the filter.

    Returns:
        Smoothed spectral cube.

    Raises:
        ValueError: If window/polyorder constraints are violated.
    """
    ...

# --- Statistics ---

def band_stats(cube: SpectralCube) -> BandStats:
    """Compute per-band statistics (min, max, mean, std, valid_count).

    Args:
        cube: Input spectral cube.

    Returns:
        Per-band statistics.
    """
    ...

def covariance(cube: SpectralCube) -> npt.NDArray[np.float64]:
    """Compute the band-to-band covariance matrix.

    Args:
        cube: Input spectral cube.

    Returns:
        Covariance matrix, shape (bands, bands).
    """
    ...

def correlation(cube: SpectralCube) -> npt.NDArray[np.float64]:
    """Compute the band-to-band correlation matrix.

    Args:
        cube: Input spectral cube.

    Returns:
        Correlation matrix, shape (bands, bands).
    """
    ...

# --- I/O ---

def read_envi(path: str) -> SpectralCube:
    """Read an ENVI file (BSQ/BIL/BIP with .hdr).

    Args:
        path: Path to the ENVI binary file (the .hdr is located automatically).

    Returns:
        Loaded spectral cube.

    Raises:
        ValueError: If the file cannot be read or the header is invalid.
    """
    ...

def write_envi(
    cube: SpectralCube,
    path: str,
    data_type: str | None = None,
    interleave: str | None = None,
    force: bool = False,
) -> None:
    """Write a spectral cube to ENVI format.

    Args:
        cube: Spectral cube to write.
        path: Output file path (a .hdr sidecar is written alongside).
        data_type: Output data type — one of ``"u8"``, ``"i16"``, ``"u16"``,
            ``"i32"``, ``"u32"``, ``"i64"``, ``"u64"``, ``"f32"``, ``"f64"``.
            Defaults to ``"f64"``.
        interleave: Band interleave — ``"bsq"``, ``"bil"``, or ``"bip"``.
            Defaults to ``"bsq"``.
        force: If True, overwrite existing files.

    Raises:
        ValueError: If data_type or interleave is invalid, or the file exists and force is False.
    """
    ...

def read_zarr(
    path: str,
    data_path: str | None = None,
    wavelength_path: str | None = None,
    fwhm_path: str | None = None,
    nodata: float | None = None,
    scale_factor: float | None = None,
    add_offset: float | None = None,
) -> SpectralCube:
    """Read a spectral cube from a Zarr V3 store.

    With no optional arguments, uses auto-discovery to locate data and wavelength arrays.
    Provide ``data_path`` and ``wavelength_path`` to specify explicit array paths within the store.

    Args:
        path: Path to the Zarr store directory.
        data_path: Path to the data array within the store.
        wavelength_path: Path to the wavelength array within the store.
        fwhm_path: Path to the FWHM array within the store.
        nodata: Nodata sentinel value.
        scale_factor: Multiplicative scale factor applied after reading.
        add_offset: Additive offset applied after reading and scaling.

    Returns:
        Loaded spectral cube.

    Raises:
        ValueError: If the store cannot be read or required arrays are missing.
    """
    ...

def zarr_cube_shape(path: str) -> tuple[int, int, int]:
    """Query the shape of a Zarr cube without loading data.

    Args:
        path: Path to the Zarr store directory.

    Returns:
        Shape as (bands, height, width).

    Raises:
        ValueError: If the store cannot be read.
    """
    ...

def read_zarr_window(
    path: str,
    bands: tuple[int, int],
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> SpectralCube:
    """Read a spatial/spectral window from a Zarr store.

    Each range is a [start, stop) half-open interval.

    Args:
        path: Path to the Zarr store directory.
        bands: Band range as (start, stop).
        rows: Row range as (start, stop).
        cols: Column range as (start, stop).

    Returns:
        Windowed spectral cube.

    Raises:
        ValueError: If the ranges are out of bounds or the store cannot be read.
    """
    ...

def write_zarr(
    cube: SpectralCube,
    path: str,
    compression: str = "gzip",
    gzip_level: int = 5,
    chunk_shape: tuple[int, int, int] | None = None,
    overwrite: bool = False,
) -> None:
    """Write a spectral cube to Zarr V3 format.

    Args:
        cube: Spectral cube to write.
        path: Output Zarr store directory path.
        compression: Compression codec — ``"none"`` or ``"gzip"``. Defaults to ``"gzip"``.
        gzip_level: Gzip compression level (0-9). Defaults to 5.
        chunk_shape: Chunk dimensions as (bands, rows, cols). If None, uses the full
            array shape. Use e.g. ``(bands, 256, 256)`` for ML-friendly spatial tiling.
        overwrite: If True, overwrite an existing store.

    Raises:
        ValueError: If compression is invalid, or the store exists and overwrite is False.
    """
    ...
