# Changelog

## 0.6.0

- **ENVI I/O**: read/write with BSQ/BIL/BIP interleave, all numeric types (u8–f64), memory-mapped reads, atomic writes
- **Zarr V3 I/O**: pure Rust read/write/window via `zarrs`; auto-discovery, dtype dispatch (f32/u16/etc. → f64), ML-friendly `chunk_shape`, windowed reads, atomic temp-dir writes
- **Python bindings**: `read_envi`, `write_envi`, `read_zarr`, `write_zarr`, `read_zarr_window`, `zarr_cube_shape`

## 0.5.0

- **Breaking**: `pca_transform` returns `Array3` / numpy array instead of `SpectralCube`; `pca_inverse` accepts `Array3`
- **Fix**: MNF denoise now uses correct inverse transform (`noise_sqrt @ Vr @ V^T @ noise_inv_sqrt`)
- **Fix**: NaN/nodata handling across all algorithms — per-pixel ops output NaN, PCA/MNF reject with error
- **Perf**: parallelized `mean_spectrum`, `normalized_difference`, `band_ratio`; MNF denoise precomputes denoise matrix

## 0.4.0

- **SpectralCube**: (bands, height, width) + wavelengths, fwhm, nodata
- **SAM**: spectral angle mapper, rayon-parallel
- **Continuum removal**: convex hull, rayon-parallel
- **Band indices**: `normalized_difference`, `band_ratio`, `ndvi`
- **PCA**: Jacobi eigendecomposition (no LAPACK), forward/inverse transform
- **MNF**: noise-whitened PCA, single-pass denoise
- **Band resampling**: linear and cubic interpolation
- **Python bindings**: all types and algorithms via PyO3 + numpy
