# Changelog

## 0.7.2

- **Fix**: MNF `noise_cov^{-1/2}` was computing `noise_cov^{-1}`, producing incorrect eigenvalues (transforms and roundtrips were unaffected)
- **Perf**: replace Jacobi eigendecomposition with `faer` in PCA and MNF (~7x faster PCA, ~3x faster MNF)
- **Perf**: fuse stats and transform in `normalize_minmax` and `normalize_zscore` (~7x faster)
- **Perf**: eliminate intermediate allocations in `normalized_difference`, `band_ratio`, `derivative` (~5-8x faster)
- **Perf**: reuse buffers in `continuum_removal`, replace triple-nested unpacking in `savitzky_golay`, `resample`, `sam`

## 0.7.1

- **Type stubs**: `.pyi` file with full signatures, return types, and docstrings for all classes and functions — enables IDE autocomplete and mypy support

## 0.7.0

- **Statistics**: `band_stats` (per-band min/max/mean/std/valid_count), `covariance`, `correlation` — Welford's algorithm, rayon-parallel, NaN/nodata-aware, population normalization
- **Normalization**: `normalize_minmax`, `normalize_zscore` — per-band scaling with non-finite stat guards for all-invalid bands
- **Derivative spectra**: `derivative(cube, order)` — 1st/2nd order finite differences at midpoint wavelengths; nodata metadata dropped from output
- **Savitzky-Golay smoothing**: `savitzky_golay(cube, window, polyorder)` — index-space SG filter with mirror-reflection boundaries, no external deps
- **Python bindings**: `BandStats` class, all new functions exposed via PyO3

## 0.6.1

- **Fix**: build wheels against numpy 2.x C ABI for runtime compatibility with numpy 2.x

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
