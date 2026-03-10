# hyperspec

[![Crates.io](https://img.shields.io/crates/v/hyperspec)](https://crates.io/crates/hyperspec)
[![PyPI](https://img.shields.io/pypi/v/hyperspec-py)](https://pypi.org/project/hyperspec-py/)
[![License](https://img.shields.io/crates/l/hyperspec)](LICENSE)

Fast hyperspectral data processing for remote sensing and ML pipelines. Rust core with Python bindings.

Built by [Hyperstate](https://hyperstate.co) — foundation models for Earth intelligence.

## Why hyperspec?

Hyperspectral imagery produces hundreds of bands per pixel, orders of magnitude more than RGB or multispectral. Most existing tools are slow, have heavy system dependencies, and were not designed for modern ML training pipelines.

hyperspec provides:

- Fast Rust implementations of standard hyperspectral algorithms
- A `SpectralCube` abstraction that keeps data, wavelengths, and metadata together
- Zarr V3 storage with chunking optimized for ML dataloaders
- Python bindings with zero-copy NumPy interop

## Install

Python:
```bash
pip install hyperspec-py
# or
uv add hyperspec-py
```

Rust:
```bash
cargo add hyperspec
```
or
```toml
[dependencies]
hyperspec = "0.7"
```

## Quickstart

```python
from hyperspec import (
    read_envi, write_zarr, read_zarr_window,
    band_stats, covariance,
    mnf_denoise, savitzky_golay, derivative, continuum_removal, normalize_zscore,
    sam, ndvi, pca,
)

# --- I/O ---

cube = read_envi("scene.hdr")

# --- Exploration ---

stats = band_stats(cube)        # .mean, .std, .min, .max, .valid_count
cov = covariance(cube)          # (bands, bands) covariance matrix

# --- Preprocessing ---

clean = mnf_denoise(cube, n_components=20)  # denoise first
smooth = savitzky_golay(clean, window=7, polyorder=2)
d1 = derivative(smooth, order=1)
cr = continuum_removal(clean)
normed = normalize_zscore(clean)

# --- Analysis ---

reference = normed.spectrum(50, 50)
angles = sam(normed, reference)             # spectral similarity
veg = ndvi(normed, nir=90, red=55)          # vegetation index
pca_result = pca(normed, n_components=10)   # dimensionality reduction

# --- Storage ---

write_zarr(normed, "chips.zarr", chunk_shape=(normed.bands, 256, 256))
tile = read_zarr_window("chips.zarr", bands=(0, normed.bands), rows=(0, 256), cols=(0, 256))
```

## SpectralCube

| Category | Methods |
|---|---|
| Dimensions | `bands`, `height`, `width`, `shape` |
| Data access | `data()`, `wavelengths()`, `fwhm()`, `nodata` |
| Pixel access | `spectrum(row, col)` |
| Band access | `band(index)`, `band_at(nm)`, `band_nearest(nm)` |
| Wavelength lookup | `wavelength(index)`, `wavelength_index(nm)`, `nearest_band_index(nm)` |
| Statistics | `mean_spectrum()` |
| Subsetting | `sel_bands([i, j])`, `sel_wavelengths(min, max)` |

## I/O

### Read

| Format | Function |
|---|---|
| ENVI | `read_envi(path)` |
| Zarr V3 | `read_zarr(path)` |
| Zarr V3 | `read_zarr_with_options(path, ...)` |
| Zarr V3 | `read_zarr_window(path, bands, rows, cols)` |
| Zarr V3 | `zarr_cube_shape(path)` |

### Write

| Format | Function |
|---|---|
| ENVI | `write_envi(cube, path)` |
| Zarr V3 | `write_zarr(cube, path)` |

## Algorithms

### Exploration

| Operation | Function |
|---|---|
| Per-band stats | `band_stats(cube)` → `BandStats` |
| Covariance matrix | `covariance(cube)` |
| Correlation matrix | `correlation(cube)` |

### Preprocessing

| Operation | Function |
|---|---|
| MNF denoise | `mnf_denoise(cube, n_components)` |
| MNF | `mnf(cube, n_components)` |
| Savitzky-Golay smoothing | `savitzky_golay(cube, window, polyorder)` |
| Derivative spectra | `derivative(cube, order)` |
| Continuum removal | `continuum_removal(cube)` |
| Min-max normalization | `normalize_minmax(cube)` |
| Z-score normalization | `normalize_zscore(cube)` |
| Resample | `resample(cube, target_wl, method)` |

### Analysis

| Operation | Function |
|---|---|
| Spectral Angle Mapper | `sam(cube, reference)` |
| Normalized difference | `normalized_difference(cube, a, b)` |
| Band ratio | `band_ratio(cube, a, b)` |
| NDVI | `ndvi(cube, nir, red)` |
| PCA | `pca(cube, n_components)` |

## Architecture

```
crates/hyperspec/           # Pure Rust library → crates.io
└── src/
    ├── cube.rs             # SpectralCube type
    ├── error.rs            # HyperspecError
    ├── io/
    │   ├── envi.rs         # ENVI read/write (BSQ, BIL, BIP)
    │   └── zarr.rs         # Zarr V3 read/write/window
    ├── linalg.rs           # Matrix ops (faer): eigen, covariance, tiled GEMM
    └── algorithms/
        ├── sam.rs
        ├── continuum.rs
        ├── indices.rs
        ├── pca.rs
        ├── mnf.rs
        ├── resample.rs
        ├── stats.rs
        ├── normalize.rs
        ├── derivative.rs
        └── smooth.rs

pyo3-hyperspec/             # PyO3 bindings → PyPI: hyperspec-py
├── src/lib.rs
└── python/hyperspec/
    └── __init__.py
```

- **Rust 2024 edition**, using `ndarray`, `rayon`, `faer`, `zarrs`, `thiserror`
- **Minimal system dependencies** — no LAPACK or GDAL
- **Python 3.12+**, NumPy bindings via PyO3

## Roadmap

- Cloud-native Zarr stores (Azure, GCS, S3)
- More algorithms and utilities

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install maturin pytest
cargo test
maturin develop
pytest
```

## License

Apache-2.0
