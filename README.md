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
import numpy as np
from hyperspec import (
    SpectralCube, read_envi, write_zarr, read_zarr_window, zarr_cube_shape,
    sam, continuum_removal, pca, mnf_denoise,
    band_stats, normalize_zscore, savitzky_golay, derivative,
)

# --- I/O ---

# Read ENVI format (AVIRIS-NG, etc.)
cube = read_envi("scene.hdr")

# Process
cr_cube = continuum_removal(cube)
denoised = mnf_denoise(cube, n_components=20)

# Write to Zarr with ML-friendly chunks
write_zarr(cube, "chips.zarr", chunk_shape=(cube.bands, 256, 256))

# Read a single training tile without loading the full cube
tile = read_zarr_window(
    "chips.zarr",
    bands=(0, cube.bands),
    rows=(0, 256),
    cols=(0, 256),
)

# Query shape without loading data
bands, lines, samples = zarr_cube_shape("chips.zarr")

# --- Preprocessing ---

# Per-band statistics
stats = band_stats(cube)        # .mean, .std, .min, .max, .valid_count

# Normalize for ML training
normed = normalize_zscore(cube)

# Smooth noisy spectra, then compute derivatives
smooth = savitzky_golay(cube, window=7, polyorder=2)
d1 = derivative(smooth, order=1)

# --- Spectral analysis ---

# Compare every pixel to a reference spectrum
reference = cube.spectrum(50, 50)
angles = sam(cube, reference)

# Dimensionality reduction
pca_result = pca(cube, n_components=10)
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

### Write

| Format | Function |
|---|---|
| ENVI | `write_envi(cube, path)` |
| Zarr V3 | `write_zarr(cube, path)` |

### Utilities

| Format | Function |
|---|---|
| Zarr V3 | `zarr_cube_shape(path)` |

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
| MNF | `mnf(cube, n_components)` |
| MNF denoise | `mnf_denoise(cube, n_components)` |

## Architecture

```
crates/hyperspec/           # Pure Rust library → crates.io
└── src/
    ├── cube.rs             # SpectralCube type
    ├── error.rs            # HyperspecError
    ├── io/
    │   ├── envi.rs         # ENVI read/write (BSQ, BIL, BIP)
    │   └── zarr.rs         # Zarr V3 read/write/window
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
