//! Linear algebra operations using faer (pure Rust, zero system deps).

use ndarray::{Array1, Array2, Array3, s};

use crate::error::{HyperspecError, Result};

// ---------------------------------------------------------------------------
// Eigendecomposition
// ---------------------------------------------------------------------------

/// Eigendecompose a symmetric ndarray `Array2<f64>` matrix.
///
/// Converts to faer `Mat`, calls `self_adjoint_eigen(Side::Lower)`, and returns
/// `(eigenvalues, eigenvectors)` where eigenvectors are columns of the returned matrix.
pub(crate) fn sym_eigen(matrix: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    debug_assert_eq!(n, matrix.ncols(), "sym_eigen requires a square matrix");

    // Convert ndarray -> faer Mat (row-major source -> owned)
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = matrix[[i, j]];
        }
    }
    let mat_ref = faer::MatRef::from_row_major_slice(&data, n, n);
    let owned: faer::Mat<f64> = mat_ref.to_owned();

    let decomp = owned
        .as_ref()
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| HyperspecError::InvalidInput("eigendecomposition failed".to_string()))?;

    let s_diag = decomp.S();
    let u = decomp.U();

    let eigenvalues: Vec<f64> = (0..n).map(|i| s_diag[i]).collect();
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            eigenvectors[[i, j]] = u[(i, j)];
        }
    }
    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// BSQ-native transform
// ---------------------------------------------------------------------------

/// Apply a linear transform to BSQ data: `matrix @ bsq_data`.
///
/// `bsq_data` has shape `(in_dim, height, width)` in BSQ layout. `matrix` has shape
/// `(out_dim, in_dim)`. Optionally subtracts `center` (per-band mean) before the
/// multiply and adds `bias` after.
///
/// Returns `Array3<f64>` of shape `(out_dim, height, width)`.
pub(crate) fn bsq_transform(
    bsq_data: &Array3<f64>,
    matrix: &Array2<f64>,
    center: Option<&[f64]>,
    bias: Option<&[f64]>,
) -> Array3<f64> {
    let in_dim = bsq_data.shape()[0];
    let height = bsq_data.shape()[1];
    let width = bsq_data.shape()[2];
    let out_dim = matrix.nrows();
    let n_pixels = height * width;

    debug_assert_eq!(matrix.ncols(), in_dim);

    let tile_size = 32768;
    let mut out_flat = vec![0.0f64; out_dim * n_pixels];

    // Convert transform matrix to faer column-major
    let mut t_buf = vec![0.0f64; out_dim * in_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            t_buf[j * out_dim + i] = matrix[[i, j]]; // column-major
        }
    }
    let t_faer = faer::MatRef::from_column_major_slice(&t_buf, out_dim, in_dim);

    let mut tile_in = vec![0.0f64; tile_size * in_dim];

    for tile_start in (0..n_pixels).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(n_pixels);
        let rows_in_tile = tile_end - tile_start;

        // Fill centered tile: column-major (rows_in_tile, in_dim)
        // Band b occupies tile_in[b * rows_in_tile .. (b+1) * rows_in_tile]
        for b in 0..in_dim {
            let band_slice = bsq_data.slice(s![b, .., ..]);
            let band_data = band_slice.as_standard_layout();
            let band_raw = band_data.as_slice().expect("band is contiguous");
            let col_start = b * rows_in_tile;
            let center_b = center.map_or(0.0, |c| c[b]);
            for i in 0..rows_in_tile {
                tile_in[col_start + i] = band_raw[tile_start + i] - center_b;
            }
        }

        let x_tile = faer::MatRef::from_column_major_slice(
            &tile_in[..rows_in_tile * in_dim],
            rows_in_tile,
            in_dim,
        );

        // result_tile = t_faer @ x_tile^T = (out_dim, rows_in_tile) column-major
        let mut res_buf = vec![0.0f64; out_dim * rows_in_tile];
        let mut res_tile =
            faer::MatMut::from_column_major_slice_mut(&mut res_buf, out_dim, rows_in_tile);

        faer::linalg::matmul::matmul(
            res_tile.as_mut(),
            faer::Accum::Replace,
            t_faer,
            x_tile.transpose(),
            1.0,
            faer::Par::rayon(0),
        );

        // Copy to out_flat in BSQ layout, adding bias
        for c in 0..out_dim {
            let bias_c = bias.map_or(0.0, |b| b[c]);
            let dst_start = c * n_pixels + tile_start;
            for i in 0..rows_in_tile {
                out_flat[dst_start + i] = res_tile[(c, i)] + bias_c;
            }
        }
    }

    Array3::from_shape_vec((out_dim, height, width), out_flat)
        .expect("shape matches total element count")
}

// ---------------------------------------------------------------------------
// Covariance (sample normalization, / (n-1))
// ---------------------------------------------------------------------------

/// Compute sample covariance of BSQ data with normalization `/ (n - 1)`.
///
/// Two paths based on cube size:
/// - Small cubes (`n_pixels <= 32768`): scalar triangle accumulation
/// - Large cubes: tiled GEMM via faer matmul
///
/// IMPORTANT: tile fill stride is `b * rows_in_tile` (NOT `b * tile_size`) for
/// correct last-tile handling.
pub(crate) fn clean_covariance(
    data: &Array3<f64>,
    mean: &[f64],
    n_pixels: usize,
    bands: usize,
    height: usize,
    width: usize,
) -> Array2<f64> {
    let n_f = (n_pixels - 1).max(1) as f64;

    if n_pixels <= 32768 {
        // Small: scalar triangle accumulation
        let tri_size = bands * (bands + 1) / 2;
        let mut tri = vec![0.0; tri_size];
        let mut centered = vec![0.0; bands];
        for row in 0..height {
            for col in 0..width {
                for b in 0..bands {
                    centered[b] = data[[b, row, col]] - mean[b];
                }
                let mut idx = 0;
                for i in 0..bands {
                    for j in i..bands {
                        tri[idx] += centered[i] * centered[j];
                        idx += 1;
                    }
                }
            }
        }
        let inv_n = 1.0 / n_f;
        let mut cov = Array2::<f64>::zeros((bands, bands));
        let mut idx = 0;
        for i in 0..bands {
            for j in i..bands {
                let v = tri[idx] * inv_n;
                cov[[i, j]] = v;
                cov[[j, i]] = v;
                idx += 1;
            }
        }
        cov
    } else {
        // Large: tiled GEMM via faer
        let tile_size = 32768;
        let mut cov_faer = faer::Mat::<f64>::zeros(bands, bands);
        let mut tile_buf = vec![0.0f64; tile_size * bands];

        for tile_start in (0..n_pixels).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n_pixels);
            let rows_in_tile = tile_end - tile_start;

            // Fill tile buffer: column-major (rows_in_tile, bands).
            // Band b occupies tile_buf[b * rows_in_tile .. (b+1) * rows_in_tile].
            // IMPORTANT: stride is b * rows_in_tile, NOT b * tile_size,
            // so the last tile (which may be smaller) is packed correctly.
            for (b, &mean_b) in mean.iter().enumerate().take(bands) {
                let band_slice = data.slice(s![b, .., ..]);
                let band_data = band_slice.as_standard_layout();
                let band_raw = band_data.as_slice().expect("band is contiguous");
                let col_start = b * rows_in_tile;
                for i in 0..rows_in_tile {
                    tile_buf[col_start + i] = band_raw[tile_start + i] - mean_b;
                }
            }

            let tile = faer::MatRef::from_column_major_slice(
                &tile_buf[..rows_in_tile * bands],
                rows_in_tile,
                bands,
            );

            // cov += tile^T @ tile
            faer::linalg::matmul::matmul(
                cov_faer.as_mut(),
                faer::Accum::Add,
                tile.transpose(),
                tile,
                1.0,
                faer::Par::rayon(0),
            );
        }

        let inv_n = 1.0 / n_f;
        let mut cov = Array2::<f64>::zeros((bands, bands));
        for i in 0..bands {
            for j in 0..bands {
                cov[[i, j]] = cov_faer[(i, j)] * inv_n;
            }
        }
        cov
    }
}

// ---------------------------------------------------------------------------
// Covariance + eigendecomposition
// ---------------------------------------------------------------------------

/// Compute sample covariance and its eigendecomposition in one call.
///
/// Returns `(covariance, eigenvalues, eigenvectors)`.
pub(crate) fn clean_covariance_eigen(
    data: &Array3<f64>,
    mean: &[f64],
    n_pixels: usize,
    bands: usize,
    height: usize,
    width: usize,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let cov = clean_covariance(data, mean, n_pixels, bands, height, width);
    let (eigenvalues, eigenvectors) = sym_eigen(&cov)?;
    Ok((cov, eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Randomized truncated PCA (Halko et al.)
// ---------------------------------------------------------------------------

/// Randomized truncated PCA via Halko et al.
///
/// Returns `(explained_variance, components)` where:
/// - `explained_variance`: shape `(n_components,)`, `sigma^2 / (n-1)`
/// - `components`: shape `(n_components, bands)`, row vectors sorted by descending variance
///
/// Parameters: oversampling = `min(10, bands - k)`, sketch dimension `l = k + oversampling`,
/// `n_power_iter = 0`.
pub(crate) fn randomized_pca(
    data: &Array3<f64>,
    mean: &[f64],
    n_pixels: usize,
    bands: usize,
    n_components: usize,
) -> Result<(Array1<f64>, Array2<f64>)> {
    use faer::linalg::solvers::Qr;

    let k = n_components;
    let oversampling = std::cmp::min(10, bands - k);
    let l = k + oversampling; // sketch dimension
    let _n_power_iter = 0; // no power iterations

    // Generate random Gaussian Omega of shape (bands, l) via Box-Muller
    let omega = box_muller_matrix(bands, l);

    // Y = A @ Omega, where A is the centered data matrix (n_pixels, bands)
    // and Omega is (bands, l). Y is (n_pixels, l).
    // Computed via tiled BSQ GEMM.
    let y = tiled_bsq_gemm_right(data, mean, n_pixels, bands, &omega, l);

    // Thin QR of Y (n_pixels, l) -> Q (n_pixels, l)
    let y_faer = faer::MatRef::from_column_major_slice(&y, n_pixels, l);
    let qr = Qr::new(y_faer);
    let q_mat = qr.compute_thin_Q(); // (n_pixels, l)

    // B = Q^T @ A, shape (l, bands).
    // Computed via tiled BSQ GEMM.
    let b = tiled_bsq_gemm_left_qt(data, mean, n_pixels, bands, &q_mat, l);

    // Thin SVD of B (l, bands)
    let b_faer = faer::MatRef::from_column_major_slice(&b, l, bands);
    let svd = b_faer
        .thin_svd()
        .map_err(|_| HyperspecError::InvalidInput("SVD failed in randomized PCA".to_string()))?;

    extract_pca_components(
        k,
        bands,
        n_pixels,
        |i| svd.S()[i],
        |i, j| svd.V()[(j, i)], // V columns are right singular vectors
    )
}

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

/// Shared helper for extracting PCA components with sign convention from SVD output.
///
/// `sigma_fn(i)` returns the i-th singular value (already in descending order).
/// `component_elem_fn(i, j)` returns element j of the i-th right singular vector.
///
/// Returns `(explained_variance, components)` where components have the sign convention
/// that the largest absolute value in each component is positive.
pub(crate) fn extract_pca_components(
    k: usize,
    bands: usize,
    n_pixels: usize,
    sigma_fn: impl Fn(usize) -> f64,
    component_elem_fn: impl Fn(usize, usize) -> f64,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n_f = (n_pixels - 1).max(1) as f64;
    let mut explained_variance = Array1::<f64>::zeros(k);
    let mut components = Array2::<f64>::zeros((k, bands));

    for i in 0..k {
        let sigma = sigma_fn(i);
        explained_variance[i] = (sigma * sigma) / n_f;

        // Copy component
        for j in 0..bands {
            components[[i, j]] = component_elem_fn(i, j);
        }

        // Sign convention: largest absolute value in each component is positive
        let mut max_abs = 0.0f64;
        let mut max_idx = 0;
        for j in 0..bands {
            let abs_val = components[[i, j]].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                max_idx = j;
            }
        }
        if components[[i, max_idx]] < 0.0 {
            for j in 0..bands {
                components[[i, j]] = -components[[i, j]];
            }
        }
    }

    Ok((explained_variance, components))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Generate a (rows, cols) column-major matrix of standard normal random values
/// via Box-Muller transform.
///
/// Uses a deterministic xoshiro256** PRNG seeded at 42. Clamps `u1` to
/// `f64::MIN_POSITIVE` to avoid `ln(0)`.
fn box_muller_matrix(rows: usize, cols: usize) -> Vec<f64> {
    use std::f64::consts::TAU;

    let total = rows * cols;
    let n_pairs = total.div_ceil(2);
    let mut out = Vec::with_capacity(total);

    let mut state = Xoshiro256StarStar::new(42);

    for _ in 0..n_pairs {
        let u1 = state.next_f64().max(f64::MIN_POSITIVE); // clamp to avoid ln(0)
        let u2 = state.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        out.push(r * theta.cos());
        if out.len() < total {
            out.push(r * theta.sin());
        }
    }

    out
}

/// Minimal xoshiro256** PRNG for deterministic random number generation.
struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into state
        let mut z = seed;
        let mut s = [0u64; 4];
        for s_i in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s_i = x ^ (x >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        // Uniform in [0, 1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Tiled BSQ GEMM: Y = A @ Omega where A is centered BSQ data (n_pixels, bands)
/// and Omega is (bands, l). Returns Y in column-major layout (n_pixels, l).
fn tiled_bsq_gemm_right(
    data: &Array3<f64>,
    mean: &[f64],
    n_pixels: usize,
    bands: usize,
    omega: &[f64], // column-major (bands, l)
    l: usize,
) -> Vec<f64> {
    let tile_size = 32768;
    let mut y = vec![0.0f64; n_pixels * l];

    let omega_ref = faer::MatRef::from_column_major_slice(omega, bands, l);
    let mut tile_buf = vec![0.0f64; tile_size * bands];

    for tile_start in (0..n_pixels).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(n_pixels);
        let rows_in_tile = tile_end - tile_start;

        // Fill tile: column-major (rows_in_tile, bands)
        for (b, &mean_b) in mean.iter().enumerate().take(bands) {
            let band_slice = data.slice(s![b, .., ..]);
            let band_data = band_slice.as_standard_layout();
            let band_raw = band_data.as_slice().expect("band is contiguous");
            let col_start = b * rows_in_tile;
            for i in 0..rows_in_tile {
                tile_buf[col_start + i] = band_raw[tile_start + i] - mean_b;
            }
        }

        let tile = faer::MatRef::from_column_major_slice(
            &tile_buf[..rows_in_tile * bands],
            rows_in_tile,
            bands,
        );

        // y_tile = tile @ omega, shape (rows_in_tile, l)
        let mut y_tile_buf = vec![0.0f64; rows_in_tile * l];
        let mut y_tile =
            faer::MatMut::from_column_major_slice_mut(&mut y_tile_buf, rows_in_tile, l);

        faer::linalg::matmul::matmul(
            y_tile.as_mut(),
            faer::Accum::Replace,
            tile,
            omega_ref,
            1.0,
            faer::Par::rayon(0),
        );

        // Copy into y (column-major n_pixels x l)
        for c in 0..l {
            let dst_offset = c * n_pixels + tile_start;
            let src_offset = c * rows_in_tile;
            y[dst_offset..dst_offset + rows_in_tile]
                .copy_from_slice(&y_tile_buf[src_offset..src_offset + rows_in_tile]);
        }
    }

    y
}

/// Tiled BSQ GEMM: B = Q^T @ A where Q is (n_pixels, l) and A is centered BSQ data.
/// Returns B in column-major layout (l, bands).
fn tiled_bsq_gemm_left_qt(
    data: &Array3<f64>,
    mean: &[f64],
    n_pixels: usize,
    bands: usize,
    q_mat: &faer::Mat<f64>, // (n_pixels, l)
    l: usize,
) -> Vec<f64> {
    let tile_size = 32768;
    let mut b_faer = faer::Mat::<f64>::zeros(l, bands);
    let mut tile_buf = vec![0.0f64; tile_size * bands];

    for tile_start in (0..n_pixels).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(n_pixels);
        let rows_in_tile = tile_end - tile_start;

        // Fill tile: column-major (rows_in_tile, bands)
        for (b, &mean_b) in mean.iter().enumerate().take(bands) {
            let band_slice = data.slice(s![b, .., ..]);
            let band_data = band_slice.as_standard_layout();
            let band_raw = band_data.as_slice().expect("band is contiguous");
            let col_start = b * rows_in_tile;
            for i in 0..rows_in_tile {
                tile_buf[col_start + i] = band_raw[tile_start + i] - mean_b;
            }
        }

        let tile = faer::MatRef::from_column_major_slice(
            &tile_buf[..rows_in_tile * bands],
            rows_in_tile,
            bands,
        );

        // Q_tile: subrows of Q from tile_start..tile_end
        let q_tile = q_mat.as_ref().get(tile_start..tile_end, ..);

        // B += Q_tile^T @ tile
        faer::linalg::matmul::matmul(
            b_faer.as_mut(),
            faer::Accum::Add,
            q_tile.transpose(),
            tile,
            1.0,
            faer::Par::rayon(0),
        );
    }

    // Extract to column-major vec
    let mut b_out = vec![0.0f64; l * bands];
    for j in 0..bands {
        for i in 0..l {
            b_out[j * l + i] = b_faer[(i, j)];
        }
    }
    b_out
}
