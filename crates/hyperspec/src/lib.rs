//! Fast processing library for hyperspectral imagery.

pub mod algorithms;
pub mod cube;
pub mod error;
pub mod io;

pub use algorithms::continuum::continuum_removal;
pub use algorithms::indices::{band_ratio, ndvi, normalized_difference};
pub use algorithms::mnf::{MnfResult, mnf, mnf_denoise};
pub use algorithms::pca::{PcaResult, pca, pca_inverse, pca_transform};
pub use algorithms::resample::{ResampleMethod, resample};
pub use algorithms::sam::sam;
pub use algorithms::stats::{BandStats, band_stats, correlation, covariance};
pub use cube::SpectralCube;
pub use error::{HyperspecError, Result};
pub use io::envi::{
    EnviHeader, EnviWriteDataType, EnviWriteOptions, Interleave, read_envi, write_envi,
    write_envi_with_options,
};
pub use io::zarr::{
    ZarrCompression, ZarrReadOptions, ZarrWriteOptions, read_zarr, read_zarr_window,
    read_zarr_with_options, write_zarr, write_zarr_with_options, zarr_cube_shape,
};
