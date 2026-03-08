//! Zarr V3 read/write support for hyperspectral data cubes.
//!
//! Reads and writes [`SpectralCube`] to/from Zarr V3 stores on the filesystem.
//! Uses the [`zarrs`] crate (pure Rust, no C dependencies).
//!
//! # Write layout
//!
//! ```text
//! store.zarr/
//! ├── zarr.json                          # root group
//! ├── reflectance/
//! │   └── zarr.json                      # (bands, lines, samples) float64
//! └── sensor/
//!     ├── zarr.json                      # sensor group
//!     ├── wavelengths/
//!     │   └── zarr.json                  # (bands,) float64
//!     └── fwhm/                          # (bands,) float64, if present
//!         └── zarr.json
//! ```
//!
//! Attributes on the reflectance array: `wavelength_units`, `_FillValue` (if nodata).

use std::path::Path;
use std::sync::Arc;

use std::ops::Range;

use ndarray::{Array1, Array3, ArrayD, IxDyn};
use zarrs::array::data_type;
use zarrs::array::{Array as ZarrArray, ArrayBuilder, ArraySubsetTraits, FillValue};
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use crate::cube::SpectralCube;
use crate::error::{HyperspecError, Result};

// ---------------------------------------------------------------------------
// Public options
// ---------------------------------------------------------------------------

/// Options for reading Zarr stores with explicit array paths.
#[derive(Debug, Clone)]
pub struct ZarrReadOptions {
    /// Zarr path to the 3-D reflectance/radiance array (e.g. "/reflectance").
    pub data_path: String,
    /// Zarr path to the 1-D wavelength array (e.g. "/sensor/wavelengths").
    pub wavelength_path: String,
    /// Zarr path to the 1-D FWHM array (optional).
    pub fwhm_path: Option<String>,
    /// Override the nodata value.
    pub nodata: Option<f64>,
    /// Multiply data by this factor after reading.
    pub scale_factor: Option<f64>,
    /// Add this offset after scaling.
    pub add_offset: Option<f64>,
}

/// Compression codec for Zarr writes.
#[derive(Debug, Clone)]
pub enum ZarrCompression {
    /// No compression.
    None,
    /// Gzip (deflate) compression with level 0–9.
    Gzip(u8),
}

/// Options for writing Zarr stores.
#[derive(Debug, Clone)]
pub struct ZarrWriteOptions {
    /// Compression codec. Default: `Gzip(5)`.
    pub compression: ZarrCompression,
    /// Chunk shape as `[bands, lines, samples]`. When `None`, defaults to
    /// `[1, lines, samples]` (one band per chunk, optimal for band-sequential
    /// access). For ML training dataloaders, use `[bands, 256, 256]` so each
    /// chunk is one training tile with the full spectrum.
    pub chunk_shape: Option<[usize; 3]>,
    /// If `true`, overwrite an existing store at the target path.
    /// Default: `false` (fail if target exists).
    pub overwrite: bool,
}

impl Default for ZarrWriteOptions {
    fn default() -> Self {
        Self {
            compression: ZarrCompression::Gzip(5),
            chunk_shape: None,
            overwrite: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// Read a hyperspectral cube from a Zarr V3 store on the filesystem.
///
/// Auto-discovers the reflectance data, wavelengths, FWHM, and nodata by
/// scanning the store hierarchy. Works with stores written by [`write_zarr`]
/// and other tools following common naming conventions.
///
/// For stores with non-standard layouts, use [`read_zarr_with_options`].
pub fn read_zarr(path: impl AsRef<Path>) -> Result<SpectralCube> {
    let path = path.as_ref();
    let store = open_store(path)?;
    let disc = discover_arrays(&store, path)?;
    read_cube_from_discovered(&store, &disc, path)
}

/// Read a hyperspectral cube from a Zarr store using explicit array paths.
pub fn read_zarr_with_options(
    path: impl AsRef<Path>,
    opts: &ZarrReadOptions,
) -> Result<SpectralCube> {
    let path = path.as_ref();
    let store = open_store(path)?;

    let data_arr = open_zarr_array(&store, &opts.data_path, path)?;
    let wl_arr = open_zarr_array(&store, &opts.wavelength_path, path)?;
    let fwhm_arr = opts
        .fwhm_path
        .as_ref()
        .map(|p| open_zarr_array(&store, p, path))
        .transpose()?;

    let mut data = retrieve_3d_f64(&data_arr, path)?;
    let wavelengths = retrieve_1d_f64(&wl_arr, path)?;
    let fwhm = fwhm_arr
        .as_ref()
        .map(|a| retrieve_1d_f64(a, path))
        .transpose()?;

    let nodata = opts.nodata.or_else(|| read_fill_value_attr(&data_arr));

    if opts.scale_factor.is_some() || opts.add_offset.is_some() {
        let scale = opts.scale_factor.unwrap_or(1.0);
        let offset = opts.add_offset.unwrap_or(0.0);
        apply_scale_offset(&mut data, scale, offset, nodata);
    }

    let data = orient_to_bsq(data, wavelengths.len(), path)?;
    SpectralCube::new(data, wavelengths, fwhm, nodata)
}

/// Query the shape of a Zarr cube without loading any data.
///
/// Returns `(bands, lines, samples)` from the auto-discovered reflectance array.
/// Useful for ML pipelines that need to know dataset dimensions before reading tiles.
pub fn zarr_cube_shape(path: impl AsRef<Path>) -> Result<(usize, usize, usize)> {
    let path = path.as_ref();
    let store = open_store(path)?;
    let disc = discover_arrays(&store, path)?;
    let data_arr = open_zarr_array(&store, &disc.data_path, path)?;
    let shape = data_arr.shape();
    if shape.len() != 3 {
        return Err(HyperspecError::Format(format!(
            "expected 3-D array, got {}-D in {}",
            shape.len(),
            path.display()
        )));
    }
    Ok((shape[0] as usize, shape[1] as usize, shape[2] as usize))
}

/// Read a spatial/spectral window from a Zarr V3 store without loading the
/// full cube.
///
/// This is the key function for ML dataloaders: read a single training tile
/// with all (or a subset of) bands in one efficient chunk-aligned read.
///
/// # Arguments
///
/// * `path` — path to the Zarr store on disk
/// * `bands` — band range to read (e.g. `0..285` for all EMIT bands)
/// * `rows` — row/line range (e.g. `0..256` for a 256-pixel tile)
/// * `cols` — column/sample range
///
/// Wavelengths and FWHM are sliced to match the requested band range.
pub fn read_zarr_window(
    path: impl AsRef<Path>,
    bands: Range<usize>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Result<SpectralCube> {
    let path = path.as_ref();
    let store = open_store(path)?;
    let disc = discover_arrays(&store, path)?;

    let data_arr = open_zarr_array(&store, &disc.data_path, path)?;
    let wl_arr = open_zarr_array(&store, &disc.wavelength_path, path)?;
    let fwhm_arr = disc
        .fwhm_path
        .as_ref()
        .map(|p| open_zarr_array(&store, p, path))
        .transpose()?;

    // Validate ranges against on-disk shape
    let shape = data_arr.shape();
    if shape.len() != 3 {
        return Err(HyperspecError::Format(format!(
            "expected 3-D array, got {}-D in {}",
            shape.len(),
            path.display()
        )));
    }
    let [d0, d1, d2] = [shape[0] as usize, shape[1] as usize, shape[2] as usize];

    if bands.is_empty() || rows.is_empty() || cols.is_empty() {
        return Err(HyperspecError::InvalidInput(
            "window ranges must be non-empty".to_string(),
        ));
    }
    if bands.end > d0 || rows.end > d1 || cols.end > d2 {
        return Err(HyperspecError::InvalidInput(format!(
            "window [{}..{}, {}..{}, {}..{}] exceeds array shape [{}, {}, {}]",
            bands.start, bands.end, rows.start, rows.end, cols.start, cols.end, d0, d1, d2
        )));
    }

    // Read the spatial/spectral window
    let subset: Vec<Range<u64>> = vec![
        bands.start as u64..bands.end as u64,
        rows.start as u64..rows.end as u64,
        cols.start as u64..cols.end as u64,
    ];
    let dyn_arr = retrieve_subset_as_f64(&data_arr, &subset, path)?;
    let shape = dyn_arr.shape();
    let (nb, nl, ns) = (shape[0], shape[1], shape[2]);
    let data: Array3<f64> = dyn_arr
        .into_shape_with_order((nb, nl, ns))
        .map_err(|e| HyperspecError::Format(format!("cannot reshape window to 3-D: {}", e)))?;

    // Slice wavelengths and fwhm to match band range
    let wl_all = retrieve_1d_f64(&wl_arr, path)?;
    let wavelengths = wl_all.slice(ndarray::s![bands.start..bands.end]).to_owned();

    let fwhm = fwhm_arr
        .as_ref()
        .map(|a| retrieve_1d_f64(a, path))
        .transpose()?
        .map(|f| f.slice(ndarray::s![bands.start..bands.end]).to_owned());

    let nodata = disc.nodata;

    SpectralCube::new(data, wavelengths, fwhm, nodata)
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write a spectral cube to a Zarr V3 store with default settings (gzip 5).
pub fn write_zarr(cube: &SpectralCube, path: impl AsRef<Path>) -> Result<()> {
    write_zarr_with_options(cube, path, &ZarrWriteOptions::default())
}

/// Write a spectral cube to a Zarr V3 store with explicit options.
///
/// Writes to a temporary directory first and renames on success to prevent
/// partial stores. Fails if the target path already exists unless
/// `opts.overwrite` is `true`.
///
/// **Overwrite note:** when replacing an existing store, the old store is
/// removed only after the new store is fully written. However, the
/// remove + rename sequence is not fully atomic — if the process crashes
/// between removing the old store and renaming the new one, data may be lost.
pub fn write_zarr_with_options(
    cube: &SpectralCube,
    path: impl AsRef<Path>,
    opts: &ZarrWriteOptions,
) -> Result<()> {
    use zarrs::array::codec::GzipCodec;

    let path = path.as_ref();

    // --- Check existing store ---
    if path.exists() {
        if !path.is_dir() {
            return Err(HyperspecError::Io(format!(
                "target path {} exists but is not a directory",
                path.display()
            )));
        }
        if !opts.overwrite {
            return Err(HyperspecError::Io(format!(
                "Zarr store already exists at {} (use overwrite: true to replace)",
                path.display()
            )));
        }
    }

    // --- Write to temp directory, rename on success ---
    let parent = path.parent().unwrap_or(Path::new("."));
    let tmp_dir = tempfile::tempdir_in(parent).map_err(|e| {
        HyperspecError::Io(format!(
            "cannot create temp directory in {}: {}",
            parent.display(),
            e
        ))
    })?;
    let tmp_path = tmp_dir.path();

    let store: ReadableWritableListableStorage = Arc::new(
        zarrs::filesystem::FilesystemStore::new(tmp_path)
            .map_err(|e| HyperspecError::Io(format!("cannot create Zarr store: {}", e)))?,
    );

    let (bands, lines, samples) = cube.shape();
    let [cb, cl, cs] = opts.chunk_shape.unwrap_or([1, lines, samples]);
    if cb == 0 || cl == 0 || cs == 0 {
        return Err(HyperspecError::InvalidInput(
            "chunk_shape dimensions must be non-zero".to_string(),
        ));
    }
    if cb > bands || cl > lines || cs > samples {
        return Err(HyperspecError::InvalidInput(format!(
            "chunk_shape [{cb}, {cl}, {cs}] exceeds cube shape [{bands}, {lines}, {samples}]"
        )));
    }
    let (cb, cl, cs) = (cb as u64, cl as u64, cs as u64);

    // --- root group ---
    let mut root_attrs = serde_json::Map::new();
    root_attrs.insert(
        "hyperspec_layout".into(),
        serde_json::Value::String("spectral-cube-v1".into()),
    );
    let mut root_builder = GroupBuilder::new();
    root_builder.attributes(root_attrs);
    let root = root_builder
        .build(store.clone(), "/")
        .map_err(|e| HyperspecError::Io(format!("cannot create root group: {e}")))?;
    root.store_metadata()
        .map_err(|e| HyperspecError::Io(format!("cannot store root group metadata: {e}")))?;

    // --- reflectance array ---
    let mut attrs = serde_json::Map::new();
    attrs.insert(
        "wavelength_units".into(),
        serde_json::Value::String("nanometers".into()),
    );
    if let Some(nd) = cube.nodata() {
        attrs.insert(
            "_FillValue".into(),
            serde_json::Number::from_f64(nd)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
        );
    }

    let mut builder = ArrayBuilder::new(
        vec![bands as u64, lines as u64, samples as u64],
        vec![cb, cl, cs],
        data_type::float64(),
        FillValue::from(0.0f64),
    );
    builder.attributes(attrs);
    builder.dimension_names(Some(["bands", "lines", "samples"].map(Some)));

    if let ZarrCompression::Gzip(level) = opts.compression {
        builder.bytes_to_bytes_codecs(vec![Arc::new(
            GzipCodec::new(level.into())
                .map_err(|e| HyperspecError::Io(format!("cannot create gzip codec: {e}")))?,
        )]);
    }

    let refl_array = builder
        .build(store.clone(), "/reflectance")
        .map_err(|e| HyperspecError::Io(format!("cannot create reflectance array: {e}")))?;
    refl_array
        .store_metadata()
        .map_err(|e| HyperspecError::Io(format!("cannot store reflectance metadata: {e}")))?;

    // Write data — clone Array3 → ArrayD for zarrs. This copies the array data;
    // zarrs requires an owned ArrayD, so a zero-copy path isn't available yet.
    let data_dyn: ArrayD<f64> = cube
        .data()
        .clone()
        .into_dimensionality::<IxDyn>()
        .map_err(|e| HyperspecError::Io(format!("cannot convert data to dynamic dims: {e}")))?;
    refl_array
        .store_array_subset(&refl_array.subset_all(), data_dyn)
        .map_err(|e| HyperspecError::Io(format!("cannot write reflectance data: {e}")))?;

    // --- sensor group ---
    let sensor = GroupBuilder::new()
        .build(store.clone(), "/sensor")
        .map_err(|e| HyperspecError::Io(format!("cannot create sensor group: {e}")))?;
    sensor
        .store_metadata()
        .map_err(|e| HyperspecError::Io(format!("cannot store sensor group metadata: {e}")))?;

    // --- wavelengths array ---
    write_1d_f64_array(&store, "/sensor/wavelengths", cube.wavelengths())?;

    // --- fwhm array (optional) ---
    if let Some(fwhm) = cube.fwhm() {
        write_1d_f64_array(&store, "/sensor/fwhm", fwhm)?;
    }

    // --- Replace: remove old store (if overwriting), then rename temp → final ---
    // persist() consumes the TempDir so it won't be cleaned up, then we rename.
    let persisted = tmp_dir.keep();

    if path.exists() {
        // overwrite=true was already checked above; safe to remove now that
        // the new store is fully written in the temp directory.
        std::fs::remove_dir_all(path).map_err(|e| {
            let _ = std::fs::remove_dir_all(&persisted);
            HyperspecError::Io(format!(
                "cannot remove existing store at {}: {}",
                path.display(),
                e
            ))
        })?;
    }

    std::fs::rename(&persisted, path).map_err(|e| {
        let _ = std::fs::remove_dir_all(&persisted);
        HyperspecError::Io(format!(
            "cannot rename temp store to {}: {}",
            path.display(),
            e
        ))
    })?;

    Ok(())
}

/// Helper to write a 1-D f64 array to the store.
fn write_1d_f64_array(
    store: &ReadableWritableListableStorage,
    arr_path: &str,
    data: &Array1<f64>,
) -> Result<()> {
    let len = data.len() as u64;
    let arr = ArrayBuilder::new(
        vec![len],
        vec![len],
        data_type::float64(),
        FillValue::from(0.0f64),
    )
    .build(store.clone(), arr_path)
    .map_err(|e| HyperspecError::Io(format!("cannot create array '{arr_path}': {e}")))?;
    arr.store_metadata()
        .map_err(|e| HyperspecError::Io(format!("cannot store metadata for '{arr_path}': {e}")))?;

    let dyn_arr: ArrayD<f64> = data.clone().into_dimensionality::<IxDyn>().map_err(|e| {
        HyperspecError::Io(format!("cannot convert 1-D data for '{arr_path}': {e}"))
    })?;
    arr.store_array_subset(&arr.subset_all(), dyn_arr)
        .map_err(|e| HyperspecError::Io(format!("cannot write data for '{arr_path}': {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Auto-discovery
// ---------------------------------------------------------------------------

struct DiscoveredArrays {
    data_path: String,
    wavelength_path: String,
    fwhm_path: Option<String>,
    nodata: Option<f64>,
    scale_factor: Option<f64>,
    add_offset: Option<f64>,
}

fn discover_arrays(
    store: &ReadableWritableListableStorage,
    path: &Path,
) -> Result<DiscoveredArrays> {
    let root = zarrs::group::Group::open(store.clone(), "/").map_err(|e| {
        HyperspecError::Format(format!(
            "cannot open Zarr root group in {}: {}",
            path.display(),
            e
        ))
    })?;

    let array_paths = collect_array_paths(store, &root);

    // --- Find 3-D data array ---
    let cube_names: &[&str] = &[
        "reflectance",
        "Reflectance_Data",
        "radiance",
        "Radiance_Data",
    ];

    let data_path = find_array_by_leaf(&array_paths, store, cube_names, 3)
        .or_else(|| find_first_nd_array(&array_paths, store, 3))
        .ok_or_else(|| {
            HyperspecError::Format(format!("no 3-D array found in {}", path.display()))
        })?;

    // --- Find wavelengths ---
    let wl_names: &[&str] = &[
        "wavelengths",
        "Wavelength",
        "wavelength",
        "wavelength_centers",
        "center_wavelengths",
    ];

    let wavelength_path =
        find_array_by_leaf(&array_paths, store, wl_names, 1).ok_or_else(|| {
            HyperspecError::Format(format!("no wavelength array found in {}", path.display()))
        })?;

    // --- FWHM (optional) ---
    let fwhm_names: &[&str] = &["fwhm", "FWHM"];
    let fwhm_path = find_array_by_leaf(&array_paths, store, fwhm_names, 1);

    // --- Metadata from data array ---
    let data_arr: Option<ZarrArray<dyn ReadableStorageTraits>> =
        ZarrArray::open(store.clone().readable(), &data_path).ok();
    let nodata = data_arr.as_ref().and_then(read_fill_value_attr);
    let scale_factor = data_arr
        .as_ref()
        .and_then(|a| read_f64_json_attr(a, "scale_factor"))
        .or_else(|| {
            data_arr
                .as_ref()
                .and_then(|a| read_f64_json_attr(a, "Scale_Factor"))
        });
    let add_offset = data_arr
        .as_ref()
        .and_then(|a| read_f64_json_attr(a, "add_offset"));

    Ok(DiscoveredArrays {
        data_path,
        wavelength_path,
        fwhm_path,
        nodata,
        scale_factor,
        add_offset,
    })
}

fn collect_array_paths(
    store: &ReadableWritableListableStorage,
    group: &zarrs::group::Group<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
) -> Vec<String> {
    let mut paths = Vec::new();
    if let Ok(children) = group.child_paths() {
        for child_path in children {
            let child_str = child_path.as_str();
            if ZarrArray::open(store.clone(), child_str).is_ok() {
                paths.push(child_str.to_string());
            } else if let Ok(child_group) = zarrs::group::Group::open(store.clone(), child_str) {
                paths.extend(collect_array_paths(store, &child_group));
            }
        }
    }
    paths
}

fn find_array_by_leaf<T: ReadableStorageTraits + ?Sized + 'static>(
    paths: &[String],
    store: &Arc<T>,
    names: &[&str],
    ndim: usize,
) -> Option<String> {
    for arr_path in paths {
        let leaf = arr_path.rsplit('/').next().unwrap_or(arr_path);
        if names.iter().any(|&n| n.eq_ignore_ascii_case(leaf))
            && let Ok(arr) = ZarrArray::open(store.clone(), arr_path)
            && arr.dimensionality() == ndim
        {
            return Some(arr_path.clone());
        }
    }
    None
}

fn find_first_nd_array<T: ReadableStorageTraits + ?Sized + 'static>(
    paths: &[String],
    store: &Arc<T>,
    ndim: usize,
) -> Option<String> {
    for arr_path in paths {
        if let Ok(arr) = ZarrArray::open(store.clone(), arr_path)
            && arr.dimensionality() == ndim
        {
            return Some(arr_path.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn open_store(path: &Path) -> Result<ReadableWritableListableStorage> {
    let store = zarrs::filesystem::FilesystemStore::new(path).map_err(|e| {
        HyperspecError::Io(format!(
            "cannot open Zarr store at {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(Arc::new(store))
}

fn open_zarr_array<T: ReadableStorageTraits + ?Sized + 'static>(
    store: &Arc<T>,
    arr_path: &str,
    file_path: &Path,
) -> Result<ZarrArray<T>> {
    ZarrArray::open(store.clone(), arr_path).map_err(|e| {
        HyperspecError::Format(format!(
            "cannot open array '{}' in {}: {}",
            arr_path,
            file_path.display(),
            e
        ))
    })
}

fn retrieve_3d_f64<T: ReadableStorageTraits + ?Sized + 'static>(
    arr: &ZarrArray<T>,
    path: &Path,
) -> Result<Array3<f64>> {
    if arr.dimensionality() != 3 {
        return Err(HyperspecError::Format(format!(
            "expected 3-D array, got {}-D in {}",
            arr.dimensionality(),
            path.display()
        )));
    }
    let subset = arr.subset_all();
    let dyn_arr = retrieve_subset_as_f64(arr, &subset, path)?;
    let shape = dyn_arr.shape();
    let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
    dyn_arr.into_shape_with_order((d0, d1, d2)).map_err(|e| {
        HyperspecError::Format(format!(
            "cannot reshape to 3-D from {}: {}",
            path.display(),
            e
        ))
    })
}

fn retrieve_1d_f64<T: ReadableStorageTraits + ?Sized + 'static>(
    arr: &ZarrArray<T>,
    path: &Path,
) -> Result<Array1<f64>> {
    if arr.dimensionality() != 1 {
        return Err(HyperspecError::Format(format!(
            "expected 1-D array, got {}-D in {}",
            arr.dimensionality(),
            path.display()
        )));
    }
    let subset = arr.subset_all();
    let dyn_arr = retrieve_subset_as_f64(arr, &subset, path)?;
    let len = dyn_arr.shape()[0];
    dyn_arr.into_shape_with_order(len).map_err(|e| {
        HyperspecError::Format(format!(
            "cannot reshape to 1-D from {}: {}",
            path.display(),
            e
        ))
    })
}

/// Read a subset of a Zarr array as f64, converting from the on-disk numeric type.
///
/// Inspects the array's dtype metadata and dispatches directly to the
/// correct retrieval type, avoiding trial-and-error reads on large arrays.
///
/// Supports: f64, f32, u8, u16, u32, u64, i8, i16, i32, i64.
/// Note: u64/i64 → f64 is lossy above 2^53.
fn retrieve_subset_as_f64<T: ReadableStorageTraits + ?Sized + 'static>(
    arr: &ZarrArray<T>,
    subset: &(dyn ArraySubsetTraits + '_),
    path: &Path,
) -> Result<ArrayD<f64>> {
    let dt = arr.data_type();
    let path_display = path.display();

    // Fast path: f64 on disk, no conversion needed.
    if *dt == data_type::float64() {
        return arr
            .retrieve_array_subset::<ArrayD<f64>>(subset)
            .map_err(|e| {
                HyperspecError::Format(format!("cannot read array from {path_display}: {e}"))
            });
    }

    // Dispatch by on-disk dtype: read as native type, cast to f64.
    // DataType is a trait object so we compare against known constructors.
    macro_rules! dispatch {
        ($($dtype_fn:path => $rust_ty:ty),+ $(,)?) => {
            $(
                if *dt == $dtype_fn() {
                    return arr
                        .retrieve_array_subset::<ArrayD<$rust_ty>>(subset)
                        .map(|a| a.mapv(|v| v as f64))
                        .map_err(|e| HyperspecError::Format(
                            format!("cannot read array from {path_display}: {e}")
                        ));
                }
            )+
        };
    }

    dispatch! {
        data_type::float32 => f32,
        data_type::uint8   => u8,
        data_type::uint16  => u16,
        data_type::uint32  => u32,
        data_type::uint64  => u64,
        data_type::int8    => i8,
        data_type::int16   => i16,
        data_type::int32   => i32,
        data_type::int64   => i64,
    }

    Err(HyperspecError::Format(format!(
        "unsupported array data type '{dt}' in {path_display}; expected a numeric type \
         (float32, float64, uint8, uint16, int16, etc.)",
    )))
}

fn read_fill_value_attr<T: ReadableStorageTraits + ?Sized + 'static>(
    arr: &ZarrArray<T>,
) -> Option<f64> {
    read_f64_json_attr(arr, "_FillValue").or_else(|| read_f64_json_attr(arr, "Data_Ignore_Value"))
}

fn read_f64_json_attr<T: ReadableStorageTraits + ?Sized + 'static>(
    arr: &ZarrArray<T>,
    name: &str,
) -> Option<f64> {
    arr.attributes().get(name)?.as_f64()
}

fn apply_scale_offset(data: &mut Array3<f64>, scale: f64, offset: f64, nodata: Option<f64>) {
    data.mapv_inplace(|v| {
        if v.is_nan() {
            return v;
        }
        if let Some(nd) = nodata
            && v == nd
        {
            return v;
        }
        v * scale + offset
    });
}

/// Orient data to (bands, lines, samples) by matching wavelength count to an axis.
///
/// Errors if multiple axes match `wl_count` (ambiguous orientation).
fn orient_to_bsq(data: Array3<f64>, wl_count: usize, path: &Path) -> Result<Array3<f64>> {
    let shape = data.shape();
    let matches: Vec<usize> = (0..3).filter(|&i| shape[i] == wl_count).collect();

    if matches.len() > 1 {
        return Err(HyperspecError::Format(format!(
            "ambiguous orientation: {} axes match wavelength count {} in shape {:?} from {}; \
             use read_zarr_with_options to specify layout explicitly",
            matches.len(),
            wl_count,
            shape,
            path.display()
        )));
    }

    match matches.first() {
        Some(&0) => Ok(data),
        Some(&2) => {
            // BIP: (lines, samples, bands) → (bands, lines, samples)
            Ok(data
                .view()
                .permuted_axes([2, 0, 1])
                .as_standard_layout()
                .to_owned())
        }
        Some(&1) => {
            // BIL: (lines, bands, samples) → (bands, lines, samples)
            Ok(data
                .view()
                .permuted_axes([1, 0, 2])
                .as_standard_layout()
                .to_owned())
        }
        _ => Err(HyperspecError::Format(format!(
            "cannot match wavelength count {} to any axis of shape {:?} in {}",
            wl_count,
            shape,
            path.display()
        ))),
    }
}

fn read_cube_from_discovered(
    store: &ReadableWritableListableStorage,
    disc: &DiscoveredArrays,
    path: &Path,
) -> Result<SpectralCube> {
    let data_arr = open_zarr_array(store, &disc.data_path, path)?;
    let wl_arr = open_zarr_array(store, &disc.wavelength_path, path)?;
    let fwhm_arr = disc
        .fwhm_path
        .as_ref()
        .map(|p| open_zarr_array(store, p, path))
        .transpose()?;

    let mut data = retrieve_3d_f64(&data_arr, path)?;
    let wavelengths = retrieve_1d_f64(&wl_arr, path)?;
    let fwhm = fwhm_arr
        .as_ref()
        .map(|a| retrieve_1d_f64(a, path))
        .transpose()?;

    let nodata = disc.nodata;

    if disc.scale_factor.is_some() || disc.add_offset.is_some() {
        let scale = disc.scale_factor.unwrap_or(1.0);
        let offset = disc.add_offset.unwrap_or(0.0);
        apply_scale_offset(&mut data, scale, offset, nodata);
    }

    let data = orient_to_bsq(data, wavelengths.len(), path)?;
    SpectralCube::new(data, wavelengths, fwhm, nodata)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_cube() -> SpectralCube {
        let data = Array3::from_shape_fn((3, 4, 5), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let fwhm = Array1::from_vec(vec![10.0, 12.0, 11.0]);
        SpectralCube::new(data, wl, Some(fwhm), Some(-9999.0)).unwrap()
    }

    #[test]
    fn roundtrip_default() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test.zarr");
        let cube = make_cube();

        write_zarr(&cube, &store_path).unwrap();
        let loaded = read_zarr(&store_path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.wavelengths(), cube.wavelengths());
        assert_eq!(loaded.fwhm().unwrap(), cube.fwhm().unwrap());
        assert_eq!(loaded.nodata(), cube.nodata());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn roundtrip_no_fwhm_no_nodata() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_minimal.zarr");
        let data = Array3::from_shape_fn((2, 3, 4), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        write_zarr(&cube, &store_path).unwrap();
        let loaded = read_zarr(&store_path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        assert!(loaded.fwhm().is_none());
        assert!(loaded.nodata().is_none());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn roundtrip_write_options() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_opts.zarr");

        // Cube with spatial dims larger than chunk size to test multi-chunk
        let data = Array3::from_shape_fn((3, 8, 8), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        // ML-style chunks, no compression
        let opts = ZarrWriteOptions {
            compression: ZarrCompression::None,
            chunk_shape: Some([3, 4, 4]),
            overwrite: false,
        };
        write_zarr_with_options(&cube, &store_path, &opts).unwrap();
        let loaded = read_zarr(&store_path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn read_with_scale_offset() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_scale.zarr");

        // Write raw DN values
        let data = Array3::from_shape_fn((2, 3, 4), |(b, _r, _c)| (b as f64 + 1.0) * 10000.0);
        let wl = Array1::from_vec(vec![500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();
        write_zarr(&cube, &store_path).unwrap();

        // Read back with scale_factor to convert DN → reflectance
        let opts = ZarrReadOptions {
            data_path: "/reflectance".to_string(),
            wavelength_path: "/sensor/wavelengths".to_string(),
            fwhm_path: None,
            nodata: None,
            scale_factor: Some(0.0001),
            add_offset: Some(0.0),
        };
        let loaded = read_zarr_with_options(&store_path, &opts).unwrap();

        assert!((loaded.data()[[0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((loaded.data()[[1, 0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn nodata_preserved_through_scale_offset() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_nodata_scale.zarr");

        let mut data = Array3::from_shape_fn((2, 3, 4), |(b, _r, _c)| (b + 1) as f64 * 100.0);
        data[[0, 0, 0]] = -9999.0;
        let wl = Array1::from_vec(vec![500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();
        write_zarr(&cube, &store_path).unwrap();

        let opts = ZarrReadOptions {
            data_path: "/reflectance".to_string(),
            wavelength_path: "/sensor/wavelengths".to_string(),
            fwhm_path: None,
            nodata: Some(-9999.0),
            scale_factor: Some(0.01),
            add_offset: None,
        };
        let loaded = read_zarr_with_options(&store_path, &opts).unwrap();

        assert_eq!(loaded.data()[[0, 0, 0]], -9999.0);
        assert!((loaded.data()[[1, 0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn orient_ambiguous_shape_errors() {
        // All axes equal wl_count — should error, not guess
        let data = Array3::from_shape_fn((4, 4, 4), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let result = orient_to_bsq(data, 4, Path::new("test"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("ambiguous"));
    }

    #[test]
    fn write_fails_if_store_exists() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_exists.zarr");
        let cube = make_cube();

        write_zarr(&cube, &store_path).unwrap();

        // Second write without overwrite should fail
        let result = write_zarr(&cube, &store_path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("already exists"));
    }

    #[test]
    fn write_overwrite_replaces_store() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_overwrite.zarr");

        // Write first cube
        let cube1 = make_cube();
        write_zarr(&cube1, &store_path).unwrap();

        // Write different cube with overwrite
        let data2 = Array3::from_shape_fn((2, 3, 4), |(b, r, c)| (b * 10 + r + c) as f64);
        let wl2 = Array1::from_vec(vec![500.0, 600.0]);
        let cube2 = SpectralCube::new(data2, wl2, None, None).unwrap();

        let opts = ZarrWriteOptions {
            overwrite: true,
            ..Default::default()
        };
        write_zarr_with_options(&cube2, &store_path, &opts).unwrap();

        let loaded = read_zarr(&store_path).unwrap();
        assert_eq!(loaded.shape(), cube2.shape());
        assert_eq!(loaded.data(), cube2.data());
    }

    #[test]
    fn open_nonexistent_store() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.zarr");
        let result = read_zarr(&missing);
        assert!(result.is_err());
    }

    /// Helper to write a non-f64 Zarr store for dtype conversion tests.
    fn write_typed_store<T: zarrs::array::Element>(
        dir: &Path,
        name: &str,
        dtype: zarrs::array::DataType,
        fill: FillValue,
        data_vals: &[T],
        wl_vals: &[f64],
    ) -> std::path::PathBuf {
        let store_path = dir.join(name);
        let store: ReadableWritableListableStorage =
            Arc::new(zarrs::filesystem::FilesystemStore::new(&store_path).unwrap());

        // Root group
        let root = GroupBuilder::new().build(store.clone(), "/").unwrap();
        root.store_metadata().unwrap();

        // Reflectance as T
        let bands = wl_vals.len() as u64;
        let lines = 3u64;
        let samples = 4u64;
        let arr = ArrayBuilder::new(
            vec![bands, lines, samples],
            vec![bands, lines, samples],
            dtype,
            fill,
        )
        .build(store.clone(), "/reflectance")
        .unwrap();
        arr.store_metadata().unwrap();
        arr.store_array_subset(&arr.subset_all(), data_vals)
            .unwrap();

        // Sensor group + wavelengths as f64
        let sensor = GroupBuilder::new().build(store.clone(), "/sensor").unwrap();
        sensor.store_metadata().unwrap();

        let wl_arr = ArrayBuilder::new(
            vec![bands],
            vec![bands],
            data_type::float64(),
            FillValue::from(0.0f64),
        )
        .build(store.clone(), "/sensor/wavelengths")
        .unwrap();
        wl_arr.store_metadata().unwrap();
        wl_arr
            .store_array_subset(&wl_arr.subset_all(), wl_vals)
            .unwrap();

        store_path
    }

    #[test]
    fn read_float32_store() {
        let dir = tempfile::tempdir().unwrap();
        let bands = 2usize;
        let pixels = 3 * 4;
        let data: Vec<f32> = (0..bands * pixels).map(|i| (i as f32) * 0.1).collect();

        let store_path = write_typed_store(
            dir.path(),
            "f32.zarr",
            data_type::float32(),
            FillValue::from(0.0f32),
            &data,
            &[500.0, 600.0],
        );

        // Use auto-discovery, not explicit paths
        let cube = read_zarr(&store_path).unwrap();

        assert_eq!(cube.shape(), (2, 3, 4));
        assert!((cube.data()[[0, 0, 0]] - 0.0).abs() < 1e-6);
        assert!((cube.data()[[1, 0, 0]] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn read_uint16_store() {
        let dir = tempfile::tempdir().unwrap();
        let bands = 2usize;
        let pixels = 3 * 4;
        let data: Vec<u16> = (0..bands * pixels).map(|i| (i as u16) * 100).collect();

        let store_path = write_typed_store(
            dir.path(),
            "u16.zarr",
            data_type::uint16(),
            FillValue::from(0u16),
            &data,
            &[500.0, 600.0],
        );

        // Use auto-discovery, not explicit paths
        let cube = read_zarr(&store_path).unwrap();

        assert_eq!(cube.shape(), (2, 3, 4));
        assert_eq!(cube.data()[[0, 0, 0]], 0.0);
        assert_eq!(cube.data()[[0, 0, 1]], 100.0);
        assert_eq!(cube.data()[[1, 0, 0]], 1200.0);
    }

    #[test]
    fn window_read_spatial_tile() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_window.zarr");

        // Write a 3-band, 8x8 cube with ML-friendly chunks
        let data = Array3::from_shape_fn((3, 8, 8), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let opts = ZarrWriteOptions {
            chunk_shape: Some([3, 4, 4]),
            ..Default::default()
        };
        write_zarr_with_options(&cube, &store_path, &opts).unwrap();

        // Read a 4x4 spatial tile — all bands
        let tile = read_zarr_window(&store_path, 0..3, 0..4, 0..4).unwrap();
        assert_eq!(tile.shape(), (3, 4, 4));
        assert_eq!(tile.wavelengths().len(), 3);
        assert_eq!(tile.data()[[0, 0, 0]], 0.0);
        assert_eq!(tile.data()[[2, 3, 3]], 233.0);

        // Read a different spatial tile
        let tile2 = read_zarr_window(&store_path, 0..3, 4..8, 4..8).unwrap();
        assert_eq!(tile2.shape(), (3, 4, 4));
        assert_eq!(tile2.data()[[0, 4 - 4, 4 - 4]], 44.0); // band 0, row 4, col 4
    }

    #[test]
    fn window_read_band_subset() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_window_bands.zarr");

        let data = Array3::from_shape_fn((5, 4, 4), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![400.0, 500.0, 600.0, 700.0, 800.0]);
        let fwhm = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0]);
        let cube = SpectralCube::new(data, wl, Some(fwhm), None).unwrap();
        write_zarr(&cube, &store_path).unwrap();

        // Read bands 1..3 (500nm, 600nm), full spatial
        let sub = read_zarr_window(&store_path, 1..3, 0..4, 0..4).unwrap();
        assert_eq!(sub.shape(), (2, 4, 4));
        assert_eq!(sub.wavelengths()[0], 500.0);
        assert_eq!(sub.wavelengths()[1], 600.0);
        assert_eq!(sub.fwhm().unwrap()[0], 10.0);
        assert_eq!(sub.data()[[0, 0, 0]], 100.0); // band 1 of original
    }

    #[test]
    fn window_read_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_window_oob.zarr");
        let cube = make_cube(); // (3, 4, 5)
        write_zarr(&cube, &store_path).unwrap();

        let result = read_zarr_window(&store_path, 0..3, 0..999, 0..5);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn window_read_empty_range() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_window_empty.zarr");
        let cube = make_cube();
        write_zarr(&cube, &store_path).unwrap();

        let result = read_zarr_window(&store_path, 0..0, 0..4, 0..5);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-empty"));
    }

    #[test]
    fn shape_query_without_loading() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_shape.zarr");
        let cube = make_cube(); // (3, 4, 5)
        write_zarr(&cube, &store_path).unwrap();

        let (bands, lines, samples) = zarr_cube_shape(&store_path).unwrap();
        assert_eq!((bands, lines, samples), (3, 4, 5));
    }

    #[test]
    fn read_bip_store_correctly_orients() {
        // Write a store where the 3D array is (lines, samples, bands) — BIP layout
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("bip.zarr");
        let store: ReadableWritableListableStorage =
            Arc::new(zarrs::filesystem::FilesystemStore::new(&store_path).unwrap());

        let root = GroupBuilder::new().build(store.clone(), "/").unwrap();
        root.store_metadata().unwrap();

        // Write reflectance as (lines=4, samples=5, bands=3) — BIP order
        let lines = 4u64;
        let samples = 5u64;
        let bands = 3u64;
        let arr = ArrayBuilder::new(
            vec![lines, samples, bands],
            vec![lines, samples, bands],
            data_type::float64(),
            FillValue::from(0.0f64),
        )
        .build(store.clone(), "/reflectance")
        .unwrap();
        arr.store_metadata().unwrap();

        // Fill: value = band*100 + line*10 + sample
        let data: Vec<f64> = (0..lines)
            .flat_map(|l| {
                (0..samples)
                    .flat_map(move |s| (0..bands).map(move |b| (b * 100 + l * 10 + s) as f64))
            })
            .collect();
        arr.store_array_subset(&arr.subset_all(), &data).unwrap();

        // Wavelengths — 3 values signals BIP axis=2
        let sensor = GroupBuilder::new().build(store.clone(), "/sensor").unwrap();
        sensor.store_metadata().unwrap();
        let wl_arr = ArrayBuilder::new(
            vec![bands],
            vec![bands],
            data_type::float64(),
            FillValue::from(0.0f64),
        )
        .build(store.clone(), "/sensor/wavelengths")
        .unwrap();
        wl_arr.store_metadata().unwrap();
        wl_arr
            .store_array_subset(&wl_arr.subset_all(), &[450.0, 550.0, 650.0])
            .unwrap();

        // Read — should auto-orient to (bands=3, lines=4, samples=5)
        let cube = read_zarr(&store_path).unwrap();
        assert_eq!(cube.shape(), (3, 4, 5));
        // Verify data: band=1, line=2, sample=3 → 1*100 + 2*10 + 3 = 123
        assert_eq!(cube.data()[[1, 2, 3]], 123.0);
    }

    #[test]
    fn write_rejects_oversized_chunk_shape() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("test_bad_chunk.zarr");
        let cube = make_cube(); // (3, 4, 5)

        let opts = ZarrWriteOptions {
            chunk_shape: Some([999, 999, 999]),
            ..Default::default()
        };
        let result = write_zarr_with_options(&cube, &store_path, &opts);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("exceeds cube shape"));
    }

    #[test]
    fn write_fails_if_target_is_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("not_a_dir.zarr");
        std::fs::write(&file_path, b"not a zarr store").unwrap();

        let cube = make_cube();
        let opts = ZarrWriteOptions {
            overwrite: true,
            ..Default::default()
        };
        let result = write_zarr_with_options(&cube, &file_path, &opts);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not a directory"));
    }
}
