use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use ndarray::{Array1, Array3};

use crate::SpectralCube;
use crate::error::{HyperspecError, Result};

/// ENVI interleave format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interleave {
    /// Band sequential: (bands, lines, samples).
    Bsq,
    /// Band interleaved by line: (lines, bands, samples).
    Bil,
    /// Band interleaved by pixel: (lines, samples, bands).
    Bip,
}

/// ENVI data type codes.
#[derive(Debug, Clone, Copy)]
enum DataType {
    U8,
    I16,
    I32,
    F32,
    F64,
    U16,
    U32,
    I64,
    U64,
}

impl DataType {
    fn from_code(code: u32) -> Result<Self> {
        match code {
            1 => Ok(Self::U8),
            2 => Ok(Self::I16),
            3 => Ok(Self::I32),
            4 => Ok(Self::F32),
            5 => Ok(Self::F64),
            12 => Ok(Self::U16),
            13 => Ok(Self::U32),
            14 => Ok(Self::I64),
            15 => Ok(Self::U64),
            _ => Err(HyperspecError::Format(format!(
                "unsupported ENVI data type code {code}"
            ))),
        }
    }

    fn to_code(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::I16 => 2,
            Self::I32 => 3,
            Self::F32 => 4,
            Self::F64 => 5,
            Self::U16 => 12,
            Self::U32 => 13,
            Self::I64 => 14,
            Self::U64 => 15,
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Parsed ENVI header.
#[derive(Debug, Clone)]
pub struct EnviHeader {
    pub samples: usize,
    pub lines: usize,
    pub bands: usize,
    pub interleave: Interleave,
    pub wavelengths: Option<Vec<f64>>,
    pub fwhm: Option<Vec<f64>>,
    pub data_ignore_value: Option<f64>,
    data_type: DataType,
    byte_order: ByteOrder,
    header_offset: usize,
}

#[derive(Debug, Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

/// Data type for ENVI output files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnviWriteDataType {
    U8,
    I16,
    I32,
    F32,
    F64,
    U16,
    U32,
    I64,
    U64,
}

impl EnviWriteDataType {
    fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }

    fn to_internal(self) -> DataType {
        match self {
            Self::U8 => DataType::U8,
            Self::I16 => DataType::I16,
            Self::I32 => DataType::I32,
            Self::F32 => DataType::F32,
            Self::F64 => DataType::F64,
            Self::U16 => DataType::U16,
            Self::U32 => DataType::U32,
            Self::I64 => DataType::I64,
            Self::U64 => DataType::U64,
        }
    }
}

/// Options for writing ENVI files.
#[derive(Debug, Clone)]
pub struct EnviWriteOptions {
    /// Output data type. Default: F64.
    pub data_type: EnviWriteDataType,
    /// Output interleave format. Default: BSQ.
    pub interleave: Interleave,
    /// Controls behavior for integer output types when values can't be represented exactly.
    /// If true: clamp out-of-range values, round fractional values, replace NaN with nodata
    /// (or 0 if unset). Nodata metadata is also clamped/rounded to fit the target type.
    /// If false (default): return an error on NaN, out-of-range, fractional values, or
    /// unrepresentable nodata. Float output types (F32, F64) always cast directly.
    pub force: bool,
}

impl Default for EnviWriteOptions {
    fn default() -> Self {
        Self {
            data_type: EnviWriteDataType::F64,
            interleave: Interleave::Bsq,
            force: false,
        }
    }
}

/// Read an ENVI file pair (.hdr + binary) into a `SpectralCube`.
///
/// `path` can point to either the .hdr file or the binary data file.
/// The companion file is located automatically.
pub fn read_envi(path: impl AsRef<Path>) -> Result<SpectralCube> {
    let (hdr_path, data_path) = resolve_envi_paths(path.as_ref())?;
    let header = parse_header(&fs::read_to_string(&hdr_path).map_err(|e| {
        HyperspecError::Io(format!("failed to read header {}: {e}", hdr_path.display()))
    })?)?;
    let data = read_data(&data_path, &header)?;

    let wavelengths = header
        .wavelengths
        .map(Array1::from_vec)
        .unwrap_or_else(|| Array1::from_vec((1..=header.bands).map(|i| i as f64).collect()));
    let fwhm = header.fwhm.map(Array1::from_vec);

    SpectralCube::new(data, wavelengths, fwhm, header.data_ignore_value)
}

/// Write a `SpectralCube` to an ENVI file pair (.hdr + BSQ binary as f64 little-endian).
///
/// For control over data type, interleave, or type conversion behavior,
/// use [`write_envi_with_options`].
pub fn write_envi(cube: &SpectralCube, path: impl AsRef<Path>) -> Result<()> {
    write_envi_with_options(cube, path, &EnviWriteOptions::default())
}

/// Write a `SpectralCube` to an ENVI file pair with explicit options.
///
/// `path` is the base/data-file path. Writes `{path}.hdr` and `{path}`.
/// If `path` ends in `.hdr`, it is treated as the header path and the data file
/// is the same path without the extension.
pub fn write_envi_with_options(
    cube: &SpectralCube,
    path: impl AsRef<Path>,
    options: &EnviWriteOptions,
) -> Result<()> {
    let (hdr_path, data_path) = normalize_output_paths(path.as_ref());
    let dtype = options.data_type.to_internal();

    // Compute effective nodata for the output type. In strict mode this rejects
    // nodata values that can't be represented exactly in the target integer type.
    // In force mode, nodata is clamped/rounded to fit.
    let effective_nodata = effective_output_nodata(cube.nodata(), options)?;

    // For strict integer writes, validate all values before writing any files.
    // This runs on the raw BSQ data (same values regardless of output interleave).
    if options.data_type.is_integer() && !options.force {
        let data = cube.data();
        let contiguous = data.as_standard_layout();
        let slice = contiguous
            .as_slice()
            .expect("standard layout should be contiguous");
        validate_integer_conversion(slice, options.data_type)?;
    }

    // Write to temp files first, then rename atomically to avoid partial output.
    let tmp_hdr = hdr_path.with_extension("hdr.tmp");
    let tmp_data = data_path.with_extension("tmp");

    let write_result = (|| {
        write_header_with_options(cube, &tmp_hdr, dtype, options.interleave, effective_nodata)?;
        write_data_with_options(cube, &tmp_data, options, effective_nodata)?;
        Ok(())
    })();

    if let Err(e) = write_result {
        // Clean up temp files on failure.
        let _ = fs::remove_file(&tmp_hdr);
        let _ = fs::remove_file(&tmp_data);
        return Err(e);
    }

    // Rename into place. Data first so header always points to complete data.
    fs::rename(&tmp_data, &data_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_hdr);
        let _ = fs::remove_file(&tmp_data);
        HyperspecError::Io(format!("failed to rename data file: {e}"))
    })?;
    fs::rename(&tmp_hdr, &hdr_path).map_err(|e| {
        // Data already renamed — not much we can do, but clean up temp header.
        let _ = fs::remove_file(&tmp_hdr);
        HyperspecError::Io(format!("failed to rename header file: {e}"))
    })?;

    Ok(())
}

// --- Path resolution ---

fn has_hdr_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("hdr"))
}

fn normalize_output_paths(path: &Path) -> (PathBuf, PathBuf) {
    if has_hdr_extension(path) {
        let data_path = path.with_extension("");
        (path.to_path_buf(), data_path)
    } else {
        (path.with_extension("hdr"), path.to_path_buf())
    }
}

fn resolve_envi_paths(path: &Path) -> Result<(PathBuf, PathBuf)> {
    if has_hdr_extension(path) {
        let data_path = path.with_extension("");
        if data_path.exists() {
            return Ok((path.to_path_buf(), data_path));
        }
        // Try common binary extensions
        for ext in &["dat", "img", "bsq", "bil", "bip", "raw"] {
            let p = path.with_extension(ext);
            if p.exists() {
                return Ok((path.to_path_buf(), p));
            }
        }
        Err(HyperspecError::Io(format!(
            "no data file found for header {}",
            path.display()
        )))
    } else {
        // Try replacing extension with .hdr (scene.dat -> scene.hdr)
        let hdr_path = path.with_extension("hdr");
        if hdr_path.exists() {
            return Ok((hdr_path, path.to_path_buf()));
        }
        // Try appending .hdr (scene.dat -> scene.dat.hdr)
        let hdr_path2 = path.with_file_name(format!(
            "{}.hdr",
            path.file_name()
                .ok_or_else(|| HyperspecError::Io("invalid path".to_string()))?
                .to_string_lossy()
        ));
        if hdr_path2.exists() {
            return Ok((hdr_path2, path.to_path_buf()));
        }
        Err(HyperspecError::Io(format!(
            "no .hdr file found for {}",
            path.display()
        )))
    }
}

// --- Header parsing ---

fn parse_header(text: &str) -> Result<EnviHeader> {
    // Strip UTF-8 BOM if present, then trim whitespace.
    let text = text.trim_start_matches('\u{feff}').trim();
    if !text.starts_with("ENVI") {
        return Err(HyperspecError::Format(
            "ENVI header must start with 'ENVI'".to_string(),
        ));
    }

    let mut samples: Option<usize> = None;
    let mut lines: Option<usize> = None;
    let mut bands: Option<usize> = None;
    let mut data_type: Option<u32> = None;
    let mut interleave: Option<Interleave> = None;
    let mut byte_order = ByteOrder::Little;
    let mut header_offset: usize = 0;
    let mut wavelengths: Option<Vec<f64>> = None;
    let mut fwhm: Option<Vec<f64>> = None;
    let mut data_ignore_value: Option<f64> = None;

    // Join continuation lines (values in { } can span multiple lines)
    let joined = join_continuation_lines(text);

    for line in joined.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_lowercase();
        let value = value.trim();

        match key.as_str() {
            "samples" => samples = Some(parse_usize(value, "samples")?),
            "lines" => lines = Some(parse_usize(value, "lines")?),
            "bands" => bands = Some(parse_usize(value, "bands")?),
            "data type" => data_type = Some(parse_usize(value, "data type")? as u32),
            "interleave" => {
                interleave = Some(match value.to_lowercase().as_str() {
                    "bsq" => Interleave::Bsq,
                    "bil" => Interleave::Bil,
                    "bip" => Interleave::Bip,
                    _ => {
                        return Err(HyperspecError::Format(format!(
                            "unknown interleave: {value}"
                        )));
                    }
                });
            }
            "byte order" => {
                byte_order = match value {
                    "0" => ByteOrder::Little,
                    "1" => ByteOrder::Big,
                    _ => {
                        return Err(HyperspecError::Format(format!(
                            "unknown byte order: {value}"
                        )));
                    }
                };
            }
            "header offset" => header_offset = parse_usize(value, "header offset")?,
            "wavelength" => wavelengths = Some(parse_brace_list(value)?),
            "fwhm" => fwhm = Some(parse_brace_list(value)?),
            "data ignore value" => {
                data_ignore_value = Some(value.parse::<f64>().map_err(|_| {
                    HyperspecError::Format(format!("invalid data ignore value: {value}"))
                })?);
            }
            _ => {} // ignore unknown keys
        }
    }

    let samples =
        samples.ok_or_else(|| HyperspecError::Format("missing 'samples' in header".into()))?;
    let lines = lines.ok_or_else(|| HyperspecError::Format("missing 'lines' in header".into()))?;
    let bands = bands.ok_or_else(|| HyperspecError::Format("missing 'bands' in header".into()))?;
    let data_type = DataType::from_code(
        data_type.ok_or_else(|| HyperspecError::Format("missing 'data type' in header".into()))?,
    )?;
    let interleave = interleave
        .ok_or_else(|| HyperspecError::Format("missing 'interleave' in header".into()))?;

    if let Some(ref wl) = wavelengths
        && wl.len() != bands
    {
        return Err(HyperspecError::Format(format!(
            "wavelength count {} does not match band count {bands}",
            wl.len()
        )));
    }
    if let Some(ref f) = fwhm
        && f.len() != bands
    {
        return Err(HyperspecError::Format(format!(
            "fwhm count {} does not match band count {bands}",
            f.len()
        )));
    }

    Ok(EnviHeader {
        samples,
        lines,
        bands,
        data_type,
        interleave,
        byte_order,
        header_offset,
        wavelengths,
        fwhm,
        data_ignore_value,
    })
}

/// Join lines that are part of a brace-delimited list into single lines.
fn join_continuation_lines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_braces = false;

    for line in text.lines() {
        if in_braces {
            result.push(' ');
            result.push_str(line.trim());
            if line.contains('}') {
                in_braces = false;
                result.push('\n');
            }
        } else if line.contains('{') && !line.contains('}') {
            result.push_str(line);
            in_braces = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn parse_usize(s: &str, field: &str) -> Result<usize> {
    s.trim()
        .parse()
        .map_err(|_| HyperspecError::Format(format!("invalid {field}: {s}")))
}

fn parse_brace_list(s: &str) -> Result<Vec<f64>> {
    let s = s.trim();
    let inner = s
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| HyperspecError::Format("expected {values} list".to_string()))?;

    inner
        .split(',')
        .map(|v| {
            v.trim().parse::<f64>().map_err(|_| {
                HyperspecError::Format(format!("invalid number in list: {}", v.trim()))
            })
        })
        .collect()
}

// --- Binary data reading ---

fn read_data(path: &Path, header: &EnviHeader) -> Result<Array3<f64>> {
    let file = fs::File::open(path)
        .map_err(|e| HyperspecError::Io(format!("failed to open {}: {e}", path.display())))?;

    // SAFETY: the file is opened read-only and we only read from the mapping.
    // Concurrent modification by another process would be UB, but that applies to
    // any file I/O and is the caller's responsibility.
    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| HyperspecError::Io(format!("failed to mmap {}: {e}", path.display())))?
    };

    if header.header_offset > mmap.len() {
        return Err(HyperspecError::Format(format!(
            "header offset {} exceeds file size {}",
            header.header_offset,
            mmap.len()
        )));
    }
    let bytes = &mmap[header.header_offset..];

    let expected_pixels = header
        .bands
        .checked_mul(header.lines)
        .and_then(|n| n.checked_mul(header.samples))
        .ok_or_else(|| HyperspecError::Format("cube dimensions overflow".to_string()))?;
    let expected_bytes = expected_pixels
        .checked_mul(header.data_type.byte_size())
        .ok_or_else(|| HyperspecError::Format("data size overflow".to_string()))?;

    if bytes.len() < expected_bytes {
        return Err(HyperspecError::Format(format!(
            "data file too small: expected {expected_bytes} bytes, got {}",
            bytes.len()
        )));
    }

    let flat = read_pixels(bytes, expected_pixels, header.data_type, header.byte_order)?;
    reshape(flat, header)
}

/// Read raw bytes into f64 values, handling data type and byte order.
fn read_pixels(bytes: &[u8], count: usize, dtype: DataType, order: ByteOrder) -> Result<Vec<f64>> {
    // Fast path: native-endian f64 — reinterpret bytes directly, no per-element conversion.
    // SAFETY: f64 has no invalid bit patterns; alignment is checked by align_to.
    #[cfg(target_endian = "little")]
    if matches!(dtype, DataType::F64) && matches!(order, ByteOrder::Little) {
        let (prefix, f64s, suffix) = unsafe { bytes[..count * 8].align_to::<f64>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(f64s.to_vec());
        }
    }
    #[cfg(target_endian = "big")]
    if matches!(dtype, DataType::F64) && matches!(order, ByteOrder::Big) {
        let (prefix, f64s, suffix) = unsafe { bytes[..count * 8].align_to::<f64>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(f64s.to_vec());
        }
    }

    let mut out = vec![0.0f64; count];
    let bsize = dtype.byte_size();

    macro_rules! convert {
        ($ty:ty) => {{
            for (chunk, val) in bytes[..count * bsize]
                .chunks_exact(bsize)
                .zip(out.iter_mut())
            {
                let arr: [u8; std::mem::size_of::<$ty>()] = chunk.try_into().unwrap();
                *val = match order {
                    ByteOrder::Little => <$ty>::from_le_bytes(arr),
                    ByteOrder::Big => <$ty>::from_be_bytes(arr),
                } as f64;
            }
        }};
    }

    match dtype {
        DataType::U8 => {
            for (src, val) in bytes.iter().zip(out.iter_mut()) {
                *val = *src as f64;
            }
        }
        DataType::I16 => convert!(i16),
        DataType::U16 => convert!(u16),
        DataType::I32 => convert!(i32),
        DataType::U32 => convert!(u32),
        DataType::F32 => convert!(f32),
        DataType::F64 => convert!(f64),
        DataType::I64 => convert!(i64),
        DataType::U64 => convert!(u64),
    }

    Ok(out)
}

/// Reshape flat pixel array into (bands, lines, samples) based on interleave.
fn reshape(flat: Vec<f64>, header: &EnviHeader) -> Result<Array3<f64>> {
    let (b, l, s) = (header.bands, header.lines, header.samples);

    match header.interleave {
        Interleave::Bsq => {
            // Already (bands, lines, samples)
            Array3::from_shape_vec((b, l, s), flat)
                .map_err(|e| HyperspecError::Format(format!("failed to reshape BSQ data: {e}")))
        }
        Interleave::Bil => {
            // Stored as (lines, bands, samples) → transpose to (bands, lines, samples)
            let src = Array3::from_shape_vec((l, b, s), flat)
                .map_err(|e| HyperspecError::Format(format!("failed to reshape BIL data: {e}")))?;
            Ok(src.permuted_axes([1, 0, 2]).as_standard_layout().to_owned())
        }
        Interleave::Bip => {
            // Stored as (lines, samples, bands) → transpose to (bands, lines, samples)
            let src = Array3::from_shape_vec((l, s, b), flat)
                .map_err(|e| HyperspecError::Format(format!("failed to reshape BIP data: {e}")))?;
            Ok(src.permuted_axes([2, 0, 1]).as_standard_layout().to_owned())
        }
    }
}

// --- Writing ---

/// Compute the nodata value that will actually be stored in the output file.
/// For float types, nodata passes through unchanged.
/// For integer types in strict mode, rejects nodata that can't be represented exactly.
/// For integer types in force mode, clamps/rounds nodata to fit the target range.
fn effective_output_nodata(
    cube_nodata: Option<f64>,
    options: &EnviWriteOptions,
) -> Result<Option<f64>> {
    let Some(nodata) = cube_nodata else {
        return Ok(None);
    };

    if !options.data_type.is_integer() {
        return Ok(Some(nodata));
    }

    macro_rules! check_nodata {
        ($ty:ty, $name:expr) => {{
            let min = <$ty>::MIN as f64;
            let max = <$ty>::MAX as f64;
            if options.force {
                Ok(Some(nodata.clamp(min, max).round()))
            } else {
                if nodata < min || nodata > max {
                    return Err(HyperspecError::InvalidInput(format!(
                        "nodata value {nodata} out of range for {} [{min}, {max}], use force=true to clamp",
                        $name
                    )));
                }
                if nodata.fract() != 0.0 {
                    return Err(HyperspecError::InvalidInput(format!(
                        "nodata value {nodata} has fractional part, cannot represent as {}, use force=true to round",
                        $name
                    )));
                }
                Ok(Some(nodata))
            }
        }};
    }

    match options.data_type {
        EnviWriteDataType::U8 => check_nodata!(u8, "u8"),
        EnviWriteDataType::I16 => check_nodata!(i16, "i16"),
        EnviWriteDataType::U16 => check_nodata!(u16, "u16"),
        EnviWriteDataType::I32 => check_nodata!(i32, "i32"),
        EnviWriteDataType::U32 => check_nodata!(u32, "u32"),
        EnviWriteDataType::I64 => check_nodata!(i64, "i64"),
        EnviWriteDataType::U64 => check_nodata!(u64, "u64"),
        EnviWriteDataType::F32 | EnviWriteDataType::F64 => Ok(Some(nodata)),
    }
}

fn write_header_with_options(
    cube: &SpectralCube,
    path: &Path,
    dtype: DataType,
    interleave: Interleave,
    effective_nodata: Option<f64>,
) -> Result<()> {
    let il_str = match interleave {
        Interleave::Bsq => "bsq",
        Interleave::Bil => "bil",
        Interleave::Bip => "bip",
    };

    let mut hdr = String::new();
    hdr.push_str("ENVI\n");
    hdr.push_str(&format!("samples = {}\n", cube.width()));
    hdr.push_str(&format!("lines = {}\n", cube.height()));
    hdr.push_str(&format!("bands = {}\n", cube.bands()));
    hdr.push_str(&format!("data type = {}\n", dtype.to_code()));
    hdr.push_str(&format!("interleave = {il_str}\n"));
    hdr.push_str("byte order = 0\n");
    hdr.push_str("header offset = 0\n");

    if let Some(nodata) = effective_nodata {
        hdr.push_str(&format!("data ignore value = {nodata}\n"));
    }

    // Wavelengths are guaranteed contiguous by SpectralCube::new validation.
    write_brace_list(
        &mut hdr,
        "wavelength",
        cube.wavelengths().as_slice().unwrap(),
    );

    if let Some(fwhm) = cube.fwhm() {
        write_brace_list(&mut hdr, "fwhm", fwhm.as_slice().unwrap());
    }

    fs::write(path, hdr)
        .map_err(|e| HyperspecError::Io(format!("failed to write header {}: {e}", path.display())))
}

fn write_brace_list(hdr: &mut String, key: &str, values: &[f64]) {
    hdr.push_str(&format!("{key} = {{\n"));
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            hdr.push_str(",\n");
        }
        hdr.push_str(&format!("  {v}"));
    }
    hdr.push_str("\n}\n");
}

fn write_data_with_options(
    cube: &SpectralCube,
    path: &Path,
    options: &EnviWriteOptions,
    effective_nodata: Option<f64>,
) -> Result<()> {
    let data = cube.data();

    // Reorder from internal BSQ (bands, lines, samples) to target interleave.
    // For BSQ, borrow directly if already contiguous to avoid a full-cube copy.
    let owned;
    let slice: &[f64] = match options.interleave {
        Interleave::Bsq => {
            if let Some(s) = data.as_slice() {
                s
            } else {
                owned = data.as_standard_layout().into_owned();
                owned
                    .as_slice()
                    .expect("standard layout should be contiguous")
            }
        }
        Interleave::Bil => {
            owned = data
                .view()
                .permuted_axes([1, 0, 2])
                .as_standard_layout()
                .into_owned();
            owned
                .as_slice()
                .expect("standard layout should be contiguous")
        }
        Interleave::Bip => {
            owned = data
                .view()
                .permuted_axes([1, 2, 0])
                .as_standard_layout()
                .into_owned();
            owned
                .as_slice()
                .expect("standard layout should be contiguous")
        }
    };

    let file = fs::File::create(path)
        .map_err(|e| HyperspecError::Io(format!("failed to create {}: {e}", path.display())))?;
    let mut writer = BufWriter::new(file);

    let write_err =
        |e: std::io::Error| HyperspecError::Io(format!("failed to write {}: {e}", path.display()));

    match options.data_type {
        EnviWriteDataType::F64 => {
            // SAFETY: &[f64] → &[u8] reinterpret is safe; slice is contiguous.
            #[cfg(target_endian = "little")]
            {
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 8)
                };
                writer.write_all(byte_slice).map_err(write_err)?;
            }
            #[cfg(target_endian = "big")]
            for v in slice {
                writer.write_all(&v.to_le_bytes()).map_err(write_err)?;
            }
        }
        EnviWriteDataType::F32 => {
            for &v in slice {
                writer
                    .write_all(&(v as f32).to_le_bytes())
                    .map_err(write_err)?;
            }
        }
        _ => {
            // Validation already happened in write_envi_with_options for strict mode.
            // effective_nodata is the single source of truth for NaN replacement.
            write_integer_data(&mut writer, slice, options, effective_nodata).map_err(write_err)?;
        }
    }

    Ok(())
}

/// Convert f64 values to integer bytes and write them.
/// Strict validation has already been done upfront in `write_envi_with_options`.
/// `effective_nodata` is the pre-computed nodata value for the target type (same value
/// written to the header), used as the NaN replacement in force mode.
fn write_integer_data(
    writer: &mut BufWriter<fs::File>,
    slice: &[f64],
    options: &EnviWriteOptions,
    effective_nodata: Option<f64>,
) -> std::io::Result<()> {
    let nan_replace = effective_nodata.unwrap_or(0.0);

    macro_rules! write_int {
        ($ty:ty) => {{
            let min = <$ty>::MIN as f64;
            let max = <$ty>::MAX as f64;
            for &v in slice {
                let converted = if v.is_nan() {
                    nan_replace.clamp(min, max).round() as $ty
                } else if options.force {
                    v.clamp(min, max).round() as $ty
                } else {
                    v as $ty
                };
                writer.write_all(&converted.to_le_bytes())?;
            }
        }};
    }

    match options.data_type {
        EnviWriteDataType::U8 => write_int!(u8),
        EnviWriteDataType::I16 => write_int!(i16),
        EnviWriteDataType::U16 => write_int!(u16),
        EnviWriteDataType::I32 => write_int!(i32),
        EnviWriteDataType::U32 => write_int!(u32),
        EnviWriteDataType::I64 => write_int!(i64),
        EnviWriteDataType::U64 => write_int!(u64),
        EnviWriteDataType::F32 | EnviWriteDataType::F64 => unreachable!(),
    }

    Ok(())
}

/// Check that all f64 values can be exactly represented in the target integer type.
fn validate_integer_conversion(slice: &[f64], dtype: EnviWriteDataType) -> Result<()> {
    macro_rules! check {
        ($ty:ty, $name:expr) => {{
            let min = <$ty>::MIN as f64;
            let max = <$ty>::MAX as f64;
            for (i, &v) in slice.iter().enumerate() {
                if v.is_nan() {
                    return Err(HyperspecError::InvalidInput(format!(
                        "NaN at index {i} cannot be represented as {}, use force=true to replace with nodata",
                        $name
                    )));
                }
                if v < min || v > max {
                    return Err(HyperspecError::InvalidInput(format!(
                        "value {v} at index {i} out of range for {} [{min}, {max}], use force=true to clamp",
                        $name
                    )));
                }
                if v.fract() != 0.0 {
                    return Err(HyperspecError::InvalidInput(format!(
                        "value {v} at index {i} has fractional part, cannot write as {}, use force=true to round",
                        $name
                    )));
                }
            }
        }};
    }

    match dtype {
        EnviWriteDataType::U8 => check!(u8, "u8"),
        EnviWriteDataType::I16 => check!(i16, "i16"),
        EnviWriteDataType::U16 => check!(u16, "u16"),
        EnviWriteDataType::I32 => check!(i32, "i32"),
        EnviWriteDataType::U32 => check!(u32, "u32"),
        EnviWriteDataType::I64 => check!(i64, "i64"),
        EnviWriteDataType::U64 => check!(u64, "u64"),
        EnviWriteDataType::F32 | EnviWriteDataType::F64 => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn make_cube() -> SpectralCube {
        let data = Array3::from_shape_fn((3, 2, 4), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![450.0, 550.0, 650.0]);
        let fwhm = Array1::from_vec(vec![10.0, 10.0, 10.0]);
        SpectralCube::new(data, wl, Some(fwhm), Some(-9999.0)).unwrap()
    }

    /// Helper: write a minimal ENVI file pair from header string + raw bytes.
    fn write_test_envi(dir: &Path, hdr: &str, data: &[u8]) -> (PathBuf, PathBuf) {
        let hdr_path = dir.join("test.hdr");
        let data_path = dir.join("test");
        fs::write(&hdr_path, hdr).unwrap();
        fs::write(&data_path, data).unwrap();
        (hdr_path, data_path)
    }

    #[test]
    fn roundtrip_bsq() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        write_envi(&cube, &path).unwrap();
        let loaded = read_envi(&path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.wavelengths(), cube.wavelengths());
        assert_eq!(loaded.fwhm().unwrap(), cube.fwhm().unwrap());
        assert_eq!(loaded.nodata(), cube.nodata());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn roundtrip_via_hdr_path() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        write_envi(&cube, &path).unwrap();
        let loaded = read_envi(dir.path().join("test.hdr")).unwrap();
        assert_eq!(loaded.shape(), cube.shape());
    }

    #[test]
    fn write_via_hdr_path() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let hdr_path = dir.path().join("scene.hdr");

        write_envi(&cube, &hdr_path).unwrap();

        // Should create scene.hdr and scene (data file)
        assert!(hdr_path.exists());
        assert!(dir.path().join("scene").exists());

        let loaded = read_envi(&hdr_path).unwrap();
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn truncated_data_file() {
        let dir = TempDir::new().unwrap();
        let hdr = "ENVI\nsamples = 10\nlines = 10\nbands = 5\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {400.0, 500.0, 600.0, 700.0, 800.0}\n";
        let (_, data_path) = write_test_envi(dir.path(), hdr, &[0u8; 8]);

        let err = read_envi(&data_path).unwrap_err();
        assert!(err.to_string().contains("too small"));
    }

    #[test]
    fn header_offset_exceeds_file_size() {
        let dir = TempDir::new().unwrap();
        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 1\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 99999\n\
                   wavelength = {500.0}\n";
        let (_, data_path) = write_test_envi(dir.path(), hdr, &[0u8; 8]);

        let err = read_envi(&data_path).unwrap_err();
        assert!(err.to_string().contains("header offset"));
    }

    #[test]
    fn read_bil() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 4\nlines = 3\nbands = 2\ndata type = 5\n\
                   interleave = bil\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {500.0, 600.0}\n";

        // BIL layout: for each line, write all bands
        let mut data = Vec::new();
        for l in 0..3usize {
            for b in 0..2usize {
                for s in 0..4usize {
                    let val = (b * 100 + l * 10 + s) as f64;
                    data.write_all(&val.to_le_bytes()).unwrap();
                }
            }
        }
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.shape(), (2, 3, 4));
        assert_eq!(cube.data()[[0, 1, 2]], 12.0);
        assert_eq!(cube.data()[[1, 1, 2]], 112.0);
    }

    #[test]
    fn read_bip() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 4\nlines = 3\nbands = 2\ndata type = 5\n\
                   interleave = bip\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {500.0, 600.0}\n";

        // BIP layout: for each pixel, write all bands
        let mut data = Vec::new();
        for l in 0..3usize {
            for s in 0..4usize {
                for b in 0..2usize {
                    let val = (b * 100 + l * 10 + s) as f64;
                    data.write_all(&val.to_le_bytes()).unwrap();
                }
            }
        }
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.shape(), (2, 3, 4));
        assert_eq!(cube.data()[[0, 1, 2]], 12.0);
        assert_eq!(cube.data()[[1, 1, 2]], 112.0);
    }

    #[test]
    fn read_f32_big_endian() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 2\nlines = 1\nbands = 1\ndata type = 4\n\
                   interleave = bsq\nbyte order = 1\nheader offset = 0\n\
                   wavelength = {500.0}\n";

        let mut data = Vec::new();
        data.write_all(&(1.5f32).to_be_bytes()).unwrap();
        data.write_all(&(2.5f32).to_be_bytes()).unwrap();
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.shape(), (1, 1, 2));
        assert!((cube.data()[[0, 0, 0]] - 1.5).abs() < 1e-6);
        assert!((cube.data()[[0, 0, 1]] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn read_f64_big_endian() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 2\nlines = 1\nbands = 1\ndata type = 5\n\
                   interleave = bsq\nbyte order = 1\nheader offset = 0\n\
                   wavelength = {500.0}\n";

        let mut data = Vec::new();
        data.write_all(&(3.25f64).to_be_bytes()).unwrap();
        data.write_all(&(2.75f64).to_be_bytes()).unwrap();
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert!((cube.data()[[0, 0, 0]] - 3.25).abs() < 1e-10);
        assert!((cube.data()[[0, 0, 1]] - 2.75).abs() < 1e-10);
    }

    #[test]
    fn read_i16() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 2\nlines = 1\nbands = 1\ndata type = 2\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {500.0}\n";

        let mut data = Vec::new();
        data.write_all(&100i16.to_le_bytes()).unwrap();
        data.write_all(&(-200i16).to_le_bytes()).unwrap();
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.data()[[0, 0, 0]], 100.0);
        assert_eq!(cube.data()[[0, 0, 1]], -200.0);
    }

    #[test]
    fn read_u16() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 2\nlines = 1\nbands = 1\ndata type = 12\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {500.0}\n";

        let mut data = Vec::new();
        data.write_all(&1000u16.to_le_bytes()).unwrap();
        data.write_all(&65535u16.to_le_bytes()).unwrap();
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.data()[[0, 0, 0]], 1000.0);
        assert_eq!(cube.data()[[0, 0, 1]], 65535.0);
    }

    #[test]
    fn read_i32() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 2\nlines = 1\nbands = 1\ndata type = 3\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {500.0}\n";

        let mut data = Vec::new();
        data.write_all(&(-100000i32).to_le_bytes()).unwrap();
        data.write_all(&200000i32.to_le_bytes()).unwrap();
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.data()[[0, 0, 0]], -100000.0);
        assert_eq!(cube.data()[[0, 0, 1]], 200000.0);
    }

    #[test]
    fn multiline_wavelength_header() {
        let hdr = "ENVI\nsamples = 2\nlines = 1\nbands = 3\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\n\
                   wavelength = {\n  400.0,\n  500.0,\n  600.0\n}\n";
        let parsed = parse_header(hdr).unwrap();
        assert_eq!(parsed.wavelengths.unwrap(), vec![400.0, 500.0, 600.0]);
    }

    #[test]
    fn multiline_fwhm_header() {
        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 2\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\n\
                   wavelength = {500.0, 600.0}\n\
                   fwhm = {\n  10.0,\n  12.5\n}\n";
        let parsed = parse_header(hdr).unwrap();
        assert_eq!(parsed.fwhm.unwrap(), vec![10.0, 12.5]);
    }

    #[test]
    fn header_with_bom() {
        let hdr = "\u{feff}ENVI\nsamples = 1\nlines = 1\nbands = 1\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nwavelength = {500.0}\n";
        let parsed = parse_header(hdr).unwrap();
        assert_eq!(parsed.bands, 1);
    }

    #[test]
    fn header_with_comments() {
        let hdr = "ENVI\n; This is a comment\nsamples = 2\n; Another comment\n\
                   lines = 1\nbands = 1\ndata type = 5\ninterleave = bsq\n\
                   byte order = 0\nwavelength = {500.0}\n";
        let parsed = parse_header(hdr).unwrap();
        assert_eq!(parsed.samples, 2);
    }

    #[test]
    fn header_with_unknown_fields() {
        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 1\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nsensor type = AVIRIS\n\
                   description = {test file}\nwavelength = {500.0}\n";
        let parsed = parse_header(hdr).unwrap();
        assert_eq!(parsed.bands, 1);
    }

    #[test]
    fn header_offset() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 1\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 16\n\
                   wavelength = {500.0}\n";

        let mut data = vec![0u8; 16]; // 16 bytes of padding
        data.extend_from_slice(&(42.0f64).to_le_bytes());
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.data()[[0, 0, 0]], 42.0);
    }

    #[test]
    fn extra_trailing_bytes() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 1\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 0\n\
                   wavelength = {500.0}\n";

        // 8 bytes of real data + 100 bytes of trailing garbage
        let mut data = Vec::new();
        data.write_all(&(7.0f64).to_le_bytes()).unwrap();
        data.extend_from_slice(&[0xFFu8; 100]);
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        // Should succeed — we only read expected_bytes
        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.data()[[0, 0, 0]], 7.0);
    }

    #[test]
    fn missing_header_field() {
        let hdr = "ENVI\nsamples = 2\nlines = 1\n";
        assert!(parse_header(hdr).is_err());
    }

    #[test]
    fn not_envi_header() {
        assert!(parse_header("NOT ENVI\nsamples = 1\n").is_err());
    }

    #[test]
    fn no_wavelengths_generates_indices() {
        let dir = TempDir::new().unwrap();

        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 3\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\nheader offset = 0\n";

        let data: Vec<u8> = (0..3).flat_map(|i| (i as f64).to_le_bytes()).collect();
        let (_, data_path) = write_test_envi(dir.path(), hdr, &data);

        let cube = read_envi(&data_path).unwrap();
        assert_eq!(cube.wavelengths().as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn wavelength_count_mismatch() {
        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 3\ndata type = 5\n\
                   interleave = bsq\nbyte order = 0\n\
                   wavelength = {400.0, 500.0}\n";
        let err = parse_header(hdr).unwrap_err();
        assert!(err.to_string().contains("wavelength count"));
    }

    // --- EnviWriteOptions tests ---

    #[test]
    fn write_f32() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::F32,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();
        let loaded = read_envi(&path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        // f32 roundtrip loses some precision but these values are exact in f32
        assert_eq!(loaded.data()[[0, 0, 0]], cube.data()[[0, 0, 0]]);
    }

    #[test]
    fn write_i16_exact_values() {
        // Cube with values that fit in i16 exactly
        let data = Array3::from_shape_fn((2, 2, 2), |(b, r, c)| (b * 100 + r * 10 + c) as f64);
        let wl = Array1::from_vec(vec![500.0, 600.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();
        let loaded = read_envi(&path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.data()[[1, 1, 1]], 111.0);
    }

    #[test]
    fn write_i16_rejects_nan_strict() {
        let mut data = Array3::zeros((1, 1, 2));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = f64::NAN;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: false,
            ..Default::default()
        };
        let err = write_envi_with_options(&cube, &path, &opts).unwrap_err();
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn write_i16_rejects_out_of_range_strict() {
        let mut data = Array3::zeros((1, 1, 1));
        data[[0, 0, 0]] = 99999.0;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: false,
            ..Default::default()
        };
        let err = write_envi_with_options(&cube, &path, &opts).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn write_i16_rejects_fractional_strict() {
        let mut data = Array3::zeros((1, 1, 1));
        data[[0, 0, 0]] = 1.5;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: false,
            ..Default::default()
        };
        let err = write_envi_with_options(&cube, &path, &opts).unwrap_err();
        assert!(err.to_string().contains("fractional"));
    }

    #[test]
    fn write_i16_force_clamps_and_rounds() {
        let mut data = Array3::zeros((1, 1, 3));
        data[[0, 0, 0]] = 99999.0; // out of range → clamp to 32767
        data[[0, 0, 1]] = -99999.0; // out of range → clamp to -32768
        data[[0, 0, 2]] = 1.7; // fractional → round to 2
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: true,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();
        let loaded = read_envi(&path).unwrap();

        assert_eq!(loaded.data()[[0, 0, 0]], 32767.0);
        assert_eq!(loaded.data()[[0, 0, 1]], -32768.0);
        assert_eq!(loaded.data()[[0, 0, 2]], 2.0);
    }

    #[test]
    fn write_i16_force_nan_becomes_nodata() {
        let mut data = Array3::zeros((1, 1, 2));
        data[[0, 0, 0]] = 42.0;
        data[[0, 0, 1]] = f64::NAN;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: true,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();
        let loaded = read_envi(&path).unwrap();

        assert_eq!(loaded.data()[[0, 0, 0]], 42.0);
        assert_eq!(loaded.data()[[0, 0, 1]], -9999.0);
    }

    #[test]
    fn write_bil_interleave() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            interleave: Interleave::Bil,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();

        // Verify header says bil
        let hdr = fs::read_to_string(dir.path().join("test.hdr")).unwrap();
        assert!(hdr.contains("interleave = bil"));

        // Roundtrip should produce same data
        let loaded = read_envi(&path).unwrap();
        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn write_bip_interleave() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            interleave: Interleave::Bip,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();

        let hdr = fs::read_to_string(dir.path().join("test.hdr")).unwrap();
        assert!(hdr.contains("interleave = bip"));

        let loaded = read_envi(&path).unwrap();
        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn write_u8_force_nodata_clamped_in_header() {
        // Cube nodata is -9999.0, which can't fit in u8. The header should
        // reflect the effective clamped value (0), not the original.
        let mut data = Array3::zeros((1, 1, 2));
        data[[0, 0, 0]] = 42.0;
        data[[0, 0, 1]] = f64::NAN;
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::U8,
            force: true,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();

        // Header should say "data ignore value = 0", not -9999
        let hdr = fs::read_to_string(dir.path().join("test.hdr")).unwrap();
        assert!(hdr.contains("data ignore value = 0"));
        assert!(!hdr.contains("-9999"));

        // Data: NaN → 0 (clamped nodata), 42 → 42
        let loaded = read_envi(&path).unwrap();
        assert_eq!(loaded.data()[[0, 0, 0]], 42.0);
        assert_eq!(loaded.data()[[0, 0, 1]], 0.0);
    }

    #[test]
    fn write_u8_strict_rejects_unrepresentable_nodata() {
        // nodata=-9999 can't be represented in u8, strict mode should reject
        let data = Array3::from_shape_fn((1, 1, 1), |_| 42.0);
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::U8,
            force: false,
            ..Default::default()
        };
        let err = write_envi_with_options(&cube, &path, &opts).unwrap_err();
        assert!(err.to_string().contains("nodata"));
    }

    #[test]
    fn write_i16_strict_accepts_representable_nodata() {
        // nodata=-9999 fits in i16 exactly
        let data = Array3::from_shape_fn((1, 1, 1), |_| 42.0);
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, Some(-9999.0)).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: false,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();

        let hdr = fs::read_to_string(dir.path().join("test.hdr")).unwrap();
        assert!(hdr.contains("data ignore value = -9999"));
    }

    #[test]
    fn write_strict_fails_before_creating_files() {
        let mut data = Array3::zeros((1, 1, 1));
        data[[0, 0, 0]] = 99999.0; // out of i16 range
        let wl = Array1::from_vec(vec![500.0]);
        let cube = SpectralCube::new(data, wl, None, None).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::I16,
            force: false,
            ..Default::default()
        };
        assert!(write_envi_with_options(&cube, &path, &opts).is_err());

        // Neither header nor data file should exist
        assert!(!dir.path().join("test.hdr").exists());
        assert!(!dir.path().join("test").exists());
    }

    #[test]
    fn write_f32_bip_combined() {
        let cube = make_cube();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test");

        let opts = EnviWriteOptions {
            data_type: EnviWriteDataType::F32,
            interleave: Interleave::Bip,
            ..Default::default()
        };
        write_envi_with_options(&cube, &path, &opts).unwrap();
        let loaded = read_envi(&path).unwrap();

        assert_eq!(loaded.shape(), cube.shape());
        assert_eq!(loaded.data(), cube.data());
    }

    #[test]
    fn unsupported_data_type() {
        let hdr = "ENVI\nsamples = 1\nlines = 1\nbands = 1\ndata type = 6\n\
                   interleave = bsq\nbyte order = 0\n";
        let err = parse_header(hdr).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }
}
