use thiserror::Error;

#[derive(Debug, Error)]
pub enum HyperspecError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("invalid wavelength: {0}")]
    InvalidWavelength(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("empty cube: {0}")]
    EmptyCube(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("format error: {0}")]
    Format(String),
}

pub type Result<T> = std::result::Result<T, HyperspecError>;
