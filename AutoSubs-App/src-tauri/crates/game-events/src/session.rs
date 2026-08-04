use std::path::Path;

use eyre::{eyre, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// CPU-only: the YAMNet graph is small (~15MB, MobileNet-v1-based) so GPU
/// execution providers aren't worth the extra Cargo feature surface here,
/// unlike the ASR/diarization models.
pub fn create_session<P: AsRef<Path>>(path: P) -> Result<Session> {
    let session = Session::builder()
        .map_err(|e| eyre!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| eyre!("{e}"))?
        .with_intra_threads(1)
        .map_err(|e| eyre!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| eyre!("{e}"))?
        .commit_from_file(path.as_ref())
        .map_err(|e| eyre!("{e}"))?;
    Ok(session)
}
