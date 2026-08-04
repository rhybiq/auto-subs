pub mod beep_heuristic;
pub mod class_map;
pub mod classify;
pub mod mel;
pub mod session;

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameEventKind {
    Gunfire,
    Explosion,
    ElectronicBeep,
}

#[derive(Debug, Clone)]
pub struct GameEvent {
    pub start: f64,
    pub end: f64,
    pub kind: GameEventKind,
    pub confidence: f32,
}

pub struct GameEventOptions {
    pub model_path: PathBuf,
    pub min_confidence: f32,
    pub enable_beep_heuristic: bool,
}

impl Default for GameEventOptions {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            min_confidence: 0.3,
            enable_beep_heuristic: false,
        }
    }
}

pub type ProgressFn<'a> = dyn Fn(i32) + Send + Sync + 'a;
pub type CancelFn<'a> = dyn Fn() -> bool + Send + Sync + 'a;

/// Detect CS2-relevant sound events (gunfire, explosions, and optionally the
/// bomb-plant/defuse electronic beep) directly from mono 16kHz PCM audio.
///
/// Experimental: the ML classifier (YAMNet, general-purpose AudioSet classes)
/// was never trained on compressed game audio, so expect false positives on
/// footsteps/UI sounds and rapid gunfire smearing into one blurred event
/// rather than discrete shots. The beep heuristic has no semantic
/// understanding and cannot distinguish planted/defusing/countdown beeps.
pub fn detect_events(
    samples: &[i16],
    sample_rate: u32,
    options: &GameEventOptions,
    progress: Option<&ProgressFn<'_>>,
    is_cancelled: Option<&CancelFn<'_>>,
) -> eyre::Result<Vec<GameEvent>> {
    let mut events =
        classify::classify_events(samples, sample_rate, options, progress, is_cancelled)?;

    if options.enable_beep_heuristic {
        events.extend(beep_heuristic::detect_beeps(samples, sample_rate));
    }

    events.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    Ok(events)
}
