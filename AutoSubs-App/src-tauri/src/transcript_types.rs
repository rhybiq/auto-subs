// use eyre::Result;
// use num::integer::div_floor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f64,
    pub end: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordTimestamp>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColorModifier {
    pub enabled: bool,
    pub color: String,
}

impl Default for ColorModifier {
    fn default() -> Self {
        ColorModifier {
            enabled: false,
            color: String::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Sample {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Speaker {
    pub name: String,
    pub fill: ColorModifier,
    pub outline: ColorModifier,
    pub border: ColorModifier,
    pub sample: Sample,
}

/// Non-speech CS2 sound event kind. See [`GameEvent`].
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GameEventKind {
    Gunfire,
    Explosion,
    ElectronicBeep,
}

/// A detected non-speech CS2 gameplay sound event (gunfire, explosion, or the
/// C4 plant/defuse electronic beep), auto-inserted as a bracketed caption tag
/// (e.g. `[GUNFIRE]`) alongside speech subtitles. Experimental: detected via a
/// general-purpose audio classifier never trained on game audio, so expect
/// false positives — kept as its own editable/deletable list, never silently
/// merged into `segments`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameEvent {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub kind: GameEventKind,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transcript {
    pub processing_time_sec: u64,
    pub language: String,
    /// Fully-formatted segments ready for display (structural line wrapping +
    /// content formatting: case, punctuation removal, censoring).
    pub segments: Vec<Segment>,
    /// Raw engine-output segments (post-translation) with untouched word data.
    /// Used as the source for reformatting: the frontend can invoke
    /// `reformat_subtitles` with new settings to regenerate `segments` without
    /// re-transcribing.
    #[serde(rename = "originalSegments")]
    pub original_segments: Vec<Segment>,
    pub speakers: Vec<Speaker>,
    /// Empty unless `enable_game_events` was requested. `#[serde(default)]` so
    /// documents saved before this field existed still deserialize.
    /// Matches `originalSegments`'s camelCase wire convention so the frontend
    /// doesn't need a snake_case/camelCase fallback chain for this field.
    #[serde(rename = "gameEvents", default)]
    pub game_events: Vec<GameEvent>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonSegment {
    id: usize,
    seek: usize,
    start: f64,
    end: f64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_id: Option<String>,
    tokens: Vec<i32>,
    temperature: f32,
    avg_logprob: f64,
    compression_ratio: f64,
    no_speech_prob: f64,
    words: Vec<JsonWordTimestamp>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonWordTimestamp {
    word: String,
    start: f64,
    end: f64,
    probability: f32,
}
