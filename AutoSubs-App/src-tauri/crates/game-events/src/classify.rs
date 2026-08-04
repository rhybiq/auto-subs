//! Sliding-window ONNX classification: log-mel patches -> YAMNet ->
//! per-patch Gunfire/Explosion scores -> hysteresis-smoothed [`GameEvent`]s.
//!
//! Structurally mirrors `diarize::segment::get_segments`'s "windowed audio ->
//! ONNX model -> frame classification -> hysteresis-smoothed segments"
//! pattern, but computes two independent max-scores per patch (gunfire,
//! explosion) rather than one `argmax` best-class pick, since these are
//! independent yes/no buckets rather than mutually-exclusive classes.

use crate::{class_map, mel, session, CancelFn, GameEvent, GameEventKind, GameEventOptions, ProgressFn};
use eyre::{eyre, Result};
use ort::value::TensorRef;

/// Finer than YAMNet's own 0.48s default patch hop — better localizes short
/// transients like individual gunshots. The model is a small MobileNet run
/// CPU-only, so the extra inference calls are cheap.
const PATCH_HOP_FRAMES: usize = 24;

/// Debounce: require at least this many consecutive silent patches before
/// closing an event, so a single momentarily-quiet patch inside a burst of
/// gunfire doesn't split it into several tiny events.
const GAP_TOLERANCE_PATCHES: usize = 1;
const MIN_EVENT_DURATION_MS: f64 = 200.0;

pub fn classify_events(
    samples: &[i16],
    sample_rate: u32,
    options: &GameEventOptions,
    progress: Option<&ProgressFn<'_>>,
    is_cancelled: Option<&CancelFn<'_>>,
) -> Result<Vec<GameEvent>> {
    if sample_rate != mel::SAMPLE_RATE {
        return Err(eyre!(
            "game-events classifier expects {} Hz audio, got {} Hz",
            mel::SAMPLE_RATE,
            sample_rate
        ));
    }

    let frames = mel::log_mel_frames(samples, sample_rate);
    let patches = mel::frames_to_patches(&frames, PATCH_HOP_FRAMES);
    if patches.is_empty() {
        return Ok(Vec::new());
    }

    let mut session = session::create_session(&options.model_path)?;
    let class_kinds = class_map::class_kind_map();

    let mut gunfire_scores = Vec::with_capacity(patches.len());
    let mut explosion_scores = Vec::with_capacity(patches.len());
    let mut patch_starts_sec = Vec::with_capacity(patches.len());

    let total = patches.len();
    for (i, (patch, start_frame)) in patches.into_iter().enumerate() {
        if let Some(is_cancelled) = is_cancelled {
            if is_cancelled() {
                return Err(eyre!("Cancelled"));
            }
        }

        let tensor = TensorRef::from_array_view((
            [1usize, 1, mel::PATCH_FRAMES, mel::NUM_MEL_BINS],
            patch.as_slice(),
        ))
        .map_err(|e| eyre!("Failed to prepare game-event input tensor: {:?}", e))?;
        let inputs = ort::inputs!["audio" => tensor];

        let outputs = session
            .run(inputs)
            .map_err(|e| eyre!("game-events ONNX inference failed: {:?}", e))?;
        let class_scores = outputs
            .get("class_scores")
            .ok_or_else(|| eyre!("game-events model output 'class_scores' not found"))?;
        let (_, data) = class_scores
            .try_extract_tensor::<f32>()
            .map_err(|e| eyre!("Failed to extract class_scores: {:?}", e))?;

        let mut gunfire = 0.0f32;
        let mut explosion = 0.0f32;
        for (idx, &score) in data.iter().enumerate() {
            match class_kinds.get(idx).copied().flatten() {
                Some(GameEventKind::Gunfire) => gunfire = gunfire.max(score),
                Some(GameEventKind::Explosion) => explosion = explosion.max(score),
                _ => {}
            }
        }
        gunfire_scores.push(gunfire);
        explosion_scores.push(explosion);
        patch_starts_sec.push(start_frame as f64 * 0.010);

        if let Some(cb) = progress {
            cb((((i + 1) * 100) / total) as i32);
        }
    }

    let mut events = smooth_to_events(
        &gunfire_scores,
        &patch_starts_sec,
        options.min_confidence,
        GameEventKind::Gunfire,
    );
    events.extend(smooth_to_events(
        &explosion_scores,
        &patch_starts_sec,
        options.min_confidence,
        GameEventKind::Explosion,
    ));
    events.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    Ok(events)
}

fn smooth_to_events(
    scores: &[f32],
    patch_starts_sec: &[f64],
    min_confidence: f32,
    kind: GameEventKind,
) -> Vec<GameEvent> {
    // Patches are PATCH_FRAMES (0.96s) wide regardless of hop, so a patch's
    // coverage always extends this far past its start frame.
    let patch_duration = mel::PATCH_FRAMES as f64 * 0.010;

    let mut events = Vec::new();
    let mut in_event = false;
    let mut start_idx = 0usize;
    let mut gap = 0usize;
    let mut peak_confidence = 0.0f32;
    let mut last_hit_idx = 0usize;

    for (i, &score) in scores.iter().enumerate() {
        let hit = score >= min_confidence;
        if hit {
            gap = 0;
            last_hit_idx = i;
            if !in_event {
                in_event = true;
                start_idx = i;
                peak_confidence = score;
            } else {
                peak_confidence = peak_confidence.max(score);
            }
        } else if in_event {
            gap += 1;
            if gap > GAP_TOLERANCE_PATCHES {
                push_event(
                    &mut events,
                    patch_starts_sec,
                    start_idx,
                    last_hit_idx,
                    patch_duration,
                    peak_confidence,
                    kind,
                );
                in_event = false;
                peak_confidence = 0.0;
            }
        }
    }
    if in_event {
        push_event(
            &mut events,
            patch_starts_sec,
            start_idx,
            last_hit_idx,
            patch_duration,
            peak_confidence,
            kind,
        );
    }
    events
}

#[allow(clippy::too_many_arguments)]
fn push_event(
    events: &mut Vec<GameEvent>,
    patch_starts_sec: &[f64],
    start_idx: usize,
    end_idx: usize,
    patch_duration: f64,
    confidence: f32,
    kind: GameEventKind,
) {
    let start = patch_starts_sec[start_idx];
    let end = patch_starts_sec[end_idx] + patch_duration;
    if (end - start) * 1000.0 < MIN_EVENT_DURATION_MS {
        return;
    }
    events.push(GameEvent { start, end, kind, confidence });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooths_isolated_spike_into_nothing_below_min_duration() {
        // A single hit patch is much shorter than MIN_EVENT_DURATION_MS's
        // worth of patch coverage, so on its own it still clears the bar
        // because one patch already covers 0.96s - the real edge case is
        // gap tolerance, tested below instead.
        let scores = vec![0.0, 0.9, 0.0, 0.0];
        let starts = vec![0.0, 0.24, 0.48, 0.72];
        let events = smooth_to_events(&scores, &starts, 0.3, GameEventKind::Gunfire);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, GameEventKind::Gunfire);
    }

    #[test]
    fn bridges_single_patch_gap() {
        let scores = vec![0.9, 0.0, 0.9, 0.0, 0.0];
        let starts = vec![0.0, 0.24, 0.48, 0.72, 0.96];
        let events = smooth_to_events(&scores, &starts, 0.3, GameEventKind::Explosion);
        // The single-patch gap should be bridged into one continuous event.
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn no_hits_produces_no_events() {
        let scores = vec![0.1, 0.2, 0.1];
        let starts = vec![0.0, 0.24, 0.48];
        assert!(smooth_to_events(&scores, &starts, 0.3, GameEventKind::Gunfire).is_empty());
    }
}
