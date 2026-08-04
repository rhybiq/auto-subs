//! Narrowband tone-burst detector for CS2's C4 plant/defuse electronic beep.
//!
//! Independent of the ML classifier: AudioSet has no class for this sound,
//! and it's a strongly narrowband periodic tone, which is exactly what
//! classic signal processing (not a general-purpose classifier) is good at.
//! Has no semantic understanding and no confidence calibration — it cannot
//! tell planted vs. defusing vs. low-countdown beeps, or distinguish the C4
//! tone from any other similar in-game UI beep. Labeled generically as
//! [`crate::GameEventKind::ElectronicBeep`], not "bomb planted".

use crate::{GameEvent, GameEventKind};
use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::PI;

const WINDOW_SIZE: usize = 1024;
const HOP_SIZE: usize = 256;

/// CS2's C4 beep pitch; a starting guess pending tuning against real
/// recordings, kept as a range to tolerate encoder/mic frequency drift.
const BAND_MIN_HZ: f32 = 700.0;
const BAND_MAX_HZ: f32 = 1100.0;

/// A frame counts as "beeping" when the target band holds at least this
/// fraction of the frame's total spectral energy.
const BAND_DOMINANCE_THRESHOLD: f32 = 0.35;

const MIN_BURST_MS: f64 = 40.0;
const MAX_BURST_MS: f64 = 400.0;

/// Consecutive tone bursts must repeat within this cadence, and there must
/// be at least `MIN_REPEATS` of them, to count as the C4 tone rather than a
/// one-off UI beep or false positive.
const MIN_REPEATS: usize = 3;
const MAX_GAP_MS: f64 = 700.0;

struct Burst {
    start: f64,
    end: f64,
}

pub fn detect_beeps(samples: &[i16], sample_rate: u32) -> Vec<GameEvent> {
    let bursts = find_tone_bursts(samples, sample_rate);
    group_into_runs(&bursts)
}

fn find_tone_bursts(samples: &[i16], sample_rate: u32) -> Vec<Burst> {
    if samples.len() < WINDOW_SIZE {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);
    let hann: Vec<f32> = (0..WINDOW_SIZE)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / WINDOW_SIZE as f32).cos())
        .collect();

    let float_samples: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
    let num_fft_bins = WINDOW_SIZE / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let band_lo = ((BAND_MIN_HZ / nyquist) * (num_fft_bins - 1) as f32)
        .round()
        .max(0.0) as usize;
    let band_hi = (((BAND_MAX_HZ / nyquist) * (num_fft_bins - 1) as f32).round() as usize)
        .min(num_fft_bins - 1)
        .max(band_lo);

    let mut buf = vec![Complex32::new(0.0, 0.0); WINDOW_SIZE];
    let num_frames = (float_samples.len() - WINDOW_SIZE) / HOP_SIZE + 1;
    let frame_duration = HOP_SIZE as f64 / sample_rate as f64;

    let mut frame_is_beep = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * HOP_SIZE;
        for (j, &w) in hann.iter().enumerate() {
            buf[j] = Complex32::new(float_samples[start + j] * w, 0.0);
        }
        fft.process(&mut buf);

        let power: Vec<f32> = buf[..num_fft_bins].iter().map(|c| c.norm_sqr()).collect();
        let total: f32 = power.iter().sum::<f32>().max(1e-9);
        let band_energy: f32 = power[band_lo..=band_hi].iter().sum();
        frame_is_beep.push(band_energy / total >= BAND_DOMINANCE_THRESHOLD);
    }

    collapse_frames_into_bursts(&frame_is_beep, frame_duration)
}

fn collapse_frames_into_bursts(frame_is_beep: &[bool], frame_duration: f64) -> Vec<Burst> {
    let mut bursts = Vec::new();
    let mut start_frame: Option<usize> = None;

    let mut maybe_push = |bursts: &mut Vec<Burst>, s: usize, e: usize| {
        let start = s as f64 * frame_duration;
        let end = e as f64 * frame_duration;
        let dur_ms = (end - start) * 1000.0;
        if dur_ms >= MIN_BURST_MS && dur_ms <= MAX_BURST_MS {
            bursts.push(Burst { start, end });
        }
    };

    for (i, &is_beep) in frame_is_beep.iter().enumerate() {
        match (is_beep, start_frame) {
            (true, None) => start_frame = Some(i),
            (false, Some(s)) => {
                maybe_push(&mut bursts, s, i);
                start_frame = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start_frame {
        maybe_push(&mut bursts, s, frame_is_beep.len());
    }
    bursts
}

/// Group individual tone bursts into runs of `MIN_REPEATS`+ periodic beeps,
/// emitting one [`GameEventKind::ElectronicBeep`] event spanning each run.
fn group_into_runs(bursts: &[Burst]) -> Vec<GameEvent> {
    let mut events = Vec::new();
    if bursts.is_empty() {
        return events;
    }

    let mut run_start_idx = 0usize;
    for i in 1..=bursts.len() {
        let run_broken = i == bursts.len()
            || (bursts[i].start - bursts[i - 1].end) * 1000.0 > MAX_GAP_MS;
        if run_broken {
            let run_len = i - run_start_idx;
            if run_len >= MIN_REPEATS {
                events.push(GameEvent {
                    start: bursts[run_start_idx].start,
                    end: bursts[i - 1].end,
                    kind: GameEventKind::ElectronicBeep,
                    // No ML confidence signal here; a fixed mid-value keeps
                    // this comparable to min_confidence-style filtering
                    // elsewhere without implying false precision.
                    confidence: 0.5,
                });
            }
            run_start_idx = i;
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_burst(freq_hz: f32, duration_ms: f64, sample_rate: u32) -> Vec<i16> {
        let n = (sample_rate as f64 * duration_ms / 1000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                ((2.0 * PI * freq_hz * t).sin() * i16::MAX as f32 * 0.8) as i16
            })
            .collect()
    }

    fn silence(duration_ms: f64, sample_rate: u32) -> Vec<i16> {
        vec![0i16; (sample_rate as f64 * duration_ms / 1000.0) as usize]
    }

    #[test]
    fn detects_periodic_beep_run() {
        let sr = 16_000;
        let mut samples = Vec::new();
        for _ in 0..4 {
            samples.extend(tone_burst(900.0, 100.0, sr));
            samples.extend(silence(200.0, sr));
        }
        let events = detect_beeps(&samples, sr);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, GameEventKind::ElectronicBeep);
    }

    #[test]
    fn ignores_white_noise() {
        let sr = 16_000;
        // Deterministic xorshift PRNG - broadband noise should never
        // concentrate 35%+ of its energy in a narrow band.
        let mut seed = 12345u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 16) as i16
        };
        let samples: Vec<i16> = (0..sr as usize * 2).map(|_| next()).collect();
        let events = detect_beeps(&samples, sr);
        assert!(events.is_empty());
    }

    #[test]
    fn ignores_single_isolated_beep() {
        let sr = 16_000;
        let mut samples = tone_burst(900.0, 100.0, sr);
        samples.extend(silence(500.0, sr));
        let events = detect_beeps(&samples, sr);
        assert!(events.is_empty(), "a single beep shouldn't count as the repeating C4 tone");
    }

    #[test]
    fn ignores_sustained_tone_too_long_to_be_a_beep() {
        let sr = 16_000;
        // A continuous 2s tone (e.g. some other sustained game sound) should
        // not register as a beep burst since it exceeds MAX_BURST_MS.
        let samples = tone_burst(900.0, 2000.0, sr);
        let events = detect_beeps(&samples, sr);
        assert!(events.is_empty());
    }
}
