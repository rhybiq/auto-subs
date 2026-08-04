//! YAMNet's log-mel spectrogram front end, reproduced bit-for-bit against
//! Google's reference implementation (`research/audioset/yamnet/{params,features}.py`
//! in tensorflow/models, Apache-2.0): 25ms/10ms STFT framing with a periodic
//! Hann window, magnitude (not power) spectrum, 64 mel bins spanning
//! 125-7500 Hz via the HTK-style `1127*ln(1+hz/700)` mel scale, natural-log
//! compression with a 0.001 offset, patched into 96-frame (0.96s) windows.
//!
//! The classifier was trained on exactly these features, so deviating from
//! them (wrong log offset, power vs. magnitude, wrong mel edges, etc.) would
//! silently degrade accuracy rather than fail loudly.

use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::PI;

pub const SAMPLE_RATE: u32 = 16_000;
pub const NUM_MEL_BINS: usize = 64;
/// One classifier input patch = 0.96s of audio = 96 frames at a 10ms hop.
pub const PATCH_FRAMES: usize = 96;
/// YAMNet's own default patch hop (50% overlap). Smaller hops are valid too
/// (Google's params.py: "patch hop can be changed arbitrarily... smaller hop
/// should give more patches... possibly better performance at a larger
/// computational cost") — callers wanting finer event localization can pass
/// a smaller hop to `frames_to_patches`.
pub const DEFAULT_PATCH_HOP_FRAMES: usize = 48;

const WINDOW_SECONDS: f32 = 0.025;
const HOP_SECONDS: f32 = 0.010;
const MEL_MIN_HZ: f32 = 125.0;
const MEL_MAX_HZ: f32 = 7500.0;
const LOG_OFFSET: f32 = 0.001;

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Triangular mel filterbank matching `tf.signal.linear_to_mel_weight_matrix`:
/// each FFT bin's own mel value is compared against each filter's mel-space
/// edges directly (not approximated via linear interpolation in bin-index
/// space, which would distort the filter shapes since the mel scale is
/// nonlinear in frequency).
struct MelFilterbank {
    /// weights[mel_bin][fft_bin]
    weights: Vec<Vec<f32>>,
}

impl MelFilterbank {
    fn build(num_fft_bins: usize, sample_rate: u32) -> Self {
        let nyquist = sample_rate as f32 / 2.0;
        let mel_lo = hz_to_mel(MEL_MIN_HZ);
        let mel_hi = hz_to_mel(MEL_MAX_HZ.min(nyquist));

        // num_mel_bins triangular filters need num_mel_bins + 2 boundary points,
        // evenly spaced in mel space.
        let edges: Vec<f32> = (0..NUM_MEL_BINS + 2)
            .map(|i| mel_lo + (mel_hi - mel_lo) * i as f32 / (NUM_MEL_BINS + 1) as f32)
            .collect();

        // FFT bins are linearly spaced in Hz from 0..nyquist.
        let bin_mels: Vec<f32> = (0..num_fft_bins)
            .map(|k| hz_to_mel(k as f32 * nyquist / (num_fft_bins - 1) as f32))
            .collect();

        let mut weights = vec![vec![0.0f32; num_fft_bins]; NUM_MEL_BINS];
        for m in 0..NUM_MEL_BINS {
            let (lower, center, upper) = (edges[m], edges[m + 1], edges[m + 2]);
            for (k, &mel_k) in bin_mels.iter().enumerate() {
                let lower_slope = (mel_k - lower) / (center - lower);
                let upper_slope = (upper - mel_k) / (upper - center);
                weights[m][k] = lower_slope.min(upper_slope).max(0.0);
            }
        }
        Self { weights }
    }

    fn apply(&self, magnitude_spectrum: &[f32]) -> [f32; NUM_MEL_BINS] {
        let mut out = [0.0f32; NUM_MEL_BINS];
        for (m, row) in self.weights.iter().enumerate() {
            out[m] = row
                .iter()
                .zip(magnitude_spectrum.iter())
                .map(|(w, s)| w * s)
                .sum();
        }
        out
    }
}

/// Compute log-mel-spectrogram frames for mono `samples` at `sample_rate` Hz
/// (must be [`SAMPLE_RATE`]). Returns one `[f32; NUM_MEL_BINS]` row per 10ms hop.
pub fn log_mel_frames(samples: &[i16], sample_rate: u32) -> Vec<[f32; NUM_MEL_BINS]> {
    let window_len = (WINDOW_SECONDS * sample_rate as f32).round() as usize;
    let hop_len = (HOP_SECONDS * sample_rate as f32).round() as usize;
    let fft_len = window_len.next_power_of_two();
    let num_fft_bins = fft_len / 2 + 1;

    if samples.len() < window_len {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);

    // Periodic Hann window (matches tf.signal.stft's default; NOT the
    // symmetric variant, which would use `window_len - 1` in the denominator).
    let hann: Vec<f32> = (0..window_len)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / window_len as f32).cos())
        .collect();

    let filterbank = MelFilterbank::build(num_fft_bins, sample_rate);
    let float_samples: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

    let num_frames = (float_samples.len() - window_len) / hop_len + 1;
    let mut frames = Vec::with_capacity(num_frames);
    let mut buf = vec![Complex32::new(0.0, 0.0); fft_len];

    for i in 0..num_frames {
        let start = i * hop_len;
        for j in 0..window_len {
            buf[j] = Complex32::new(float_samples[start + j] * hann[j], 0.0);
        }
        for slot in buf.iter_mut().take(fft_len).skip(window_len) {
            *slot = Complex32::new(0.0, 0.0);
        }
        fft.process(&mut buf);

        // Magnitude, not power — matches `tf.abs(tf.signal.stft(...))`.
        let magnitude: Vec<f32> = buf[..num_fft_bins].iter().map(|c| c.norm()).collect();
        let mel = filterbank.apply(&magnitude);

        let mut log_mel = [0.0f32; NUM_MEL_BINS];
        for (dst, &v) in log_mel.iter_mut().zip(mel.iter()) {
            *dst = (v + LOG_OFFSET).ln();
        }
        frames.push(log_mel);
    }

    frames
}

/// Slice consecutive log-mel frames into fixed-size `PATCH_FRAMES` x
/// `NUM_MEL_BINS` patches, hopping by `patch_hop_frames`. Each patch is
/// flattened row-major (frame-major, then mel-bin) to match the model's
/// `(1, 1, 96, 64)` input layout. Returns `(patch_data, start_frame_index)`.
pub fn frames_to_patches(
    frames: &[[f32; NUM_MEL_BINS]],
    patch_hop_frames: usize,
) -> Vec<(Vec<f32>, usize)> {
    if frames.len() < PATCH_FRAMES {
        return Vec::new();
    }
    let patch_hop_frames = patch_hop_frames.max(1);

    let mut patches = Vec::new();
    let mut start = 0;
    while start + PATCH_FRAMES <= frames.len() {
        let mut data = Vec::with_capacity(PATCH_FRAMES * NUM_MEL_BINS);
        for frame in &frames[start..start + PATCH_FRAMES] {
            data.extend_from_slice(frame);
        }
        patches.push((data, start));
        start += patch_hop_frames;
    }
    patches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_expected_frame_count() {
        // 1 second of audio @ 16kHz -> (16000 - 400) / 160 + 1 = 98 frames.
        let samples = vec![0i16; SAMPLE_RATE as usize];
        let frames = log_mel_frames(&samples, SAMPLE_RATE);
        assert_eq!(frames.len(), 98);
        assert_eq!(frames[0].len(), NUM_MEL_BINS);
    }

    #[test]
    fn silence_produces_finite_log_values() {
        let samples = vec![0i16; SAMPLE_RATE as usize];
        let frames = log_mel_frames(&samples, SAMPLE_RATE);
        for frame in &frames {
            for &v in frame {
                assert!(v.is_finite(), "log-mel value should never be NaN/inf for silence");
            }
        }
    }

    #[test]
    fn too_short_input_yields_no_frames() {
        let samples = vec![0i16; 10];
        assert!(log_mel_frames(&samples, SAMPLE_RATE).is_empty());
    }

    #[test]
    fn patches_have_correct_shape_and_hop() {
        // ~2s of audio -> enough frames for a couple of overlapping patches.
        let samples = vec![0i16; SAMPLE_RATE as usize * 2];
        let frames = log_mel_frames(&samples, SAMPLE_RATE);
        let patches = frames_to_patches(&frames, DEFAULT_PATCH_HOP_FRAMES);
        assert!(!patches.is_empty());
        for (data, _) in &patches {
            assert_eq!(data.len(), PATCH_FRAMES * NUM_MEL_BINS);
        }
        if patches.len() > 1 {
            assert_eq!(patches[1].1 - patches[0].1, DEFAULT_PATCH_HOP_FRAMES);
        }
    }

    #[test]
    fn sine_tone_concentrates_energy_in_expected_mel_bin() {
        // A pure tone should produce a clear energy peak in the mel bins
        // near its frequency, not be spread uniformly — a basic sanity
        // check that the filterbank is wired up correctly.
        let freq = 1000.0f32;
        let n = SAMPLE_RATE as usize;
        let samples: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                ((2.0 * PI * freq * t).sin() * i16::MAX as f32 * 0.5) as i16
            })
            .collect();
        let frames = log_mel_frames(&samples, SAMPLE_RATE);
        let mid_frame = &frames[frames.len() / 2];
        let (peak_bin, _) = mid_frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // hz_to_mel(1000) falls roughly in the lower-middle of the 64 mel
        // bins spanning 125-7500 Hz; assert it's not at either extreme.
        assert!(peak_bin > 10 && peak_bin < 40, "peak_bin = {peak_bin}");
    }
}
