import numpy as np
import pretty_midi

import numpy as np
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def resample_descriptors(descriptors: dict,
                         target_length: int,
                         smoothing_window: int = 17,
                         polyorder: int = 3) -> dict:
    """
    Smooth and resample descriptor curves using Savitzky-Golay filtering and linear interpolation.

    Parameters:
    - descriptors: dict of descriptor_name -> np.ndarray
    - target_length: int, number of desired samples
    - smoothing_window: int, window size for Savitzky-Golay filter (must be odd)
    - polyorder: int, polynomial order for smoothing

    Returns:
    - resampled_descriptors: dict with same keys, resampled and smoothed values
    """
    resampled = {}

    for name, signal in descriptors.items():
        N = len(signal)
        if N == 0 or target_length == 0:
            resampled[name] = np.array([])
            continue
        if np.all(np.isnan(signal)):
            resampled[name] = np.full(target_length, np.nan)
            continue

        # Handle NaNs: simple interpolation before smoothing
        if np.any(np.isnan(signal)):
            valid = ~np.isnan(signal)
            if np.sum(valid) < 2:
                resampled[name] = np.full(target_length, np.nan)
                continue
            interp_func = interp1d(np.flatnonzero(valid),
                                   signal[valid],
                                   kind='linear',
                                   fill_value='extrapolate')
            signal = interp_func(np.arange(N))

        # Ensure smoothing window is valid
        if smoothing_window >= N:
            smoothing_window = N - 1 if N % 2 == 0 else N
        if smoothing_window % 2 == 0:
            smoothing_window -= 1
        if smoothing_window < polyorder + 2:
            smoothing_window = polyorder + 2 | 1  # ensure odd

        # Apply Savitzky–Golay filter
        try:
            smoothed = savgol_filter(signal,
                                     window_length=smoothing_window,
                                     polyorder=polyorder)
        except ValueError:
            smoothed = signal  # fallback if filtering fails

        # Interpolation to target length
        x_old = np.linspace(0, 1, N)
        x_new = np.linspace(0, 1, target_length)
        interp_func = interp1d(x_old, smoothed, kind='linear')
        resampled[name] = interp_func(x_new)

    return resampled


def compute_midi_descriptors(midi: pretty_midi.PrettyMIDI,
                             window_size: float,
                             hop_size: float,
                             playing_notes: bool = False,
                             target_length: int = None,
                             total_time: float = 0.) -> dict:
    """
    Compute time-varying MIDI descriptors over a sliding window.

    Returns a dict of:
    - note_density
    - central_pitch
    - average_duration
    - pitch_range_abs
    - pitch_range_var

    Parameters:
    - midi: PrettyMIDI object
    - window_size: window size in seconds
    - hop_size: hop size in seconds
    - playing_notes: if True, considers sounding notes; otherwise, note onsets

    Returns:
    - descriptors: dict of name -> np.ndarray (one value per window)
    """
    # Collect all notes
    all_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        all_notes.extend(instrument.notes)

    # total_time = midi.get_end_time()
    times = np.arange(0, total_time - window_size + hop_size,
                      hop_size) + window_size / 2
    num_frames = len(times)

    # Initialize outputs
    note_density = np.zeros(num_frames)
    central_pitch = np.full(num_frames, np.nan)
    average_duration = np.full(num_frames, np.nan)
    pitch_range_abs = np.full(num_frames, np.nan)
    pitch_range_var = np.full(num_frames, np.nan)

    for i, center in enumerate(times):
        t_start = center - window_size / 2
        t_end = center + window_size / 2

        if playing_notes:
            notes = [
                n for n in all_notes if n.start <= t_end and n.end >= t_start
            ]
        else:
            notes = [n for n in all_notes if t_start <= n.start < t_end]

        if notes:
            pitches = np.array([n.pitch for n in notes])
            durations = np.array([n.end - n.start for n in notes])

            note_density[i] = len(notes)
            central_pitch[i] = np.mean(pitches)
            average_duration[i] = np.mean(durations)
            pitch_range_abs[i] = np.max(pitches) - np.min(pitches)
            pitch_range_var[i] = np.var(pitches) / (
                12.0**2)  # normalized to 1 octave variance

    out = {
        "note_density": note_density,
        "central_pitch": central_pitch,
        "average_duration": average_duration,
        "pitch_range_abs": pitch_range_abs,
        "pitch_range_var": pitch_range_var,
    }

    if target_length is not None:
        out = resample_descriptors(out, target_length)

    return out
