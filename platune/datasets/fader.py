import numpy as np
import pretty_midi


def extract_pitch_class_and_octave_signal(pm,
                                          num_frames,
                                          sample_rate=44100,
                                          hop_length=512):
    """
    Args:
        pm (pretty_midi.PrettyMIDI): Loaded MIDI file
        num_frames (int): Number of time frames (usually audio_len // hop_length)
        sample_rate (int): Sampling rate of the corresponding audio
        hop_length (int): Hop size in samples (for frame spacing)
    
    Returns:
        pitch_class_signal: (num_frames,) array with pitch class [0–11] or -1 if no note
        octave_signal: (num_frames,) array with octave (0–10) or -1 if no note
    """
    pitch_class_signal = np.full(num_frames, fill_value=12, dtype=int)
    octave_signal = np.full(num_frames, fill_value=8, dtype=int)

    # times = np.arange(num_frames) * hop_length / sample_rate
    times = np.linspace(
        0 + hop_length / sample_rate / 2,
        num_frames * hop_length / sample_rate - hop_length / sample_rate / 2,
        num_frames)
    # print(num_frames * hop_length / sample_rate)

    for i, t in enumerate(times):
        active_notes = []

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.start <= t < note.end:
                    active_notes.append((note.pitch, note.velocity))

        if active_notes:
            # Pick most prominent (loudest) note
            pitch, _ = max(active_notes, key=lambda x: x[1])
            pitch_class = pitch % 12
            octave = min(pitch // 12, 8)

            pitch_class_signal[i] = pitch_class
            octave_signal[i] = octave

    return pitch_class_signal, octave_signal


import gin
import json
import torch


@gin.configurable
def make_collate_fn(
    descriptor_file,
    descriptor_names,
    num_signal,
    ae_ratio,
    transforms=[],
    return_full=False,
):

    hop_length = ae_ratio  # assumed from earlier

    with open(descriptor_file) as f:
        descriptor_stats = json.load(f)

    def collate_fn(batch):
        waveforms = []
        cont_features = []
        binned_features = []

        for item in batch:
            x = item["waveform"].reshape(1, -1)
            total_time = x.shape[-1]

            # Crop waveform
            if x.shape[-1] > num_signal:
                i0 = np.random.randint(
                    0, x.shape[-1] // hop_length - num_signal // hop_length)
            else:
                i0 = 0  # no crop if too short
            x = x[..., i0 * hop_length:i0 * hop_length + num_signal]
            waveforms.append(x)

            # Feature hop parameters

            feature_start = i0
            feature_len = num_signal // hop_length

            normed = []
            binned = []
            if "pitch" in descriptor_names:
                pitch_class_signal, octave_signal = extract_pitch_class_and_octave_signal(
                    item["midi"],
                    total_time // hop_length,
                    sample_rate=44100,
                    hop_length=hop_length)

                cropped_pitch = pitch_class_signal[
                    feature_start:feature_start + feature_len]
                cropped_octave = octave_signal[feature_start:feature_start +
                                               feature_len]

            for name in descriptor_names:

                if name == "pitch":
                    bucket = cropped_pitch
                    norm = (cropped_pitch) / 12
                elif name == "octave":
                    bucket = cropped_octave
                    norm = (cropped_octave) / 8
                else:
                    values = np.array(item[name]).flatten()
                    cropped = values[feature_start:feature_start + feature_len]
                    # Normalize
                    stats = descriptor_stats[name]
                    min_val, max_val = stats["min"], stats["max"]
                    norm = (cropped - min_val) / (max_val - min_val)
                    norm = np.clip(norm, 0.0, 1.0)

                    # Bin
                    bins = np.array(stats["quantile_bins"])
                    bucket = np.stack([
                        int(np.digitize(c, bins, right=False)) - 1
                        for c in cropped
                    ])
                    bucket = np.clip(bucket, 0, len(bins) - 2)

                normed.append(norm)
                binned.append(bucket)

            cont_features.append(normed)
            binned_features.append(binned)

        # Stack and convert
        waveforms = np.stack(waveforms)
        for transform in transforms:
            waveforms = transform(waveforms)
        waveforms = torch.from_numpy(waveforms).reshape(
            waveforms.shape[0], 1, -1).float()

        cont_features = np.stack(cont_features)
        binned_features = np.stack(binned_features)

        cont_features = torch.tensor(cont_features, dtype=torch.float32)
        binned_features = torch.tensor(binned_features, dtype=torch.long)

        return {
            "waveform": waveforms,
            "continuous": cont_features,
            "binned": binned_features
        }

    return collate_fn
