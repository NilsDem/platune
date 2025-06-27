import gin
import numpy as np
import torch
from scipy.interpolate import interp1d


def get_midi_notes(midi):
    # only for monophonic solo midi
    assert len(midi.instruments) == 1
    for instrument in midi.instruments:
        pitches = []
        onsets = []
        offsets = []
        velocities = []
        for note in instrument.notes:
            pitches.append(note.pitch)
            onsets.append(note.start)
            offsets.append(note.end)
            velocities.append(note.velocity)

    return np.array(pitches).astype(np.int32), np.array(onsets), np.array(offsets), np.array(velocities)


def get_melody_onsets(midi, audio_length, sr = 44100):
    pitch, onset, offset, velocity = get_midi_notes(midi)

    melody = np.zeros((audio_length, )).astype(np.int32)
    onsets_signal = np.zeros(audio_length).astype(np.int32)
    dynamics = np.zeros(audio_length).astype(np.int32)

    onset_sample_positions = np.round(onset * sr).astype(int)
    offset_sample_positions = np.round(offset * sr).astype(int)
    for i in range(len(pitch)):
        melody[onset_sample_positions[i]:offset_sample_positions[i]] = pitch[i]
        dynamics[onset_sample_positions[i]:offset_sample_positions[i]] = velocity[i]
    
    onsets_signal[onset_sample_positions] = 1

    return melody, onsets_signal, dynamics


def process_melody(melody, default_midi_note=60):
    last_note = default_midi_note
    new_melody = melody.copy()
    for i in range(melody.shape[-1]):
        current_note = melody[i]
        if current_note > 0:
            last_note = current_note
        new_melody[i] = last_note
    return new_melody


def downsample_to_latent_sample_rate(melody, z_length):
    z_indices = np.linspace(0, 1, z_length)
    current_indices = np.linspace(0, 1, melody.shape[0])
    interp_func = interp1d(current_indices, melody, kind='nearest')
    melody_downsampled = interp_func(z_indices)
    return melody_downsampled.astype(np.int32)


@gin.configurable
def process_midi_attributes(x, instrument_val, z_length, num_signal, ae_ratio, pitch_note_values, octave_boundaries, instruments_values, dynamics_boundaries):
    attr = {}

    melody, onsets_signal, dynamics = get_melody_onsets(midi=x, audio_length=num_signal, sr=44100)

    melody_processed = process_melody(melody)

    split_per_frame = onsets_signal.reshape(len(onsets_signal) // ae_ratio, -1)
    onsets_downsampled = np.any(split_per_frame[:] == 1, axis=1).astype(int)
    attr['onsets'] = onsets_downsampled[:z_length]

    melody_downsampled = downsample_to_latent_sample_rate(melody, z_length)
    attr['melody'] = melody_downsampled

    melody_processed_downsampled = downsample_to_latent_sample_rate(melody_processed, z_length)
    attr['melody_processed'] = melody_processed_downsampled

    attr['pitch'] = (melody_downsampled % len(pitch_note_values)).astype(int)
    attr['octave'] = (torch.bucketize(torch.from_numpy(melody_downsampled), torch.tensor(octave_boundaries)) - 1).numpy()

    attr['pitch_processed'] = (melody_processed_downsampled % len(pitch_note_values)).astype(int)
    attr['octave_processed'] = (torch.bucketize(torch.from_numpy(melody_processed_downsampled), torch.tensor(octave_boundaries)) - 1).numpy()

    attr['velocity'] = downsample_to_latent_sample_rate(dynamics, z_length)
    attr['dynamics'] = (torch.bucketize(torch.from_numpy(attr['velocity']), torch.tensor(dynamics_boundaries))).numpy()

    attr['instrument'] = np.full((z_length,), instruments_values.index(instrument_val))

    return attr
