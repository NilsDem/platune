import os
import gin
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List

from .audio_example import AudioExample
from .base import SimpleDataset
from .midi_descriptors import compute_midi_descriptors

CONTINUOUS_ATTRIBUTES = [
    'rms', 'loudness1s', 'integrated_loudness', 'centroid', 'bandwidth',
    'booming', 'sharpness', 'arousal', 'valence', "dark", "epic", "retro",
    "fast", "loudness1s", "energetic", "melodic", "emotional", "dark",
    "average_duration", "note_density", "central_pitch", "pitch_range",
    "children", "energetic", "emotional", "dark", "pitch_range_var"
]
DISCRETE_ATTRIBUTES = [
    'onsets',
    'melody',
    'pitch',
    'octave',
    'melody_processed',
    'pitch_processed',
    'octave_processed',
    'instrument',
    'velocity',
    'dynamics',
    'Mode',
    "key",
]

MIDI_DESCRIPTORS = [
    "note_density", "average_duration", "pitch_range", "central_pitch",
    "pitch_range_var"
]
AE_RATIO = 4096
SAMPLE_RATE = 44100
WINDOW_SIZE = 12.
HOP_SIZE = 3.


class LatentsContinuousDiscreteAttritbutesDataset(SimpleDataset):

    def __init__(
        self,
        path,
        keys=['z'] + CONTINUOUS_ATTRIBUTES + DISCRETE_ATTRIBUTES,
        use_hardcodec_keys=False,
        lmdb_keys_file: str = None,
        dataset_name: str = None,
        crop=None,
    ):

        if use_hardcodec_keys:
            keys = keys + CONTINUOUS_ATTRIBUTES + DISCRETE_ATTRIBUTES
        else:
            if "z" not in keys:
                keys = ["z"] + keys
        super().__init__(path, keys)

        if lmdb_keys_file is not None:
            with open(os.path.join(path, f"{lmdb_keys_file}.pkl"), "rb") as f:
                lmdb_keys = pickle.load(f)
            self.keys = lmdb_keys

        self.dataset_name = dataset_name
        self.crop = crop
        print(self.crop)

    def __getitem__(self, idx):
        if self.cache:
            return self.data[idx]

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[idx]))

        z = torch.from_numpy(ae.get("z"))

        # if z.shape[-1] not power of 2 replicate last frame
        z_length = z.shape[-1]
        is_pow_2 = (z_length > 0 and (z_length & (z_length - 1)) == 0)

        if not is_pow_2:
            z = torch.nn.functional.pad(z, (0, 1), mode='replicate')

        attr_discrete = []
        attr_continuous = []

        metadata = None
        w = None
        midi = None

        for key in self.buffer_keys:
            if key == 'z':
                continue

            if key in DISCRETE_ATTRIBUTES:
                try:
                    attr = torch.from_numpy(ae.get(key))
                except:
                    attr = torch.zeros(z_length)

                if attr.shape[-1] != z_length:
                    #     print(
                    #         "warning, you are using interpolation on discrete data, maybe try it first - not tested"
                    #     )
                    attr = torch.nn.functional.interpolate(
                        attr.reshape(1, 1, -1),
                        mode='nearest',
                        # align_corners=True,
                        size=z_length).reshape(-1).long()

                attr_discrete.append(attr)
            elif key in CONTINUOUS_ATTRIBUTES:
                if self.dataset_name == 'jamendo':
                    try:
                        attr = ae.get(key)

                        z_time_indices = np.linspace(0, (z_length * AE_RATIO) /
                                                     SAMPLE_RATE, z_length)
                        attr_time_indices = np.asarray([
                            (WINDOW_SIZE / 2) + i * HOP_SIZE
                            for i in range(attr.shape[-1])
                        ])
                        a = np.interp(x=z_time_indices,
                                      xp=attr_time_indices,
                                      fp=attr).astype(np.float32)
                        attr = torch.from_numpy(a)
                    except:
                        print("error in jamendo")
                        attr = torch.zeros(z_length)
                else:
                    try:
                        attr = torch.from_numpy(ae.get(key))
                    except:
                        attr = torch.zeros(z_length)
                if attr.shape[-1] != z_length:
                    attr = torch.nn.functional.interpolate(
                        attr.reshape(1, 1, -1),
                        mode='linear',
                        align_corners=True,
                        size=z_length).reshape(-1)

                attr_continuous.append(attr)
            elif key == 'metadata':
                metadata = ae.get_metadata()
                metadata['lmdb_key'] = self.keys[idx]
            elif key == 'waveform':
                w = torch.from_numpy(ae.get("waveform"))
            elif key == 'midi':
                midi = ae.get("midi")
            else:
                raise ValueError(
                    f'Need to specify if attribute is discrete or continuous for key={key}'
                )

        if len(attr_discrete) > 0:
            attr_discrete = torch.stack(attr_discrete)
            if not is_pow_2:
                attr_discrete = torch.nn.functional.pad(attr_discrete, (0, 1),
                                                        mode='replicate')
        else:
            # attr_discrete = np.array(attr_discrete)
            attr_discrete = torch.tensor(attr_discrete)

        if len(attr_continuous) > 0:
            attr_continuous = torch.stack(attr_continuous)
            if not is_pow_2:
                attr_continuous = torch.nn.functional.pad(attr_continuous,
                                                          (0, 1),
                                                          mode='replicate')
        else:
            # attr_continuous = np.array(attr_continuous)
            attr_continuous = torch.tensor(attr_continuous)

        if metadata is not None and self.crop is None:
            if w is not None:
                if midi is not None:
                    return z, attr_discrete, attr_continuous, metadata, w, midi
                else:
                    return z, attr_discrete, attr_continuous, metadata, w
            elif midi is not None:
                return z, attr_discrete, attr_continuous, metadata, midi
            else:
                return z, attr_discrete, attr_continuous, metadata
        if self.crop is not None:
            id_crop = np.random.randint(0, z_length - self.crop)
            z = z[:, id_crop:id_crop + self.crop]
            if len(attr_continuous) > 0:
                attr_continuous = attr_continuous[:,
                                                  id_crop:id_crop + self.crop]
            if len(attr_discrete) > 0:
                attr_discrete = attr_discrete[:, id_crop:id_crop + self.crop]
            if metadata is not None and midi is not None:
                return z, attr_discrete, attr_continuous, metadata, midi
        return z, attr_discrete, attr_continuous


def extract_pitch_class_and_octave_signal_old(pm,
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


import numpy as np


@gin.configurable
def extract_pitch_class_and_octave_signal(pm,
                                          num_frames,
                                          velocity_groups=6,
                                          sample_rate=44100,
                                          hop_length=512,
                                          hold_last_note=False,
                                          dummy_value="max"):
    """
    Args:
        pm (pretty_midi.PrettyMIDI): Loaded MIDI file
        num_frames (int): Number of time frames (usually audio_len // hop_length)
        sample_rate (int): Sampling rate of the corresponding audio
        hop_length (int): Hop size in samples (for frame spacing)
        hold_last_note (bool): If True, holds last pitch/octave when no note is active

    Returns:
        pitch_class_signal: (num_frames,) array with pitch class [0–11] or 12 (or held value)
        octave_signal: (num_frames,) array with octave (0–10) or 8 (or held value)
    """
    if dummy_value == "0":
        pitch_class_signal = np.full(num_frames, fill_value=0, dtype=int)
        octave_signal = np.full(num_frames, fill_value=0, dtype=int)
        velocity_group_signal = np.full(num_frames, fill_value=0, dtype=int)
        # print("USING 0 VALUE ")
    elif dummy_value == "max":
        pitch_class_signal = np.full(num_frames, fill_value=12, dtype=int)
        octave_signal = np.full(num_frames, fill_value=8, dtype=int)
        velocity_group_signal = np.full(num_frames,
                                        fill_value=velocity_groups + 1,
                                        dtype=int)
        # print("USING MAX VALUE ")

    last_pitch_class = 0
    last_octave = 0
    last_velocity = 0
    velocity_bins = np.linspace(40, 128, velocity_groups + 1)

    times = np.linspace(
        0 + hop_length / sample_rate / 2,
        num_frames * hop_length / sample_rate - hop_length / sample_rate / 2,
        num_frames)

    for i, t in enumerate(times):
        active_notes = []

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.start <= t < note.end:
                    active_notes.append((note.pitch, note.velocity))

        if active_notes:
            pitch, velocity = max(active_notes, key=lambda x: x[1])
            pitch_class = pitch % 12
            octave = min(pitch // 12, 7)
            velocity_group = np.digitize(velocity, velocity_bins) - 1

            pitch_class_signal[i] = pitch_class
            octave_signal[i] = octave
            velocity_group_signal[i] = velocity_group

            last_pitch_class = pitch_class
            last_octave = octave
            last_velocity = velocity_group
        elif hold_last_note:
            pitch_class_signal[i] = last_pitch_class
            octave_signal[i] = last_octave
            velocity_group_signal[i] = last_velocity

    return pitch_class_signal, octave_signal, velocity_group_signal


import json


@gin.configurable
def make_collate_fn(
    descriptor_names,
    num_signal,
    ae_ratio,
    velocity_groups=0,
    use_instrument=False,
):

    hop_length = ae_ratio  # assumed from earlier

    def collate_fn(batch):
        zs = []
        attr_continous = []
        attr_discrete = []

        for item in batch:
            x = item["z"]
            x = np.pad(x, ((0, 0), (0, 1)), mode='edge')  # pad to power of 2
            total_time = x.shape[-1]

            # Crop waveform
            if x.shape[-1] > num_signal:
                i0 = np.random.randint(0, x.shape[-1] - num_signal)
            else:
                i0 = 0  # no crop if too short

            x = x[..., i0:i0 + num_signal]
            zs.append(x)

            # Feature hop parameters

            feature_start = i0
            feature_len = num_signal

            cropped_continuous = []
            cropped_discrete = []
            if "pitch" in descriptor_names:
                pitch_class_signal, octave_signal, velocity_signal = extract_pitch_class_and_octave_signal(
                    item["midi"],
                    total_time,
                    velocity_groups=velocity_groups,
                    sample_rate=44100,
                    hop_length=hop_length)

                cropped_pitch = pitch_class_signal[
                    feature_start:feature_start + feature_len]
                cropped_octave = octave_signal[feature_start:feature_start +
                                               feature_len]
                cropped_velocity = velocity_signal[
                    feature_start:feature_start + feature_len]

            if any([n in MIDI_DESCRIPTORS for n in descriptor_names]):
                midi_descriptors = compute_midi_descriptors(
                    item["midi"],
                    window_size=1,
                    hop_size=0.05,
                    playing_notes=True,
                    target_length=total_time,
                    total_time= total_time* hop_length / 44100)

                for descr in midi_descriptors:
                    midi_descriptors[descr] = midi_descriptors[descr][
                        feature_start:feature_start + feature_len]

            for name in descriptor_names:
                if name in MIDI_DESCRIPTORS:
                    cropped = midi_descriptors[name]
                elif name == "pitch":
                    cropped = cropped_pitch
                    # norm = (cropped_pitch) / 12
                elif name == "octave":
                    cropped = cropped_octave
                    # norm = (cropped_octave) / 8
                elif name == "velocity":
                    cropped = cropped_velocity
                else:
                    values = np.array(item[name]).flatten()
                    cropped = values[feature_start:feature_start + feature_len]

                if name in CONTINUOUS_ATTRIBUTES:
                    cropped_continuous.append(cropped)
                elif name in DISCRETE_ATTRIBUTES:
                    cropped_discrete.append(cropped)
                # binned.append(bucket)

            attr_continous.append(cropped_continuous)
            attr_discrete.append(cropped_discrete)
            # binned_features.append(binned)

        # Stack and convert
        # waveforms = np.stack(waveforms)
        # for transform in transforms:
        #     waveforms = transform(waveforms)
        # waveforms = torch.from_numpy(waveforms).reshape(
        #     waveforms.shape[0], 1, -1).float()
        # print([z.shape for z in attr_continous])
        attr_continous = np.stack(attr_continous)
        attr_discrete = np.stack(attr_discrete)
        # binned_features = np.stack(binned_features)

        attr_continous = torch.tensor(attr_continous, dtype=torch.float32)
        attr_discrete = torch.tensor(attr_discrete, dtype=torch.long)

        zs = np.stack(zs)
        zs = torch.tensor(zs).float()
        if use_instrument:
            instrument = [
                b["metadata"]["instrument"].replace("_synthetic", "")
                for b in batch
            ]
            return zs, attr_discrete, attr_continous, instrument
        return zs, attr_discrete, attr_continous, ["none"] * len(zs)

    return collate_fn


from after.dataset import SimpleDataset, CombinedDataset


@gin.configurable
def load_data(data_path: List[str],
              freqs: List[float] = None,
              discrete_keys: List[str] = [],
              continuous_keys: List[str] = [],
              batch_size: int = 8,
              n_workers: int = 0,
              cache: bool = False,
              velocity_groups: int = 6,
              lmdb_keys_file: str = None,
              dataset_name: str = None,
              crop: int = None,
              ae_ratio: int = None,
              descriptor_file: str = None,
              use_instrument: bool = True):
    data_keys = ["z", "midi", "metadata"] + [
        k for k in discrete_keys
        if k not in ["pitch", "octave", "velocity"] + MIDI_DESCRIPTORS
    ] + [c for c in continuous_keys if c not in MIDI_DESCRIPTORS]

    if len(data_path) > 1:
        path_dict = {f: {"name": f, "path": f} for f in data_path}

        dataset = CombinedDataset(
            path_dict=path_dict,
            keys=data_keys,
            freqs="estimate" if freqs is None else freqs,
            config="train",
            init_cache=cache,
        )

        train_sampler = dataset.get_sampler()

        valset = CombinedDataset(
            path_dict=path_dict,
            config="validation",
            freqs="estimate" if freqs is None else freqs,
            keys=data_keys,
            init_cache=cache,
        )
        val_sampler = valset.get_sampler()

    else:
        dataset = SimpleDataset(path=data_path[0],
                                keys=data_keys,
                                init_cache=cache,
                                split="train")

        valset = SimpleDataset(path=data_path[0],
                               keys=data_keys,
                               split="validation",
                               init_cache=cache)
        train_sampler, val_sampler = None, None

    # dataset = SimpleDataset(
    #     path=data_path,
    #     keys=["z"] + discrete_keys + continuous_keys,
    #     lmdb_keys_file=lmdb_keys_file,
    #     dataset_name=dataset_name,
    #     crop=crop)
    # z, ad, ac = dataset[0]
    # if cache:
    #     dataset.build_cache()
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))

    # train_ids, valid_ids = train_test_split(indices,
    #                                         test_size=int(0.05 * dataset_size),
    #                                         random_state=42)

    # train_dataset = torch.utils.data.Subset(dataset, train_ids)
    # val_dataset = torch.utils.data.Subset(dataset, valid_ids)

    # print("dataset sizes : ", len(train_dataset), len(val_dataset))
    collate_fn = make_collate_fn(descriptor_names=continuous_keys +
                                 discrete_keys,
                                 num_signal=crop,
                                 ae_ratio=ae_ratio,
                                 use_instrument=use_instrument,
                                 velocity_groups=velocity_groups)
    train_loader = DataLoader(dataset,
                              batch_size,
                              shuffle=True if train_sampler is None else False,
                              num_workers=n_workers,
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    val_loader = DataLoader(valset,
                            batch_size,
                            shuffle=False,
                            num_workers=n_workers,
                            sampler=val_sampler,
                            collate_fn=collate_fn)

    print("dataloader sizes : ", len(train_loader), len(val_loader))

    print(100 * "#")

    print(f"using {n_workers} workers")

    print(100 * "#")

    return train_loader, val_loader
