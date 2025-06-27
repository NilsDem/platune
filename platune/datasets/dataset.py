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


CONTINUOUS_ATTRIBUTES = [
    'rms', 'loudness1s', 'integrated_loudness', 'centroid', 'bandwidth',
    'booming', 'sharpness', 'arousal', 'valence', "dark", "epic", "retro",
    "fast", "loudness1s", "energetic", "melodic", "emotional", "dark",
    "average_duration", "note_density", "central_pitch", "pitch_range",
    "children", "energetic", "emotional", "dark"
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


@gin.configurable
def load_data(data_path: str,
              discrete_keys: List[str] = [],
              continuous_keys: List[str] = [],
              batch_size: int = 8,
              n_workers: int = 0,
              cache: bool = False,
              lmdb_keys_file: str = None,
              dataset_name: str = None,
              crop: int = None):

    dataset = LatentsContinuousDiscreteAttritbutesDataset(
        path=data_path,
        keys=["z"] + discrete_keys + continuous_keys,
        lmdb_keys_file=lmdb_keys_file,
        dataset_name=dataset_name,
        crop=crop)
    # z, ad, ac = dataset[0]
    if cache:
        dataset.build_cache()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    train_ids, valid_ids = train_test_split(indices,
                                            test_size=int(0.05 * dataset_size),
                                            random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_ids)
    val_dataset = torch.utils.data.Subset(dataset, valid_ids)

    print("dataset sizes : ", len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True,
                              num_workers=n_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size,
                            shuffle=False,
                            num_workers=n_workers)

    print("dataloader sizes : ", len(train_loader), len(val_loader))

    return train_loader, val_loader
