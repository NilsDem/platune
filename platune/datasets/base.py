import gin
import torch
import lmdb
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from platune.datasets.audio_example import AudioExample


class SimpleDataset(Dataset):

    def __init__(
        self,
        path,
        keys="all",
        transforms=None,
        readonly=True,
        map_size=None,
        num_samples=None,
    ) -> None:
        super().__init__()

        if map_size is not None:
            self.env = lmdb.open(path,
                                 lock=False,
                                 readonly=readonly,
                                 readahead=False,
                                 map_async=False,
                                 map_size=1024**3 * map_size)
        else:
            self.env = lmdb.open(
                path,
                lock=False,
                readonly=readonly,
                readahead=False,
                map_async=False,
            )

        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        if keys == "all":
            self.buffer_keys = self.get_keys()
        else:
            self.buffer_keys = keys
        self.transforms = transforms

        if num_samples is not None and num_samples < len(self.keys):
            np.random.seed(42)
            self.keys = list(
                np.random.choice(self.keys, num_samples, replace=False))

        self.cache = False

    def build_cache(self):
        self.cache = False
        print("building cache")
        self.data = []
        for i in tqdm(range(len(self))):
            self.data.append(self.__getitem__(i))
        self.cache = True

    def __len__(self):
        return len(self.keys)

    def get_keys(self):
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[0]))
        return ae.get_keys()

    def __getitem__(self, index=None, key=None):
        if self.cache == True:
            return self.data[index]

        with self.env.begin() as txn:
            if key is not None:
                ae = AudioExample(txn.get(key))
            else:
                ae = AudioExample(txn.get(self.keys[index]))
        out = {}
        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    pass
                    #print("key: ", key, " not found")
        return out
