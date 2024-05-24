from __future__ import absolute_import
from __future__ import print_function

import random
import os
import torch

import numpy as np

from torchmimic.data.preprocessing import Discretizer, Normalizer
from torchmimic.data.readers import DecompensationReader
from torchmimic.data.utils import read_chunk
from torchmimic.data.base_dataset import BaseDataset

from torch.utils.data import Dataset


class DecompensationDataset(Dataset):
    """
    Decompensation dataset that can be directly used by PyTorch dataloaders. This class preprocesses the data the same way as "Multitask learning and benchmarking with clinical time series data": https://github.com/YerevaNN/mimic3-benchmarks

    :param root: directory where data is located
    :type root: str
    :param train: if true, the training split of the data will be used. Otherwise, the validation dataset will be used
    :type train: bool
    :param n_samples: number of samples to use. If None, all the data is used
    :type n_samples: int
    :param customListFile: listfile to use. If None, use train_listfile.csv
    :type customListFile: str
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        n_samples=None,
        customListFile=None,
    ):
        """
        Initialize DecompensationDataset

        :param root: directory where data is located
        :type root: str
        :param train: if true, the training split of the data will be used. Otherwise, the validation dataset will be used
        :type train: bool
        :param n_samples: number of samples to use. If None, all the data is used
        :type n_samples: int
        :param customListFile: listfile to use. If None, use train_listfile.csv
        :type customListFile: str
        """
        # super().__init__(transform=transform)

        listfile = "train_listfile.csv" if train else "val_listfile.csv"

        if customListFile is not None:
            listfile = customListFile

        self._read_data(root, listfile)
        self._load_data(n_samples)

        self.n_samples = len(self.data)
        self.transform = transform

    def _read_data(self, root, listfile):
        if "test" in listfile:
            self.reader = DecompensationReader(
                dataset_dir=os.path.join(root, "test"),
                listfile=os.path.join(root, listfile),
            )
        else:
            self.reader = DecompensationReader(
                dataset_dir=os.path.join(root, "train"),
                listfile=os.path.join(root, listfile),
            )

        self.discretizer = Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
        )

        discretizer_header = self.discretizer.transform(
            self.reader.read_example(0)["X"]
        )[1].split(",")
        cont_channels = [
            i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
        ]

        self.normalizer = Normalizer(fields=cont_channels)
        normalizer_state = "../normalizers/decomp_ts1.0.input_str:previous.n1e5.start_time:zero.normalizer"
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        self.normalizer.load_params(normalizer_state)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        sl = len(x)
        y = self.labels[idx]
        m = self.mask[idx]
        index = idx
        # index = torch.tensor(index, dtype=torch.int)

        if self.transform:
            x = self.transform(x)

        return x, y, sl, m, index

    def __len__(self):
        return self.n_samples

    def _load_data(self, sample_size):
        N = self.reader.get_number_of_examples()

        if sample_size is None:
            sample_size = N

        # print(sample_size)
        ret = read_chunk(self.reader, sample_size)

        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]

        data_tmp = []
        self.mask = []

        for X, t in zip(data, ts):
            d = self.discretizer.transform(X, end=t)[0]
            data_tmp.append(d)
            self.mask.append(self.expand_mask(d[:, 59:]))

        # data = [self.discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if self.normalizer is not None:
            self.data = [self.normalizer.transform(X) for X in data_tmp]
        self.labels = ys
        self.ts = ts
        self.names = names
        self.targets = ys

    def expand_mask(self, mask):
        expanded_mask = torch.ones((mask.shape[0], 59))

        for i, pv in enumerate(self.discretizer._possible_values.values()):
            n_values = len(pv) if not pv == [] else 1
            for p in range(n_values):
                expanded_mask[:, p + i] = torch.from_numpy(mask[:, i])

        return expanded_mask
