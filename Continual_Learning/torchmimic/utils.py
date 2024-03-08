import os
import shutil
import torch
import random

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torchmimic.data import IHMDataset
from torchmimic.data import DecompensationDataset
from torchmimic.data import LOSDataset
from torchmimic.data import PhenotypingDataset
from torch.utils.data import DataLoader


def pad_colalte(batch):
    xx, yy, lens, mask = zip(*batch)
    x = pad_sequence(xx, batch_first=True)
    y = torch.FloatTensor(yy)

    return x, y, lens, mask


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print(f"Experiment dir: {path}")

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, "scripts")):
            os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def get_samples(sample_size, buffer_size, train_loader):
    # get specified number of random samples

    count = 0
    random_samples = []
    sample_idx = random.sample(range(len(train_loader)), buffer_size)
    for idx, (data, label, lens, mask) in enumerate(train_loader):
        if idx in sample_idx:
            random_samples.append((data, label, lens, mask))

    random.shuffle(random_samples)
    return random_samples


# Takes dict of random samples per task, returns a shuffled list of buffer_size length with an equal distribution of random samples from each task
def get_random_samples(task_num, task_samples, buffer_size):
    random_samples = []

    # Calculate the number of samples to take from each list
    samples_per_list = buffer_size // task_num

    print(samples_per_list)
    for idx, lst in enumerate(task_samples.values()):
        # Randomly sample values from the list
        if idx == task_num:
            break
        sampled_values = random.sample(lst, samples_per_list)
        random_samples.extend(sampled_values)

    # Shuffle the list
    random.shuffle(random_samples)
    return random_samples


# returns list of training/testing loaders for each task
def get_loaders(
    task_name,
    tasks,
    train_batch_size,
    test_batch_size,
    buffer_size,
    sample_size,
    workers,
    device,
):
    train_loaders = []
    test_loaders = []
    task_samples = {}

    for task_num, task_data in enumerate(tasks):

        if task_name == "ihm":
            train_dataset = IHMDataset(
                task_data,
                train=True,
                n_samples=sample_size,
            )

            test_dataset = IHMDataset(
                task_data,
                train=False,
                n_samples=sample_size,
            )
        elif task_name == "decomp":
            train_dataset = DecompensationDataset(
                task_data,
                train=True,
                n_samples=sample_size,
            )

            test_dataset = DecompensationDataset(
                task_data,
                train=False,
                n_samples=sample_size,
            )
        elif task_name == "los":
            train_dataset = LOSDataset(
                task_data,
                train=True,
                n_samples=sample_size,
            )

            test_dataset = LOSDataset(
                task_data,
                train=False,
                n_samples=sample_size,
            )
        elif task_name == "phen":
            train_dataset = PhenotypingDataset(
                task_data,
                train=True,
                n_samples=sample_size,
            )

            test_dataset = PhenotypingDataset(
                task_data,
                train=False,
                n_samples=sample_size,
            )

        kwargs = {"num_workers": workers, "pin_memory": True} if device else {}

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=pad_colalte,
            **kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        task_samples[task_num] = (
            []
            if buffer_size == 0
            else get_samples(sample_size, buffer_size, train_loader)
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders, task_samples
