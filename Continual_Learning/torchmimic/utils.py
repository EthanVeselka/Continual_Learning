import os
import shutil
import torch
import random

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchmimic.data import IHMDataset
from torchmimic.data import DecompensationDataset
from torchmimic.data import LOSDataset
from torchmimic.data import PhenotypingDataset


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


# Takes dict of random samples per task, returns a shuffled list of buffer_size length with an equal distribution of random samples from each task
def update_buffer(task_num, task_samples, buffer_size):
    buff = []

    # Calculate the number of samples to take from each list
    samples_per_list = buffer_size // (task_num + 1)

    for _, lst in enumerate(task_samples.values()):
        # Randomly sample values from the list
        sampled_values = random.sample(lst, samples_per_list)
        buff.extend(sampled_values)

    # Shuffle the list
    random.shuffle(buff)
    return buff


def get_samples(sample_size, buffer_size, train_loader):
    # get specified number of random samples

    random_samples = []
    sample_idx = random.sample(range(len(train_loader)), buffer_size)
    for idx, (data, label, lens, mask) in enumerate(train_loader):
        if idx in sample_idx:
            random_samples.append((data, label, lens, mask))

    random.shuffle(random_samples)
    return random_samples


# returns task training loader
def get_train_loader(
    task_num,
    task_name,
    tasks,
    lf_map,
    train_batch_size,
    sample_size,
    workers,
    device,
):

    clf = (
        (lf_map[task_num - 1] + "_train.csv")
        if (task_num > 0 and len(tasks) > 2)
        else "train_listfile.csv"
    )
    if task_name == "ihm":
        train_dataset = IHMDataset(
            tasks[task_num],
            train=True,
            n_samples=sample_size,
            customListFile=clf,
        )
    elif task_name == "decomp":
        train_dataset = DecompensationDataset(
            tasks[task_num],
            train=True,
            n_samples=sample_size * 0.7,  # 100000
            customListFile=clf,
        )
    elif task_name == "los":
        train_dataset = LOSDataset(
            tasks[task_num], train=True, n_samples=sample_size * 0.7, customListFile=clf
        )
    elif task_name == "phen":
        train_dataset = PhenotypingDataset(
            tasks[task_num], train=True, n_samples=sample_size, customListFile=clf
        )

    kwargs = {"num_workers": workers, "pin_memory": True} if device else {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=pad_colalte,
        **kwargs,
    )

    return train_loader


# Returns validation loaders for each task
def get_val_loaders(
    task_name,
    tasks,
    lf_map,
    val_batch_size,
    sample_size,
    workers,
    device,
):
    val_loaders = []
    clf = None

    for task_num, task_data in enumerate(tasks):
        clf = (
            (lf_map[task_num - 1] + "_val.csv")
            if (task_num > 0 and len(tasks) > 2)
            else "val_listfile.csv"
        )
        if task_name == "ihm":
            val_dataset = IHMDataset(
                task_data,
                train=False,
                n_samples=sample_size,
                customListFile=clf,
            )
        elif task_name == "decomp":
            val_dataset = DecompensationDataset(
                task_data,
                train=False,
                n_samples=sample_size * 0.15,  # 100000
                customListFile=clf,
            )
        elif task_name == "los":
            val_dataset = LOSDataset(
                task_data, train=False, n_samples=sample_size * 0.15, customListFile=clf
            )
        elif task_name == "phen":
            val_dataset = PhenotypingDataset(
                task_data, train=False, n_samples=sample_size, customListFile=clf
            )

        kwargs = {"num_workers": workers, "pin_memory": True} if device else {}

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        val_loaders.append(val_loader)

    return val_loaders


def get_test_loaders(
    task_name,
    tasks,
    lf_map,
    test_batch_size,
    sample_size,
    workers,
    device,
):
    test_loaders = []
    clf = None

    for task_num, task_data in enumerate(tasks):
        clf = (
            (lf_map[task_num - 1] + "_test.csv")
            if (task_num > 0 and len(tasks) > 2)
            else "test_listfile.csv"
        )
        if task_name == "ihm":
            test_dataset = IHMDataset(
                task_data,
                train=False,
                n_samples=sample_size,
                customListFile=clf,
            )
        elif task_name == "decomp":
            test_dataset = DecompensationDataset(
                task_data,
                train=False,
                n_samples=sample_size * 0.15,  # 100000
                customListFile=clf,
            )
        elif task_name == "los":
            test_dataset = LOSDataset(
                task_data,
                train=False,
                n_samples=sample_size * 0.15,
                customListFile=clf,
            )
        elif task_name == "phen":
            test_dataset = PhenotypingDataset(
                task_data,
                train=False,
                n_samples=sample_size,
                customListFile=clf,
            )

        kwargs = {"num_workers": workers, "pin_memory": True} if device else {}

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        test_loaders.append(test_loader)

    return test_loaders
