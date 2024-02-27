import os
import shutil
import torch
import random

import numpy as np
from torch.nn.utils.rnn import pad_sequence


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


# Takes dict of random samples per task, returns a shuffled list of er_memory length with an equal distribution of random samples from each task
def get_random_samples(task_samples, er_memory):
    random_samples = []

    # Calculate the number of samples to take from each list
    samples_per_list = er_memory // len(task_samples)

    for lst in task_samples.values():
        # Randomly sample values from the list
        sampled_values = random.sample(lst, samples_per_list)
        random_samples.extend(sampled_values)

    # Shuffle the list
    random.shuffle(random_samples)
    return random_samples
