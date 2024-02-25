import unittest

import data

from benchmarks import (
    IHMBenchmark,
    DecompensationBenchmark,
    LOSBenchmark,
    PhenotypingBenchmark,
)

from models import WUPERR
import random

ihm_tasks = [
    "/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
    "/data/datasets/eICU-benchmarks/data/in-hospital-mortality",
]


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


class TestWUPERR(unittest.TestCase):
    def test_WUPERR_phenotype(self):
        device = 0

        model = WUPERR(
            n_classes=25,
            hidden_dim1=40,
            hidden_dim2=25,
            dropout_rate=0,
        )

        trainer = PhenotypingBenchmark(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/phenotyping",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            sample_size=1000,
            wandb=False,
        )

        trainer.fit(2)

    def test_WUPERR_ihm(self, tasks=ihm_tasks, er_memory=1000):
        device = 0
        task_samples = {}
        random_samples = []

        model = WUPERR(
            n_classes=1,
            hidden_dim1=40,
            hidden_dim2=25,
            dropout_rate=0,
        )

        # iterate through tasks and create benchmark for each, return model after train/test, and use for next task
        for task_num, task in enumerate(ihm_tasks):
            samples = []
            trainer = IHMBenchmark(
                model=model,
                train_batch_size=8,
                test_batch_size=256,
                task_data=task,
                learning_rate=0.001,
                weight_decay=0,
                report_freq=200,
                device=device,
                sample_size=1000,
                er_memory=er_memory,
                wandb=False,
            )

            model = trainer.fit(2, task_num, random_samples)

            task_samples[task_num] = trainer.get_samples()
            random_samples = get_random_samples(task_samples, er_memory)

    def test_WUPERR_los(self, num_tasks=1):
        device = 0

        model = WUPERR(
            n_classes=10,
            hidden_dim1=40,
            hidden_dim2=25,
            dropout_rate=0,
        )

        trainer = LOSBenchmark(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/length-of-stay",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            sample_size=1000,
            partition=10,
            wandb=False,
        )

        trainer.fit(2)

    def test_WUPERR_decomp(self, num_tasks=1):
        device = 0

        model = WUPERR(
            n_classes=25,
            hidden_dim1=40,
            hidden_dim2=25,
            dropout_rate=0,
        )

        trainer = DecompensationBenchmark(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data="/data/datasets/mimic3-benchmarks/data/decompensation",
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=device,
            sample_size=1000,
            wandb=False,
        )

        trainer.fit(2)
