import unittest
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

import torchmimic.data

from torchmimic.benchmarks import (
    IHMBenchmark,
    DecompensationBenchmark,
    LOSBenchmark,
    PhenotypingBenchmark,
)

from torchmimic.models import StandardLSTM
from torchmimic.utils import get_loaders, get_random_samples


# ihm_tasks = [
#     "../../datasets/mimic3-benchmarks/in-hospital-mortality",
#     "../../datasets/eICU-benchmarks/data_mimicformat/in-hospital-mortality",
# ]

# decomp_tasks = [
#     "../../datasets/mimic3-benchmarks/decompensation",
#     "../../datasets/eICU-benchmarks/data_mimicformat/decompensation",
# ]

ihm_tasks = [
    "/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
    "../../datasets/eICU-benchmarks/data_mimicformat/in-hospital-mortality",
]

decomp_tasks = [
    "/data/datasets/mimic3-benchmarks/data/decompensation",
    "../../datasets/eICU-benchmarks/data_mimicformat/decompensation",
]


class TestLSTM(unittest.TestCase):
    def test_standard_lstm_phenotype(self):
        device = 0

        model = StandardLSTM(
            n_classes=25,
            hidden_dim=256,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
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

    def test_standard_lstm_ihm(
        self, task_list=[0], buffer_size=0, replay=False, ewc_penalty=False
    ):
        # specify tasks
        tasks = [ihm_tasks[i] for i in task_list]
        if buffer_size == 0:
            ewc_penalty = False
            replay = False
            print("NOTE: Buffer size is 0, EWC and Replay will not be used")

        task_name = "ihm"
        device = 0
        sample_size = None
        train_batch_size = 8
        test_batch_size = 256
        learning_rate = 0.001
        weight_decay = 0
        report_freq = 200
        workers = 5
        wandb = False

        model = StandardLSTM(
            n_classes=1,
            hidden_dim=16,
            num_layers=2,
            dropout_rate=0.3,
            bidirectional=True,
        )

        # get data loaders for each task
        train_loaders, test_loaders, task_samples = get_loaders(
            task_name,
            tasks,
            train_batch_size,
            test_batch_size,
            buffer_size,
            sample_size,
            workers,
            device,
        )

        # train on current task, test on all tasks
        prev_models = []
        random_samples = []
        for task_num, task_data in enumerate(tasks):

            # use model from previous trainer for additional tasks
            if task_num > 0:
                model = prev_models[task_num - 1]
                # random_samples = get_random_samples(task_num, task_samples, buffer_size)

            # # get random samples for ewc/replay
            if task_num > 0 and (ewc_penalty or replay):
                random_samples = get_random_samples(task_num, task_samples, buffer_size)

            trainer = IHMBenchmark(
                model=model,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                data=task_data,
                buffer_size=buffer_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                report_freq=report_freq,
                device=device,
                wandb=wandb,
            )

            # train model, evaluate on all testing data
            trainer.fit(
                2,
                train_loaders[task_num],
                test_loaders,
                task_num,
                random_samples,
                replay=replay,
                ewc_penalty=ewc_penalty,
            )

            prev_models.append(trainer.model)

        # torch.cuda.empty_cache()

    def test_standard_lstm_los(self):
        device = 0

        model = StandardLSTM(
            n_classes=10,
            hidden_dim=64,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
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

    def test_standard_lstm_decomp(
        self, task_list=[0], buffer_size=0, replay=False, ewc_penalty=False
    ):
        # specify tasks
        tasks = [decomp_tasks[i] for i in task_list]
        if buffer_size == 0:
            ewc_penalty = False
            replay = False
            print("NOTE: Buffer size is 0, EWC and Replay will not be used")

        task_name = "decomp"
        device = 0
        sample_size = 1000
        train_batch_size = 8
        test_batch_size = 256
        learning_rate = 0.001
        weight_decay = 0
        report_freq = 200
        workers = 5
        wandb = False

        model = StandardLSTM(
            n_classes=1,
            hidden_dim=128,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
        )

        # get data loaders for each task
        train_loaders, test_loaders, task_samples = get_loaders(
            task_name,
            tasks,
            train_batch_size,
            test_batch_size,
            buffer_size,
            sample_size,
            workers,
            device,
        )

        # train on current task, test on all tasks
        prev_models = []
        random_samples = []
        for task_num, task_data in enumerate(tasks):

            # use model from previous trainer for additional tasks
            if task_num != 0:
                model = prev_models[task_num - 1]
                # random_samples = get_random_samples(task_num, task_samples, buffer_size)

            # get random samples for ewc/replay
            if task_num > 0 and (ewc_penalty or replay):
                random_samples = get_random_samples(task_num, task_samples, buffer_size)

            trainer = DecompensationBenchmark(
                model=model,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                data=task_data,
                buffer_size=buffer_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                report_freq=report_freq,
                device=device,
                wandb=wandb,
            )

            # train model, evaluate on all testing data
            trainer.fit(
                1,
                train_loaders[task_num],
                test_loaders,
                task_num,
                random_samples,
                replay=replay,
                ewc_penalty=ewc_penalty,
            )

            prev_models.append(trainer.model)
        # torch.cuda.empty_cache()
