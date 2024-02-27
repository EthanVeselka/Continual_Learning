import unittest
import torchmimic.data

from torchmimic.benchmarks import (
    IHMBenchmark,
    DecompensationBenchmark,
    LOSBenchmark,
    PhenotypingBenchmark,
)

from torchmimic.models import StandardLSTM
from torchmimic.utils import get_random_samples

ihm_tasks = [
    "/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
    "/data/datasets/eICU-benchmarks/data/in-hospital-mortality",
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
        self, tasks=ihm_tasks, er_memory=1000, ewc_penalty=False
    ):
        device = 0
        task_samples = {}
        random_samples = []

        model = StandardLSTM(
            n_classes=1,
            hidden_dim=16,
            num_layers=2,
            dropout_rate=0.3,
            bidirectional=True,
        )

        # iterate through tasks and create benchmark for each, return model after train/test, and use for next task
        for task_num, task_data in enumerate(ihm_tasks):
            trainer = IHMBenchmark(
                model=model,
                train_batch_size=8,
                test_batch_size=256,
                data=task_data,
                er_memory=er_memory,
                learning_rate=0.001,
                weight_decay=0,
                report_freq=200,
                device=device,
                sample_size=1000,
                wandb=False,
            )

            # Make sure model assignment here works properly
            model = trainer.fit(2, task_num, random_samples, ewc_penalty=ewc_penalty)

            task_samples[task_num] = trainer.get_samples()
            random_samples = get_random_samples(task_samples, er_memory)

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

    def test_standard_lstm_decomp(self):
        device = 0

        model = StandardLSTM(
            n_classes=1,
            hidden_dim=128,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
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


#     def test_standard_lstm_multi(self):
#         device = get_free_gpu()


#         model = StandardLSTM(
#             n_classes=1,
#             hidden_dim=256,
#             num_layers=1,
#             dropout_rate=0.3,
#             bidirectional=True,
#         )

#         trainer = Multitask_Trainer(
#             model=model,
#             train_batch_size=8,
#             test_batch_size=256,
#             data="/data/datasets/mimic3-benchmarks/data/multitask",
#             learning_rate=0.001,
#             weight_decay=0,
#             report_freq=200,
#             device=device,
#             small_part=True,
#             partition=10,
#         )

#         trainer.fit(2)
