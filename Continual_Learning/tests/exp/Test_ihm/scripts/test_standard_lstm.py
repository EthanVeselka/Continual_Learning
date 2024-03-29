import unittest
import sys
import os
import torch
import gc
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

import torchmimic.data

from torchmimic.benchmarks import (
    IHMBenchmark,
    DecompensationBenchmark,
    LOSBenchmark,
    PhenotypingBenchmark,
)

from torchmimic.loggers import (
    IHMLogger,
    DecompensationLogger,
    LOSLogger,
    PhenotypingLogger,
)

from torchmimic.models import StandardLSTM
from torchmimic.utils import get_test_loaders, get_train_loader, update_buffer


ihm_tasks = [
    "../../datasets/mimic3-benchmarks/in-hospital-mortality",
    "../../datasets/eICU-benchmarks/data_mimicformat/in-hospital-mortality",
]

# decomp_tasks = [
#     "../../datasets/mimic3-benchmarks/decompensation",
#     "../../datasets/eICU-benchmarks/data_mimicformat/decompensation",
# ]

# ihm_tasks = [
#     "/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
#     "../../datasets/eICU-benchmarks/data_mimicformat/in-hospital-mortality",
# ]

decomp_tasks = [
    "/data/datasets/mimic3-benchmarks/data/decompensation",
    "../../datasets/eICU-benchmarks/data_mimicformat/decompensation",
]

los_tasks = [
    "/data/datasets/mimic3-benchmarks/data/length-of-stay",
    "../../datasets/eICU-benchmarks/data_mimicformat/length-of-stay",
]
phen_tasks = [
    "/data/datasets/mimic3-benchmarks/data/phenotyping",
    "../../datasets/eICU-benchmarks/data_mimicformat/phenotyping",
]


def get_config(
    test_batch_size,
    train_batch_size,
    learning_rate,
    weight_decay,
    ewc_penalty=False,
    importance=0,
    replay=False,
    buffer_size=0,
):
    return {
        "Buffer size": buffer_size,
        "EWC": ewc_penalty,
        "Importance": importance,
        "Replay": replay,
        "test_batch_size": test_batch_size,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


class TestLSTM(unittest.TestCase):
    def test_standard_lstm_phenotype(
        self,
        task_list=[0],
        buffer_size=0,
        replay=False,
        ewc_penalty=False,
        importance=0,
    ):

        # specify tasks
        tasks = [phen_tasks[i] for i in task_list]
        if buffer_size == 0:
            ewc_penalty = False
            replay = False
            print("NOTE: Buffer size is 0, EWC and Replay will not be used")

        task_name = "phen"
        device = 0
        sample_size = None
        train_batch_size = 8
        test_batch_size = 256
        learning_rate = 0.001
        weight_decay = 0
        report_freq = 200
        workers = 5
        wandb = True

        model = StandardLSTM(
            n_classes=25,
            hidden_dim=256,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
        )

        config = {}
        config.update(model.get_config())
        config.update(
            get_config(
                test_batch_size,
                train_batch_size,
                learning_rate,
                weight_decay,
                ewc_penalty,
                importance,
                replay,
                buffer_size,
            )
        )

        logger = PhenotypingLogger("Test_phen", config, wandb)

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
        buffer = []
        for task_num, task_data in enumerate(tasks):

            # use model from previous trainer for additional tasks
            if task_num > 0:
                model = prev_models[task_num - 1]

            # get random samples for ewc/replay
            if task_num > 0 and (ewc_penalty or replay):
                buffer = get_buffer(task_num, task_samples, buffer_size)

            trainer = PhenotypingBenchmark(
                model=model,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                data=task_data,
                buffer_size=buffer_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                report_freq=report_freq,
                logger=logger,
                device=device,
            )

            # train model, evaluate on all testing data
            trainer.fit(
                2,
                train_loaders[task_num],
                test_loaders,
                task_num,
                buffer,
                replay=replay,
                ewc_penalty=ewc_penalty,
                importance=importance,
            )

            prev_models.append(trainer.model)

    def test_standard_lstm_ihm(
        self,
        task_list=[0],
        buffer_size=0,
        replay=False,
        ewc_penalty=False,
        importance=0,
    ):
        # specify tasks
        tasks = [ihm_tasks[i] for i in task_list]
        if buffer_size == 0:
            ewc_penalty = False
            replay = False
            print("NOTE: Buffer size is 0, EWC and Replay will not be used")

        task_name = "ihm"
        device = 0
        sample_size = 100
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

        config = {}
        config.update(model.get_config())
        config.update(
            get_config(
                test_batch_size,
                train_batch_size,
                learning_rate,
                weight_decay,
                ewc_penalty,
                importance,
                replay,
                buffer_size,
            )
        )

        logger = IHMLogger("Test_ihm", config, wandb)

        # get test loaders for each task
        test_loaders = get_test_loaders(
            task_name,
            tasks,
            test_batch_size,
            sample_size,
            workers,
            device,
        )

        # train on current task, test on all tasks
        prev_model = None
        results = []
        buffer = []
        for task_num, task_data in enumerate(tasks):

            # use model from previous trainer for additional tasks
            if task_num > 0:
                model = prev_model

            trainer = IHMBenchmark(
                model=model,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                data=task_data,
                buffer_size=buffer_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                report_freq=report_freq,
                logger=logger,
                device=device,
            )

            # train model, evaluate on all testing data
            train_loader = get_train_loader(
                "ihm",
                task_data,
                train_batch_size,
                sample_size,
                workers,
                device,
            )
            result = trainer.fit(
                2,
                train_loader,
                test_loaders,
                task_num,
                buffer,
                replay=replay,
                ewc_penalty=ewc_penalty,
                importance=importance,
            )

            results.append(result)

            # get random samples for ewc/replay
            if ewc_penalty or replay:
                buffer = update_buffer(buffer, task_num, train_loader, buffer_size)
            prev_model = trainer.model
            del trainer, train_loader

        logger.update_wandb(results)

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
            data="../../datasets/eICU-benchmarks/data_mimicformat/length-of-stay",
            # data="/data/datasets/mimic3-benchmarks/data/length-of-stay",
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
        sample_size = 24000  # try 36 chunks, each 8000 samples long, so 300k samples total (should use ~65 gigs of memory)
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

        gc.collect()
        # train on current task, test on all tasks
        prev_models = []
        buffer = []
        for task_num, task_data in enumerate(tasks):

            # use model from previous trainer for additional tasks
            if task_num != 0:
                model = prev_models[task_num - 1]
                # buffer = get_buffer(task_num, task_samples, buffer_size)

            # get random samples for ewc/replay
            if task_num > 0 and (ewc_penalty or replay):
                buffer = get_buffer(task_num, task_samples, buffer_size)

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
                buffer,
                replay=replay,
                ewc_penalty=ewc_penalty,
            )

            prev_models.append(trainer.model)
            gc.collect()
        # torch.cuda.empty_cache()
