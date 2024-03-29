import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from torchmimic.EWC import EWC

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
from torchmimic.utils import (
    get_test_loaders,
    get_val_loaders,
    get_train_loader,
    update_buffer,
    get_samples,
)


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
    def test_standard_lstm(
        self,
        task,
        epochs=2,
        task_list=[0],
        buffer_size=0,
        replay=False,
        ewc_penalty=False,
        importance=0,
        test=False,
    ):

        device = 0
        sample_size = None  # use all samples
        train_batch_size = 8
        test_batch_size = 256
        learning_rate = 0.001
        weight_decay = 0
        report_freq = 200
        workers = 5
        wandb = True
        config = {}

        if buffer_size == 0:
            ewc_penalty = False
            replay = False
            print("NOTE: Buffer size is 0, EWC and Replay will not be used")

        # specify tasks
        if task == "ihm":
            tasks = [ihm_tasks[i] for i in task_list]
            model = StandardLSTM(
                n_classes=1,
                hidden_dim=16,
                num_layers=2,
                dropout_rate=0.3,
                bidirectional=True,
            )
            logger = IHMLogger
            benchmark = IHMBenchmark

        elif task == "phen":
            tasks = [phen_tasks[i] for i in task_list]
            model = StandardLSTM(
                n_classes=25,
                hidden_dim=256,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=True,
            )
            logger = PhenotypingLogger
            benchmark = PhenotypingBenchmark

        elif task == "decomp":
            tasks = [decomp_tasks[i] for i in task_list]
            model = StandardLSTM(
                n_classes=1,
                hidden_dim=128,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=True,
            )
            logger = DecompensationLogger
            benchmark = DecompensationBenchmark

        elif task == "los":
            tasks = [los_tasks[i] for i in task_list]
            model = StandardLSTM(
                n_classes=10,
                hidden_dim=64,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=True,
            )
            logger = LOSLogger
            benchmark = LOSBenchmark

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

        exp_name = ("Test_" + task) if test else ("Train_" + task)
        logger = logger(exp_name, config, wandb)

        # get test loaders for each task
        val_loaders = get_val_loaders(
            task,
            tasks,
            test_batch_size,
            sample_size,
            workers,
            device,
        )

        if test:
            test_loaders = get_test_loaders(
                task,
                tasks,
                test_batch_size,
                sample_size,
                workers,
                device,
            )

        # train on current task, test on all tasks
        prev_model = None
        samples = {}
        val_results = []
        test_results = {}
        buffer = []

        for task_num, task_data in enumerate(tasks):

            # use model from previous trainer for additional tasks
            if task_num > 0:
                model = prev_model

            trainer = benchmark(
                model=model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                report_freq=report_freq,
                logger=logger,
                device=device,
            )

            # train model, evaluate on all testing data
            train_loader = get_train_loader(
                task,
                task_data,
                train_batch_size,
                sample_size,
                workers,
                device,
            )
            result = trainer.fit(
                epochs,
                train_loader,
                val_loaders,
                test_loaders,
                task_num,
                buffer,
                replay=replay,
                ewc_penalty=ewc_penalty,
                importance=importance,
            )

            if test:
                test_results["Task " + str(task_num + 1)] = result["test"]

            val_results.append(result["val"])

            # get random samples for ewc/replay
            if ewc_penalty or replay:
                samples["Task " + str(task_num + 1)] = get_samples(
                    sample_size, buffer_size, train_loader
                )
                buffer = update_buffer(task_num, samples, buffer_size)
            prev_model = trainer.model
            del trainer, train_loader

        if task == ("decomp" or "los"):
            os.system("ps aux | grep veselka | grep dec")
        logger.update_wandb_val(val_results)
        logger.update_wandb_test(test_results)
