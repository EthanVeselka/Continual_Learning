import glob
import os

from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb

from torchmimic.metrics import AverageMeter, MetricMeter
from torchmimic.utils import create_exp_dir


class BaseLogger(ABC):
    """
    Base Logger class. Used for logging, printing, and saving information about the run. Contains built-in wandb support.

    :param config: A dictionary of the run configuration
    :type config: dict
    :param log_wandb: If true, wandb will be used to log metrics and configuration
    :type log_wandb: bool
    """

    def __init__(self, exp_name, config, log_wandb=False):
        """
        Initialize BaseLogger

        :param config: A dictionary of the run configuration
        :type config: dict
        :param log_wandb: If true, wandb will be used to log metrics and configuration
        :type log_wandb: bool
        """
        self.log_wandb = log_wandb
        self.perf = {}
        self.task_results = []

        if self.log_wandb:
            wandb.init(project="MIMIC Benchmark", name=exp_name)
            wandb.config.update(config)
            wandb.run.log_code("*.py")

        self.experiment_path = f"./exp/{exp_name}"

        if not os.path.exists("./exp"):
            os.mkdir("./exp")

        create_exp_dir(self.experiment_path, scripts_to_save=glob.glob("*.py"))
        np.save(os.path.join(self.experiment_path, "config"), config)

        self.metrics = {
            "Loss": AverageMeter(),
        }

    def __del__(self):
        """
        Destructor for BaseLogger. Finishes wandb if log_wandb is true
        """
        if self.log_wandb:
            wandb.finish()

    @abstractmethod
    def update(self, outputs, labels, loss):
        """
        Abstract class for updating metrics
        """
        pass

    def reset(self):
        """
        Resets metrics
        """
        for item in self.metrics.values():
            item.reset()

    def get_loss(self):
        """
        Returns average loss

        :return: Average Loss
        :rtype: float
        """
        return self.metrics["Loss"].avg

    def print_metrics(self, epoch, split="Train", task=None, test=False):
        """
        Prints and logs metrics. If log_wandb is True, wandb run will be updated

        :param epoch: The current epoch
        :type epoch: int
        :param split: The split of the data. "Train" or "Eval"
        :type split: str
        """

        assert split in ("Train", "Eval", "Test")

        result_str = split + ": "

        if self.log_wandb:
            wandb.log({"Epochs": epoch + 1}, commit=False)

        result_str += f" Epoch {epoch+1}"
        for name, meter in self.metrics.items():
            if isinstance(meter, MetricMeter):
                result = meter.score()
                if task != None:
                    self.perf.update({task + " " + name: result})
                if test:
                    self.task_results.append((name, result))
            elif isinstance(meter, AverageMeter):
                result = meter.avg
                if task != None:
                    self.perf.update({task + " Avg " + name: result})
                if test:
                    self.task_results.append(("Avg " + name, result))

            # if self.log_wandb:
            #     wandb.log(
            #         {split + " " + name: result},
            #         commit=False,
            #     )
            result_str += f", {name}={result}"

        print(result_str)
        if split == "Eval" and self.log_wandb:
            wandb.log({})

    def save(self, model):
        """
        Saves the provided model to the experiment path
        """
        torch.save(
            model.state_dict(),
            os.path.join(self.experiment_path, "weights.pt"),
        )

    def save_results(self):
        task_res = self.task_results
        self.task_results = []
        return task_res

    def get_results(self):
        perf = self.perf
        self.perf = {}
        return perf

    def update_wandb_val(self, results):
        differences = []
        perf_summary = []
        prev = None
        count = 1

        metric1 = ""
        metric2 = ""
        total_diff_m1 = 0
        total_diff_m2 = 0

        for result in results:
            res = {}
            words = []
            if count != 1:
                for idx, (key, val) in enumerate(result.items()):
                    if (idx + 1) % 3 == 2:
                        diff = val - prev[key]
                        total_diff_m1 += diff
                        res[key] = diff
                    elif (idx + 1) % 3 == 0:
                        diff = val - prev[key]
                        total_diff_m2 += diff
                        res[key] = diff

            else:
                for idx, (key, val) in enumerate(result.items()):
                    if idx == 1:
                        words = key.split()
                        metric1 = " ".join(words[3:])
                        res[key] = val
                    elif idx == 2:
                        words = key.split()
                        metric2 = " ".join(words[3:])
                        res[key] = val
                    elif (idx + 1) % 3 == 2:
                        words = key.split()
                        res[key] = val
                        wandb.run.summary[
                            "Initial " + metric1 + " (Task " + words[2] + ")"
                        ] = val
                    elif (idx + 1) % 3 == 0:
                        words = key.split()
                        res[key] = val
                        wandb.run.summary[
                            "Initial " + metric2 + " (Task " + words[2] + ")"
                        ] = val

            name = "{Task " + str(count) + "}"
            perf_summary.append({name: result})
            differences.append(res)

            count += 1
            prev = result

        wandb.run.summary["Differences"] = differences
        wandb.run.summary["Average " + metric1 + " Delta"] = total_diff_m1 / (count - 1)
        wandb.run.summary["Average " + metric2 + " Delta"] = total_diff_m2 / (count - 1)
        wandb.run.summary[
            "Validation Performance Summary " + "(Tasks 1-" + str(count - 1) + ")"
        ] = perf_summary

    def update_wandb_test(self, results):
        all_res = []

        for _, (k, v) in enumerate(results.items()):
            name = "{" + k + " Tests}"
            perf_summary = []
            for _, (key, val) in enumerate(v.items()):
                k1 = val[0][0]
                v1 = val[0][1]
                k2 = val[1][0]
                v2 = val[1][1]
                k3 = val[2][0]
                v3 = val[2][1]

                res = {k1: v1, k2: v2, k3: v3}
                perf_summary.append({key: res})
            all_res.append({name: perf_summary})

        wandb.run.summary["Test Performance Summary"] = all_res
