import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
import random

from loggers import IHMLogger
from data import IHMDataset
from utils import pad_colalte
from LossFunctions import EWC


class IHMBenchmark:
    def __init__(
        self,
        model,
        train_batch_size=8,
        test_batch_size=256,
        task_data="/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        exp_name="Test",
        device="cpu",
        sample_size=None,
        er_memory=1000,
        workers=5,
        wandb=False,
    ):
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.task_data = task_data
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.device = device
        self.report_freq = report_freq
        self.er_memory = er_memory

        config = {}
        config.update(model.get_config())
        config.update(self.get_config())

        self.logger = IHMLogger(exp_name + "_ihm", config, wandb)

        torch.cuda.set_device(self.device)

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

        kwargs = {"num_workers": workers, "pin_memory": True} if self.device else {}

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=pad_colalte,
            **kwargs,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=pad_colalte,
            **kwargs,
        )

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # self.crit = nn.BCELoss()

    def fit(self, epochs, task_num, random_samples, gamma=0.99, beta=0.80):

        for epoch in range(epochs):
            self.model.train()
            self.logger.reset()

            ewc = None if task_num == 0 else EWC(self.model, random_samples, beta)

            for batch_idx, (data, label) in enumerate(self.train_loader):
                # read from random samples periodically if task_num > 0
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model(data)

                if task_num == 0:
                    loss = nn.cross_entropy(output, label[:, None])
                else:
                    loss = nn.cross_entropy(
                        output, label[:, None]
                    ) + gamma * ewc.penalty(self.model)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.logger.update(output, label, loss)
                if (batch_idx + 1) % self.report_freq == 0:
                    print(f"Train: epoch: {epoch+1}, loss = {self.logger.get_loss()}")

            self.logger.print_metrics(epoch, split="Train")

            self.model.eval()
            self.logger.reset()
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(self.test_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    output = self.model(data)

                    if task_num == 0:
                        loss = nn.cross_entropy(output, label[:, None])
                    else:
                        loss = nn.cross_entropy(
                            output, label[:, None]
                        ) + gamma * ewc.penalty(self.model)

                    self.logger.update(output, label, loss)
                    if (batch_idx + 1) % self.report_freq == 0:
                        print(
                            f"Eval: epoch: {epoch+1}, loss = {self.logger.get_loss()}"
                        )

                self.logger.print_metrics(epoch, split="Eval")

        return self.model

    def get_config(self):
        return {
            "test_batch_size": self.test_batch_size,
            "train_batch_size": self.train_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def get_samples(self):
        # get specified number of random samples
        count = 0
        sample_idx = random.sample(range(len(self.sample_size)), self.er_memory)
        for data, label in self.train_loader:
            if count in sample_idx:
                self.random_samples.append((data, label))
            count += 1

        random.shuffle(self.random_samples)
        return self.random_samples
