import torch
import torch.nn as nn
import random

from torch import optim
from torch.utils.data import DataLoader

from torchmimic.loggers import IHMLogger
from torchmimic.data import IHMDataset
from torchmimic.utils import pad_colalte
from torchmimic import EWC


class IHMBenchmark:
    def __init__(
        self,
        model,
        train_batch_size=8,
        test_batch_size=256,
        data="/data/datasets/mimic3-benchmarks/data/in-hospital-mortality/",
        er_memory=1000,
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        exp_name="Test",
        device="cpu",
        sample_size=None,
        workers=5,
        wandb=False,
    ):
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.device = device
        self.report_freq = report_freq

        config = {}
        config.update(model.get_config())
        config.update(self.get_config())

        self.logger = IHMLogger(exp_name + "_ihm", config, wandb)

        torch.cuda.set_device(self.device)

        train_dataset = IHMDataset(
            data,
            train=True,
            n_samples=sample_size,
        )

        test_dataset = IHMDataset(
            data,
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

        self.crit = nn.BCELoss()

    def fit(self, epochs, task_num, random_samples, ewc_penalty=False, gamma=0.99):
        # need to implement replay for LSTM model
        for epoch in range(epochs):
            self.model.train()
            self.logger.reset()

            ewc = (
                EWC(self.model, random_samples)
                if task_num != 0 and ewc_penalty
                else None
            )

            for batch_idx, (data, label, lens, mask) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model((data, lens))

                if task_num == 0:
                    loss = self.crit(output, label[:, None])
                else:
                    loss = self.crit(output, label[:, None]) + gamma * ewc.penalty(
                        self.model
                    )

                # add replay loss here
                # if self.er_memory != 0:
                #     loss = (1 / (task_num + 1)) * loss + (
                #         1 - (1 / (task_num + 1))
                #     ) * self.replay_loss(self, random_samples)

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
                for batch_idx, (data, label, lens, mask) in enumerate(self.test_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    output = self.model((data, lens))

                    if task_num == 0:
                        loss = self.crit(output, label[:, None])
                    else:
                        loss = self.crit(output, label[:, None]) + gamma * ewc.penalty(
                            self.model
                        )

                    # add replay loss here
                    # if self.er_memory != 0:
                    #     loss = (1 / (task_num + 1)) * loss + (
                    #         1 - (1 / (task_num + 1))
                    #     ) * self.replay_loss(self, random_samples)

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

    # might need to change sample index range to be within batch size
    def get_samples(self):
        # get specified number of random samples
        count = 0
        sample_idx = random.sample(range(len(self.sample_size)), self.er_memory)
        for _, (data, label, lens, mask) in enumerate(self.train_loader):
            if count in sample_idx:
                self.random_samples.append((data, label, lens))
            count += 1

        random.shuffle(self.random_samples)
        return self.random_samples

    # # return average loss on replay samples
    # def replay_loss(self, random_samples):
    #     loss = 0
    #     for _, (data, label, lens, mask) in enumerate(random_samples):
    #         data = data.to(self.device)
    #         label = label.to(self.device)
    #         output = self.model((data, lens))

    #         loss += self.crit(output, label[:, None])
    #     return loss / len(random_samples)
