import torch
import torch.nn as nn
import random


from torch import optim
from torchmimic.loggers import DecompensationLogger
from torchmimic.EWC import EWC


class DecompensationBenchmark:
    def __init__(
        self,
        model,
        train_batch_size=8,
        test_batch_size=256,
        data="../../datasets/mimic3-benchmarks/decompensation",
        buffer_size=1000,
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        exp_name="Test",
        device="cpu",
        wandb=False,
    ):
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.device = device
        self.report_freq = report_freq

        self.buffer_size = buffer_size
        self.task = "decomp"

        config = {}
        config.update(model.get_config())
        config.update(self.get_config())

        self.logger = DecompensationLogger(exp_name + "_decomp", config, wandb)

        torch.cuda.set_device(self.device)
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        self.crit = nn.BCELoss()

    def fit(
        self,
        epochs,
        train_loader,
        test_loaders,
        task_num,
        random_samples,
        replay=False,
        ewc_penalty=False,
        importance=2,
    ):

        for epoch in range(epochs):
            print("------------------------------------")
            print(f"Task: {task_num + 1}, Epoch: {epoch + 1}")

            model_copy = self.model
            self.model.train()
            self.logger.reset()

            ewc = (
                EWC(model_copy, random_samples, self.device, self.task)
                if task_num != 0 and ewc_penalty
                else None
            )

            for batch_idx, (data, label, lens, mask) in enumerate(train_loader):
                print(f"Progress: {batch_idx/len(train_loader) * 100:.2f}%", end="\r")
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model((data, lens))

                # add ewc penalty
                if task_num == 0 or not ewc_penalty:
                    loss = self.crit(output[:, 0], label)
                elif task_num > 0 and ewc_penalty:
                    loss = self.crit(output[:, 0], label) + importance * ewc.penalty(
                        self.model
                    )

                # add replay loss
                if task_num > 0 and replay:
                    loss = (1 / (task_num + 1)) * loss + (
                        1 - (1 / (task_num + 1))
                    ) * self.replay_loss(random_samples)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.logger.update(output, label, loss)

                if (batch_idx + 1) % self.report_freq == 0:
                    print(f"Train: epoch: {epoch+1}, loss = {self.logger.get_loss()}")

            self.logger.print_metrics(epoch, split="Train")
            print("-------------")

            # evaluate model on all tasks
            with torch.no_grad():
                for eval_task, test_loader in enumerate(test_loaders):
                    self.model.eval()
                    self.logger.reset()
                    for batch_idx, (data, label, lens, mask) in enumerate(test_loader):
                        data = data.to(self.device)
                        label = label.to(self.device)
                        output = self.model((data, lens))

                        # add ewc penalty
                        if task_num == 0 or not ewc_penalty:
                            loss = self.crit(output[:, 0], label)
                        elif task_num > 0 and ewc_penalty:
                            loss = self.crit(
                                output[:, 0], label
                            ) + importance * ewc.penalty(self.model)

                        # add replay loss
                        if task_num > 0 and replay:
                            loss = (1 / (task_num + 1)) * loss + (
                                1 - (1 / (task_num + 1))
                            ) * self.replay_loss(random_samples)

                        self.logger.update(output, label, loss)

                        if (batch_idx + 1) % self.report_freq == 0:
                            print(
                                f"Eval: epoch: {epoch+1}, loss = {self.logger.get_loss()}"
                            )
                    print(f"Eval task: {eval_task + 1}")
                    self.logger.print_metrics(epoch, split="Eval")

    def get_config(self):
        return {
            "test_batch_size": self.test_batch_size,
            "train_batch_size": self.train_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def replay_loss(self, random_samples):
        idx = random.randint(0, len(random_samples) - 1)
        data, label, lens, mask = random_samples[idx]

        data = data.to(self.device)
        label = label.to(self.device)
        output = self.model((data, lens))
        replay_loss = self.crit(output[:, 0], label)

        return replay_loss
