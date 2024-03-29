import torch
import torch.nn as nn
import random

from torch import optim
from torchmimic.EWC import EWC


class IHMBenchmark:
    def __init__(
        self,
        model,
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        logger=None,
        device="cpu",
    ):
        self.model = model
        self.device = device
        self.report_freq = report_freq
        self.task = "ihm"
        self.logger = logger

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
        val_loaders,
        test_loaders,
        task_num,
        random_samples,
        replay=False,
        ewc_penalty=False,
        importance=1,
    ):
        results = {}

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
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model((data, lens))

                # add ewc penalty
                if task_num == 0 or not ewc_penalty:
                    loss = self.crit(output, label[:, None])
                elif task_num > 0 and ewc_penalty:
                    loss = self.crit(output, label[:, None]) + importance * ewc.penalty(
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

            self.logger.print_metrics(epoch, split="Train", task=None)
            print("-------------")

            # evaluate model on all tasks
            with torch.no_grad():
                for eval_task, val_loader in enumerate(val_loaders):
                    self.logger.reset()
                    self.model.eval()
                    for batch_idx, (data, label, lens, mask) in enumerate(val_loader):
                        data = data.to(self.device)
                        label = label.to(self.device)
                        output = self.model((data, lens))

                        # add ewc penalty
                        if task_num == 0 or not ewc_penalty:
                            loss = self.crit(output, label[:, None])
                        elif task_num > 0 and ewc_penalty:
                            loss = self.crit(
                                output, label[:, None]
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
                    self.logger.print_metrics(
                        epoch,
                        split="Eval",
                        task="Eval Task " + str(eval_task + 1),
                    )

            if epoch == (epochs - 1):
                results["val"] = self.logger.get_results()

        tests = {}
        if test_loaders != None:
            # test model on all tasks
            with torch.no_grad():
                for eval_task, test_loader in enumerate(test_loaders):
                    self.logger.reset()
                    self.model.eval()
                    for batch_idx, (data, label, lens, mask) in enumerate(test_loader):
                        data = data.to(self.device)
                        label = label.to(self.device)
                        output = self.model((data, lens))
                        loss = self.crit(output, label[:, None])

                        self.logger.update(output, label, loss)

                        if (batch_idx + 1) % self.report_freq == 0:
                            print(
                                f"Eval: epoch: {epoch+1}, loss = {self.logger.get_loss()}"
                            )

                    print("\n")
                    print("-------------------------")
                    print(f"Testing task: {eval_task + 1}")
                    print("-------------------------")
                    self.logger.print_metrics(
                        (epochs - 1),
                        split="Test",
                        task="Eval Task " + str(eval_task + 1),
                        test=True,
                    )

                    tests["Eval Task " + str(eval_task + 1)] = (
                        self.logger.save_results()
                    )

            # self.logger.save(self.model)

        results["test"] = tests
        return results

    def replay_loss(self, random_samples):
        idx = random.randint(0, len(random_samples) - 1)
        data, label, lens, mask = random_samples[idx]

        data = data.to(self.device)
        label = label.to(self.device)
        output = self.model((data, lens))
        replay_loss = self.crit(output, label[:, None])

        return replay_loss
