import torch
import torch.nn as nn
import random

from torch import optim
from torchmimic.EWC import EWC
from libauc.losses import pAUC_DRO_Loss
from libauc.optimizers import SOPAs


class PhenotypingBenchmark:
    def __init__(
        self,
        model,
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        logger=None,
        device="cpu",
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = model
        self.device = device
        self.report_freq = report_freq
        self.task = "phen"
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
        # rsc = len(random_samples) * train_loader.batch_size
        rsc = len(random_samples)
        self.crit_rep = pAUC_DRO_Loss(data_len=rsc)
        self.crit = pAUC_DRO_Loss(data_len=len(train_loader.dataset))
        self.optimizer = SOPAs(
            self.model.parameters(),
            mode="adam",
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )

        results = {}
        step = (
            len(train_loader) // len(random_samples)
            if len(random_samples) > 0
            else len(train_loader)
        )
        if step == 0:
            step = 1  # in the case of less training samples than buffer size
        for epoch in range(epochs):
            print("------------------------------------")
            print(f"Task: {task_num + 1}, Epoch: {epoch + 1}")

            model_copy = self.model
            self.model.train()
            self.logger.reset()

            ewc = (
                EWC(
                    model_copy,
                    random_samples,
                    rsc,
                    self.device,
                    self.task,
                )
                if task_num != 0 and ewc_penalty
                else None
            )

            idx = 0
            for batch_idx, (data, label, lens, mask, index) in enumerate(train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                index = torch.tensor(index, dtype=torch.int)
                index = index.to(self.device)
                output = self.model((data, lens))
                # output = torch.sigmoid(output)
                # print(len(train_loader.dataset))
                # print(index)
                print(label)
                pos_mask = (label == 1).squeeze()
                print(pos_mask)
                print(sum(pos_mask))

                # add ewc penalty
                if task_num == 0 or not ewc_penalty:
                    loss = self.crit(output, label, index)
                elif task_num > 0 and ewc_penalty:
                    loss = self.crit(output, label, index) + importance * ewc.penalty(
                        self.model
                    )

                # add replay loss
                # if task_num > 0 and replay:
                #     loss = (1 / (task_num + 1)) * loss + (
                #         1 - (1 / (task_num + 1))
                #     ) * self.replay_loss(random_samples)

                if (
                    task_num > 0
                    and replay
                    and (batch_idx % step == 0)
                    and idx < len(random_samples)
                ):
                    loss = (1 - (1 / (task_num + 1))) * loss + (
                        1 / (task_num + 1)
                    ) * self.replay_loss(random_samples, idx, rsc)
                    idx += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.logger.update(output, label, loss)

                if (batch_idx + 1) % self.report_freq == 0:
                    print(f"Train: epoch: {epoch+1}, loss = {self.logger.get_loss()}")

            self.logger.print_metrics(epoch, split="Train", task=None)
            print("-------------")
            idx = 0
            # evaluate model on all tasks
            with torch.no_grad():
                for eval_task, val_loader in enumerate(val_loaders):
                    self.logger.reset()
                    self.model.eval()
                    for batch_idx, (data, label, lens, mask, index) in enumerate(
                        val_loader
                    ):
                        data = data.to(self.device)
                        label = label.to(self.device)
                        index = torch.tensor(index, dtype=torch.int)
                        index = index.to(self.device)
                        output = self.model((data, lens))
                        # output = torch.sigmoid(output)

                        # add ewc penalty
                        if task_num == 0 or not ewc_penalty:
                            loss = self.crit(output, label, index)
                        elif task_num > 0 and ewc_penalty:
                            loss = self.crit(
                                output, label, index
                            ) + importance * ewc.penalty(self.model)

                        # add replay loss
                        # if task_num > 0 and replay:
                        #     loss = (1 / (task_num + 1)) * loss + (
                        #         1 - (1 / (task_num + 1))
                        #     ) * self.replay_loss(random_samples)
                        if (
                            task_num > 0
                            and replay
                            and (batch_idx % step == 0)
                            and idx < len(random_samples)
                        ):
                            loss = (1 - (1 / (task_num + 1))) * loss + (
                                1 / (task_num + 1)
                            ) * self.replay_loss(random_samples, idx, rsc)
                            idx += 1

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
                    for batch_idx, (data, label, lens, mask, index) in enumerate(
                        test_loader
                    ):
                        data = data.to(self.device)
                        label = label.to(self.device)
                        index = torch.tensor(index, dtype=torch.int)
                        index = index.to(self.device)
                        output = self.model((data, lens))

                        loss = self.crit(output, label, index)
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

    def replay_loss(self, random_samples, idx=0, data_len=0):
        # idx = random.randint(0, len(random_samples) - 1)
        data, label, lens, mask, index = random_samples[idx]
        index = [idx] * 8
        index = torch.tensor(index, dtype=torch.int)
        index = index.to(self.device)

        data = data.to(self.device)
        label = label.to(self.device)
        output = self.model((data, lens))
        replay_loss = self.crit_rep(output, label, index)

        return replay_loss
