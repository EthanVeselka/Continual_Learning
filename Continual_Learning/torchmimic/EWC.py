from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from libauc.losses import pAUC_DRO_Loss

# from torch.autograd import Variable
import torch.utils.data


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, loss, shift_map, device, task):

        self.model = model
        self.dataset = dataset
        self.device = device
        self.task = task
        self.loss = loss
        self.shift_map = shift_map

        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        # self.model.eval()
        self.model.train()
        for idx, (data, label, lens, mask, index, task_num) in enumerate(self.dataset):
            self.model.zero_grad()
            data = data.to(self.device)
            # index = [idx] * 8
            index = torch.tensor(index, dtype=torch.int)
            index += self.shift_map[task_num]
            index = index.to(self.device)

            if self.task == "los":  # multiclass classification
                label = label.type(torch.LongTensor)
                label = label.to(self.device)
                output = self.model((data, lens))

                loss = nn.CrossEntropyLoss()
                loss = loss(output, label)

            else:  # binary classification (phen uses ovr)
                label = label.to(self.device)
                output = self.model((data, lens))
                loss = self.loss
                # loss = nn.BCELoss()
                if self.task == "ihm":
                    loss = loss(output, label[:, None], index)
                elif self.task == "decomp":
                    loss = loss(output[:, 0], label, index)
                elif self.task == "phen":
                    loss = loss(output, label, index)

            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data**2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
