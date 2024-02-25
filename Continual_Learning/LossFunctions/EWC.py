import torch
import torch.nn as nn
import torch.utils.data

from torch.autograd import Variable
from copy import deepcopy


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


# fix diag and penalty calc according to WUPERR
class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = nn.nll_loss(nn.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                # This is information matrix calc
                # Each parameter p for matrix n is sum of p.grad ** 2 / len(data)
                precision_matrices[n].data += p.grad.data**2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # F(n) * (theta - mean(thetas)) ** 2   Theta is ith parameter, mean(theta) is mean of all p for parameter name n
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
