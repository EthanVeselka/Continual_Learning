import torch
import torch.nn as nn


class WUPERR(nn.Module):
    def __init__(
        self,
        n_classes,
        hidden_dim1=40,
        hidden_dim2=25,
        dropout_rate=0,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate

        self.model = nn.Sequential(
            nn.Linear(76, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim2, self.n_classes),
            nn.sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def get_config(self):
        return {
            "n_classes": self.n_classes,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "dropout_rate": self.dropout_rate,
        }
