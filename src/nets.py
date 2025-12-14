import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        logits = self.model(x)
        return {
            "logits": logits,
            "loss": self.loss(logits, y) if y is not None else None,
        }
