import torch
import torch.nn as nn

class AnomalyNet(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, D)
        scores = self.net(x)
        scores = scores.view(B, T)
        return scores