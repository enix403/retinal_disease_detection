import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # x: (3, 224, 224)
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            # x: (32, 222, 222)
            nn.MaxPool2d(2),
            # x: (32, 111, 111)
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            # x: (64, 109, 109)
            nn.MaxPool2d(2),
            # x: (64, 54, 54)
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            # x: (128, 52, 52)
            nn.MaxPool2d(2),
            # x: (128, 26, 26)
            nn.Flatten(start_dim=-3, end_dim=-1),
            # x: (128*26*26 = 86528,)
            nn.Linear(128*26*26, 256),
            nn.ReLU(),
            # x: (512,)
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            # x: (256,)
            nn.Linear(256, 2),
            # x: (2,)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits