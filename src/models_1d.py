# src/models_1d.py

import torch
import torch.nn as nn


def conv_block(in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    """
    Conv1d -> BatchNorm1d -> ReLU.
    Small kernels + BN (Bag-of-Tricks style).
    """
    padding = kernel_size // 2  # keep length with stride 1
    return nn.Sequential(
        nn.Conv1d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
    )


class AMC1DCNN_v2(nn.Module):
    """
    1D CNN for (batch, 2, 128) I/Q inputs.
    Uses stacked small kernels + BN + GAP + MLP head.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Input: (B, 2, 128)
        self.features = nn.Sequential(
            # Block 1: 2 -> 32, length 128 -> 64
            conv_block(2, 32, kernel_size=3),
            conv_block(32, 32, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64

            # Block 2: 32 -> 64, length 64 -> 32
            conv_block(32, 64, kernel_size=3),
            conv_block(64, 64, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),  # 64 -> 32

            # Block 3: 64 -> 128, length 32 -> 16
            conv_block(64, 128, kernel_size=3),
            conv_block(128, 128, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),  # 32 -> 16
        )

        # Global average over time dimension -> (B, 128)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # He init for conv/linear, BN gamma=1, beta=0
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 128)
        x = self.features(x)          # -> (B, 128, 16)
        x = self.gap(x).squeeze(-1)   # -> (B, 128)
        x = self.classifier(x)        # -> (B, num_classes)
        return x
