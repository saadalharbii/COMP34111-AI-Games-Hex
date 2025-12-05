# agents/Group8/PolicyValueNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """
    AlphaZero-style policy + value network for 11x11 Hex.

    Input:  (batch, 2, 11, 11)
    Output: policy_logits (batch, 121)
            value        (batch, 1)
    """

    def __init__(self, board_size=11):
        super().__init__()
        self.board_size = board_size
        channels = 64

        # Shared CNN trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.trunk(x)

        # Policy: (B,1,11,11) → (B,121)
        policy_logits = self.policy_head(h).flatten(1)

        # Value: (B, C, 1, 1) → (B,1)
        value = self.value_head(h)

        return policy_logits, value
