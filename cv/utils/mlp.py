"""
Author: Redal
Date: 2025-12-13
Todo: Intergrated all kinds of MLP class to analysis how diffence in them
Homepage: https://github.com/Rtwotwo/CoAI.git
"""
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_chans: int,
        dropout: float = 0.0
        out_chans: int = None,
        hidden_chans: int = None,
        act_layer: nn.Module = nn.GELU,
    )->None:
        super().__init__()
        out_chans = in_chans or out_chans
        hidden_chans = in_chans or hidden_chans
        self.fc1 = nn.Linear(in_chans, hidden_chans)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_chans, out_chans)
        self.dropout = nn.Dropout(dropout)
    def forward(
        self, 
        x: torch.Tensor
    )->torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


