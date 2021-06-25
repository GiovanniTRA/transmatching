import torch
from torch import nn


class PositionalEncoderLearnt(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))

    def forward(self, x):
        seq_len = x.size(-2)
        x = x + self.pos[:seq_len]
        return x

