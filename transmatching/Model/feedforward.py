from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=32, dropout=0.05):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

