from torch import nn
from transmatching.Model.attention import MultiHeadAttention
from transmatching.Model.feedforward import FeedForward
from transmatching.Model.layernorm import AddNorm
from transmatching.Model.norm import Norm
from transmatching.Model.pos_enc import PositionalEncoderLearnt
import torch
from transmatching.Utils.utils import get_clones


class EncoderLayer(nn.Module):

    def __init__(self, d_latent, d_channels, heads, d_middle, d_origin, dropout=0):
        super().__init__()

        self.ln1 = AddNorm(d_latent, dropout=dropout)
        self.ln2 = AddNorm(d_latent, dropout=dropout)
        self.linear1 = nn.Linear(d_channels, d_latent)
        self.attn1 = MultiHeadAttention(heads, d_latent, d_channels, dropout=dropout)
        self.attn2 = MultiHeadAttention(heads, d_latent, d_latent, dropout=dropout)
        self.ff = FeedForward(d_latent, d_middle, dropout=dropout)
        self.embedder = nn.Sequential(nn.Linear(d_origin, d_channels // 8), nn.ReLU(),
                                      nn.Linear(d_channels // 8, d_channels // 4), nn.ReLU(),
                                      nn.Linear(d_channels // 4, d_channels // 2), nn.ReLU(),
                                      nn.Linear(d_channels // 2, d_channels))

    def forward(self, x, src, weights1=None, weights2=None):
        src = self.embedder(src)
        latent = self.attn1(x, src, src, weights=weights1)
        x = self.ln1(x, self.linear1(latent))

        latent2 = self.attn2(x, x, x, weights=weights2)
        latent2 = self.ff(latent2)
        x = self.ln2(x, latent2)

        return x


class Encoder(nn.Module):

    def __init__(self, d_latent, d_channels, d_middle, N, heads, max_seq_len, d_origin, dropout=0):
        super().__init__()

        self.N = N
        self.pe = PositionalEncoderLearnt(d_latent, max_seq_len)
        self.norm = Norm(d_latent)
        self.encoder_layer = get_clones(EncoderLayer(d_latent, d_channels, heads, d_middle, d_origin, dropout), self.N)

    def forward(self, x, src, weights=None):

        x = self.pe(x)
        for i in range(self.N):
                x = self.encoder_layer[i](x, src, weights1=weights)

        return self.norm(x)


if __name__ == '__main__':
    x = torch.zeros(10, 10, 5)
    src = torch.ones(10, 100, 3)
    enc = Encoder(5, 3, 128, 4, 1, 1000)
    print(enc(x, src).shape)

