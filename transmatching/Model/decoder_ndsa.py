import torch
from torch import nn
from transmatching.Model.attention import MultiHeadAttention
from transmatching.Model.feedforward import FeedForward
from transmatching.Model.layernorm import AddNorm
from transmatching.Model.norm import Norm
from transmatching.Model.pos_enc import PositionalEncoderLearnt
from transmatching.Utils.utils import get_clones


class DecoderLayerNoDecSa(nn.Module):

    def __init__(self, d_channels, heads, d_middle, dropout=0):
        super().__init__()
        self.ln1 = AddNorm(d_channels, dropout=dropout)
        self.ln2 = AddNorm(d_channels, dropout=dropout)
        self.linear1 = nn.Linear(d_channels, d_channels)
        self.attn1 = MultiHeadAttention(heads, d_channels, d_channels, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_channels, d_channels * 2), nn.ReLU(),
            nn.Linear(d_channels * 2, d_channels * 4), nn.ReLU(),
            nn.Linear(d_channels * 4, d_channels * 2), nn.ReLU(),
            nn.Linear(d_channels * 2, d_channels), nn.ReLU(),
        )

    def forward(self, x, src):

        latent = self.attn1(x, src, src)
        x = self.ln1(x, self.linear1(latent))

        latent2 = self.mlp(x)
        x = self.ln2(x, latent2)

        return x


class DecoderNoDecSa(nn.Module):

    def __init__(self, d_latent, d_channels, d_middle, N, heads, max_seq_len, d_origin, dropout=0):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoderLearnt(d_channels, max_seq_len)
        self.norm = Norm(d_channels)
        self.decoder_layer = get_clones(DecoderLayerNoDecSa(d_channels, heads, d_middle, dropout), self.N)
        self.embedder = nn.Sequential(nn.Linear(d_origin, d_channels // 8), nn.ReLU(),
                                      nn.Linear(d_channels // 8, d_channels // 4), nn.ReLU(),
                                      nn.Linear(d_channels // 4, d_channels // 2), nn.ReLU(),
                                      nn.Linear(d_channels // 2, d_channels))

    def forward(self, x, src):

        src = self.pe(src)
        x = self.embedder(x)

        for i in range(self.N):
            x = self.decoder_layer[i](x, src)

        return self.norm(x)


if __name__ == '__main__':
    src = torch.zeros(10, 20, 5)
    x = torch.ones(10, 100, 3)
    dec = Decoder(3, 5, 128, 4, 1, 1000)
    print(dec(x, src).shape)

