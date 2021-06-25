import torch
from torch import nn
from transmatching.Model.decoder_ndsa import DecoderNoDecSa
from transmatching.Model.encoder import Encoder


class ModelNoDecSa(nn.Module):

    def __init__(self, d_bottleneck, d_latent, d_channels, d_middle, N, heads, max_seq_len, d_origin, dropout=0):
        super().__init__()

        self.encoder = Encoder(d_latent, d_channels, d_middle, N, heads, max_seq_len, d_origin, dropout)
        self.decoder = DecoderNoDecSa(d_channels, d_latent, d_middle, N, heads, max_seq_len, d_origin, dropout)
        self.out = nn.Sequential(nn.Linear(d_channels, d_origin * 16), nn.ReLU(),
                                 nn.Linear(d_origin * 16, d_origin * 8), nn.ReLU(),
                                 nn.Linear(d_origin * 8, d_origin * 4), nn.ReLU(),
                                 nn.Linear(d_origin * 4, d_origin * 2), nn.ReLU(),
                                 nn.Linear(d_origin * 2, d_origin))
        self.tokens = nn.Parameter(torch.randn((d_bottleneck, d_latent)))

    def forward(self, src, trg):

        x = self.tokens.expand(src.size(0), self.tokens.size(0), self.tokens.size(1))
        e_out = self.encoder(x, src)
        d_out = self.decoder(trg, e_out)
        out = self.out(d_out)
        return out


if __name__ == '__main__':
    src = torch.ones(1, 100, 3)
    trg = torch.ones(1, 100, 3)
    model = Model2(100, 32, 32, 128, 4, 1, 100, 3)
    print(model(src).shape)


