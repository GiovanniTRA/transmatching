import torch
from torch import nn
from transmatching.Model.decoder import Decoder
from transmatching.Model.encoder import Encoder


from transmatching.Model.attention import MultiHeadAttention
from transmatching.Model.feedforward import FeedForward
from transmatching.Model.layernorm import AddNorm
from transmatching.Model.norm import Norm
from transmatching.Model.pos_enc import PositionalEncoderLearnt
import torch
from transmatching.Utils.utils import get_clones, est_area
from transmatching.Model.debug import Debug


class Model(nn.Module):
    def __init__(self, d_bottleneck, d_latent, d_channels, d_middle, N, heads, max_seq_len, d_origin, dropout=0, estimate_area=True):
        super().__init__()

        self.encoder = Encoder(d_latent, d_channels, d_middle, N, heads, max_seq_len, d_origin, dropout)
        self.decoder = Decoder(d_channels, d_latent, d_middle, N, heads, max_seq_len, d_origin, dropout)
        self.out = nn.Sequential(nn.Linear(d_channels, d_origin * 16), nn.ReLU(),
                                 nn.Linear(d_origin * 16, d_origin * 8), nn.ReLU(),
                                 nn.Linear(d_origin * 8, d_origin * 4), nn.ReLU(),
                                 nn.Linear(d_origin * 4, d_origin * 2), nn.ReLU(),
                                 nn.Linear(d_origin * 2, d_origin))
        self.tokens = nn.Parameter(torch.randn((d_bottleneck, d_latent)))
        self.estimate_area = estimate_area

    def forward(self, src, trg):
        Ds = Dt = None
        if self.estimate_area:
            with torch.no_grad():
                Ds = est_area(src)
                Dt = est_area(trg)
        if Debug.debug:
                self.Ds=Ds.cpu()
                self.Dt=Dt.cpu()
            
        x = self.tokens.expand(src.size(0), self.tokens.size(0), self.tokens.size(1))
        e_out = self.encoder(x, src, Ds)
            
        d_out = self.decoder(trg, e_out, Dt)
        out = self.out(d_out)
        return out

if __name__ == '__main__':
    src = torch.ones(1, 100, 3)
    trg = torch.ones(1, 100, 3)
    model = Model2(100, 32, 32, 128, 4, 1, 100, 3)
    print(model(src).shape)


