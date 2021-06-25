import math
import torch
from torch import nn
import torch.nn.functional as F
from transmatching.Model.debug import Debug

try:
    from pykeops.torch import LazyTensor
except ImportError:
    Debug.keops=False



def attention(q, k, v, d_k, mask=None, dropout=None, weights=None, w=1):
    if Debug.keops:
        bs =  q.shape[0] # b x h x nq x d

        lq = LazyTensor(q.reshape(-1,q.shape[2],q.shape[3])[:,None,:,:].contiguous()) #(b x h) x 1 x nq x d
        lk = LazyTensor(k.reshape(-1,k.shape[2],k.shape[3])[:,:,None,:].contiguous()) #(b x h) x nk x 1 x d
        lv = v.reshape(-1,v.shape[2],v.shape[3]).contiguous()

        scores = (lq*lk).sum(-1)/ math.sqrt(d_k) # b x nk x nq
        
        scores = scores.exp()
        if weights is not None:
            lw = LazyTensor(weights[:,None,:].repeat(1,q.shape[1],1).reshape(-1,weights.shape[-1])[:,:,None,None].contiguous())
            scores = scores*lw

        output = scores.t()@lv
        output = output/scores.sum(1)

        assert(dropout.p==0 or dropout is None)
        
        output = output.reshape(q.shape[0],q.shape[1],output.shape[1],output.shape[2])
            
        #hotfix [:bs]: backprob through lazytensors fail if bs=1
        return output[:bs], None
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=(-1))
    wscores = scores
    
    if weights is not None:
        wscores = wscores*weights[:,None,None,:]
        wscores = wscores/wscores.sum(-1,keepdims=True)
        
    if dropout is not None:
        scores = dropout(wscores)

    output = torch.matmul(wscores, v)    
    
    return output, scores


class MultiHeadAttention(nn.Module):
    weighted=False
    
    def __init__(self, heads, d_latent, d_model, dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_latent, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, weights=None):

        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores, _p = attention(q, k, v, self.d_k, mask, self.dropout, weights)
        
        if Debug.debug:
            self.scores = _p.detach().cpu()  
        
        
        concat = scores.transpose(1, 2).reshape(bs, -1, self.d_model)
        output = self.out(concat)

        return output

