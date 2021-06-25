import torch
# from transmatching.Model.model import Model
import matplotlib.pyplot as plt
import time
import gc

from transmatching.Utils.utils import get_clones, est_area, chamfer_loss
from transmatching.Model.debug import Debug

def chamfer(y_hat,src):
    dist = torch.cdist(y_hat,src)
    loss = dist.min(-2)[0].mean(-1)+dist.min(-1)[0].mean(-1)
    return loss

def subsamp(X, samp=10000):
    if X.shape[1]<samp:
        return X
    subsamp = torch.randperm(X.shape[1])[:samp]
    return X[:,subsamp,:]


def get_splits(Y,maxvert=15000):
    splits = (Y.shape[0]+1)//maxvert+1
    selected = [[] for i in range(splits)]

    subsamp = torch.randperm(Y.shape[0]).numpy()
    n = Y.shape[0]//splits+1
    for i,sel in enumerate(selected):
        selected[i] = list(subsamp[i*n:(i+1)*n])
        
    return selected

def register(model,X,Y,maxvert=350000):
    if Y.shape[1]<maxvert:
        return model(X,Y)
    
    assert(X.shape[0]==1)
    X = X[0]
    Y = Y[0]
    
    selected = get_splits(Y,maxvert)
    
    y_hats = Y*0
    for sel in selected:
        with torch.no_grad():
            y_hat = model(X[None,...],Y[None,sel,:].cuda())
        y_hats[sel,:] = y_hat[0]
        del y_hat
        gc.collect()
        
    return y_hats[None,...]    
        

def bidirectional_match(model,shape1,shape2, dorefine=False,extra_data=None):
        ref_steps = 10        
        if dorefine:            
            y_hats,loss1 = refine(model,shape1,shape2,max_iter=ref_steps)
            y_hat1=y_hats[-1].to(shape1.device)

            y_hats,loss2 = refine(model,shape2,shape1,max_iter=ref_steps)
            y_hat2=y_hats[-1].to(shape1.device)
        else:            
            y_hat1 = register(model,subsamp(shape1), shape2)
            loss1 = chamfer(subsamp(y_hat1),subsamp(shape1))

            y_hat2 = register(model,subsamp(shape2), shape1)
            loss2 = chamfer(subsamp(y_hat2),subsamp(shape2))
        
#         print('%.2e - %.2e' % (loss1,loss2))
#             swap shapes if the error decreases        
        better = torch.stack([loss1, loss2],-1).argmin(-1)
    
        if shape1.shape[0]==1:
            if extra_data is not None and len(extra_data)==2:
                return [shape1,shape2][better],\
                       [shape2,shape1][better],\
                       [y_hat1,y_hat2][better],\
                       (extra_data[better],extra_data[1-better])                        
            return [shape1,shape2][better],\
                   [shape2,shape1][better],\
                   [y_hat1,y_hat2][better]
            
        if shape1.shape==shape2.shape: #handles batches
            a1 = shape1*(1-better)[:,None,None] + shape2*better[:,None,None]
            return shape1*(1-better)[:,None,None] + shape2*better[:,None,None],\
                   shape2*(1-better)[:,None,None] + shape1*better[:,None,None],\
                   y_hat1*(1-better)[:,None,None] + y_hat2*better[:,None,None]
        
        
    
def refine(model,src,trg,max_iter=50, samp=1000, lr=5e-3, saveall=False):
    
    src = subsamp(src)
    with torch.no_grad():
#         D = torch.cdist(src,src)
#         D = 1/(-50*D).exp().sum(-1).to(trg.device).detach()
        D = est_area(src)
        
        x = model.tokens.expand(src.size(0), model.tokens.size(0), model.tokens.size(1))
        e_out = model.encoder(x, src, D)
        x_d_pe = model.decoder.embedder(trg)

#         D = torch.cdist(trg,trg)
#         D = 1/(-50*D).exp().sum(-1).to(trg.device).detach()
        D = est_area(trg)
        
    e_opt = torch.autograd.Variable(e_out.detach(), requires_grad=True)
    opt = torch.optim.Adam([e_opt], lr=lr)
    
    y_hats=[]
    for it in range(max_iter): 
        #decoding
        src_d = model.decoder.pe(e_opt)
        ssamp = torch.randperm(x_d_pe.shape[1])[:samp]
        x_d = x_d_pe[:,ssamp,:].clone()

        for i in range(model.decoder.N):
            x_d = model.decoder.decoder_layer[i](x_d, src_d, weights2=D[:,ssamp])
        d_out = model.decoder.norm(x_d)

        y_hat = model.out(d_out)
        if it==0 or it==max_iter-1 or saveall:
            y_hats.append(y_hat.detach().cpu())
        
#         dist = torch.cdist(y_hat,src)
#         losses = dist.min(-1)[0].mean(-1)+dist.min(-2)[0].mean(-1)
        losses = chamfer_loss(y_hat,src)
        loss = losses.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses = losses.detach()
    
    del loss, src_d, x_d, d_out, y_hat, opt
    gc.collect()
    
    with torch.no_grad():
        src_d = model.decoder.pe(e_opt)
        x_d = x_d_pe[:,:,:].clone()

        for i in range(model.decoder.N):
            x_d = model.decoder.decoder_layer[i](x_d, src_d, weights2=D[:,:])
        d_out = model.decoder.norm(x_d)
        y_hat = model.out(d_out)
        y_hats[-1]=y_hat.detach().cpu()

    return y_hats, losses


def refine_hires(model,src,trg,max_iter=50, samp=3000, lr=5e-3, saveall=False):
    
    src = subsamp(src,30000)
    with torch.no_grad():
#         D = torch.cdist(src,src)
#         D = 1/(-50*D).exp().sum(-1).to(trg.device).detach()
        D = est_area(src)

        x = model.tokens.expand(src.size(0), model.tokens.size(0), model.tokens.size(1))
        e_out = model.encoder(x, src, D)
        x_d_pe = model.decoder.embedder(trg)



    e_opt = torch.autograd.Variable(e_out.detach(), requires_grad=True)
    opt = torch.optim.Adam([e_opt], lr=lr)
    
    y_hats=[]
    for it in range(max_iter): 
        #decoding
        src_d = model.decoder.pe(e_opt)
        ssamp = torch.randperm(x_d_pe.shape[1])[:samp]
        x_d = x_d_pe[:,ssamp,:].clone().contiguous()

#         D = torch.cdist(trg[:,ssamp,:],trg[:,ssamp,:])
#         D = 1/(-50*D).exp().sum(-1).to(trg.device).detach()
        D = est_area(trg[:,ssamp,:]).detach()
        for i in range(model.decoder.N):
            x_d = model.decoder.decoder_layer[i](x_d, src_d, weights2=D)
        d_out = model.decoder.norm(x_d)

        y_hat = model.out(d_out)
        if it==0 or it==max_iter-1 or saveall:
            y_hats.append(y_hat.detach().cpu())
        
        dist = torch.cdist(y_hat,src)
        losses = dist.min(-1)[0].mean(-1)+dist.min(-2)[0].mean(-1)
        loss = losses.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses = losses.detach()
    
    del dist, loss, src_d, x_d, d_out, y_hat, opt
    gc.collect()
    
    assert(trg.shape[0]==1)
    
    with torch.no_grad():
        src_d = model.decoder.pe(e_opt)
        
        selected = get_splits(trg[0],300000)
        y_hat = trg.cpu()*0
        for sel in selected:
#             D = torch.cdist(trg[:,sel,:],trg[:,sel,:])
#             D = 1/(-50*D).exp().sum(-1).to(trg.device).detach()
            D = est_area(trg[:,sel,:])
            
            x_d = x_d_pe[:,sel,:].clone()
            for i in range(model.decoder.N):
                x_d = model.decoder.decoder_layer[i](x_d, src_d, weights2=D)
            d_out = model.decoder.norm(x_d)
            _y_hat = model.out(d_out)
            y_hat[0,sel,:] = _y_hat.cpu()

        y_hats[-1]=y_hat.detach().cpu()

    return y_hats, losses
