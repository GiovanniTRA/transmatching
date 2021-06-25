import numpy as np
import os
import torch
import trimesh
from torch.utils.data import Dataset
from scipy.io import loadmat
from transmatching.Utils.utils import RandomRotateCustom, est_area

class FaustDataset(Dataset):

    def __init__(self, in_path, area=True):

        self.in_path = in_path
        self.area = area
        
        self.mat = loadmat(self.in_path+"Fausts/FAUSTS_rem.mat")
        self.data = torch.from_numpy(self.mat["vertices"]).float()

        self.reference = torch.from_numpy(trimesh.load_mesh((os.path.join(self.in_path, '12ktemplate.ply')), process=False).vertices).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        shape = self.data[index]
        shape = shape * 0.7535
        ref = self.reference
    
        if self.area:         
            A = est_area(ref[None,...])[0]
            ref = ref - (ref*(A/A.sum(-1,keepdims=True))[...,None]).sum(-2,keepdims=True)

            A = est_area(shape[None,...])[0]
            shape = shape - (shape*(A/A.sum(-1,keepdims=True))[...,None]).sum(-2,keepdims=True)
        else:
            shape = shape - torch.mean(shape, dim=(-2))
            ref = self.reference - torch.mean(self.reference, dim=-2)
        
        return {'x': shape, 'y': ref}


if __name__ == '__main__':

    a = loadmat("../../test/dataset/Fausts/FAUSTS_rem.mat")
    print(a)

    # d = FaustDataset("../../test/dataset/")
    # print(d.mat.keys())
    # faces = trimesh.load_mesh((os.path.join("../../test/dataset/", '12ktemplate.ply')), process=False).faces
    # mesh = trimesh.Trimesh(vertices=d[0]["x"], faces=d.mat["faces"] - 1, process=False)
    # mesh.show()
    # print(len(d))