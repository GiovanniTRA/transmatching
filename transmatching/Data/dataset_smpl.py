import numpy as np
import os
import torch
import trimesh
from torch.utils.data import Dataset

from transmatching.Utils.utils import RandomRotateCustom, est_area


class SMPLDataset(Dataset):

    def __init__(self, in_path, train=True, area=True):

        self.in_path = in_path
        self.train = train
        self.area = area
        self.train_data = torch.from_numpy(np.load(os.path.join(self.in_path, '12k_shapes_train.npy'))).float()
        self.test_data = torch.from_numpy(np.load(os.path.join(self.in_path, '12k_shapes_test.npy'))).float()
        self.reference = torch.from_numpy(trimesh.load_mesh((os.path.join(self.in_path, '12ktemplate.ply')),
                                                            process=False).vertices).float()

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        return self.test_data.shape[0]

    def __getitem__(self, index):

        if self.train:
            shape = self.train_data[index]
            shape = RandomRotateCustom(shape, 180, 1)
        else:
            shape = self.test_data[index]
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
    pass
