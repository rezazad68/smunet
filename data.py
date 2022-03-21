#!/usr/bin/env python3
# encoding: utf-8
# Code modified from https://github.com/Wangyixinxin/ACN
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random

class Brats2018(Dataset):

    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + "_" + mode + '.nii')
            volume = nib.load(volume_path).get_data()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        ed_volume = (seg_volume == 2) # peritumoral edema ED
        net_volume = (seg_volume == 1) # enhancing tumor core NET
        et_volume = (seg_volume == 4) # enhancing tumor ET
        bg_volume = (seg_volume == 0)
        
        seg_volume = [ed_volume, net_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))


    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized

def split_dataset(data_root, test_p):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    N = int(len(patients_dir)*test_p)
    train_patients_list =  patients_dir[N:]
    val_patients_list   =  patients_dir[:N]

    return train_patients_list, val_patients_list
    
def make_data_loaders(config):
    train_list, val_list = split_dataset(config['path_to_data'], float(config['test_p']))  
    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]      
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)
    train_ds = Brats2018(train_list, crop_size=crop_size, modes=config['modalities'], train=True)
    val_ds = Brats2018(val_list,     crop_size=crop_size, modes=config['modalities'], train=False)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=False)
    return loaders

