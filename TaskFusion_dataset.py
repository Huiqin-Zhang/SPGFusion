# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import glob
import os
from util import randrot,randfilp

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.npy"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = 'assets/data/MSRS/Train_vi'
            data_dir_ir = 'assets/data/MSRS/Train_ir'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            
        elif split == 'val' or split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
            
        self.crop = torchvision.transforms.RandomCrop(224)
        print(min(len(self.filenames_vis), len(self.filenames_ir)))
    def _prepare_patches(self):
        patches = []
        for index in range(len(self.filenames_vis)):
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            h, w = image_vis.shape[:2]
            for x in range(0, h - self.image_size + 1, self.stride):
                for y in range(0, w - self.image_size + 1, self.stride):
                    patch_info = (index, x, y)
                    patches.append(patch_info)
        return patches
    
    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        
        image_vis = Image.open(vis_path).convert('RGB')
        image_ir  = Image.open(ir_path).convert('RGB')
        
        w_vi, h_vi = image_vis.size  # (width, height)
        w_ir, h_ir = image_ir.size
        new_w = max(16, (w_vi // 16) * 16)
        new_h = max(16, (h_vi // 16) * 16)
        if (w_vi != new_w) or (h_vi != new_h):
            image_vis = image_vis.resize((new_w, new_h), resample=Image.BICUBIC)
        if (w_ir != new_w) or (h_ir != new_h):
            image_ir = image_ir.resize((new_w, new_h), resample=Image.BICUBIC)
        
        image_vis = TF.to_tensor(image_vis).unsqueeze(0)  # [1, 3, H, W]
        image_ir  = TF.to_tensor(image_ir).unsqueeze(0)   # [1, 3, H, W]
            
        if self.split=='train':
            vis_ir = torch.cat([image_vis, image_ir],dim=1)
            
            vis_ir = randfilp(vis_ir)
            vis_ir = randrot(vis_ir)
            vis_ir = self.crop(vis_ir)
            
            vis, ir = torch.split(vis_ir, [3, 3], dim=1)
            
            return vis.squeeze(0), ir.squeeze(0), self.filenames_vis[index]
        else:
            return image_vis.squeeze(0), image_ir.squeeze(0), self.filenames_vis[index]

    def __len__(self):
        return min(len(self.filenames_vis), len(self.filenames_ir))
