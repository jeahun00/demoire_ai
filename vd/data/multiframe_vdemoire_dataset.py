"""
VDemoire dataset 을 위한 dataloader
input: 3 sRGB frames
output: 1 sRGB frame
"""

import torch
from torch.utils import data as data

from vd.data.data_util import *
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
    
@DATASET_REGISTRY.register()
class MultiFrameVDemoireDataset(data.Dataset):
    def __init__(self, opt):
        super(MultiFrameVDemoireDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if self.opt['phase'] == 'train':
            self.paths = vdemoire_data_loader_train([self.lq_folder, self.gt_folder])
        else:
            self.paths = vdemoire_data_loader_test([self.lq_folder, self.gt_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.paths[index]['key']

        # Load gt image and alignratio.
        gt_path = self.paths[index]['gt_path']
        img_gt = read_img(gt_path)

        # Load lq images.
        lq_0_path = self.paths[index]['lq_0_path']
        img_lq_0 = read_img(lq_0_path)

        lq_1_path = self.paths[index]['lq_1_path']
        img_lq_1 = read_img(lq_1_path)

        lq_2_path = self.paths[index]['lq_2_path']
        img_lq_2 = read_img(lq_2_path)

        img_results = [img_gt, img_lq_0, img_lq_1, img_lq_2]

        if self.opt['phase'] == 'train':
            rgb_size_imgs, raw_size_imgs = paired_random_crop(img_results[:1], img_results[1:4], self.opt['crop_size'], 1)
            imgs = [rgb_size_imgs, raw_size_imgs[0], raw_size_imgs[1], raw_size_imgs[2]]
            imgs = augment(imgs, self.opt['use_hflip'], self.opt['use_rot'])
                
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt = img2tensor(imgs[0], bgr2rgb=True, float32=True)
            img_lq_0, img_lq_1, img_lq_2 = img2tensor(imgs[1:4], bgr2rgb=True, float32=True)
            
        else:
            img_gt = img2tensor(img_results[0], bgr2rgb=True, float32=True)
            img_lq_0, img_lq_1, img_lq_2 = img2tensor(img_results[1:4], bgr2rgb=True, float32=True) 
        
        img_lqs = torch.stack((img_lq_0, img_lq_1, img_lq_2), dim=0)

        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.paths)