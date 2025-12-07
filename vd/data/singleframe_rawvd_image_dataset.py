import torch
from torch.utils import data as data

from vd.data.data_util import *
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
    

@DATASET_REGISTRY.register()
class SingleFrameVDPairedN2NRawDataset(data.Dataset):
    """
    단일 Raw moired frame과 단일 clean sRGB 이미지를 쌍으로 처리하는 데이터셋 클래스.
    기존 Multi-Frame 구조를 Single-Frame으로 수정한 버전입니다.
    """
    def __init__(self, opt):
        super(SingleFrameVDPairedN2NRawDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        # train/val 구분 없이 단일 프레임 경로를 생성하는 함수를 사용합니다.
        self.paths = single_rawrgb_frame_from_folders([self.lq_folder, self.gt_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # 경로에서 키(key)와 파일 경로를 가져옵니다.
        key = self.paths[index]['key']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # GT(sRGB) 이미지와 LQ(Raw) 이미지를 로드합니다.
        img_gt_rgb = read_img(gt_path)
        img_lq_raw = read_npz(lq_path)

        # 학습(Train) 단계일 경우, 데이터 증강을 수행합니다.
        if self.opt['phase'] == 'train':
            # 단일 이미지 쌍에 대해 랜덤 크롭을 수행합니다.
            # paired_random_crop 함수는 이미지 리스트를 입력으로 받으므로, 리스트로 감싸줍니다.
            gt_size_imgs, raw_size_imgs = paired_random_crop([img_gt_rgb], [img_lq_raw], self.opt['crop_size'], 2)
            
            # 크롭된 이미지에 대해 좌우 반전(hflip) 및 회전(rot)을 적용합니다.
            # gt_size_imgs와 raw_size_imgs를 리스트로 결합
            imgs = [gt_size_imgs, raw_size_imgs]
            imgs = augment(imgs, self.opt['use_hflip'], self.opt['use_rot'])
            
            # numpy 배열을 텐서로 변환합니다. (BGR -> RGB, HWC -> CHW)
            # img2tensor 함수가 리스트를 반환한다고 가정하고 첫 번째 원소를 선택합니다.
            img_gt_rgb = img2tensor([imgs[0]], bgr2rgb=True, float32=True)[0]
            img_lq_raw = img2tensor([imgs[1]], bgr2rgb=True, float32=True)[0]
            
        # 검증(Validation) 단계일 경우, 텐서 변환만 수행합니다.
        else:
            img_gt_rgb = img2tensor([img_gt_rgb], bgr2rgb=True, float32=True)[0]
            img_lq_raw = img2tensor([img_lq_raw], bgr2rgb=True, float32=True)[0]

        # 반환 형태: {'lq': [C, H, W], 'gt': [C, H, W], 'key': str}
        # torch.stack을 사용하지 않아 시간 축(T)이 없습니다.
        return {'lq': img_lq_raw, 'gt': img_gt_rgb, 'key': key}

    def __len__(self):
        return len(self.paths)
