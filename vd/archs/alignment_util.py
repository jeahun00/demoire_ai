import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from vd.utils.test_visualize import update_out_vis, store_vis_data




def backwarp(x, flow, objBackwarpcache):
    # x: [B, C, H, W]
    # flow: [B, 2, H, W]
    if 'grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3]) not in objBackwarpcache:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=flow.shape[3], dtype=flow.dtype,
                                device=flow.device).view(1, 1, 1, -1).repeat(1, 1, flow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=flow.shape[2], dtype=flow.dtype,
                                device=flow.device).view(1, 1, -1, 1).repeat(1, 1, 1, flow.shape[3])

        objBackwarpcache['grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3])] = torch.cat([tenHor, tenVer], 1)

    if flow.shape[3] == flow.shape[2]:
        flow = flow * (2.0 / ((flow.shape[3] and flow.shape[2]) - 1.0))

    elif flow.shape[3] != flow.shape[2]:
        flow = flow * torch.tensor(data=[2.0 / (flow.shape[3] - 1.0), 2.0 / (flow.shape[2] - 1.0)], dtype=flow.dtype, device=flow.device).view(1, 2, 1, 1)

    return nn.functional.grid_sample(input=x, grid=(objBackwarpcache['grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3])] + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)



class MultiFlowBWarp(torch.nn.Module):
    def __init__(self, dim, num_seq, num_flow):
        super(MultiFlowBWarp, self).__init__()
        self.dim = dim
        self.num_seq = num_seq
        self.num_flow = num_flow
        self.objBackwarpcache = {}
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, F, f):
        # F: [B, C*num_flow, T, H, W]
        # f: [B, 3*num_flow, T, H, W]

        F = rearrange(F, 'b (n c) t h w -> (b n t) c h w', c=self.dim//self.num_flow, n=self.num_flow)    # [B*num_flow*T, C//num_flow, H, W]
        f = rearrange(f, 'b (n c) t h w -> (b n t) c h w', c=3, n=self.num_flow)    # [B*num_flow*T, 2+1, H, W]

        weight = f[:, 2:3, :, :]  # [b, 1, h, w]
        flow = f[:, :2, :, :]  # [b, 2, h, w]

        weight = self.sigmoid(weight)

        F = backwarp(F, flow, self.objBackwarpcache)
        F = F * weight

        F = rearrange(F, '(b n t) c h w -> b (n c) t h w', t=self.num_seq, n=self.num_flow)    # [B, C, T, H, W]

        return F