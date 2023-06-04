# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:08:56 2021
@ author: Kun Li
@ refer to: https://github.com/hw2hwei/SSRNET
"""

import sys
sys.path.append("..")
from utils_.GSA_GLP import GSA

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GSA_SSRNET(nn.Module): 
    def __init__(self, in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands):
        super(GSA_SSRNET, self).__init__()
        # self.in_channels    = in_channels
        # self.mid_channels   = mid_channels
        # self.out_channels   = out_channels
        self.hp_ratio       = hp_ratio
        self.n_select_bands = n_select_bands
        self.n_bands        = n_bands   
        self.conv_fus = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spat = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spec = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr      = F.interpolate(x_lr, scale_factor=self.hp_ratio, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands-1.0)
        for i in range(0, self.n_select_bands-1):
            x_lr[:, int(gap_bands*i)   , ::] = x_hr[:, i                    , ::]
        x_lr    [:, int(self.n_bands-1), ::] = x_hr[:, (self.n_select_bands-1), ::]
        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2)-1, :            ] - x[:, :, 1:x.size(2), :          ]         
        edge2 = x[:, :, :            , 0:x.size(3)-1] - x[:, :,  :         , 1:x.size(3)]
        return edge1, edge2
    
    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]
        return edge
  
    def forward(self, batch_h, batch_m, batch_p):  
    
        batch_m, batch_p   = batch_m.permute(0, 2, 3, 1).cpu().numpy(), batch_p.permute(0, 2, 3, 1).cpu().numpy()
        batch_mp           = np.zeros([batch_p.shape[0],batch_p.shape[1],batch_p.shape[2],batch_m.shape[3]], dtype = float)
        # print(batch_mp.shape)
        for i in range(batch_mp.shape[0]):
            batch_mp[i,:,:,:]  = GSA(batch_p[i,:,:,:], batch_m[i,:,:,:])
        batch_mp         = torch.from_numpy(batch_mp).permute(0, 3, 1, 2)          
        x_lr, x_hr = batch_h, batch_mp
        x = self.lrhr_interpolate(x_lr, x_hr) # lrhr_interpolate()  
        x = self.conv_fus(x) # conv_fus()
        x_spat = x + self.conv_spat(x) # conv_spat()
        spat_edge1, spat_edge2 = self.spatial_edge(x_spat) # spatial_edge()
        x_spec = x_spat + self.conv_spec(x_spat) # conv_spec()
        spec_edge = self.spectral_edge(x_spec) # spectral_edge()

        x = x_spec
        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge # used in SSR-Net

