# -*- coding: utf-8 -*-
"""
Created by likun

@ title: Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening.
@ DOI: 10.1109/LGRS.2019.2945424
@ TF_version: https://github.com/yxzheng24/IEEE_GRSL_DRCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DRCNN(nn.Module):    
    def __init__(self, in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands):
        super(DRCNN, self).__init__()
        self.in_channels    = in_channels
        self.mid_channels   = mid_channels
        self.out_channels   = out_channels
        self.hp_ratio       = hp_ratio

        self.conv1 = nn.Sequential(
                      nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      )     
        self.conv2 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv3 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv4 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv5 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv6 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv7 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv8 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv9 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv10 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv11 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv12 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv13 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv14 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv15 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )     
        self.conv16 = nn.Conv2d(mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, batch_h, batch_m, batch_p):  
        X_HS  = batch_h
        X_PAN = batch_p  
        X_HS_UP = F.interpolate(X_HS, scale_factor=(self.hp_ratio,self.hp_ratio),mode ='bilinear')
        H_ini = X_HS_UP

        H_block1 = self.conv1(H_ini)
        H_block2 = self.conv2(H_block1)
        H_block2 = self.conv3(H_block2)
        H_block2 = self.conv4(H_block2)
        H_block2 = self.conv5(H_block2)
        H_block2 = self.conv6(H_block2)
        H_block2 = self.conv7(H_block2)
        H_block2 = self.conv8(H_block2)
        H_block2 = self.conv9(H_block2)
        H_block2 = self.conv10(H_block2)
        H_block2 = self.conv11(H_block2)
        H_block2 = self.conv12(H_block2)
        H_block2 = self.conv13(H_block2)
        H_block2 = self.conv14(H_block2)
        H_block2 = self.conv15(H_block2)
        H_res    = self.conv16(H_block2)

        # print(H_ini.size(), H_res.size())
        res = H_ini + H_res
        
        return res


""" Guided filter: https://github.com/haochange/imguidedfilter-opencv """
import cv2 
import sys
import numpy as np
sys.path.append("..")
from utils_.imguidedfilter import imguidedfilter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Guided_DRCNN(nn.Module):     
    def __init__(self, in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands):
        super(Guided_DRCNN, self).__init__()
        self.in_channels    = in_channels
        self.mid_channels   = mid_channels
        self.out_channels   = out_channels
        self.hp_ratio       = hp_ratio
        self.conv1 = nn.Sequential(
                      nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      )
        self.conv2 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv3 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv4 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv5 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv6 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv7 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv8 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv9 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv10 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv11 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv12 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv13 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv14 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv15 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv16 = nn.Conv2d(mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, batch_h, batch_m, batch_p, batch_r):  
        # 1-Enhancing Edge Details of the PAN Image With CLAHE: D_PAN = adapthisteq(I_PAN)        
        # 2-Generating the Initialized HSI via Guided Filter:mguidedfilter(I_HS2(:,:,j),D_PAN,...        
        batch_h_up = F.interpolate(batch_h, scale_factor=(self.hp_ratio,self.hp_ratio),mode ='bicubic')  
        batch_h_up, batch_p   = batch_h_up.permute(0, 2, 3, 1).cpu().numpy(), batch_p.permute(0, 2, 3, 1).cpu().numpy()

        # adapthisteq
        for i in range(batch_p.shape[0]):
            batch_p[i,:,:,0] = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(8, 8)).apply((batch_p[i,:,:,0]*255).astype('uint8'))
            batch_p[i,:,:,0] = np.float32(batch_p[i,:,:,0] - np.min(batch_p[i,:,:,0])) / (np.max(batch_p[i,:,:,0]) - np.min(batch_p[i,:,:,0]))
        # Guided Filter
        batch_Hini = batch_h_up
        for i in range(batch_h_up.shape[0]):
            for j in range(batch_h_up.shape[3]):
                batch_Hini[i,:,:,j] = imguidedfilter(batch_h_up[i,:,:,j], batch_p[i,:,:,0], (15,15), 0.000001)        
        H_ini = torch.from_numpy(batch_Hini).permute(0, 3, 1, 2).to(device) 
        H_ref = torch.sub(batch_r, H_ini)

        H_block1 = self.conv1(H_ini)
        H_block2 = self.conv2(H_block1)
        H_block2 = self.conv3(H_block2)
        H_block2 = self.conv4(H_block2)
        H_block2 = self.conv5(H_block2)
        H_block2 = self.conv6(H_block2)
        H_block2 = self.conv7(H_block2)
        H_block2 = self.conv8(H_block2)
        H_block2 = self.conv9(H_block2)
        H_block2 = self.conv10(H_block2)
        H_block2 = self.conv11(H_block2)
        H_block2 = self.conv12(H_block2)
        H_block2 = self.conv13(H_block2)
        H_block2 = self.conv14(H_block2)
        H_block2 = self.conv15(H_block2)
        H_res    = self.conv16(H_block2)
        fusion_out = H_ini + H_res
        return H_res, H_ref, fusion_out 
    
    

class Guided_DRCNN_Test(nn.Module):     
    def __init__(self, in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands):
        super(Guided_DRCNN_Test, self).__init__()
        self.in_channels    = in_channels
        self.mid_channels   = mid_channels
        self.out_channels   = out_channels
        self.hp_ratio       = hp_ratio
        
        self.conv1 = nn.Sequential(
                      nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      )
        self.conv2 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv3 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv4 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv5 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv6 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv7 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv8 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv9 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv10 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv11 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv12 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv13 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv14 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv15 = nn.Sequential(
                      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(self.mid_channels),
                      nn.ReLU(),
                      )
        self.conv16 = nn.Conv2d(mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
          
    def forward(self, batch_h, batch_m, batch_p):  
        batch_h_up = F.interpolate(batch_h, scale_factor=(self.hp_ratio,self.hp_ratio),mode ='bicubic')  
        batch_h_up, batch_p   = batch_h_up.permute(0, 2, 3, 1).cpu().numpy(), batch_p.permute(0, 2, 3, 1).cpu().numpy()
        # adapthisteq
        for i in range(batch_p.shape[0]):
            batch_p[i,:,:,0] = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(8, 8)).apply((batch_p[i,:,:,0]*255).astype('uint8'))
            batch_p[i,:,:,0] = np.float32(batch_p[i,:,:,0] - np.min(batch_p[i,:,:,0])) / (np.max(batch_p[i,:,:,0]) - np.min(batch_p[i,:,:,0]))
        # Guided Filter
        batch_Hini = batch_h_up
        
        for i in range(batch_h_up.shape[0]):
            for j in range(batch_h_up.shape[3]):
                batch_Hini[i,:,:,j] = imguidedfilter(batch_h_up[i,:,:,j], batch_p[i,:,:,0], (15,15), 0.000001)
        H_ini         = torch.from_numpy(batch_Hini).permute(0, 3, 1, 2).to(device)
        # print(H_ini.size())
        H_block1 = self.conv1(H_ini)
        H_block2 = self.conv2(H_block1)
        H_block2 = self.conv3(H_block2)
        H_block2 = self.conv4(H_block2)
        H_block2 = self.conv5(H_block2)
        H_block2 = self.conv6(H_block2)
        H_block2 = self.conv7(H_block2)
        H_block2 = self.conv8(H_block2)
        H_block2 = self.conv9(H_block2)
        H_block2 = self.conv10(H_block2)
        H_block2 = self.conv11(H_block2)
        H_block2 = self.conv12(H_block2)
        H_block2 = self.conv13(H_block2)
        H_block2 = self.conv14(H_block2)
        H_block2 = self.conv15(H_block2)
        H_res    = self.conv16(H_block2)
        fusion_out = H_ini + H_res
        return H_res, fusion_out
