# -*- coding: utf-8 -*-
"""
Created by likun

@ title: HyperPNN: Hyperspectral Pansharpening via Spectrally Predictive Convolutional Neural Networks.
@ refer to: https://github.com/wgcban/DIP-HyperKite/blob/main/models/HyperPNN.py
"""

from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F 

### HyperPNN3 IMPLEMENTATION ###
class HyperPNN(nn.Module):    
    def __init__(self, in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands):
        super(HyperPNN, self).__init__() 
        self.in_channels    = in_channels
        self.mid_channels   = mid_channels
        self.out_channels   = out_channels
        self.hp_ratio       = hp_ratio

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,  out_channels=self.mid_channels, kernel_size=1) 
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels+1, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1) 
        self.conv4 = nn.Conv2d(in_channels=self.mid_channels,   out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.mid_channels,   out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
    
        self.conv6 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, batch_h, batch_m, batch_p):  
        
        X_HS  = batch_h
        X_PAN = batch_p
        X_HS_UP = F.interpolate(X_HS, scale_factor=(self.hp_ratio,self.hp_ratio),mode ='bilinear')
        x = F.relu(self.conv1(X_HS_UP))
        x = F.relu(self.conv2(x))
        x = torch.cat((x, X_PAN), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        output   = x
        return output
    
class HyperPNNNN(nn.Module):
    def __init__(self, config):
        super(HyperPNN, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]
        self.mid_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,  out_channels=self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels+1, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.mid_channels,   out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.mid_channels,   out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=1) 
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, X_HS, X_PAN):
        X_HS_UP = F.interpolate(X_HS, scale_factor=(self.factor,self.factor),mode ='bilinear')
        x       = F.relu(self.conv1(X_HS_UP))
        x_conv2 = F.relu(self.conv2(x))
        x = torch.cat((x_conv2, X_PAN.unsqueeze(1)), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = x+x_conv2 

        output = {  "pred": x}
        return output
