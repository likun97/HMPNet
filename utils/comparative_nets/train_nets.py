# -*- coding: utf-8 -*-
"""
Created on Mon Oct 4 19:14:34 2021
@author: likun
"""

import os 
import time
import torch     
import torch.nn as nn
import torch.nn.functional as F   
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import scipy.io as sio  

from argparse import ArgumentParser
parser = ArgumentParser(description='Comparable-Nets')

parser.add_argument('--Network',       type=str,   default='Guided_DRCNN', help='from {HyperPNN, Guided_DRCNN , GSA_SSR, DRCNN, SSR}')
parser.add_argument('--Dataset',       type=str,   default='CAVE',  help='training dataset from {CAVE, Harvard, Chikusei_v2, GF_simu}')
parser.add_argument('--batch_size',    type=int,   default =8, help='trained or pre-trained model directory') 

parser.add_argument('--start_epoch',   type=int,   default =0,       help='epoch number of start training')
parser.add_argument('--end_epoch',     type=int,   default =551,    help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default =5*1e-4,  help='learning rate') 
parser.add_argument('--gpu_list',      type=str,   default ='0',     help='gpu index') 
parser.add_argument('--WEIGHT_DECAY',  type=float, default =1e-8,    help='params of ADAM') 
parser.add_argument('--model_dir',     type=str,   default ='train_models', help='trained or pre-trained model directory') 
parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')     
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
if args.Dataset =='CAVE':
    path      = '../../data/CAVE/version_1/'
    # path    = 'E:\\0 HyperModel\\1_dealed_data\\CAVE\\version_1\\'
    train_hsi = np.load(path+'CAVE_data_simu_train_cropH.npy')
    train_msi = np.load(path+'CAVE_data_simu_train_cropM.npy')
    train_pan = np.load(path+'CAVE_data_simu_train_cropP.npy')
    train_ref = np.load(path+'CAVE_data_simu_train_cropR.npy') 
elif args.Dataset =='Harvard':
    path      = '../../data/CAVE/version_1/'
    train_hsi = np.load(path+'Harvard_data_simu_train_cropH.npy')
    train_msi = np.load(path+'Harvard_data_simu_train_cropM.npy')
    train_pan = np.load(path+'Harvard_data_simu_train_cropP.npy')
    train_ref = np.load(path+'Harvard_data_simu_train_cropR.npy')   
# elif args.Dataset =='Harvard': # Chikusei_v2; GF5_real
else:
    raise NotImplementedError("Dataset [%s] is not recognized." % args.Dataset)
    
    
Name      = "Net_%s_Data_%s_Epoch_%d_" %(args.Network, args.Dataset, args.end_epoch)  #  
model_pth =  "./%s/" %(args.model_dir)+Name
log_dir   = "./%s/" %(args.model_dir)+Name+'.txt' 
if not os.path.exists(model_pth):
    os.makedirs(model_pth) 

import sys
sys.path.append(".") 
from models.HyperPNN  import HyperPNN 
from models.DRCNN   import Guided_DRCNN  
from models.GSA_SSR  import  GSA_SSRNET
# from models.DRCNN   import DRCNN


if args.Network == "HyperPNN":
    Train_Network = HyperPNN
elif args.Network == "Guided_DRCNN":
    Train_Network = Guided_DRCNN      
elif args.Network == "GSA_SSR":
    Train_Network = GSA_SSRNET 
else:
    raise NotImplementedError("Network [%s] model error occurs." % args.Network) 

patch_num  = train_hsi.shape[0]
# HP_fusion
hs_channel = train_hsi.shape[3]
hp_ratio   = train_pan.shape[2]//train_hsi.shape[2]
# HM_fusion
ms_channel = train_msi.shape[3]
hm_ratio   = train_msi.shape[2]//train_hsi.shape[2]

in_channels  = hs_channel
out_channels = hs_channel
mid_channels = hs_channel*2+2                    
n_select_bands, n_bands = ms_channel, hs_channel  

model = Train_Network(in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands)      
model = nn.DataParallel(model)
model = model.to(device)  

print        ("Train_Network is:",model)
model      = nn.DataParallel(model).to(device) 
optimizer  = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.WEIGHT_DECAY) # - SGD 
scheduler  = MultiStepLR(optimizer, milestones=[25,50,100], gamma=0.5)
criterion = nn.MSELoss().cuda()


class RandomDataset(Dataset):
    def __init__(self, hsi, msi, pan, ref, length):
        self.len   = length        
        self.hsi  = torch.tensor(hsi).permute(0, 3, 1, 2) 
        self.msi  = torch.tensor(msi).permute(0, 3, 1, 2)
        self.pan  = torch.tensor(pan).permute(0, 3, 1, 2)
        self.ref  = torch.tensor(ref).permute(0, 3, 1, 2) 
        
    def __getitem__(self, index):
        return self.hsi[index].float(), self.msi[index].float(), self.pan[index].float(), self.ref[index].float()
    def __len__(self):
        return self.len
random_loader = DataLoader(dataset     =  RandomDataset(train_hsi, train_msi, train_pan, train_ref, patch_num), 
                           batch_size  = args.batch_size, 
                           shuffle     = True, 
                           num_workers = 0
                           )
import skimage.measure
from torch.utils.tensorboard import SummaryWriter
print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
tb_writer = SummaryWriter(comment=args.Network+'_'+args.Dataset+args.name)

# Training loop
start = time.time()   
for epoch_i in range(args.start_epoch+1, args.end_epoch+1):
    PSNR, MSE, i = 0, 0, 0
    for data in random_loader:
        batch_h = data[0].to(device)
        batch_m = data[1].to(device)
        batch_p = data[2].to(device)
        batch_r = data[3].to(device)
        if args.Network == 'HyperPNN':
            batch_pred = model(batch_h, batch_m, batch_p)
            loss   = torch.mean(torch.pow(batch_r - batch_pred, 2))
        elif args.Network == 'Guided_DRCNN':
            H_res, H_ref, batch_pred = model(batch_h, batch_m, batch_p, batch_r)
            loss                     = 0.5*torch.mean(torch.pow(H_res - H_ref, 2))
        elif args.Network == 'GSA_SSR':
            batch_pred, _, _,  spat_edge1, spat_edge2, spec_edge = model(batch_h, batch_m, batch_p)
            ref_edge_spat1 = batch_r[:, :, 0:batch_r.size(2)-1, :] - batch_r[:, :, 1:batch_r.size(2), :]         
            ref_edge_spat2 = batch_r[:, :, :, 0:batch_r.size(3)-1] - batch_r[:, :, :, 1:batch_r.size(3)]
            ref_edge_spec  = batch_r[:, 0:batch_r.size(1)-1, :, :] - batch_r[:, 1:batch_r.size(1), :, :] 
            # https://github.com/hw2hwei/SSRNET/blob/master/train.py  
            loss_fus       = criterion(batch_pred, batch_r)
            loss_spec_edge = criterion(spec_edge , ref_edge_spec)
            loss_spat_edge = criterion(spat_edge1, ref_edge_spat1)*0.5+ criterion(spat_edge2, ref_edge_spat2)*0.5
            loss       = loss_fus + loss_spat_edge + loss_spec_edge
        else:
            raise NotImplementedError("Network [%s] model error occurs." % args.Network)
        # loss_all = loss + 0.3*loss_grad  
        loss_all = loss  

        psnr     = skimage.metrics.peak_signal_noise_ratio(batch_r.cpu().numpy(), batch_pred.cpu().data.numpy(), data_range=1 )        
        mse      = skimage.metrics.mean_squared_error     (batch_r.cpu().numpy(), batch_pred.cpu().data.numpy())    
        PSNR     = (PSNR*i+psnr)*1.0/(i+1)
        MSE      = (MSE*i+mse)*1.0/(i+1) 

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step() 
        scheduler.step()
    
    output_data = "[%02d/%02d] Total Loss: %.6f MSE %.6f PSNR %.3f\n" % (epoch_i, args.end_epoch, loss_all.item(), MSE,PSNR)
    print(output_data)

    output_file = open(log_dir, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 15 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_pth, epoch_i))  # save only the parameters
 
    # Write into tensorboard                                                    
    now_lr = optimizer.param_groups[0]["lr"]
    if tb_writer:
        tags = ['train/loss_all', "learning_rate", "PSNR", "MSE"]
        for x, tag in zip([loss_all, now_lr, PSNR, MSE], tags):
            tb_writer.add_scalar(tag, x, epoch_i) 

end  = time.time()
tim  = end - start  
print(tim)
curtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(curtime)

