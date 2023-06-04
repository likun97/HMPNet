# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage  


def upsample_bicubic(image, ratio):
    h,w,c = image.shape
    re_image = cv2.resize(image, (w*ratio, h*ratio), cv2.INTER_CUBIC)
    return re_image


def upsample_interp23(image, ratio):
    image = np.transpose(image, (2, 0, 1))
    b,r,c = image.shape
    CDF23 = 2*np.array(
        [0.5, 0.305334091185, 
         0, -0.072698593239, 
         0, 0.021809577942, 
         0, -0.005192756653, 
         0, 0.000807762146, 
         0, -0.000060081482]
    )
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image = I1LRU
    re_image=np.transpose(I1LRU, (1, 2, 0))
    return re_image



def estimation_alpha(pan, hs, mode='global'):
    if mode == 'global':
        IHC = np.reshape(pan, (-1, 1))
        ILRC = np.reshape(hs, (hs.shape[0]*hs.shape[1], hs.shape[2]))
        
        alpha = np.linalg.lstsq(ILRC, IHC)[0]
        
    elif mode == 'local':
        patch_size = 32
        all_alpha = []
        print(pan.shape)
        for i in range(0, hs.shape[0]-patch_size, patch_size):
            for j in range(0, hs.shape[1]-patch_size, patch_size):
                patch_pan = pan[i:i+patch_size, j:j+patch_size, :]
                patch_hs = hs[i:i+patch_size, j:j+patch_size, :]
                
                IHC = np.reshape(patch_pan, (-1, 1))
                ILRC = np.reshape(patch_hs, (-1, hs.shape[2]))
                
                local_alpha = np.linalg.lstsq(ILRC, IHC)[0]
                all_alpha.append(local_alpha)
                
        all_alpha = np.array(all_alpha)
        
        alpha = np.mean(all_alpha, axis=0, keepdims=False)
        
    return alpha

# ---- GSA ----
"""
refer to https://github.com/codegaj/py_pansharpening/blob/master/methods/GSA.py
Created on Tue Oct  5 21:45:32 2021
Paper References:
    [1] B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan data,” 
        IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230-3239, October 2007.
    [2] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
""" 
# from utils import upsample_interp23
import numpy as np
import cv2


def GSA(pan, hs):
    
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    if ratio==3 or ratio==5:
        u_hs = upsample_bicubic(hs, ratio)
    else:
        u_hs = upsample_interp23(hs, ratio)
    #remove means from u_hs
    means = np.mean(u_hs, axis=(0, 1))
    image_lr = u_hs-means
    #remove means from hs
    image_lr_lp = hs-np.mean(hs, axis=(0,1))
    #sintetic intensity
    image_hr = pan-np.mean(pan)
    image_hr0 = cv2.resize(image_hr, (n, m), cv2.INTER_CUBIC)
    image_hr0 = np.expand_dims(image_hr0, -1)
    
    alpha = estimation_alpha(
        image_hr0, np.concatenate((image_lr_lp, np.ones((m, n, 1))), axis=-1), mode='global')
    I = np.dot(np.concatenate((image_lr, np.ones((M, N, 1))), axis=-1), alpha)
    I0 = I-np.mean(I)

    #computing coefficients
    g = []
    g.append(1)
    for i in range(C):
        temp_h = image_lr[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(c[0,1]/np.var(I0))
    g = np.array(g)
    #detail extraction
    delta = image_hr-I0
    deltam = np.tile(delta, (1, 1, C+1))
    #fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    g = np.tile(g, (M, N, 1))
    V_hat = V + g*deltam
    I_GSA = V_hat[:, :, 1:]
    I_GSA = I_GSA - np.mean(I_GSA, axis=(0, 1)) + means
    #adjustment
    I_GSA[I_GSA<0]=0
    I_GSA[I_GSA>1]=1
    # return np.uint8(I_GSA*255)
    return I_GSA



# ---- MTF_GLP ----
"""
https://github.com/codegaj/py_pansharpening/blob/master/methods/MTF_GLP.py

Paper References:
    [1] B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis,” 
        IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October 2002.
    [2] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,”
        Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
    [3] G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral image pansharpening,” 
        IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
    [4] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
"""

# from utils import upsample_interp23
from scipy import signal
import numpy as np 
import cv2
  
def kaiser2d(N, beta):
    
    t=np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w

def fir_filter_wind(Hd,w):
    """
	compute fir filter with window method
	Hd: 	desired freqeuncy response (2D)
	w: 		window (2D)
	"""
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    h=h/np.sum(h)
    
    return h

def MTF_GLP(pan, hs, sensor='gaussian'):
    
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))   
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    #equalization
    image_hr = np.tile(pan, (1, 1, C))
    image_hr = (image_hr - np.mean(image_hr, axis=(0,1)))*(np.std(u_hs, axis=(0, 1), ddof=1)/np.std(image_hr, axis=(0, 1), ddof=1))+np.mean(u_hs, axis=(0,1))

    pan_lp = np.zeros_like(u_hs)
    N =31
    fcut = 1/ratio
    match = 0
    if sensor == 'gaussian':
        sig = (1/(2*(2.772587)/ratio**2))**0.5
        kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9,sig).T)
        t=[]
        for i in range(C):
            temp = signal.convolve2d(image_hr[:, :, i], kernel, mode='same', boundary = 'wrap')
            temp = temp[0::ratio, 0::ratio]
            temp = np.expand_dims(temp, -1)
            t.append(temp)
        
        t = np.concatenate(t, axis=-1)
        pan_lp = upsample_interp23(t, ratio)
    
    elif sensor == None:
        match=1
        GNyq = 0.3*np.ones((C,))
    elif sensor=='QB':
        match=1
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        match=1           #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1':
        match=1             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR   
    elif sensor=='WV2':
        match=1            # MTF usage
        GNyq = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27]
    elif sensor=='WV3':
        match=1             #MTF usage
        GNyq = 0.29 * np.ones(8)
    
    if match==1:
        t = []
        for i in range(C):
            alpha = np.sqrt(N*(fcut/2)**2/(-2*np.log(GNyq)))
            H = np.multiply(cv2.getGaussianKernel(N, alpha[i]), cv2.getGaussianKernel(N, alpha[i]).T)
            HD = H/np.max(H)
            
            h = fir_filter_wind(HD, kaiser2d(N, 0.5))
            
            temp = signal.convolve2d(image_hr[:, :, i], np.real(h), mode='same', boundary = 'wrap')
            temp = temp[0::ratio, 0::ratio]
            temp = np.expand_dims(temp, -1)
            t.append(temp)
        
        t = np.concatenate(t, axis=-1)
        pan_lp = upsample_interp23(t, ratio)
        
    I_MTF_GLP = u_hs + image_hr - pan_lp        
    
    #adjustment
    I_MTF_GLP[I_MTF_GLP<0]=0
    I_MTF_GLP[I_MTF_GLP>1]=1
    
    return np.uint8(I_MTF_GLP*255)