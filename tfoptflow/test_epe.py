import numpy as np
import os,sys
# import torch
import glob
from cv2 import imread
from tqdm import tqdm

TAG_FLOAT = 202021.25

def EPE(input_flow, target_flow):
    # return torch.norm(target_flow-input_flow,p=2,dim=1).mean()
    return np.linalg.norm(target_flow-input_flow, ord=2, axis=1).mean()

def flow_read(file):
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)
    assert flo_number[0] == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)

    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow

def im2patch(im, pch_size, stride=1):
    import numpy as np
    import sys
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    
    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    import numpy as np
    if im.ndim == 3:
        # print(im.shape)
        im = im.transpose((2, 0, 1))
        # print(im.shape)
    else:
        print("see dim=1")
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)

all_tt_flo = glob.glob('./pwc-net/intensity/99/*.flo')
all_gt_flo = glob.glob('../data_all_nikon_half/*.flo')
all_tt_flo.sort()
all_gt_flo.sort()
# all_tt_flo = all_tt_flo[:400]
# all_gt_flo = all_gt_flo[:400]
assert(len(all_gt_flo)==len(all_tt_flo))

epelist = []
noiselist = []
for i in tqdm(range(len(all_gt_flo))):
    assert(os.path.basename(all_gt_flo[i]) == os.path.basename(all_tt_flo[i]))
    flow_tt = flow_read(all_tt_flo[i])
    flow_gt = flow_read(all_gt_flo[i])
    flow_tt = flow_tt / (np.abs(flow_tt).max())
    flow_gt = flow_gt / (np.abs(flow_gt).max())

    epe = EPE(flow_tt,flow_gt)
    epelist.append(epe)
    
    noisy_image_str = '../data_all_nikon_half/%s_img1.jpg'%os.path.basename(all_gt_flo[i])[0:10]
    noisy_image = imread(noisy_image_str)
    noise_level = noise_estimate(noisy_image)
    noiselist.append(noise_level)

combinelist = []
for i in range(len(epelist)):
    combinelist.append((noiselist[i],epelist[i]))

combinelist.sort()

list1 = []
for i in range(int(len(combinelist)/6*1)):
    list1.append(combinelist[i][1])
print(np.mean(list1))

list2 = []
for i in range(int(len(combinelist)/6*2)):
    list2.append(combinelist[i][1])
print(np.mean(list2))

list3 = []
for i in range(int(len(combinelist)/6*3)):
    list3.append(combinelist[i][1])
print(np.mean(list3))

list4 = []
for i in range(int(len(combinelist)/6*4)):
    list4.append(combinelist[i][1])
print(np.mean(list4))

list5 = []
for i in range(int(len(combinelist)/6*5)):
    list5.append(combinelist[i][1])
print(np.mean(list5))

list6 = []
for i in range(int(len(combinelist)/6*6)):
    list6.append(combinelist[i][1])
print(np.mean(list6))

print('& %.2f & %.2f & %.2f & %.2f & %.2f & %.2f'%(np.mean(list1),np.mean(list2),np.mean(list3),np.mean(list4),np.mean(list5),np.mean(list6)))
