from __future__ import absolute_import, division, print_function
from copy import deepcopy
# from skimage.io import imread
from cv2 import imread
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import zmf_save_flows_c
import rawpy
import numpy as np
import glob,os,sys
import matplotlib.image as mp
import cv2 as cv

gpu_devices = ['/device:GPU:0']  
controller = '/device:GPU:0'

ckpt_path = '../../../data/pretrained_model/zmf_models/zmf-pwc-model/zmf-halfwhite/pwcnet.ckpt-50000'

img_pairs = []
names = []

canon_im1num = [1,1,2,10,10,11,21,21,21,21,22,22,22,24,25,31,31,31,32,32,33,41,43,51,54] # 25
canon_im2num = [2,3,3,11,12,12,22,24,25,26,23,24,27,26,27,32,33,34,33,34,34,42,45,52,55]
sony_im1num = [1,1,1,2,11,11,12,15] # 8
sony_im2num = [2,3,4,4,12,13,13,16]

for i in range(20,25):
    foldernum1 = canon_im1num[i]
    foldernum2 = canon_im2num[i]
    rawnum = len(glob.glob('/home/zhangmf/Documents/data/dataset/newOptFlow/canon/processed/scale/%d_*.jpg'%(foldernum1)))
    for j in range(rawnum):
        image_path1 = '/home/zhangmf/Documents/data/dataset/newOptFlow/canon/processed/scale/%d_%d.jpg'%(foldernum1,j+1)
        image_path2 = '/home/zhangmf/Documents/data/dataset/newOptFlow/canon/processed/scale/%d_%d.jpg'%(foldernum2,j+1)
        image1, image2 = imread(image_path1), imread(image_path2)
        image1 = image1/255.0
        image2 = image2/255.0

        img_pairs.append((image1, image2))
        n = '%d-%d_%d'%(foldernum1,foldernum2,j+1)
        names.append(n)



# Configure the model for inference, starting with the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# We're running the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 436, 1024, 2)

nn = ModelPWCNet(mode='test', options=nn_opts)
# nn.print_config()

pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
zmf_save_flows_c(names, img_pairs, pred_labels)
