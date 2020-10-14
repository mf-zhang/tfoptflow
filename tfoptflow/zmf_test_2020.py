from __future__ import absolute_import, division, print_function
from copy import deepcopy
# from skimage.io import imread
from cv2 import imread
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import zmf_save_flows
import rawpy
import numpy as np
import glob,os,sys
import matplotlib.image as mp
import cv2 as cv

gpu_devices = ['/device:GPU:0']  
controller = '/device:GPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
# au
# ckpt_path1 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/FC/au/pwcnet.ckpt-595000'
ckpt_path1 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/FC/me/pwcnet.ckpt-50000'
# cvpr
ckpt_path2 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/DFC/cvpr_halfwhite/pwcnet.ckpt-50000'
# gauss
ckpt_path3 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/DFC/gauss020/pwcnet.ckpt-50000'
# real
ckpt_path4 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/Real/zmf-scratch/pwcnet.ckpt-43000'
# finetune
ckpt_path5 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/Finetune/FCau_VBOF/pwcnet.ckpt-46000'
# mix
ckpt_path6 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/Mix/FC_Real/pwcnet.ckpt-48000'
# temp
ckpt_path7 = '/home/zhangmf/Documents/data/checkpoints/Optical_Flow/PWCNet/Mix/DFC_halfReal_long/pwcnet.ckpt-490000'

ckpt_path = ckpt_path7
# save_addr = '../realcar/mix_dfc_halfreal_allcameras/' # don't worry if you didn't create it yet
save_addr = '../intensity/allmight_people/'

img_pairs = []
names = []

# PNG
# image_path1 = '../../../workplace/1-prove-bad/eva/eva-base/my-simple/6_scale_img1.png'
# image_path2 = '../../../workplace/1-prove-bad/eva/eva-base/my-simple/6_scale_img2.png'
# image_path1 = '../../../data/dataset/newOptFlow/1/(1).JPG'
# image_path2 = '../../../data/dataset/newOptFlow/2/(1).JPG'
# image1, image2 = imread(image_path1), imread(image_path2)
# img_pairs.append((image1, image2))

# TODO 
# allimg1 = glob.glob('../../../data_all_nikon_half_99/*_img1.jpg')
# allimg2 = glob.glob('../../../data_all_nikon_half_99/*_img2.jpg')
# allimg1 = glob.glob('../../../YOMO-half/fig1/*_img1.jpg')
# allimg2 = glob.glob('../../../YOMO-half/fig1/*_img2.jpg')
allimg1 = glob.glob('../../../realpeople_data_half/*_img1.jpg')
allimg2 = glob.glob('../../../realpeople_data_half/*_img2.jpg')
allimg1.sort()
allimg2.sort()

length = len(allimg1)
if length > 1000: length = 1000
for i in range(length):
    image_path1 = allimg1[i]
    image_path2 = allimg2[i]

    image1, image2 = imread(image_path1), imread(image_path2)
    img_pairs.append((image1, image2))
    bname = os.path.basename(image_path1)
    bnum = bname[0:10]
    n = '%s_flow.flo'%bnum # 4100102001_img2
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

pred_labels = nn.predict_from_img_pairs(img_pairs[0:50], batch_size=1, verbose=False)
zmf_save_flows(names[0:50], pred_labels, save_addr)
print(1)
pred_labels = nn.predict_from_img_pairs(img_pairs[50:100], batch_size=1, verbose=False)
zmf_save_flows(names[50:100], pred_labels, save_addr)
print(2)
pred_labels = nn.predict_from_img_pairs(img_pairs[100:150], batch_size=1, verbose=False)
zmf_save_flows(names[100:150], pred_labels, save_addr)
print(3)
pred_labels = nn.predict_from_img_pairs(img_pairs[150:200], batch_size=1, verbose=False)
zmf_save_flows(names[150:200], pred_labels, save_addr)
print(4)
pred_labels = nn.predict_from_img_pairs(img_pairs[200:250], batch_size=1, verbose=False)
zmf_save_flows(names[200:250], pred_labels, save_addr)
print(5)
pred_labels = nn.predict_from_img_pairs(img_pairs[250:300], batch_size=1, verbose=False)
zmf_save_flows(names[250:300], pred_labels, save_addr)
print(6)
pred_labels = nn.predict_from_img_pairs(img_pairs[300:350], batch_size=1, verbose=False)
zmf_save_flows(names[300:350], pred_labels, save_addr)
print(7)
pred_labels = nn.predict_from_img_pairs(img_pairs[350:400], batch_size=1, verbose=False)
zmf_save_flows(names[350:400], pred_labels, save_addr)
print(8)
pred_labels = nn.predict_from_img_pairs(img_pairs[400:450], batch_size=1, verbose=False)
zmf_save_flows(names[400:450], pred_labels, save_addr)
print(9)
pred_labels = nn.predict_from_img_pairs(img_pairs[450:500], batch_size=1, verbose=False)
zmf_save_flows(names[450:500], pred_labels, save_addr)
print(10)
pred_labels = nn.predict_from_img_pairs(img_pairs[500:550], batch_size=1, verbose=False)
zmf_save_flows(names[500:550], pred_labels, save_addr)
print(11)
pred_labels = nn.predict_from_img_pairs(img_pairs[550:600], batch_size=1, verbose=False)
zmf_save_flows(names[550:600], pred_labels, save_addr)
print(12)
pred_labels = nn.predict_from_img_pairs(img_pairs[600:650], batch_size=1, verbose=False)
zmf_save_flows(names[600:650], pred_labels, save_addr)
print(13)
pred_labels = nn.predict_from_img_pairs(img_pairs[650:700], batch_size=1, verbose=False)
zmf_save_flows(names[650:700], pred_labels, save_addr)
print(14)
pred_labels = nn.predict_from_img_pairs(img_pairs[700:750], batch_size=1, verbose=False)
zmf_save_flows(names[700:750], pred_labels, save_addr)
print(15)
pred_labels = nn.predict_from_img_pairs(img_pairs[750:800], batch_size=1, verbose=False)
zmf_save_flows(names[750:800], pred_labels, save_addr)
print(16)
pred_labels = nn.predict_from_img_pairs(img_pairs[800:850], batch_size=1, verbose=False)
zmf_save_flows(names[800:850], pred_labels, save_addr)
print(17)
pred_labels = nn.predict_from_img_pairs(img_pairs[850:900], batch_size=1, verbose=False)
zmf_save_flows(names[850:900], pred_labels, save_addr)
print(18)
pred_labels = nn.predict_from_img_pairs(img_pairs[900:950], batch_size=1, verbose=False)
zmf_save_flows(names[900:950], pred_labels, save_addr)
print(19)
pred_labels = nn.predict_from_img_pairs(img_pairs[950:1000], batch_size=1, verbose=False)
zmf_save_flows(names[950:1000], pred_labels, save_addr)
print(20)
