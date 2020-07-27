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

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
# ckpt_path = '../../../data/pretrained_model/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

# ckpt_path = '../../../workplace/3-noise-model/final/zmf-origChair/pwcnet.ckpt-50000'
# ckpt_path = '../../../workplace/3-noise-model/final/zmf-try-my-raw-6-pwcnet-lg-6-2-cyclic/pwcnet.ckpt-50000'
# ckpt_path = '../../../workplace/3-noise-model/final/train-on-white-noise/pwcnet.ckpt-50000'
# ckpt_path = '../../../workplace/3-noise-model/zmf-finetune/pwcnet.ckpt-48000'

# ckpt_path = '../../../workplace/3-noise-model/mix/zmf-allmyppm/pwcnet.ckpt-50000'
ckpt_path = '../../../workplace/3-noise-model/mix/zmf-mixwhite/pwcnet.ckpt-50000'

img_pairs = []
names = []

# PNG
# image_path1 = '../../../workplace/1-prove-bad/eva/eva-base/my-simple/6_scale_img1.png'
# image_path2 = '../../../workplace/1-prove-bad/eva/eva-base/my-simple/6_scale_img2.png'
# image_path1 = '../../../data/dataset/newOptFlow/1/(1).JPG'
# image_path2 = '../../../data/dataset/newOptFlow/2/(1).JPG'
# image1, image2 = imread(image_path1), imread(image_path2)
# img_pairs.append((image1, image2))

# ARW
# canon_im1num = [1,1,2,10,10,11,21,21,21,21,22,22,22,24,25,31,31,31,32,32,33,41,43,51,54] # 25
# canon_im2num = [2,3,3,11,12,12,22,24,25,26,23,24,27,26,27,32,33,34,33,34,34,42,45,52,55]
# canon_im1num = [1,1] # 25
# canon_im2num = [2,3]
sony_im1num = [1,1,1,2,11,11,12,15] # 8
sony_im2num = [2,3,4,4,12,13,13,16]
# sony_im1num = [1,1] # 8
# sony_im2num = [2,3]
fuji_im1num = [1,1,1,2,2,3,5,6,11,11,11,12,12,12,14,14,15] # 17
fuji_im2num = [2,3,4,3,4,4,8,7,12,16,17,13,14,15,16,17,17]

def crop(im,h,w,sample_every_n_pixels):
    h0 = im.shape[0]
    w0 = im.shape[1]
    if (h0 < h or w0 < w):
        print("bad crop")
        return im
    newim = im[0:h:sample_every_n_pixels,0:w:sample_every_n_pixels,:]
    return newim

s = int(sys.argv[1])
for i in range(s,s+4):
    foldernum2 = sony_im1num[i]
    foldernum1 = sony_im2num[i]
    rawnum = len(glob.glob('../../../data/dataset/newOptFlow/sony/%d/*.ARW'%(foldernum1)))
    for j in range(rawnum):
        image_path1 = '../../../data/dataset/newOptFlow/sony/%d/(%d).ARW'%(foldernum1,j+1)
        image_path2 = '../../../data/dataset/newOptFlow/sony/%d/(%d).ARW'%(foldernum2,j+1)
        # image_path1 = '../../../data/dataset/newOptFlow/sony/1/(1).ARW'
        # image_path2 = '../../../data/dataset/newOptFlow/sony/2/(1).ARW'

        raw1 = rawpy.imread(image_path1)
        raw2 = rawpy.imread(image_path2)

        im1 = raw1.raw_image_visible.astype(np.float32)
        im1 = (im1 - 512) / (16383 - 512)
        ratio = 0.3 / np.mean(im1)
        im1 = np.minimum(np.maximum(im1*ratio,0.0),1.0)

        im2 = raw2.raw_image_visible.astype(np.float32)
        im2 = (im2 - 512) / (16383 - 512)
        ratio = 0.3 / np.mean(im2)
        im2 = np.minimum(np.maximum(im2*ratio,0.0),1.0)
        # print(ratio)

        im1 = np.expand_dims(im1, axis=2)
        H = im1.shape[0]
        W = im1.shape[1]
        image1 = np.concatenate((im1[0:H:2, 0:W:2,:], #r
                              (im1[0:H:2, 1:W:2,:]+im1[1:H:2, 0:W:2,:])/2.0, #g
                              im1[1:H:2, 1:W:2,:]), axis=2) #b
        im2 = np.expand_dims(im2, axis=2)
        H = im2.shape[0]
        W = im2.shape[1]
        image2 = np.concatenate((im2[0:H:2, 0:W:2,:], #r
                              (im2[0:H:2, 1:W:2,:]+im2[1:H:2, 0:W:2,:])/2.0, #g
                              im2[1:H:2, 1:W:2,:]), axis=2) #b
        image1 = crop(image1,1920,2944,2)
        image2 = crop(image2,1920,2944,2)

        # NLM
        # image1 = (image1*255).astype('uint8')
        # image2 = (image2*255).astype('uint8')

        # image1 = cv.fastNlMeansDenoisingColored(image1,None,10,10,7,21)
        # image2 = cv.fastNlMeansDenoisingColored(image2,None,10,10,7,21)

        # image1 = image1/255.0
        # image2 = image2/255.0


        img_pairs.append((image1, image2))
        n = '%d-%d_%d'%(foldernum1,foldernum2,j+1)
        names.append(n)

        # add1 = './finalresult/sony/%d_%d.jpg'%(foldernum1,j+1)
        # add2 = './finalresult/sony/%d_%d.jpg'%(foldernum2,j+1)
        # mp.imsave(add1,image1)
        # mp.imsave(add2,image2)



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
del img_pairs
del pred_labels