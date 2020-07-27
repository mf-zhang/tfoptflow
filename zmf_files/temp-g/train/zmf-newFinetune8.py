#!/usr/bin/env python
# coding: utf-8

# PWC-Net-large model training (with cyclical learning rate schedule)
# =======================================================
# 
# In this notebook, we:
# - Use a PWC-Net-large model (with dense and residual connections), 6 level pyramid, uspample level 2 by 4 as the final flow prediction
# - Train the model on a mix of the `FlyingChairs` and `FlyingThings3DHalfRes` dataset using a Cyclic<sub>short</sub> schedule of our own
# - The Cyclic<sub>short</sub> schedule oscillates between `5e-04` and `1e-05` for 200,000 steps
# 
# Below, look for `TODO` references and customize this notebook based on your own machine setup.

# ## Reference
# 
# [2018a]<a name="2018a"></a> Sun et al. 2018. PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume. [[arXiv]](https://arxiv.org/abs/1709.02371) [[web]](http://research.nvidia.com/publication/2018-02_PWC-Net%3A-CNNs-for) [[PyTorch (Official)]](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) [[Caffe (Official)]](https://github.com/NVlabs/PWC-Net/tree/master/Caffe)

# In[1]:


"""
pwcnet_train.ipynb

PWC-Net model training.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Tensorboard:
    [win] tensorboard --logdir=E:\\repos\\tf-optflow\\tfoptflow\\pwcnet-lg-6-2-cyclic-chairsthingsmix
    [ubu] tensorboard --logdir=/media/EDrive/repos/tf-optflow/tfoptflow/pwcnet-lg-6-2-cyclic-chairsthingsmix
"""
from __future__ import absolute_import, division, print_function
import sys
from copy import deepcopy

from dataset_base import _DEFAULT_DS_TUNE_OPTIONS
from dataset_flyingchairs import FlyingChairsDataset
from dataset_flyingthings3d import FlyingThings3DHalfResDataset
from dataset_mixer import MixedDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_FINETUNE_OPTIONS


# ## TODO: Set this first!

# In[2]:


# TODO: You MUST set dataset_root to the correct path on your machine!
if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = '../../../data/dataset/'
_FLYINGCHAIRS_ROOT = _DATASET_ROOT + 'newGnoise'
# _FLYINGTHINGS3DHALFRES_ROOT = _DATASET_ROOT + 'FlyingThings3D_HalfRes'
    
# TODO: You MUST adjust the settings below based on the number of GPU(s) used for training
# Set controller device and devices
# A one-gpu setup would be something like controller='/device:GPU:0' and gpu_devices=['/device:GPU:0']
# Here, we use a dual-GPU setup, as shown below
gpu_devices = ['/device:GPU:0', '/device:GPU:1']
controller = '/device:CPU:0'

# TODO: You MUST adjust this setting below based on the amount of memory on your GPU(s)
# Batch size
batch_size = 8


# # Pre-train on `FlyingChairs+FlyingThings3DHalfRes` mix

# ## Load the dataset

# In[3]:


# TODO: You MUST set the batch size based on the capabilities of your GPU(s) 
#  Load train dataset
ds_opts = deepcopy(_DEFAULT_DS_TUNE_OPTIONS)
ds_opts['in_memory'] = False                          # Too many samples to keep in memory at once, so don't preload them
ds_opts['aug_type'] = 'heavy'                         # Apply all supported augmentations
ds_opts['batch_size'] = batch_size * len(gpu_devices) # Use a multiple of 8; here, 16 for dual-GPU mode (Titan X & 1080 Ti)
ds_opts['crop_preproc'] = (256,448)                  # Crop to a smaller input size
ds1 = FlyingChairsDataset(mode='train_with_val', ds_root=_FLYINGCHAIRS_ROOT, options=ds_opts)
ds_opts['type'] = 'into_future'
# ds2 = FlyingThings3DHalfResDataset(mode='train_with_val', ds_root=_FLYINGTHINGS3DHALFRES_ROOT, options=ds_opts)
ds = MixedDataset(mode='train_with_val', datasets=[ds1], options=ds_opts)


# In[4]:


# Display dataset configuration
ds.print_config()


# ## Configure the training

# In[5]:


# Start from the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_FINETUNE_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_dir'] = './zmf-newFinetune8/'
nn_opts['ckpt_path'] = './zmf-newFinetune7/pwcnet.ckpt-50000'
nn_opts['batch_size'] = ds_opts['batch_size']
nn_opts['x_shape'] = [2, ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 3]
nn_opts['y_shape'] = [ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 2]
nn_opts['use_tf_data'] = True # Use tf.data reader
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# Use the PWC-Net-large model in quarter-resolution mode
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# Robust loss as described doesn't work, so try the following:
nn_opts['loss_fn'] = 'loss_multiscale' # 'loss_multiscale' # 'loss_robust' # 'loss_robust'
nn_opts['q'] = 1. # 0.4 # 1. # 0.4 # 1.
nn_opts['epsilon'] = 0. # 0.01 # 0. # 0.01 # 0.


# In[6]:


# Set the learning rate schedule. This schedule is for a single GPU using a batch size of 8.
# Below,we adjust the schedule to the size of the batch and the number of GPUs.
nn_opts['lr_policy'] = 'cyclic'
nn_opts['cyclic_lr_max'] = 5e-04 # Anything higher will generate NaNs
nn_opts['cyclic_lr_base'] = 1e-05
nn_opts['cyclic_lr_stepsize'] = 20000
nn_opts['max_steps'] = 200000

# Below,we adjust the schedule to the size of the batch and our number of GPUs (2).
nn_opts['cyclic_lr_stepsize'] /= len(gpu_devices)
nn_opts['max_steps'] /= len(gpu_devices)
nn_opts['cyclic_lr_stepsize'] = int(nn_opts['cyclic_lr_stepsize'] / (float(ds_opts['batch_size']) / 8))
nn_opts['max_steps'] = int(nn_opts['max_steps'] / (float(ds_opts['batch_size']) / 8))


# In[7]:


# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_with_val', options=nn_opts, dataset=ds)
nn.print_config()


# ## Train the model

# In[8]:


# Train the model
nn.train()


# ## Training log

# Here are the training curves for the run above:
# 
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/loss.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/epe.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/lr.png)

# Here are the predictions issued by the model for a few validation samples:
# 
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val1.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val2.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val3.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val4.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val5.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val6.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val7.png)
# ![](img/pwcnet-lg-6-2-cyclic-chairsthingsmix/val8.png)
