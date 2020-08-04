import zmf
import math
import numpy as np

addr_imclear = '../../../data/dataset/VBOF/myoptflow/YOMO-full/test/10001_img1.jpg'
addr_imnoisy = '../../../data/dataset/VBOF/myoptflow/YOMO-full/test/10009_img1.jpg'

# 0_Real: Noisy -> 60 62 65 67 69 70 71 73 361 -> clear
imclear = zmf.imread(addr_imclear)/255.
imnoisy = zmf.imread(addr_imnoisy)/255.

# 1_Normal: Noisy -> std = 0.3 - 58, 0.05 - 74 -> clear
std = 0.25
noise_Normal = zmf.normal(0,std,imclear.shape)

# 2_BIT: Noisy -> logK = 1.8 - 60, 0.6 - 70  -> clear
# 2_BIT_Poisson
logK = 1.8#zmf.uniform(0.6,1.8)
K = math.exp(logK)
noise_1 = (zmf.poisson(imclear*255.,imclear.shape)-imclear*255.) / 255. * K

# 2_BIT_TukeyLambda
lam = -0.26
mean = 0.
logscale = (5./6.) * logK + (0.6-5./6.*1.4) + zmf.uniform(-0.05,0.05)
scale = math.exp(logscale)
noise_2 = zmf.tukeylambda(lam,mean,scale/60.,imclear.shape)

# 2_BIT_ROW
noise_3 = np.zeros(imclear.shape)
for rgb in range(noise_3.shape[2]):
    for row in range(noise_3.shape[0]):
        logdev = 0.75 * logK - 2.2 + zmf.uniform(-0.375,0.375)
        dev = math.exp(logdev)
        row_shift = zmf.normal(0,dev)
        noise_3[row,:,rgb] = row_shift/15.

noise_BIT = noise_1 + noise_2 + noise_3

# 3_Mine
a = zmf.uniform(0.05,0.2)
b = zmf.uniform(0.05,0.2)
std = (a * imclear + b)
noise_Mine = zmf.normal(0,std,imclear.shape)

print(zmf.im_psnr(imclear,imclear+noise_BIT))

zmf.imshow(imclear+noise_BIT)