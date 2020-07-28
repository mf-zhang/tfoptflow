import zmf

add_imclear = '../../../data/dataset/VBOF/myoptflow/YOMO-full/test/10001_img1.jpg'
add_imnoisy = '../../../data/dataset/VBOF/myoptflow/YOMO-full/test/10009_img1.jpg'

imclear = zmf.imread(add_imclear)/255.
# imnoisy = zmf.imread(add_imnoisy)/255.
noise = zmf.normal(0,0.1,imclear.shape)

print(zmf.im_psnr(imclear,imclear+noise))

zmf.imshow(imclear+noise)