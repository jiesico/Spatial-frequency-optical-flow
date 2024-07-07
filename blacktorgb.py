# import cv2
# import numpy as np
# import torch
# import os
import cv2
import numpy as np
import math
from PIL import Image
# import os
# import math
# import random
from glob import glob
import os.path as osp
# 方法2：公式换算各个像素点
def Pixel_rules(gray_pixel):
    if(gray_pixel >=0 and gray_pixel <=63):
        r = 125
        g = 254-4*gray_pixel
        b = 0
    elif(gray_pixel >=64 and gray_pixel <=127):
        r = 125
        g = 4*gray_pixel-254
        b = 0#400-4*gray_pixel
    elif(gray_pixel >=128 and gray_pixel <=191):
        r = 4*gray_pixel-510
        g = 255
        b = 0
    elif(gray_pixel >=192 and gray_pixel <=255):
        r = 125
        g = 1022-4*gray_pixel
        b = 0
    return [r, g, b]
# 方法2：公式换算各个像素点
def Pixel_rule(gray_pixel):
    # print(gray_pixel)
    if(gray_pixel >=0 and gray_pixel <=63):
        r = 255
        g = 254-4*gray_pixel
        b = 255
    elif(gray_pixel >=64 and gray_pixel <=127):
        r = 255
        g = 4*gray_pixel-254
        b = 255
    elif(gray_pixel >=128 and gray_pixel <=191):
        r = 255
        g = 4*gray_pixel-510
        b = 255
    elif(gray_pixel >=192 and gray_pixel <=255):
        r = 255
        g = 1022-4*gray_pixel
        b = 255
    return [r, g, b]

# 方法1：暴力换算,直接让r,g,b与单通道像素点相等
def Pixel_rule2(gray_pixel):
    r = gray_pixel
    g = gray_pixel
    b = gray_pixel
    return [r, g, b]

def Gray2RGB(img):
    gray = img   # 单通道灰度图读入
    W,H = gray.shape[:2]   # 保存原图像的宽度与高度

    d0 = img#np.array(Image.open(path1))  # 将原单通道图像转换成像素值
    # print(d0.shape, d0.dtype)
    # print(d0)


    dr = np.zeros([W, H])  # 与图大小相同的数组分别存r,g,b三个像素的像素值矩阵
    dg = np.zeros([W, H])
    db = np.zeros([W, H])
    three_Channel = np.zeros([W, H, 3])  # 定义三维数组存储新的三通道像素值


    for i in range(1, W - 1):
        for j in range(1, H-1):  # 遍历原灰度图的每个像素点
            [dr[i, j], dg[i, j], db[i, j]] = Pixel_rule(gray_pixel=d0[i, j])  # 将每个像素点的灰度值转换成rgb值(此处可以选择转换规则1或者2)
            three_Channel[i, j] = np.array([dr[i, j], dg[i, j], db[i, j]])

    # print(three_Channel.shape, three_Channel.dtype)
    # print(three_Channel)
    result = Image.fromarray(three_Channel.astype('uint8'))
    return result

# 读入路径
path1 = '/home/zhou/deep_learning_slam/RAFT/CamLiFlow-mainnew/flythings/setting3/186-5.9515.png'    #  path1路径改为文件夹上一层路径
path2 = '/home/zhou/186-5.9515.png' 
# path2 = "/home/zhou/file8T/raft/flowresults-KITTI/CRAFT-1-6.121.png"  # path2改为保存目录的根路径
# image_list = []
# image_list = sorted(glob(osp.join(path1, '*.jpg')))
# for i in range(len(image_list)-1):
img = cv2.imread(path1)
# img = cv2.imread(path1)
# print(img[:,:,0].shape)
# imgs = (img[:,:,0] + img[:,:,1] + img[:,:,2])/int(3)
imgs = img[:,:,0]
print(imgs.max())
# gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
result = Gray2RGB(img=imgs)
# 生成RGB图像保存路径
result.save(path2)
