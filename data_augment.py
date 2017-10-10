# -*- coding:utf-8 -*-
"""
 数据增强
    1. 翻转变换 flip
    2. 随机修剪 random crop
    3. 色彩抖动 color jittering
    4. 平移变换 shift
    5. 尺度变换 scale
    6. 对比度变换 contrast
    7. 噪声扰动 noise
    8. 旋转变换/反射变换 Rotation/reflection
    author: XiJun.Gong
    date:2016-11-29
"""
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
from scipy import misc
 
class DataAugmentation:
    """
    包含数据增强的八种方式
    """
    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
    
    @staticmethod
    def resizeImage(image, new_size=224):
        """
         对图像进行大小变换
        :param new_size 变换后图像的大小
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        
        return image.resize((new_size, new_size), Image.BILINEAR)

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(45, 135)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像

        """
        image_width = image.size[0]
        image_height = image.size[1]
        
        x0 = np.random.randint(0, 32)
        y0 = np.random.randint(0, 32)
        random_region = (x0, y0, x0+224, y0+224)
        return image.crop(random_region)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))
    
    @staticmethod
    def PCAJittering(img):    
        img = np.asanyarray(img, dtype = 'float32')  
          
        img = img / 255.0  
        img_size = img.size / 3  
        img1 = img.reshape(img_size, 3)  
        img1 = np.transpose(img1)  
        img_cov = np.cov([img1[0], img1[1], img1[2]])  
        lamda, p = np.linalg.eig(img_cov)  
          
        p = np.transpose(p)  
          
        alpha1 = random.normalvariate(0,3)  
        alpha2 = random.normalvariate(0,3)  
        alpha3 = random.normalvariate(0,3)  
          
        v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2]))      
        add_num = np.dot(p,v)  
          
        img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])  
          
        img2 = np.swapaxes(img2,0,2)  
        img2 = np.swapaxes(img2,0,1)
        return img2

    @staticmethod
    def saveImage(image, path):
        image.save(path)
 
DEBUG = False
if __name__ == '__main__':
	if DEBUG:
		img_path = "imgs/lean.jpg"
		img = DataAugmentation.openImage(img_path)
		new_img = DataAugmentation.resizeImage(img, 256)
		img = DataAugmentation.resizeImage(img, 224)
		rotate_img = DataAugmentation.randomRotation(img)
		crop_img = DataAugmentation.randomCrop(new_img)
		randcolor_img = DataAugmentation.randomColor(img)
		gaussian_img = DataAugmentation.randomGaussian(img)
		pca_img = DataAugmentation.PCAJittering(img)
		DataAugmentation.saveImage(img, 'imgs/new_img.jpg')
		DataAugmentation.saveImage(rotate_img, 'imgs/rorate_img.jpg')
		DataAugmentation.saveImage(crop_img, 'imgs/crop_img.jpg')
		DataAugmentation.saveImage(randcolor_img, 'imgs/randcolor_img.jpg')
		DataAugmentation.saveImage(gaussian_img, 'imgs/gaussian_img.jpg')
		misc.imsave('imgs/pca_img.jpg', pca_img)
	else:
		f = open('files/trainval.txt', 'r')
		filenames = f.readlines()
		f.close()
		
		names = []
		labels = []
		for row in filenames:
			row = row.strip('\n').split(' ')
			names.append(row[0])
			labels.append(row[1])
		
		funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian,
			   "PCAJittering": DataAugmentation.PCAJittering
               }
		
		count = 0
		fw = open('files/new_subject_train.txt', 'w')
		for idx in range(len(names)):
			img = DataAugmentation.openImage(os.path.join('data/imgs/subject_train',names[idx]))
			new_img = DataAugmentation.resizeImage(img, 256)
			img = DataAugmentation.resizeImage(img, 224)
			
			new_name = "data/imgs/new_subject_train/{0:0>7}.jpg".format(count)
			DataAugmentation.saveImage(img, new_name)
			fw.write(new_name + ' ' + labels[idx] + '\n')
			count += 1
			for op in funcMap.keys():
				if op == "randomCrop":
					new_img1 = funcMap[op](new_img)
				else:
					new_img1 = funcMap[op](img)
				
				new_name = "data/imgs/new_subject_train/{0:0>7}.jpg".format(count)
				fw.write(new_name + ' ' + labels[idx] + '\n')
				if op == "PCAJittering":
					misc.imsave(new_name, new_img1)
				else:
					DataAugmentation.saveImage(new_img1, new_name)
				
				count += 1
