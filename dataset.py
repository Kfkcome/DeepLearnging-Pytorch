import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):  #继承Dataset
    def __init__(self, path_dir, transform=None):  #初始化一些属性
        self.path_dir = path_dir  #文件路径,如'.\data\cat-dog'
        self.transform = transform  #对图形进行处理，如标准化、截取、转换等
        self.images = os.listdir(self.path_dir)  #把路径下的所有文件放在一个列表中

    def __len__(self):  #返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  #根据索引 index返回图像及标签
        image_index = self.images[index]  #根据索引获取图像文件名称
        img_path = os.path.join(self.path_dir, image_index)  #获取图像的路径或目录
        img = Image.open(img_path).convert('RGB')

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.reshape(1, img.shape[0], img.shape[1])
        # img = np.concatenate([img, img, img], axis=0)

        # 根据目录名称获取图像标签（cat或dog）
        label = img_path.split('/')[-1].split('.')[0][0:6]
        # print(label)
        #把字符转换为数字noraml-1，potholes-0
        label = 1 if 'normal' in label else 0

        if self.transform is not None:
            img = self.transform(img)

        # img=torch.mean(img,dim=0)
        # img=img.reshape(-1)
        # # 将图片处理为单通道
        # def rgb2gray(rgb):
        #     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        # gray = rgb2gray(np.array(img))
        # gray_img = Image.fromarray(gray)

        return img, label




# dataset = MyDataset('.\OnlyRoad', transform=None)  #将启动魔法方法__getitem__(0)