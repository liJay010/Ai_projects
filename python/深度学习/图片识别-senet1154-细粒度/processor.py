# -*- coding: utf-8 -*

import numpy
from flyai.processor.base import Base
import os
import cv2
from flyai.processor.download import check_download
from torchvision import transforms as T
from path import DATA_PATH
import numpy as np
from PIL import Image
import torch
import scipy.misc
'''
把样例项目中的processor.py件复制过来替换即可
'''
label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery', 'church', 'commercial-area', 'desert', 'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake', 'meadow', 'mine', 'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway', 'residential-area', 'river', 'road', 'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant', 'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court', 'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']
label_num = len(label_list)
img_size = [256, 256]

class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def __init__(self):
        # get a set of unique text labels
        self.list_labels = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery', 'church', 'commercial-area', 'desert', 'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake', 'meadow', 'mine', 'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway', 'residential-area', 'river', 'road', 'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant', 'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court', 'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']
        # integer encode


        # one hot encode


        # Map integer value to text labels
        self.label_to_int = {k: v for v, k in enumerate(self.list_labels)}

        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}
    def convert_to_label(self, onehot):
        return np.argmax(onehot)
    def input_x(self, img_path):
        # 根据 img_path 读取图片
        #img_path = os.path.join(DATA_PATH, img_path)

        #img = Image.open(img_path).convert('RGB')
        #img = cv2.resize(img, (img_size[0], img_size[1]))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = scipy.misc.toimage(img)
        '''self.transforms = T.Compose([
            T.Resize((256,256)),
            T.CenterCrop((224,224)),#之后试试用center loss
            T.RandomRotation(90),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(45),
            T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            T.RandomGrayscale(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), ])
        image_data = self.transforms(img)'''
        path = check_download(img_path, DATA_PATH)
        path = path.replace('\\', '/')
        image = cv2.imread(path)
        #image = Image.open(path).convert('RGB')
        image = cv2.resize(image,(256,256))
        return image

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''
    def input_y(self, label):
        label_onehot = np.zeros((label_num), dtype=int)
        index = label_list.index(label)
        label_onehot[index] = 1
        return label_onehot

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, image_path):
        img_path = os.path.join(DATA_PATH, image_path)
        img_path = img_path.replace('\\', '/')
        img = cv2.imread(img_path)
        #img = Image.open(img_path).convert('RGB')
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image_data = self.transforms(img)
        return img

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, pred_onehot_label):
        # 将预测的 one-hot类型的输出转换成 具体类别
        '''pred_onehot_label = pred_onehot_label[0]
        pred_index = np.argmax(pred_onehot_label)
        pred_label = label_list[pred_index]'''
        label = self.convert_to_label(pred_onehot_label)
        y = self.int_to_label[label]
        return y
