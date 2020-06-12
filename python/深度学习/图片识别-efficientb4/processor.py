# -*- coding: utf-8 -*
from flyai.processor.base import Base
import os
from path import DATA_PATH
import numpy as np
'''
把样例项目中的processor.py件复制过来替换即可
'''
label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge',
 'cemetery', 'church', 'commercial-area', 'desert', 'dry-field', 'forest', 'golf-course', 
 'greenhouse', 'helipad', 'ice-land', 'island', 'lake', 'meadow', 'mine', 'mountain', 'oil-field', 
 'paddy-field', 'park', 'parking-lot', 'port', 'railway', 'residential-area', 'river', 'road', 
 'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant', 
 'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court', 'terraced-field', 
 'train-station', 'viaduct', 'wind-turbine', 'works']
label_num = len(label_list)
import  torch
from PIL import Image
import torchvision.transforms as transforms
test_process1 = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0),
            transforms.RandomVerticalFlip(0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
test_process2 = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0),
            transforms.RandomVerticalFlip(1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
test_process3 = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomVerticalFlip(0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
test_process4 = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomVerticalFlip(1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
class Processor(Base):
    def input_x(self, img_path):
        # 根据 img_path 读取图片
        img_path = os.path.join(DATA_PATH, img_path)
        return img_path

    def input_y(self, label):
        label_int=label_list.index(label)
        return label_int

    def output_x(self, img_path):
        # 根据 img_path 读取图片
        img_path = os.path.join(DATA_PATH, img_path)
        img = Image.open(img_path).convert('RGB')
        img = torch.cat([test_process1(img).unsqueeze(0),test_process2(img).unsqueeze(0),
                         test_process3(img).unsqueeze(0),test_process4(img).unsqueeze(0)],0).numpy()

        return img
    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, data):
        return  label_list[np.argmax(data)]
