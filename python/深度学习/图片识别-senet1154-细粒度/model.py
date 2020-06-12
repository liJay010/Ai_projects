# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
import cv2
from path import MODEL_PATH
from torchvision import transforms as T
from torch.autograd import Variable
import torch.nn.functional as F
__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        self.transforms = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), ])
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)


    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.transforms = T.Compose([
            T.Resize((28, 28)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), ])
        outputs = self.net(x_data)
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)

            #x_data = T.ToPILImage()(x_data).convert('RGB')
            #x_data=self.transforms(x_data)
            x_data = numpy.multiply(x_data, 1.0 / 255.0)  ## scale to [0,1] from [0,255]
            x_data = numpy.transpose(x_data, (0, 3, 1, 2))  ## reshape
            x_data = torch.from_numpy(x_data)
            x_data = x_data.float()
            x_data = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x_data.squeeze(0))
            x_data=x_data.unsqueeze(0)
            x_data = Variable(x_data).cuda()
            outputs = self.net(x_data)
            prediction = outputs.cpu().data.numpy()
            #print(prediction,'prediction1')
            prediction = self.data.to_categorys(prediction)
            #print(prediction, 'prediction2')

            labels.append(prediction)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))
