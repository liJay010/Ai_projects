# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base

from path import MODEL_PATH
torch.manual_seed(10010)
torch.cuda.manual_seed(10010)
torch.cuda.manual_seed_all(10010)  # if you are using multi-GPU.
numpy.random.seed(10010)  # Numpy module.
torch.manual_seed(10010)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)
            print("loading model...")
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        outputs = self.net(x_data)
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        self.net = torch.load(self.net_path)
        print("loading model to predict...")
        self.net.eval()
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            # #flip the image
            # x1 = x_data
            # x2 = x_data[:, :, :, ::-1]
            # x3 = x_data[:, :, ::-1]
            # x4 = x2[:, :, ::-1]
            # x_data = numpy.concatenate((x1,x2,x3,x4),0)
            x_data = torch.from_numpy(x_data)
            x_data = x_data.float().to(self.device).squeeze(0)
            outputs0 = self.net(x_data[0].unsqueeze(0))
            outputs1 = self.net(x_data[1].unsqueeze(0))
            outputs2 = self.net(x_data[2].unsqueeze(0))
            outputs3 = self.net(x_data[3].unsqueeze(0))
            outputs = (outputs0+outputs1+outputs2+outputs3)/4.
            prediction = outputs.data.cpu().numpy()
            prediction = self.data.to_categorys(prediction)
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
