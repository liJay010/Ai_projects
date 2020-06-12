# -*- coding: utf-8 -*-
import argparse
from flyai.dataset import Dataset
from model import Model
from net import *
from path import MODEL_PATH
import math
import random
import numpy as np
import torch
import torch.utils.data as Data
import os
import json
from PIL import Image
import torchvision.transforms as transforms
torch.manual_seed(10010)
torch.cuda.manual_seed(10010)
torch.cuda.manual_seed_all(10010)  # if you are using multi-GPU.
np.random.seed(10010)  # Numpy module.
random.seed(10010)  # Python random module.
torch.manual_seed(10010)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
class TrainDataset(Data.Dataset):
    def __init__(self,datas):
        #mode: train val test
        self.data_path = datas[0].tolist() + datas[2].tolist()
        print('the train dataset has %d images...' % (len(self.data_path)))
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gts = datas[1].tolist() + datas[3].tolist()
    def __getitem__(self, idx):
        img_pth =self.data_path[idx]

        image = Image.open(img_pth).convert('RGB')
        image = self.preprocess(image)
        gt    = self.gts[idx]
        return {
                'image':image,
                'cate':torch.from_numpy(np.array(gt))
            }
    def __len__(self):###how to calculate the length of the dataset
        return len(self.data_path)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=40, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
args.EPOCHS=40
args.BATCH=48

dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

all_datas=dataset.get_all_processor_data()

train_dataset = TrainDataset(all_datas)
train_loader = Data.DataLoader(dataset=train_dataset,shuffle=True,
                               batch_size=args.BATCH)
'''
实现自己的网络机构
'''

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

#init cnn model
net = GetPretrainedNet('efficientnet-b3')
net.to(device)

LR=3e-4

optimizer = AdamW(net.parameters(), lr=LR , weight_decay=1e-5, amsgrad=True)  # 选用AdamOptimizer
loss_fn =LabelSoothCrossEntropyLoss()  # 定义损失函数
scheduler = lr_sch(lr_start=LR, warm_up_iters=train_dataset.__len__() * 1 // args.BATCH,
                    warm_up_lr=1e-7,
                    end_iters=train_dataset.__len__()*(args.EPOCHS) // args.BATCH)

def eval_in_training(loader,net):
    net.eval()
    is_right=0
    y_sample=0
    for i, data in enumerate(loader):
        I = data['image'].cuda()
        gt = data['cate'].cuda()
        out=net(I)
        _, prediction = torch.max(out.data, 1)
        is_right += (prediction == gt).sum().item()
        y_sample += I.size(0)
    acc = float(is_right)/float(y_sample)
    return acc

correct=0
num_sample=0
best_accuracy = 0.
running_loss = 0.
iter=0
acc=0.01
for e in range(args.EPOCHS):
    net.train()
    for i, data in enumerate(train_loader):
        lr = scheduler(iter)
        set_lr(optimizer, lr)
        I = data['image'].cuda()
        gt = data['cate'].cuda()
        outputs = net(I)

        optimizer.zero_grad()
        loss = loss_fn(outputs, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iter+=1
        if i % 10 == 0:
            print('[%d, %d]   loss: %.5f    lr:%.7f'%
                  (e, i, 10 * running_loss/I.size(0),lr))
            running_loss = 0.
    model.save_model(net, MODEL_PATH, overwrite=True)
    print('**************************EPOCH  %d****************************' % (e+1))
    print('saving model...')

