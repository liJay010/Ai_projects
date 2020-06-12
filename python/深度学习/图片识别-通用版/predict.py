# -*- coding: utf-8 -*
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.autograd import Variable
from utils import *
import glob
import cv2
from torchvision.models import shufflenet_v2_x1_0,resnet18
subm = pd.DataFrame()
path_name = glob.glob(r'E:\竞赛\看图识车\深度学习新模型\car\test/*.jpg')

from torchvision import transforms as T
import torch
torch.manual_seed(123)
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
class Detection2(Dataset):
    """
    对数据的处理
    """
    def __init__(self):
        self.data = path_name
        self.transforms = T.Compose([
                T.Resize((240, 320)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), ])
    def __getitem__(self,index):
        data = Image.open(self.data[index]).convert("RGB")
        image_data = self.transforms(data)
        return image_data,self.data[index].split("\\")[-1]
    def __len__(self):
        return len(self.data)


def predict():
    dataset = Detection2()

    trained_model = shufflenet_v2_x1_0()
    cnn = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                  Flatten(),  # [b, 512, 1, 1] => [b, 512]
                    nn.Linear(81920, 3)
                  ).to(device)
    cnn.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        na = []
        pr = []
        i = 0
        for  input,name in dataset:
            input = input.type(torch.FloatTensor)
            input = input.unsqueeze(0)
            input = Variable(input).cuda()

            output = cnn(input)
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1).cpu()[0].numpy()
            na.append(name)
            pr.append(output)
            print(output,name,"--",i,"--",i / len(path_name))
            i+=1
    subm["filename"] = na
    subm["label"] = pr
    subm.to_csv("sub1.csv",index = None)




if __name__ == "__main__":
    predict()