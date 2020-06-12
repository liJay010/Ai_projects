from torch.utils.data import DataLoader,random_split
from torchvision import transforms as T
from  torchvision import datasets
import torch
torch.manual_seed(123)
from torch.utils.data import DataLoader,random_split
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
class Detection2(Dataset):
    def __init__(self,test=False,train=True):
        self.data1 = pd.read_csv(r"E:\竞赛\看图识车\深度学习新模型\car\train.csv")
        self.data = np.array(self.data1['filename'].T)
        self.labels = np.array(self.data1['label'].T)
        self.path =r'E:\竞赛\看图识车\深度学习新模型\car\train'
        self.test = test
        self.train = train
        if self.test or not self.train:
            self.transforms = T.Compose([
                T.Resize((240, 320)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), ])
        else:
            print('hh')
            self.transforms = T.Compose([
                #T.Resize((28,28)),
                # T.RandomCrop((224,224)),#之后试试用center loss
                T.Resize((240, 320)),
                # T.RandomRotation(15),
                # T.RandomHorizontalFlip(),
                #T.RandomVerticalFlip(),
                #T.RandomAffine(45),
                #T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                #T.RandomGrayscale(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])#随机擦除
            ])
    def __getitem__(self,index):
        data = Image.open(os.path.join(self.path,self.data[index])).convert("RGB")
        label = self.labels[index]
        image_labels = label
        image_data = self.transforms(data)
        return image_data, image_labels
    def __len__(self):
        return len(self.labels)

class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.33, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def main():
    loder  = Detection2(train=True)
    train_size = int(0.8 * len(loder))
    test_size = len(loder) - train_size
    train_dataset, val_dataset = random_split(loder, [train_size, test_size])

    for i,j in loder:
        print(i.shape)
        print(j)
        # break

if __name__ == '__main__':
    main()