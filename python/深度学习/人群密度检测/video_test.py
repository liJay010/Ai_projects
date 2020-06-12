import torch
import os
from torchvision import transforms
import numpy as np
from models.vgg import vgg19
import argparse
from torch.utils.data import DataLoader
args = None
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np

class Crowd(data.Dataset):
    def __init__(self, path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = 'E:\竞赛\城市\人群密度测试集A\Test'
        self.im_list = path

        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
#        self.d_ratio = downsample_ratio
#        assert self.c_size % self.d_ratio == 0
#        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                # transforms.Resize((512,512)),
                transforms.ToTensor(),

                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.im_list[item]+'.jpg')
        #gd_path = img_path.replace('jpg', 'npy')
        gd_path =img_path
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            #keypoints = np.load(gd_path)
            pass
            #return self.train_transform(img, keypoints)
        elif self.method == 'val':
            #keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            # return img, len(keypoints), name
            return img, 0, name


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(name, 512, 8, is_gray=False, method='val')
    dataloader = DataLoader(datasets, 1, shuffle=False,
                                             num_workers=4, pin_memory=False)
    # for (img,label),k in enumerate(dataloader):
    #     print(img.shape)
    model = vgg19()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    for inputs, count, name in dataloader:
        try:
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    outputs = model(inputs)
                    print(round(torch.sum(outputs).item()))
        except:
            pass

