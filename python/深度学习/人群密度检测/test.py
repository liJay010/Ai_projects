import torch
import os
from torchvision import transforms
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from torch.utils.data import DataLoader
args = None

"""
E:\竞赛\城市\人群密度测试集A\A
"""
def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='E:\竞赛\城市\人群密度测试集A',
                        help='training data directory')
    parser.add_argument('--save-dir', default=r'E:\竞赛\Bayesian-Crowd-Counting',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'Test'), 512, 8, is_gray=False, method='val')
    dataloader = DataLoader(datasets, 1, shuffle=False,
                                             num_workers=4, pin_memory=False)
    # for (img,label),k in enumerate(dataloader):
    #     print(img.shape)
    model = vgg19()
    names = []
    co = []
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda:0')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    #model.load_state_dict('best_model.pth', device)
    epoch_minus = []
    k = 0
    n = 0
    n_name = []
    for inputs, count, name in dataloader:
        try:
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    outputs = model(inputs)
                    print(round(torch.sum(outputs).item()),name[0],"*****",k)
                    k +=1
                    names.append(name[0])
                    co.append(round(torch.sum(outputs).item()))

        except:
            n+=1
            n_name.append(name[0])
            print('*************************************************',n)


    print(n)
    # epoch_minus = np.array(epoch_minus)
    # mse = np.sqrt(np.mean(np.square(epoch_minus)))
    # mae = np.mean(np.abs(epoch_minus))
    # log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    # print(log_str)
    import pandas as pd
    d = pd.DataFrame()
    f = open('a.txt','w')
    for i in n_name:
        f.write(i)
    f.close()
    d['file'] = names
    d['man_count'] = co
    d.to_csv('sub.csv',index=None)
