import torch.nn as nn
from sklearn.metrics import f1_score
import argparse
import numpy as np
import torch
torch.manual_seed(123)
import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader,random_split
from utils import *
from net import Net,Resnet50v1
from net_revised import senet154
from datasets import Detection2
from transforms import RandomErasing
from flyai.utils import remote_helper
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset as Datahh
from sklearn.model_selection import train_test_split, StratifiedKFold,StratifiedShuffleSplit, KFold
from torch.autograd import Variable
from torchvision.models import shufflenet_v2_x1_0,resnet18
from torchvision import transforms as T
import visdom
import os
"""
保存模型参数，若无此文件，则创造路径
"""
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

viz = visdom.Visdom() #数据可视化
torch.manual_seed(123)
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=15, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=3e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()
#path = remote_helper.get_remote_date("https://www.flyai.com/m/resnet50-19c8e357.pth")
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
'''print('batch_size: %d, epoch_size: %d'%(args.BATCH, args.EPOCHS))
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)'''
nb_classes = 3

def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        # distribution of classes in the dataset
        label_to_count = {}
        for _,label in dataset:
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[idx]
                   for _,idx in dataset]
        print(weights)
        self.weights = torch.DoubleTensor(weights)
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


loder = Detection2(train=True)
train_size = int(0.8 * len(loder))
test_size = len(loder) - train_size
train_dataset, val_dataset = random_split(loder, [train_size, test_size])
train_dataset = loder
# 模型操作辅助类

'''
实现自己的网络机构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

device = torch.device(device)
#net = Net().to(device)
#net = Resnet50v1().to(device)
#实现自己的网络层
trained_model = shufflenet_v2_x1_0(pretrained=True)
net = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                  Flatten(),  # [b, 512, 1, 1] => [b, 512]
                    nn.Linear(81920, 3)
                  ).to(device)
# trained_model = resnet18(pretrained=True)
# net = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
#                   Flatten(),  # [b, 512, 1, 1] => [b, 512]
#                     nn.Linear(512, 3)
#                   ).to(device)
go_on = True
if go_on:
    #导入模型和最大精确度
    net.load_state_dict(torch.load(model_path))
    try:
        with open(best_acc_path,'r') as f:
            r = f.readline()
        best_acc = float(r)
    except:
        best_acc = 0
        print("no such a flie")
        pass
else:
    best_acc = 0


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

gamma = 0.99312  # 0.99312
warmup = 3e-6
best_prec1=0
iteration=0
loc_loss=0
best_prec1=0
#criterion = nn.CrossEntropyLoss().cuda()
criterion = CrossEntropyLabelSmooth(num_classes=nb_classes) .cuda()

optimizer = torch.optim.Adam(net.parameters(), args.lr,
                            weight_decay=args.weight_decay)


def main():
    best_prec1 = 0
    viz.line([0], [-1], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0], [-1], win='train_acc', opts=dict(title='train_acc'))
    viz.line([0], [-1], win='val_loss', opts=dict(title='val_loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    # train_loader = DataLoader(train_dataset, batch_size=args.BATCH, shuffle=True)  # ,shuffle=True
    train_loader=DataLoader(train_dataset, batch_size=args.BATCH, sampler=ImbalancedDatasetSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH)
    # for k,(img,  label) in enumerate(train_dataset):
    #     print(label) # label 为 int
    #     print(img.shape)
    for epoch in range(args.EPOCHS):
        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch)
        best_acc = validate(val_loader, net, criterion,epoch,best_acc)



def train(train_loader, model, criterion, optimizer, epoch):
    global iteration,loc_loss,top1
    # switch to train mode
    model.train()
    accs = []
    for i, (input, target) in enumerate(train_loader):
        iteration=iteration+1
        # adjust_learning_rate(optimizer, iteration)
        device = 'cuda'
        # measure data loading time
        # target = target.cuda(async=True)

        input = input.type(torch.FloatTensor)
        input_var = Variable(input).cuda()
        # compute output
        '''inputs, targets_a, targets_b, lam = mixup_data(
            input_var, target, 0.4, device)'''
        output = model(input_var)

        '''loss = mixup_criterion(
            criterion, output, torch.max(targets_a,1)[1], torch.max(targets_b,1)[1], lam)'''
        target = target.to(device)
        loss = criterion(output, target)
        viz.line([loss.item()], [iteration], win='train_loss', update='append')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loc_loss += loss.item()
        # measure elapsed time

        acc = acc_predict( output, target)  # 预测效果
        accs.append(acc)
        viz.line([loss.item()], [iteration], win='train_loss', update='append')
    viz.line([torch.mean(torch.Tensor(accs))], [epoch], win='train_acc', update='append')


def validate(val_loader, model, criterion,epoch,best_acc):
    model.eval()
    model.cuda()
    global val_s
    accs = []
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            #target = target.cuda(async=True)
            input = input.type(torch.FloatTensor)
            input = Variable(input).cuda()
            target = target.to(device)
            #input_var = torch.autograd.Variable(input, volatile=True).cuda()
            #target_var = torch.autograd.Variable(target, volatile=True).cuda()
            # compute output
            output = model(input)
            #loss = criterion(output, target.squeeze())
            try:
                loss = criterion(output, target)
                viz.line([loss.item()], [val_s], win='val_loss', update='append')
                val_s+=1
            except:
                pass

            acc = acc_predict(output, target)  # 预测效果
        accs.append(acc)
        viz.line([torch.mean(torch.Tensor(accs))], [epoch], win='val_acc', update='append')
        if best_acc < torch.mean(torch.Tensor(accs)) :
            best_acc = torch.mean(torch.Tensor(accs))
            torch.save(net.state_dict(), model_path)
            with open(best_acc_path,'w') as f:
                f.write(str(float(best_acc.cpu())))
            print(float(best_acc.cpu()))
            print("这是第{:}个epoch".format(epoch))
    try :
        return float(best_acc.cpu())
    except:
        return best_acc
# def adjust_learning_rate(optimizer, iteration):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if iteration <= 500:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 2.5*warmup * iteration
#     elif iteration > 500:
#         if iteration % 50 == 0:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = param_group['lr'] * gamma

if __name__ == '__main__':
    main()