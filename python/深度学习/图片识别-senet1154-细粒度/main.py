# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import scipy.misc
import torch.nn as nn
from sklearn.metrics import f1_score
import argparse
import numpy as np
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net,Resnet50v1
from net_revised import senet154
from path import MODEL_PATH
from transforms import RandomErasing
from flyai.utils import remote_helper
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset as Datahh
from sklearn.model_selection import train_test_split, StratifiedKFold,StratifiedShuffleSplit, KFold
from torch.autograd import Variable
from torchvision import transforms as T
'''
1.训练出多模型
2.test with augmentation
3.mix up (OK)  是不是要多个epoch才会有效果呢
4.改进网络
5.看看双fc可不可以用在这里鸭
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
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
nb_classes = 45

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
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

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
        for idx in self.indices:
            try:
                label = self._get_label(dataset, idx)
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
            except:
                pass

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        #print(dataset_type,'dataset_type')
        #print((dataset.labels[idx]), 'lable')
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return np.argmax(dataset.labels[idx])


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
class Detection2(Datahh):

    def __init__(self,data, labels,test=False,train=True):
        self.data = data
        self.labels = labels
        self.test = test
        self.train = train
        if self.test or not self.train:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), ])
        else:
            print('hh')
            self.transforms = T.Compose([
                #T.Resize((28,28)),
                T.RandomCrop((224,224)),#之后试试用center loss
                T.RandomRotation(90),
                T.RandomHorizontalFlip(),
                #T.RandomVerticalFlip(),
                #T.RandomAffine(45),
                #T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                #T.RandomGrayscale(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])

            ])
    def __getitem__(self,index):

        data = self.data[index]
        label = self.labels[index]
        data=T.ToPILImage()(data).convert('RGB')
        image_labels = label
        image_data = self.transforms(data)
        return image_data, torch.LongTensor(image_labels)
    def __len__(self):
        return len(self.labels)

data = Dataset()
# 模型操作辅助类
model = Model(data)
# 数据信息

#x_train, y_train, x_val, y_val = data.get_all_data()
x_train, y_train, x_val, y_val = data.get_all_processor_data()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
images = np.concatenate((x_train,x_val))
labels = np.concatenate((y_train,y_val))

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
net = senet154().to(device)

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
#net.load_state_dict(torch.load(path))
#net.load_state_dict(torch.load(path))
gamma = 0.99312  # 0.99312
warmup = 3e-6
best_prec1=0
iteration=0
loc_loss=0
best_prec1=0
#criterion = nn.CrossEntropyLoss().cuda()
criterion = CrossEntropyLabelSmooth(num_classes=45) .cuda()

optimizer = torch.optim.SGD(net.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

#x_train, x_val, y_train, y_val = train_test_split(images,labels,test_size=0.2, random_state=66)
#sss = StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=1024)

kf = KFold(n_splits=10,random_state=1122)
model_count = 0
is_inception = False
for train_index, test_index in kf.split(images,labels):
    print("Start model_%d training process......"%model_count)
    x_train,y_train, x_val, y_val = images[train_index],labels[train_index],images[test_index], labels[test_index]
    print("train img shape: ",x_train.shape,"val image shape: ",x_val.shape)
def main():
    best_prec1 = 0
    train_dataset = Detection2(x_train, y_train, train=True)
    val_dataset = Detection2(x_val, y_val, test=True)
    #train_loader = DataLoader(train_dataset, batch_size=args.BATCH, shuffle=True)  # ,shuffle=True
    train_loader=DataLoader(train_dataset, batch_size=args.BATCH, sampler=ImbalancedDatasetSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH)
    for epoch in range(0, args.EPOCHS):

        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch)
        prec1 = validate(val_loader, net, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        """save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)"""
        if (is_best):
            model.save_model(net, MODEL_PATH, overwrite=True)
            print('--------------- saved model !!!!')


def train(train_loader, model, criterion, optimizer, epoch):
    global iteration,loc_loss,top1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        iteration=iteration+1;
        adjust_learning_rate(optimizer, iteration)
        device = 'cuda'
        # measure data loading time
        target = target.cuda(async=True)

        input = input.type(torch.FloatTensor)
        input_var = Variable(input).cuda()
        # compute output
        '''inputs, targets_a, targets_b, lam = mixup_data(
            input_var, target, 0.4, device)'''
        output = model(input_var)

        '''loss = mixup_criterion(
            criterion, output, torch.max(targets_a,1)[1], torch.max(targets_b,1)[1], lam)'''

        loss = criterion(output, torch.max(target, 1)[1])
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data,  torch.max(target,1)[1], topk=(1, 1))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loc_loss += loss.item()
        # measure elapsed time


        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
 
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
            for param_group in optimizer.param_groups:
                print(param_group['lr'],'Ir:\t')


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    f1_s = AverageMeter()
    # switch to evaluate mode
    model.eval()
    model.cuda()

    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.type(torch.FloatTensor)
            input = Variable(input).cuda()
            #input_var = torch.autograd.Variable(input, volatile=True).cuda()
            #target_var = torch.autograd.Variable(target, volatile=True).cuda()
            # compute output
            output = model(input)
            #loss = criterion(output, target.squeeze())
            try:
                loss = criterion(output, torch.max(target, 1)[1])
            except:
                print(torch.max(target, 1)[1],'hh')

            prec1, prec5 = accuracy(output.data, torch.max(target,1)[1], topk=(1, 1))
            temp_val_pred = np.argmax(output.cpu().data.numpy(), axis=-1)
            #print(temp_val_pred, ' temp_val_pred')
            #print(torch.max(target, 1)[1], 'target')
            try:
                temp_val_f1 = f1_score(torch.max(target,1)[1].cpu(), temp_val_pred, average='macro')
            except:
                print(torch.max(target,1)[1].cpu(),'target')
                print(temp_val_pred, 'temp_val_pred')
            losses.update(loss.item(), input.size(0))
            f1_s.update(temp_val_f1, input.size(0))

            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            #temp_val_f1 = f1_score(output.data, torch.max(target,1)[1], average='macro')
            # measure elapsed time

            if i % args.print_freq == 0:
                print('Validate: [{0}/{1}]\t'
                     'F1_score {f1_s.val:.4f} ({f1_s.avg:.4f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader),  loss=losses,f1_s=f1_s,
                    top1=top1))
        print(' * validate:Total Prec@1 {top1.avg:.3f} '
              .format(top1=top1))
        print(' * validate:Total F1_score {f1_s.avg:.3f} '
              .format(f1_s=f1_s))

    return f1_s.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iteration <= 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 2.5*warmup * iteration
    elif iteration > 1500:
        if iteration % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * gamma

if __name__ == '__main__':
    main()