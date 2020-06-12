from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from  torchvision.transforms import Compose,ToTensor,Resize
batch_size = 128
model_path = r'./checkpoints/model.pth' #定义保存模型参数的路径
go_on =True#是否继续上次的训练
best_acc_path = './checkpoints/best_modle_acc.txt' #最好的精度存储文件路径
device = torch.device('cuda')  #转移到cuda上训练
#以RGB的形式读取图片
def load_img(one_img_path):
    img = Image.open(one_img_path)
    return img.convert('RGB')

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

"""
计算精度并返回
"""
def acc_predict(logits,y):
    logits = F.softmax(logits, dim=1)
    logits = torch.argmax(logits,dim=1)
    #以4 个打包成一个
    correct = []
    #计算正确的个数
    for i ,j in zip(logits,y):
        if torch.equal(i,j):
            correct.append(1)
        else:
            correct.append(0)
    acc = sum(correct) / len(correct)

    return acc