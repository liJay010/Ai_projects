from PIL import Image
import os

import torch
import torch.nn.functional as F
from  torchvision.transforms import Compose,ToTensor,Resize

transforms = Compose([ToTensor()]) #图片的处理-变换成Tensor

alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" #定义72个字符表-区分大小写验证码

train_data_path = 'E:\\竞赛\\yzm\\my_capach\\data\\验证码图片内容识别竞赛数据\\train\\' #训练集路径

test_data_path = 'E:\\竞赛\\yzm\\my_capach\\data\\验证码图片内容识别竞赛数据\\test\\' #测试集路径

trian_label_path = r'E:\竞赛\my_capach\data\验证码图片内容识别竞赛数据\train_label.csv'


test_data_label = r'E:\竞赛\yzm\s.csv'

sub_path = r'E:\竞赛\my_capach\data\验证码图片内容识别竞赛数据\submission.csv'

batch_size = 128

lr = 0.0001

epochs = 100

weight_decay = 0.00005

model_path = './checkpoints/model.pth' #定义保存模型参数的路径

go_on = True  #是否继续上次的训练

best_acc_path = './checkpoints/best_modle_acc.txt' #最好的精度存储文件路径

device = torch.device('cpu') # 转移到cuda上训练

#以RGB的形式读取图片
def load_img(one_img_path):
    img = Image.open(one_img_path)
    return img.convert('RGB')

"""
做数据集
data_path：数据集路径
alphabet：字母表
num_class：总字符数
num_char：预测字符数
"""

"""def data_maker(data_path,label_path,alphabet,num_class,num_char):
    img_names = os.listdir(data_path) # 读入图片名字
    lis_data = [] #建立样本列表

    for img_name in img_names:
        img_path = os.path.join(data_path,img_name)
        target_str = img_name.split('.')[0]
        assert len(target_str) == num_char
        target = []

        'one-hot编码'
        for char in target_str:
            one_hot= [0] * num_class
            one_hot[alphabet.find(char)] = 1
            target += one_hot
        lis_data.append((img_path,target))
    return lis_data"""

def one_hot(target_str,num_class):
    target = []
    'one-hot编码'
    for char in target_str:
        one_hot= [0] * num_class
        one_hot[alphabet.find(char)] = 1
        target += one_hot

    return target


"""
计算精度并返回
"""
def acc_predict(logits,y):
    logits,y = logits.view(-1,62) ,y.view(-1,62)
    logits = F.softmax(logits, dim=1)
    logits = torch.argmax(logits,dim=1)
    y = torch.argmax(y,dim=1)
    #以4 个打包成一个
    output, target = logits.view(-1, 4), y.view(-1, 4)
    correct = []

    #计算正确的个数
    for i ,j in zip(target,output):
        if torch.equal(i,j):
            correct.append(1)
        else:
            correct.append(0)
    acc = sum(correct) / len(correct)

    return acc