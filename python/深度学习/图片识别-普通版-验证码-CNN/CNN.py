# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Base(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Base, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self,x):
        return self.base(x)
class Net(nn.Module):
    def __init__(self, num_size=62, char_num=4):
        super(Net, self).__init__()
        self.num_size = num_size
        self.char_num = char_num
        self.conv1 = Base(3,16)
        self.conv2 = Base(16,64)
        self.conv3 = Base(64,512)
        self.conv4 = Base(512,512)
        #打平后的线性层
        self.fc = nn.Linear(512 * 7 * 2, self.num_size * self.char_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 512 * 7 * 2)
        x = self.fc(x)
        return x


def main():
    x = torch.randn(64, 3, 120, 40)
    cnn = Net()
    y = cnn(x)
    print(y.shape)

if __name__ == '__main__':
    main()