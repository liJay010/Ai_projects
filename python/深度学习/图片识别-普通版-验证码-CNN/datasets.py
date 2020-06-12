from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *
import pandas as pd
import numpy as np

from utils import *
class GetData(Dataset):
    def __init__(self,data_path,num_class = 62,label_path=None,num_charm = 4,train = True,
                 transform=None, target_transform=None, alphabet=alphabet):
        super(GetData, self).__init__()
        #初始化参数
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_charm
        #得到样本
        #self.data = data_maker(self.data_path,self.alphabet,
                                   # self.num_class,self.num_char)
        data = pd.read_csv(label_path)
        self.data = np.array(data)

        if train==True:
            self.data = self.data
        elif train==False:
            self.data = self.data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_path,target = self.data[index]
        target = one_hot(target,62)
        target = torch.Tensor(target) #one_hot 编码转化为Tensor

        image = load_img(self.data_path+img_path)
        if self.transform is not  None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image,target

def main():
    data = GetData(train_data_path,label_path=trian_label_path,train=True,transform=transforms)
    db = DataLoader(data,batch_size=64,shuffle=True,num_workers=4)
    print(len(db.dataset))
    for img, target in db:
        print(img)




if __name__ == '__main__':
    main()