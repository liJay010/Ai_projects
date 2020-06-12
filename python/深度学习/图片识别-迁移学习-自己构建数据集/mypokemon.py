from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from  torchvision import datasets

class Loader():
    def __init__(self,size = [0.6,0.2,0.2],root =r'E:\搜狗高速下载\Flink_export_20190812\爬虫'):
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        db = datasets.ImageFolder(root=root,transform=tf)

        #数据集划分

        train_size = int(size[0]* len(db))
        val_size = int(size[1] * len(db))
        test_size = len(db) - train_size - val_size
        self.train_dataset, self.val_dataset,self.test_dataset =random_split(db, [train_size,val_size  ,test_size])

    def load(self ):
        return self.train_dataset, self.val_dataset,self.test_dataset


def main():
    loder  = Loader()
    a,b,c = loder.load()
    print(len(a))

if __name__ == '__main__':
    main()