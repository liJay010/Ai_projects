from CNN import Net
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.autograd import Variable
from utils import *
import glob
import cv2


subm = pd.read_csv(r'E:\竞赛\yzm\my_capach\data\验证码图片内容识别竞赛数据\submission.csv')

path_name = glob.glob(r'E:\竞赛\yzm\my_capach\data\验证码图片内容识别竞赛数据\test/*.jpg')
print(len(path_name[0]))
num = []
for i in path_name:
    num.append(eval(i[44:-4]))

data = pd.DataFrame({"path":path_name,"key":num})
data = data.sort_values(by=['key'])
path_name = (np.array(data.T)[0])
class GetData1(Dataset):
    def __init__(self,num_class = 62,num_charm = 4,
                 transform=None, alphabet=alphabet):
        super(GetData1, self).__init__()
        #初始化参数
        self.transform = transform
        self.alphabet = alphabet
        self.num_class = num_class
        self.num_char = num_charm
        self.data = path_name
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_path = self.data[index]

        image = load_img(img_path)
        if self.transform is not  None:
            image = self.transform(image)

        return image


def predict():
    dataset = GetData1(transform=transforms)

    cnn = Net().to(device)

    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))
    #cnn = torch.load('model7.pth', map_location=torch.device('cpu'))
    torch.save(cnn, 'model7.pth')
    sub = []
    for k, img in enumerate(dataset):
        img = img.unsqueeze(0)
        img = Variable(img)
        img = img.to(device)
        output = cnn(img)
        output= output.view(-1, 62)
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        # 以4 个打包成一个
        output= output.view(-1, 4)
        output = np.array(output.cpu())[0]
        result = ''
        for i in output:
            result = result+alphabet[i]
        print(result,k / len(path_name))
        sub.append(result)
    subm['label'] = sub

    subm.to_csv('result5.csv',index=None)





if __name__ == "__main__":
    predict()