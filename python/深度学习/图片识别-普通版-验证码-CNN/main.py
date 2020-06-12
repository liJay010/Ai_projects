"""
自适应二值化
"""
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from CNN import Net
from utils import *
from torch import optim
from datasets import GetData
import visdom
"""
保存模型参数，若无此文件，则创造路径
"""
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

viz = visdom.Visdom() #数据可视化
torch.manual_seed(123)
def main():
    """

    导入训练数据集，测试数据集，并处理
    :return:
    """

    train_data = GetData(test_data_path,label_path=test_data_label ,train=True,transform=transforms)
    train_data_loader= DataLoader(train_data,batch_size=64,shuffle=True,num_workers=4)

    test_data = GetData(test_data_path,label_path=test_data_label ,train=False, transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                  num_workers=0, shuffle=True)


    cnn = Net(62,4).to(device)

    """
    是否训练过，若训练过，则导入训练的模型
    """

    if go_on:
        #导入模型和最大精确度
        cnn.load_state_dict(torch.load(model_path))
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

    optimizer = optim.Adam(cnn.parameters(), lr=lr,weight_decay = weight_decay)
    criterion = nn.MultiLabelSoftMarginLoss().to(device)

    #做loss，acc的可视化

    global_step_train = 0
    viz.line([0], [-1], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0], [-1], win='train_acc', opts=dict(title='train_acc'))
    viz.line([0], [-1], win='test_acc', opts=dict(title='test_acc'))

    for epoch in range(epochs):

        all_loss = []
        accs = []

        cnn.train()
        for img ,target in train_data_loader:
            img, target = Variable(img), Variable(target)
            img,target = img.to(device),target.to(device)

            logits = cnn(img)

            loss = criterion(logits,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = acc_predict(logits,target) #预测效果
            accs.append(acc)
            all_loss.append(loss)

            viz.line([loss.item()], [global_step_train], win='train_loss', update='append')
            global_step_train += 1

        """print('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))"""
        viz.line([torch.mean(torch.Tensor(accs))], [epoch], win='train_acc', update='append')


        accs = []
        cnn.eval()
        for img, target in test_data_loader:
            img ,target= Variable(img),Variable(target)
            img,target = img.to(device),target.to(device)
            output = cnn(img)

            acc = acc_predict(output,target)
            accs.append(acc)
            all_loss.append(loss)


        viz.line([torch.mean(torch.Tensor(accs))], [epoch], win='test_acc', update='append')

        #if best_acc < torch.mean(torch.Tensor(accs)) or epoch == 10:
        if best_acc < torch.mean(torch.Tensor(accs)) :
            best_acc = torch.mean(torch.Tensor(accs))
            torch.save(cnn.state_dict(), model_path)
            with open(best_acc_path,'w') as f:
                f.write(str(float(best_acc.cpu())))
            print(float(best_acc.cpu()))
            print("这是第{:}个epoch".format(epoch))

if __name__ == '__main__':
    main()