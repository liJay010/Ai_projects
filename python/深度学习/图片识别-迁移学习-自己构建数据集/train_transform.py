import  torch
from    torch import optim, nn
import  visdom
import  torchvision
from    torch.utils.data import DataLoader

from    mypokemon import Loader
# from    resnet import ResNet18
from    torchvision.models import resnet18

from    utils import Flatten

bachsize = 128
lr = 1e-3
epochs = 10
device = torch.device('cuda')
torch.manual_seed(0)

from mypokemon import Loader
loader = Loader()
train,val,test= loader.load()

train_loader = DataLoader(train,batch_size=bachsize,shuffle=True,num_workers=4)
test_loader = DataLoader(test,batch_size=bachsize,num_workers=2)
val_loader = DataLoader(val, batch_size=bachsize, num_workers=2)
viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def main():
    train_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(train_model.children())[:-1],
                          Flatten(),
                          nn.Linear(512,5)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss().to(device)


    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))


    for epoch in range(epochs):

        for step, (x,y) in enumerate(train_loader):

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc> best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append')


    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)

if __name__ == '__main__':
    main()